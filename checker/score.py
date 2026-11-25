"""Scorer of rail-standard-shift-bench (rules fixed in PREREGISTRATION.md before the first model run).

Usage (from the repository root):
  python checker/score.py --raw <dir with model outputs> --out my_scores.json
  python checker/score.py --compiled results/scores_p3.json --out rescored.json     # re-score the published run

Raw layout: <raw>/<model_tag>/<variant>.jsonl ; rows {"item_id", "output"} for one answer per item,
or {"item_id", "outputs": [5 strings]} for the self-consistency variant (file name V1SC.jsonl).

Per item the model's condition is compiled to one limit value for the project's cell; the six fixed candidate
design values are then judged with that limit.
  primary analysis:      an invalid or abstaining answer is a hold ('보류', routed to the designer);
                         unsafe pass = the model's condition lets a candidate pass ('적합') that the gold condition rejects ('위반').
  sensitivity analysis:  the invalid or abstaining answer is treated as an omission: for C items the rule is
                         unconstrained (every candidate passes); for A and T items the current official value is used.
  self-consistency:      the five samples are compiled; the modal value is used if at least 4 of 5 agree, otherwise hold.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import railcheck as C  # noqa: E402

try:  # Korean labels in the output: make sure a Windows console does not fail on them
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
ROOT = Path(__file__).resolve().parents[1]
SC_N, SC_MIN = 5, 4
BOOT_N, SEED = 2000, 20261121
MAX_ABSTAIN = 0.20          # above this item-level hold rate no conclusion is drawn
THRESH = 0.05               # threshold of the fixed wording (a decision threshold, not a target)
PRIMARY_MODEL, PRIMARY_VARIANTS = "qwen3_8b", ("V1", "V1SC")


def read_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_gold(path=None):
    gold = {it["item_id"]: it for it in read_jsonl(path or ROOT / "data" / "gold.jsonl")}
    off = json.load(open(ROOT / "data" / "official_values.json", encoding="utf-8"))["values"]
    for it in gold.values():
        it["official_value"] = off[f'{it["rule"]}|{it["V"]}|{it["track"]}']
    return gold


def compile_text(it, text):
    return C.compile_condition(it, C.extract_json(text))


def decide(it, raw_row, variant):
    """-> (status, value, reason)"""
    if raw_row is None:
        return ("missing", None, "no_output")
    if variant == "V1SC":
        outs = raw_row.get("outputs") or []
        comp = [compile_text(it, o) for o in outs[:SC_N]]
        vals = [round(v, 6) for s, v, _ in comp if s == "ok"]
        if not vals:
            return ("abstain", None, "sc_no_valid")
        val, cnt = collections.Counter(vals).most_common(1)[0]
        if cnt >= SC_MIN:
            return ("ok", val, f"sc_{cnt}of{len(outs)}")
        return ("abstain", None, f"sc_agree_{cnt}of{len(outs)}")
    return compile_text(it, raw_row.get("output"))


def item_result(it, status, value):
    rows = []
    for c in it["candidates"]:
        mv = C.verdict(it["dir"], c["x"], value) if status == "ok" else C.HOLD
        if mv != C.HOLD:
            sens = mv
        elif it["set"] == "C":
            sens = C.COMPLY
        else:
            sens = C.verdict(it["dir"], c["x"], it["official_value"])
        rows.append({"x": c["x"], "gold": c["gold"], "model": mv, "model_sens": sens})
    return rows


def sc_curve(gold, rows):
    """Self-consistency variant: primary metrics for every agreement threshold tau = 1..5
    (tau = 4 is the pre-registered operating point)."""
    out = {}
    for tau in range(1, SC_N + 1):
        res = []
        for iid, it in gold.items():
            outs = (rows.get(iid) or {}).get("outputs") or []
            vals = [round(v, 6) for s, v, _ in (compile_text(it, o) for o in outs[:SC_N]) if s == "ok"]
            st, val = "abstain", None
            if vals:
                v, cnt = collections.Counter(vals).most_common(1)[0]
                if cnt >= tau:
                    st, val = "ok", v
            res.append({"item_id": iid, "rule": it["rule"], "set": it["set"], "status": st, "value": val,
                        "gold_value": it["gold_value"], "pairs": item_result(it, st, val)})
        m = metrics(res)
        out[str(tau)] = {k: m[k] for k in ("unsafe_pass", "unsafe_pass_rate", "false_alarm", "abstain_items", "fa_plus_hold_rate_on_pass")}
    return out


def metrics(results, key="model"):
    gv = [r for res in results for r in res["pairs"] if r["gold"] == C.VIOLATE]
    gp = [r for res in results for r in res["pairs"] if r["gold"] == C.COMPLY]
    up = sum(r[key] == C.COMPLY for r in gv)
    fa = sum(r[key] == C.VIOLATE for r in gp)
    hold_v = sum(r[key] == C.HOLD for r in gv)
    hold_p = sum(r[key] == C.HOLD for r in gp)
    n_items = len(results)
    exact = sum(1 for r in results if r["status"] == "ok" and abs(r["value"] - r["gold_value"]) <= 1e-6 * max(1.0, abs(r["gold_value"])))
    return {"items": n_items, "abstain_items": sum(r["status"] != "ok" for r in results),
            "gold_violation_pairs": len(gv), "unsafe_pass": up, "unsafe_pass_rate": up / len(gv) if gv else None,
            "gold_pass_pairs": len(gp), "false_alarm": fa, "false_alarm_rate": fa / len(gp) if gp else None,
            "hold_on_violation": hold_v, "hold_on_pass": hold_p,
            "fa_plus_hold_rate_on_pass": (fa + hold_p) / len(gp) if gp else None,
            "value_exact": exact, "value_exact_rate": exact / n_items if n_items else None}


def boot_ci(results, key="model", by="rule"):
    """95% percentile interval of the unsafe-pass rate, bootstrap over rule clusters (2,000 draws, fixed seed)."""
    clusters = collections.defaultdict(list)
    for r in results:
        clusters[r[by]].append(r)
    keys = sorted(clusters)
    rng = random.Random(SEED)
    vals = []
    for _ in range(BOOT_N):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in clusters[k]]
        m = metrics(sample, key)["unsafe_pass_rate"]
        if m is not None:
            vals.append(m)
    vals.sort()
    if not vals:
        return None
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def wording(prim, rate_ci):
    """Reporting wording fixed in the pre-registration (section 6)."""
    if rate_ci is None or not prim["items"]:
        return "판정 불가(자료 없음)"
    ab = prim["abstain_items"] / prim["items"]
    if ab > MAX_ABSTAIN:
        return f"판정 불가(보류 {ab:.0%} > {MAX_ABSTAIN:.0%})"
    lo, hi = rate_ci
    if lo > THRESH:
        return (f"최선의 프롬프트에서도 위반 후보 {prim['gold_violation_pairs']}건 중 {prim['unsafe_pass']}건 통과"
                f"(보류 {ab:.0%})")
    if hi < THRESH:
        return "프롬프트로 충분, 학습 효과는 부칙·협의 의견에서 검정, '추가 학습 불필요'도 결론"
    return "판정 불가(신뢰구간이 5%를 포함)"


def primary_claim(models):
    """Qwen3-8B with best-effort prompting: of V1 and V1SC with a hold rate of at most 20%, the variant with the
    LOWER unsafe-pass rate is reported (conservative for the claim that prompting is not enough)."""
    vs = models.get(PRIMARY_MODEL, {})
    ok = [(v, vs[v]) for v in PRIMARY_VARIANTS if v in vs and vs[v]["primary"]["items"]
          and vs[v]["primary"]["abstain_items"] / vs[v]["primary"]["items"] <= MAX_ABSTAIN]
    if not ok:
        cand = [(v, vs[v]) for v in PRIMARY_VARIANTS if v in vs]
        if not cand:
            return {"variant": None, "wording": "판정 불가(자료 없음)"}
        v, r = cand[0]
        return {"variant": v, "wording": wording(r["primary"], r["primary_unsafe_ci95_cluster_rule"])}
    v, r = min(ok, key=lambda t: (t[1]["primary"]["unsafe_pass_rate"], t[0]))
    return {"variant": v, "unsafe_pass": r["primary"]["unsafe_pass"], "gold_violation_pairs": r["primary"]["gold_violation_pairs"],
            "ci95": r["primary_unsafe_ci95_cluster_rule"], "abstain_items": r["primary"]["abstain_items"],
            "wording": wording(r["primary"], r["primary_unsafe_ci95_cluster_rule"])}


def summarise(results):
    prim = metrics(results)
    ci = boot_ci(results)
    return {
        "primary": prim, "primary_unsafe_ci95_cluster_rule": ci,
        "sensitivity_omission": metrics(results, "model_sens"),
        "by_set": {s: metrics([r for r in results if r["set"] == s]) for s in sorted({r["set"] for r in results})},
        "by_set_sub": {s: metrics([r for r in results if (r["set"] + ":" + str(r["sub"])) == s])
                       for s in sorted({r["set"] + ":" + str(r["sub"]) for r in results})},
        "not_ok_reasons": dict(collections.Counter(r["reason"] for r in results if r["status"] != "ok")),
        "missing_items": sum(r["status"] == "missing" for r in results),
        "wording_if_primary": wording(prim, ci),
        "items": results}


def score_decisions(gold, decisions):
    """decisions: {item_id: (status, value, reason)} -> summary dict."""
    results = []
    for iid, it in gold.items():
        st, val, reason = decisions.get(iid, ("missing", None, "no_output"))
        res = {"item_id": iid, "rule": it["rule"], "set": it["set"], "sub": it.get("sub"), "status": st,
               "reason": reason, "value": val, "gold_value": it["gold_value"]}
        res["pairs"] = item_result(it, st, val)
        results.append(res)
    return summarise(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=None, help="gold file (default: data/gold.jsonl)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", help="directory with model outputs: <raw>/<model_tag>/<variant>.jsonl")
    src.add_argument("--compiled", help="a scores file; its per-item status and value are scored again")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gold = load_gold(a.gold)
    out = {"models": {}, "rules": {"sc": [SC_N, SC_MIN], "boot": [BOOT_N, SEED]}}
    if a.raw:
        for mdir in sorted(p for p in Path(a.raw).iterdir() if p.is_dir()):
            out["models"][mdir.name] = {}
            for vf in sorted(mdir.glob("V*.jsonl")):
                if vf.stem.endswith(".errors"):
                    continue
                rows = {r["item_id"]: r for r in read_jsonl(vf)}
                out["models"][mdir.name][vf.stem] = score_decisions(gold, {iid: decide(it, rows.get(iid), vf.stem) for iid, it in gold.items()})
                if vf.stem == "V1SC":
                    out["models"][mdir.name][vf.stem]["sc_curve"] = sc_curve(gold, rows)
    else:
        src_scores = json.load(open(a.compiled, encoding="utf-8"))
        for m, vs in src_scores["models"].items():
            out["models"][m] = {v: score_decisions(gold, {r["item_id"]: (r["status"], r["value"], r["reason"]) for r in d["items"]})
                                for v, d in vs.items()}
    for m, vs in out["models"].items():
        for v, d in vs.items():
            p = d["primary"]
            ci = d["primary_unsafe_ci95_cluster_rule"]
            print(f"{m:16s} {v:5s} unsafe {p['unsafe_pass']}/{p['gold_violation_pairs']} ({(p['unsafe_pass_rate'] or 0):.1%}) "
                  f"CI {None if ci is None else [round(c, 3) for c in ci]} false alarm {p['false_alarm']}/{p['gold_pass_pairs']} "
                  f"hold {p['abstain_items']}/{p['items']} value-exact {p['value_exact']}")
    out["primary_claim"] = primary_claim(out["models"])
    print("PRIMARY CLAIM:", json.dumps(out["primary_claim"], ensure_ascii=False))
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
