"""P4 scorer (rules fixed in research/pilot_P4/PREREG_P4_ko.md before the first model run).

Usage (from the p4 folder of this repository, any Python 3.9+; no engine needed; raw outputs are not published):
  python checker/score_p4.py --gold data/gold_test.jsonl --raw <results>/raw --out results/scores_p4.json
             [--closedbook <results>/raw/qwen3_8b__base/CB.jsonl]
Raw layout: <raw>/<run_tag>/<variant>.jsonl with rows {item_id, output}.  run_tag examples: qwen3_8b__base,
qwen3_4b_2507__base, qwen3_4b_2507__sft_s1.

Per item the model's condition is compiled to one limit value; the six fixed candidates are judged with it by the
same inequality that reproduced every engine verdict of the gold (checker railcheck.py, copied from the public
repository).  Measures are those of P3:
  primary:      invalid or abstaining answer = hold; unsafe pass = gold '위반' and model '적합'
  sensitivity:  invalid or abstaining answer = omission (C items: rule absent, everything complies; other items:
                the base value of the table is used)
Groups: all items; official table (T0 = the P3 set) / new tables (TE1, TE2); clauses used in training / held out.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "checker"))   # public layout: p4/checker -> <repo>/checker/railcheck.py
import railcheck as C  # noqa: E402

BOOT_N, SEED = 2000, 20261126
MAX_HOLD = 0.20
SFT_MODEL, SFT_SEEDS = "qwen3_4b_2507", ("sft_s1", "sft_s2", "sft_s3")


def read_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def decide(it, row):
    if row is None or row.get("output") is None:
        return ("missing", None, "no_output")
    return C.compile_condition(it, C.extract_json(row["output"]))


def item_result(it, status, value):
    rows = []
    for c in it["candidates"]:
        mv = C.verdict(it["dir"], c["x"], value) if status == "ok" else C.HOLD
        if mv != C.HOLD:
            sens = mv
        elif it["set"] == "C":
            sens = C.COMPLY
        else:
            sens = C.verdict(it["dir"], c["x"], it["base_value"])
        rows.append({"x": c["x"], "gold": c["gold"], "model": mv, "model_sens": sens})
    return rows


def metrics(results, key="model"):
    gv = [r for res in results for r in res["pairs"] if r["gold"] == C.VIOLATE]
    gp = [r for res in results for r in res["pairs"] if r["gold"] == C.COMPLY]
    up = sum(r[key] == C.COMPLY for r in gv)
    fa = sum(r[key] == C.VIOLATE for r in gp)
    hold_p = sum(r[key] == C.HOLD for r in gp)
    n = len(results)
    exact = sum(1 for r in results if r["status"] == "ok" and abs(r["value"] - r["gold_value"]) <= 1e-6 * max(1.0, abs(r["gold_value"])))
    return {"items": n, "hold_items": sum(r["status"] != "ok" for r in results),
            "gold_violation_pairs": len(gv), "unsafe_pass": up, "unsafe_pass_rate": up / len(gv) if gv else None,
            "gold_pass_pairs": len(gp), "false_alarm": fa, "hold_on_pass": hold_p,
            "fa_plus_hold_rate_on_pass": (fa + hold_p) / len(gp) if gp else None,
            "value_exact": exact, "value_exact_rate": exact / n if n else None}


def boot_ci(results, key="model"):
    clusters = collections.defaultdict(list)
    for r in results:
        clusters[r["rule"]].append(r)
    keys = sorted(clusters)
    if len(keys) < 4:
        return None                      # too few clusters for an interval (held-out clauses: three)
    rng = random.Random(SEED)
    vals = []
    for _ in range(BOOT_N):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in clusters[k]]
        m = metrics(sample, key)["unsafe_pass_rate"]
        if m is not None:
            vals.append(m)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


GROUPS = {
    "all": lambda r: True,
    "official_table": lambda r: r["table"] == "T0",
    "new_tables": lambda r: r["table"] != "T0",
    "new_tables_trained_clauses": lambda r: r["table"] != "T0" and r["rule_group"] == "trained",
    "new_tables_heldout_clauses": lambda r: r["table"] != "T0" and r["rule_group"] == "heldout",
}


def summarise(results):
    out = {"missing_items": sum(r["status"] == "missing" for r in results),
           "not_ok_reasons": dict(collections.Counter(r["reason"] for r in results if r["status"] != "ok")), "groups": {}}
    for g, f in GROUPS.items():
        sub = [r for r in results if f(r)]
        out["groups"][g] = {"primary": metrics(sub), "unsafe_ci95_cluster_rule": boot_ci(sub), "sensitivity_omission": metrics(sub, "model_sens")}
    new = [r for r in results if r["table"] != "T0"]
    out["new_tables_by_set_sub"] = {k: metrics([r for r in new if (r["set"] + ":" + str(r["sub"])) == k])
                                    for k in sorted({r["set"] + ":" + str(r["sub"]) for r in new})}
    out["new_tables_by_rule"] = {k: metrics([r for r in new if r["rule"] == k]) for k in sorted({r["rule"] for r in new})}
    out["items"] = results
    return out


def score_run(gold, rows):
    results = []
    for iid, it in gold.items():
        st, val, reason = decide(it, rows.get(iid))
        results.append({"item_id": iid, "rule": it["rule"], "set": it["set"], "sub": it.get("sub"), "table": it["table"],
                        "rule_group": it["rule_group"], "status": st, "reason": reason, "value": val, "gold_value": it["gold_value"],
                        "pairs": item_result(it, st, val)})
    return summarise(results)


def sft_decision(runs):
    """Pre-registered reading of the supervised fine-tuning arm (PREREG_P4 section 5).  Reference: the untrained
    Qwen3-4B with the same prompt (V2).  Group by group: 'lower' only if every seed has fewer unsafe passes than
    the reference AND no seed has a higher false alarm + hold rate; 'higher' if every seed has more unsafe passes;
    otherwise 'no clear difference'.  No statistical test (three seeds, at most seven clusters)."""
    ref = runs.get(f"{SFT_MODEL}__base", {}).get("V2")
    seeds = [runs.get(f"{SFT_MODEL}__{s}", {}).get("V2") for s in SFT_SEEDS]
    if ref is None or any(s is None for s in seeds):
        return {"status": "incomplete", "have_reference": ref is not None, "seeds_found": sum(s is not None for s in seeds)}
    out = {"status": "complete", "groups": {}}
    for g in ("new_tables", "new_tables_trained_clauses", "new_tables_heldout_clauses", "official_table"):
        r = ref["groups"][g]["primary"]; ss = [s["groups"][g]["primary"] for s in seeds]
        hold_ok = all(s["hold_items"] / s["items"] <= MAX_HOLD for s in ss + [r])
        if not hold_ok:
            word = "no conclusion (hold above 20% in a run)"
        elif all(s["unsafe_pass"] < r["unsafe_pass"] for s in ss) and all(s["fa_plus_hold_rate_on_pass"] <= r["fa_plus_hold_rate_on_pass"] for s in ss):
            word = "lower"
        elif all(s["unsafe_pass"] > r["unsafe_pass"] for s in ss):
            word = "higher"
        else:
            word = "no clear difference"
        out["groups"][g] = {"reading": word, "violation_pairs": r["gold_violation_pairs"], "untrained_unsafe": r["unsafe_pass"],
                            "sft_unsafe_by_seed": [s["unsafe_pass"] for s in ss],
                            "untrained_fa_plus_hold": r["fa_plus_hold_rate_on_pass"], "sft_fa_plus_hold_by_seed": [s["fa_plus_hold_rate_on_pass"] for s in ss],
                            "untrained_value_exact": r["value_exact"], "sft_value_exact_by_seed": [s["value_exact"] for s in ss], "items": r["items"]}
    return out


def score_closedbook(path):
    """44 closed-book questions (P1/P2 items): current value / 2020 value / other / unusable."""
    gold = {r["item_id"]: r for r in read_jsonl(HERE.parents[2] / "pilot_P1P2" / "p1_eval_set" / "data" / "closedbook_items.jsonl")}
    cnt = collections.Counter(); rows = read_jsonl(path)
    for r in rows:
        g = gold[r["item_id"]]; obj = C.extract_json(r.get("output"))
        v = C._num(obj.get("answer")) if isinstance(obj, dict) else None
        if v is None:
            cnt["unusable"] += 1
        elif abs(v - float(g["answer_k0"])) <= 1e-6 * max(1.0, abs(float(g["answer_k0"]))):
            cnt["current_value"] += 1
        elif g.get("answer_2020") is not None and abs(v - float(g["answer_2020"])) <= 1e-6 * max(1.0, abs(float(g["answer_2020"]))):
            cnt["value_of_2020_edition"] += 1
        else:
            cnt["other"] += 1
    return {"items": len(rows), **dict(cnt)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--closedbook", default=None)
    a = ap.parse_args()
    gold = {it["item_id"]: it for it in read_jsonl(a.gold)}
    runs = {}
    for d in sorted(p for p in Path(a.raw).iterdir() if p.is_dir()):
        for vf in sorted(d.glob("V*.jsonl")):
            if vf.stem.endswith(".errors"):
                continue
            rows = {r["item_id"]: r for r in read_jsonl(vf)}
            runs.setdefault(d.name, {})[vf.stem] = score_run(gold, rows)
            g = runs[d.name][vf.stem]["groups"]
            for name in ("official_table", "new_tables", "new_tables_heldout_clauses"):
                p = g[name]["primary"]
                print(f"{d.name:26s} {vf.stem:3s} {name:27s} unsafe {p['unsafe_pass']:>3d}/{p['gold_violation_pairs']:<4d} "
                      f"fa+hold {(p['fa_plus_hold_rate_on_pass'] or 0):.1%} hold {p['hold_items']}/{p['items']} exact {p['value_exact']}")
    out = {"runs": runs, "sft_decision": sft_decision(runs), "rules": {"boot": [BOOT_N, SEED], "max_hold": MAX_HOLD}}
    if a.closedbook:
        out["closedbook"] = score_closedbook(a.closedbook)
        print("closed book:", out["closedbook"])
    print("SFT reading:", json.dumps({k: v["reading"] for k, v in out["sft_decision"].get("groups", {}).items()} or out["sft_decision"], ensure_ascii=False))
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
