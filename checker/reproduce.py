"""Check that the minimal checker reproduces the published, engine-scored results.

  python checker/reproduce.py

1. Gold verdicts: for every item, the six candidate design values are judged with the gold limit value;
   all 540 verdicts must equal the gold verdicts (which were produced by the full verification engine).
   Round trip: the gold condition written as a model answer goes through the compiler and the checker;
   all 90 values must be exact and all 540 verdicts must equal the gold verdicts.
2. Published run: for each of the seven model runs in results/scores_p3.json the per-item status and compiled
   value are scored again with this repository's checker; every per-pair verdict, every summary number, the
   bootstrap interval and the fixed wording must be identical to the stored ones.
Exit code 0 means everything matched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import railcheck as C  # noqa: E402
import score as S      # noqa: E402
import test_scenarios as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
try:  # Korean labels in the output: make sure a Windows console does not fail on them
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    gold = S.load_gold()
    bad = n = 0
    for it in gold.values():
        for c in it["candidates"]:
            n += 1
            bad += C.verdict(it["dir"], c["x"], it["gold_value"]) != c["gold"]
    print(f"1. gold verdicts reproduced: {n - bad}/{n}")
    rt = S.score_decisions(gold, {i: S.decide(g, {"output": T.cond(g, g["gold_value"])}, "V1") for i, g in gold.items()})["primary"]
    rt_ok = rt["unsafe_pass"] == 0 and rt["false_alarm"] == 0 and rt["abstain_items"] == 0 and rt["value_exact"] == rt["items"]
    bad += not rt_ok
    print(f"   round trip (gold condition -> compiler -> checker): exact values {rt['value_exact']}/{rt['items']}, "
          f"unsafe passes {rt['unsafe_pass']}, false alarms {rt['false_alarm']}, holds {rt['abstain_items']}")
    pub = json.load(open(ROOT / "results" / "scores_p3.json", encoding="utf-8"))
    fails = bad
    models = {}
    print("2. published run, scored again with the minimal checker")
    print(f"   {'model':14s} {'var':5s} {'unsafe pass':15s} {'95% interval':13s} {'false alarm':>11s} {'hold':>6s} {'exact':>6s}  same as published")
    for m, vs in pub["models"].items():
        models[m] = {}
        for v, d in vs.items():
            mine = S.score_decisions(gold, {r["item_id"]: (r["status"], r["value"], r["reason"]) for r in d["items"]})
            models[m][v] = mine
            same_pairs = all(a["pairs"] == b["pairs"] for a, b in zip(mine["items"], d["items"]))
            same_sum = all(mine[k] == d[k] for k in ("primary", "primary_unsafe_ci95_cluster_rule", "sensitivity_omission",
                                                     "by_set", "by_set_sub", "wording_if_primary"))
            ok = same_pairs and same_sum
            fails += not ok
            p, ci = mine["primary"], mine["primary_unsafe_ci95_cluster_rule"]
            up = f"{p['unsafe_pass']}/{p['gold_violation_pairs']} ({p['unsafe_pass_rate']:.1%})"
            iv = f"{ci[0]:.1%}–{ci[1]:.1%}"
            fa = f"{p['false_alarm']}/{p['gold_pass_pairs']}"
            hd = f"{p['abstain_items']}/{p['items']}"
            print(f"   {m:14s} {v:5s} {up:15s} {iv:13s} {fa:>11s} {hd:>6s} {p['value_exact']:>6d}  {'yes' if ok else 'NO'}")
    claim = S.primary_claim(models)
    same_claim = claim == pub["primary_claim"]
    fails += not same_claim
    print("3. primary claim:", json.dumps(claim, ensure_ascii=False), "| same as published:", "yes" if same_claim else "NO")
    print("RESULT:", "REPRODUCED" if not fails else f"{fails} MISMATCH(ES)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
