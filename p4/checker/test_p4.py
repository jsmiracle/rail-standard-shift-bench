"""Scorer scenarios for P4 (run before the pre-registration is frozen; result goes to reports/p4_tests.json).

  python3.12 code/p4gen/test_p4.py data/gold_test.jsonl <tmp dir>

  gold       every item answered with its gold condition -> 0 unsafe passes, 0 false alarms, 0 holds, all values exact
  keep_base  every item answered with the base value of its table -> unsafe passes exactly where a tightening applies
  abstain    every item abstains -> everything is a hold
  sft        reference = keep_base, three 'seeds' = gold -> the reading is 'lower' wherever the reference has unsafe passes
  same       reference and seeds identical -> 'no clear difference'
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_p4 as S  # noqa: E402
C = S.C
TRK = {"ballast": "자갈도상", "slab": "콘크리트도상"}


def cond(it, value):
    R = C.RULES[it["rule"]]
    return json.dumps({"parameter": it["label"], "op": ">=" if R["dir"] == "min" else "<=", "value": value, "unit": R["unit"],
                       "applies": {"design_speed_kmh": it["V"], "track": TRK[it["track"]] if R["track_dep"] else "무관"},
                       "edition": "현행", "source": "-", "abstain": False}, ensure_ascii=False)


def write(raw, tag, variant, rows):
    d = raw / tag; d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{variant}.jsonl", "w", encoding="utf-8") as fh:
        for iid, o in rows.items():
            fh.write(json.dumps({"item_id": iid, "output": o}, ensure_ascii=False) + "\n")


def run(gold_path, tmp):
    gold = {it["item_id"]: it for it in S.read_jsonl(gold_path)}
    ans = {"gold": {i: cond(g, g["gold_value"]) for i, g in gold.items()},
           "keep_base": {i: cond(g, g["base_value"]) for i, g in gold.items()},
           "abstain": {i: '{"abstain": true}' for i in gold}}
    sc = {k: S.score_run(gold, {i: {"item_id": i, "output": o} for i, o in rows.items()}) for k, rows in ans.items()}
    checks = {}
    g = sc["gold"]["groups"]["all"]["primary"]
    checks["gold_zero_errors"] = g["unsafe_pass"] == 0 and g["false_alarm"] == 0 and g["hold_items"] == 0 and g["value_exact"] == g["items"] == len(gold)
    exp = 0
    for it in gold.values():
        old, new = it["base_value"], it["gold_value"]
        tight = (new > old) if it["dir"] == "min" else (new < old)
        if tight:
            exp += sum(1 for c in it["candidates"] if c["gold"] == C.VIOLATE and ((old <= c["x"]) if it["dir"] == "min" else (c["x"] <= old)))
    k = sc["keep_base"]["groups"]["all"]["primary"]
    checks["keep_base_unsafe_matches_expectation"] = k["unsafe_pass"] == exp and exp > 0
    checks["keep_base_unsafe_count"] = [k["unsafe_pass"], exp]
    a = sc["abstain"]["groups"]["all"]
    checks["abstain_all_hold"] = a["primary"]["hold_items"] == len(gold) and a["primary"]["unsafe_pass"] == 0 and a["sensitivity_omission"]["unsafe_pass"] > 0
    parts = sc["gold"]["groups"]
    checks["groups_partition"] = parts["official_table"]["primary"]["items"] + parts["new_tables"]["primary"]["items"] == len(gold) and \
        parts["new_tables_trained_clauses"]["primary"]["items"] + parts["new_tables_heldout_clauses"]["primary"]["items"] == parts["new_tables"]["primary"]["items"]
    checks["heldout_has_no_interval"] = parts["new_tables_heldout_clauses"]["unsafe_ci95_cluster_rule"] is None and parts["all"]["unsafe_ci95_cluster_rule"] is not None
    # file-level run through main(): reference keep_base, seeds gold
    raw = Path(tmp) / "raw"
    write(raw, "qwen3_4b_2507__base", "V2", ans["keep_base"])
    for s in S.SFT_SEEDS:
        write(raw, f"qwen3_4b_2507__{s}", "V2", ans["gold"])
    out = Path(tmp) / "scores.json"
    sys.argv = ["score_p4.py", "--gold", str(gold_path), "--raw", str(raw), "--out", str(out)]
    S.main()
    d = json.load(open(out, encoding="utf-8"))["sft_decision"]
    checks["sft_reading_lower"] = d["status"] == "complete" and all(v["reading"] == "lower" for v in d["groups"].values())
    for s in S.SFT_SEEDS:
        write(raw, f"qwen3_4b_2507__{s}", "V2", ans["keep_base"])
    S.main()
    d = json.load(open(out, encoding="utf-8"))["sft_decision"]
    checks["sft_reading_same"] = all(v["reading"] == "no clear difference" for v in d["groups"].values())
    return checks


if __name__ == "__main__":
    res = run(Path(sys.argv[1]), sys.argv[2])
    Path("reports").mkdir(exist_ok=True)
    json.dump(res, open("reports/p4_tests.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
    ok = all(v for k, v in res.items() if not k.endswith("_count"))
    print("ALL PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
