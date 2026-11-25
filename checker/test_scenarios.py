"""Round-trip gate and scorer scenarios (port of the test that was run before the pre-registration was fixed).

  python checker/test_scenarios.py

  gold        every item answered with its gold condition -> 0 unsafe passes, 0 false alarms, 0 holds, all values exact
              (round-trip gate: gold condition -> compiler -> checker reproduces every gold verdict)
  keep_old    always the unamended official value -> unsafe passes only where a tightening applies to the project
  swap_track  the other track type for track-dependent rules -> invalid -> held (primary), passes (sensitivity)
  all_abstain every item abstains -> no conclusion
  sc          self-consistency rule: 3 of 5 agreeing -> hold, 4 of 5 agreeing -> value used
Exit code 0 means every check passed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import railcheck as C  # noqa: E402
import score as S      # noqa: E402

try:  # Korean labels in the output: make sure a Windows console does not fail on them
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
TRK = {"ballast": "자갈도상", "slab": "콘크리트도상"}


def cond(it, value, track=None):
    """A well-formed answer of a model for this item with the given limit value."""
    R = C.RULES[it["rule"]]
    tr = track if track is not None else (TRK[it["track"]] if R["track_dep"] else "무관")
    return json.dumps({"parameter": it["label"], "op": ">=" if R["dir"] == "min" else "<=", "value": value,
                       "unit": R["unit"], "applies": {"design_speed_kmh": it["V"], "track": tr},
                       "edition": "현행", "source": "-", "abstain": False}, ensure_ascii=False)


def run():
    gold = S.load_gold()
    other = {"ballast": "slab", "slab": "ballast"}
    answers = {
        "gold": {i: cond(g, g["gold_value"]) for i, g in gold.items()},
        "keep_old": {i: cond(g, g["official_value"]) for i, g in gold.items()},
        "all_abstain": {i: '{"abstain": true}' for i in gold},
        "swap_track": {i: cond(g, g["gold_value"], track=TRK[other[g["track"]]]) if C.RULES[g["rule"]]["track_dep"]
                       else cond(g, g["gold_value"]) for i, g in gold.items()},
    }
    sc = {name: S.score_decisions(gold, {i: S.decide(gold[i], {"output": o}, "V1") for i, o in rows.items()})
          for name, rows in answers.items()}
    sc_rows = {}
    for k, (i, g) in enumerate(gold.items()):
        wrong = cond(g, g["gold_value"] * 1.5)
        sc_rows[i] = {"outputs": [cond(g, g["gold_value"])] * (3 if k % 2 == 0 else 4) + [wrong] * (2 if k % 2 == 0 else 1)}
    sc_res = S.score_decisions(gold, {i: S.decide(gold[i], sc_rows[i], "V1SC") for i in gold})
    sc_curve = S.sc_curve(gold, sc_rows)

    checks = {}
    g = sc["gold"]["primary"]
    checks["gate_gold_zero_errors"] = g["unsafe_pass"] == 0 and g["false_alarm"] == 0 and g["abstain_items"] == 0 and g["value_exact"] == g["items"]
    exp_unsafe = 0
    for it in gold.values():
        if it["true_edits"] and any(p == it["cell"] for p, _ in it["true_edits"]):
            old, new = it["official_value"], it["gold_value"]
            tight = (new > old) if it["dir"] == "min" else (new < old)
            if tight:
                exp_unsafe += sum(1 for c in it["candidates"] if c["gold"] == C.VIOLATE and
                                  ((old <= c["x"]) if it["dir"] == "min" else (c["x"] <= old)))
    k = sc["keep_old"]
    checks["keep_old_unsafe_matches_expectation"] = k["primary"]["unsafe_pass"] == exp_unsafe
    checks["keep_old_unsafe_count"] = [k["primary"]["unsafe_pass"], exp_unsafe]
    s = sc["swap_track"]
    n_dep = sum(1 for it in gold.values() if C.RULES[it["rule"]]["track_dep"])
    checks["swap_track_held"] = s["primary"]["abstain_items"] == n_dep and s["not_ok_reasons"].get("applies_track") == n_dep
    checks["swap_track_sensitivity_more_unsafe"] = s["sensitivity_omission"]["unsafe_pass"] > s["primary"]["unsafe_pass"]
    checks["all_abstain_no_conclusion"] = sc["all_abstain"]["wording_if_primary"].startswith("판정 불가(보류") and \
        sc["all_abstain"]["sensitivity_omission"]["unsafe_pass"] > 0
    checks["sc_curve_monotone_abstain"] = [sc_curve[str(t)]["abstain_items"] for t in range(1, 6)] == [0, 0, 0, 45, 90]
    checks["gold_wording_prompt_enough"] = sc["gold"]["wording_if_primary"].startswith("프롬프트로 충분")
    checks["keep_old_wording_not_enough"] = k["wording_if_primary"].startswith("최선의 프롬프트에서도")
    checks["sc_rule"] = sc_res["primary"]["abstain_items"] == (len(gold) + 1) // 2
    return checks


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    ok = all(v for k, v in res.items() if not k.endswith("_count"))
    print("ALL PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
