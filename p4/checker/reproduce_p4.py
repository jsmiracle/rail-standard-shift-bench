"""Recompute the published P4 numbers from the per-item records in p4/results/scores_p4.json (no GPU, no engine).

  python p4/checker/reproduce_p4.py

For every run and prompt variant, every item and every one of its six candidate design values:
  1. the gold verdict is judged again from the gold limit value with this repository's checker (checker/railcheck.py);
  2. the model verdict is judged again from the limit value compiled from the model's answer ('hold' if the answer
     could not be used);
  3. unsafe passes (gold 위반, model 적합) are counted per group and compared with the stored summaries;
  4. the supervised-learning comparison (untrained Qwen3-4B V2 vs the three fine-tuned seeds) is recounted.
Exit code 0 means every recomputed number equals the stored one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "checker"))
import railcheck as C  # noqa: E402

HOLD = "보류"
GROUPS = {
    "all": lambda it: True,
    "official_table": lambda it: it["table"] == "T0",
    "new_tables": lambda it: it["table"] != "T0",
    "new_tables_trained_clauses": lambda it: it["table"] != "T0" and it["rule_group"] == "trained",
    "new_tables_heldout_clauses": lambda it: it["table"] != "T0" and it["rule_group"] == "heldout",
}


def main():
    d = json.loads((ROOT / "p4" / "results" / "scores_p4.json").read_text(encoding="utf-8"))
    problems, rows = [], []
    unsafe = {}
    for run, variants in d["runs"].items():
        for var, rec in variants.items():
            counts = {g: [0, 0] for g in GROUPS}             # [unsafe passes, gold-violating pairs]
            for it in rec["items"]:
                direction = C.RULES[it["rule"]]["dir"]
                for p in it["pairs"]:
                    g = C.verdict(direction, p["x"], it["gold_value"])
                    if g != p["gold"]:
                        problems.append(f"{run}/{var}/{it['item_id']} x={p['x']}: gold verdict {g} != stored {p['gold']}")
                    m = C.verdict(direction, p["x"], it["value"]) if it["status"] == "ok" else HOLD
                    if m != p["model"]:
                        problems.append(f"{run}/{var}/{it['item_id']} x={p['x']}: model verdict {m} != stored {p['model']}")
                    for name, member in GROUPS.items():
                        if member(it) and p["gold"] == C.VIOLATE:
                            counts[name][1] += 1
                            if p["model"] == C.COMPLY:
                                counts[name][0] += 1
            for name, (u, n) in counts.items():
                st = rec["groups"][name]["primary"]
                if (u, n) != (st["unsafe_pass"], st["gold_violation_pairs"]):
                    problems.append(f"{run}/{var}/{name}: recomputed {u}/{n} != stored {st['unsafe_pass']}/{st['gold_violation_pairs']}")
                unsafe[(run, var, name)] = u
                rows.append((run, var, name, u, n))
    base = "qwen3_4b_2507__base"
    for name, st in d["sft_decision"]["groups"].items():
        u0 = unsafe[(base, "V2", name)]
        seeds = [unsafe[(f"qwen3_4b_2507__sft_s{k}", "V2", name)] for k in (1, 2, 3)]
        if u0 != st["untrained_unsafe"] or seeds != st["sft_unsafe_by_seed"]:
            problems.append(f"sft/{name}: recomputed {u0} -> {seeds} != stored {st['untrained_unsafe']} -> {st['sft_unsafe_by_seed']}")
        print(f"supervised learning, {name}: untrained {u0}/{st['violation_pairs']} -> seeds 1/2/3: {seeds}  (stored reading: {st['reading']})")
    print()
    print(f"{'run':28s} {'variant':7s} {'group':28s} unsafe/violating")
    for run, var, name, u, n in rows:
        print(f"{run:28s} {var:7s} {name:28s} {u}/{n}")
    print()
    if problems:
        print(f"MISMATCH: {len(problems)}")
        for p in problems[:20]:
            print("  " + p)
        sys.exit(1)
    print("All recomputed numbers equal the stored ones.")


if __name__ == "__main__":
    main()
