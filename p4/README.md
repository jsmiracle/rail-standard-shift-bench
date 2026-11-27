# P4: follow-up pilot on standards tables with changed values

[한국어](README.ko.md) · [Pre-registration (Korean)](PREREGISTRATION_P4.ko.md) · [Results summary (Korean)](results/RESULTS_SUMMARY_ko.md)

> **Status: pilot, run on 2026-11-26 with the package and rules frozen before the first model run.** These are small, controlled checks with synthetic documents. They are **not** a test of the main hypothesis of the authors' research plan (that reward learning with the checker's verdict lowers unsafe passes more than supervised learning). Read the limits below before quoting a number.

## Questions

| No. | Question | Models |
|---|---|---|
| 0 | Closed book: without any document, does the model know the Korean limit values? | Qwen3-8B |
| 1 | Baseline: how often does an untrained model hand the checker a limit that lets a violating design pass, when the standards table has **changed values** (synthetic tables TE1, TE2) and on the official table (T0, the P3 items)? | Qwen3-8B (prompt V1), Qwen3-4B-Instruct-2507 (V1, V2) |
| 2 | Supervised fine-tuning (QLoRA) on items generated from **other** synthetic tables: does it lower unsafe passes on the unseen tables, including three clauses never used in training? | Qwen3-4B-Instruct-2507, 3 seeds |
| 3 | Reward-learning procedure (GRPO, reward = the checker's verdict): does it run on an 8 GB GPU? No effect is asked. | Qwen3-4B-Instruct-2507 |

Conditions common to all runs: 4-bit weights, greedy decoding, synthetic tables and template sentences, 7 clauses of 「철도의 건설기준에 관한 규정」 (3 held out from training), one laptop GPU (RTX 4070 Laptop, 8 GB).

## Results

| Question | Result |
|---|---|
| 0. Closed book (44 questions) | current value 0, 2020-edition value 0, other values 44 |
| 1. Baseline, Qwen3-8B (V1) | official table 16/270 unsafe passes (identical to P3); new tables 122/972. By document type on the new tables: current clause 0/108, amendment of the project's cell 4/215, amendment of another cell 22/217, **transitional provision 96/432** |
| 2. Supervised learning, new tables | untrained 118/972 → seeds 1/2/3: **0 / 7 / 11** (reading: *lower*) |
| 2. Supervised learning, held-out clauses | untrained 58/440 → 0 / 7 / 11 (*lower*); every remaining unsafe pass is in one held-out clause (R-11-04) |
| 2. Supervised learning, official table | untrained 17/270 → 0 / 0 / 8 (*lower*) |
| 3. Reward-learning procedure | ran: 150 of 150 steps, 0 failures, 9 updates, 14.7 s per step, peak 4.6 GB allocated; in 141 of 150 groups (94%) all six answers had the same reward, so there was nothing to learn from |

Cells give unsafe passes / design candidates that violate the gold condition. Reading rule fixed in advance: *lower* means all three seeds are below the untrained model and no seed has a higher rate of false alarms plus holds; a group where any run holds more than 20% of items is *no conclusion*; no statistical test (3 seeds, 7 clauses). Per-clause and per-document-type readings are in the Korean summary.

In every run no answer abstained by itself; every 'hold' was an answer that could not be used.

## Limits

- Not a test of the research plan's main hypothesis; reward learning was only checked to run.
- The trained model is Qwen3-4B, not the 8B model of the plan.
- Tables, amendments and transitional provisions are synthetic, and the documents are short template sentences, easier than real documents.
- 7 clauses, 3 of them held out: one clause can decide a result.
- The supervised targets contain no abstentions, so trained models may stop abstaining.

## Files

| Path | Content |
|---|---|
| `PREREGISTRATION_P4.ko.md` | Questions, fixed items, models, scoring and reading rules (Korean). Frozen before the first model run (commit `7e882bc` of the authors' private research log, also in `run_package/P4_COMMIT.txt`) |
| `data/tables.json` | The synthetic standards tables (training tables TR*, test tables TE1, TE2, official T0) |
| `data/gold_test.jsonl` | Gold limit values and the six candidate design values of every test item |
| `data/gen_report.json` | Checks run when the items were generated (all empty lists = no failure) |
| `run_package/` | The package run on the laptop: scripts, config, `data/items_test.jsonl` (rendered prompts V1 and V2), `data/train.jsonl` (supervised targets and rewards of the training tables), `data/closedbook_items.jsonl` |
| `checker/score_p4.py` | The fixed scorer (uses `../checker/railcheck.py` of this repository) |
| `checker/test_p4.py` | Scorer scenarios (gold answers, base values, abstention, readings) |
| `checker/reproduce_p4.py` | Recomputes every published number from the per-item records |
| `results/scores_p4.json` | Per-item records: compiled limit value, status, and the gold and model verdict of every candidate |
| `results/run_records_p4.json` | Run status, environment, training settings, time and memory (fields that could hold personal data removed) |
| `results/RESULTS_SUMMARY_ko.md` | The summary written by the summary script (Korean) |

Not included: raw model outputs and run logs (they contain machine and user names), and the item generator (it calls the authors' verification engine).

## Reproduce

```
python p4/checker/reproduce_p4.py                                  # recompute every number; exit code 0 = all equal
python p4/checker/test_p4.py p4/data/gold_test.jsonl <empty folder> # scorer scenarios; prints ALL PASS
```

Both need only Python 3.9+ and this repository.
