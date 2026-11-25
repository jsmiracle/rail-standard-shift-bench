# Pre-registration (English translation)

> This is a translation of [`PREREGISTRATION.ko.md`](PREREGISTRATION.ko.md). The Korean text is the original and prevails where the two differ.
> - The original was committed to the authors' private research repository on 2026-11-21 02:18 (KST), commit `742820f9a57b0bbda84ac44aa037e91e4df4a127`, and has not been changed since. SHA-256 of the original file: `a9df5d3c621e9ef99b3e10ab9ef8df39dc912b60fe896755afc1988520c6f838`.
> - Models were run on 2026-11-23 and scored on 2026-11-24.
> - Because that repository is private, readers cannot check the commit time from this repository alone. The hash is given so that the claim can be checked if the history is disclosed later.
> - Internal labels in the text: `D-188` is a decision number in the research log; `C_ML_engineer_plan.md` is an internal planning note; "P1" and "P2" are earlier pilots; `engine_io`, `models`, `gen_rules` are helper modules that call the public API of the verification engine. The engine (railassist) and those helpers are not in this repository; `checker/` gives the same verdicts for the seven rules (`python checker/reproduce.py`).
> - File names: `code/p3gen/score_p3.py` → `checker/score.py`; `data/items_public.jsonl` → `data/items.jsonl`; `data/gold.jsonl` → `data/gold.jsonl`.

---

# Pilot P3 pre-registration: unsafe passes of untrained open models when converting design-condition documents into conditions

- Written on 2026-11-21 (D-188). The hashes of the generator, the scorer and the data in the same commit as this document are what is fixed. Model outputs are produced only after this commit.
- Run: the researcher's notebook (RTX 4070 Laptop 8 GB, Windows, same environment as P2). Scoring: the researcher's Mac (pinned engine).
- This experiment does not test hypothesis H1, because nothing is trained. It measures in advance the level of what will be comparison group 2 (the untrained model) in the main study, and the difficulty of the task.

## 1. Question

An open language model receives a design-condition document (a current clause, the amending text of an amendment notice, a transitional provision in the addenda) and the project conditions, and writes the condition that applies to this project as one JSON object. When the engine judges with that condition, what share of the design candidates that violate the gold condition pass (unsafe-pass rate)? Does best-effort prompting (rules, a list of conditions, three worked examples, self-consistency) bring this rate below 5%?

## 2. Fixed items

| Item | Value |
|---|---|
| Verdict engine | railassist, tag engine-ref-7fed2c6 (commit 7fed2c62e0765fda1349bb1d9135b74200c925f6), public API only (P1 helper modules engine_io, models, gen_rules) |
| Rules | The seven single-limit rules of P1: R-06-01 minimum curve radius, R-07-01 maximum applied cant, R-09-01 minimum length of straight and circular curve, R-10-01 maximum gradient, R-11-02 minimum vertical curve radius, R-11-04 upper limit of vertical curve radius, R-11-05 minimum vertical curve length |
| Project contexts | 18: per rule, design speed 200/250 km/h × track type (ballast/slab). All are mixed passenger–freight lines, new double-track electrified main lines |
| Items | 90: C current clause 18; A_target 36 (per context one tightening and one relaxation); A_neighbour 18 (an amendment that changes another cell; the answer is "no change"); T 18 (transitional provision: the former rule if the order date is before the enforcement date, the amended rule if after) |
| Candidate design values | 6 per item, 540 pairs in all (270 violate the gold condition, 270 comply). 219 pairs lie between the gold value and the value before or after the amendment |
| Gold verdicts | The engine judges with a profile in which the gold condition is put into the project's cell. Check at generation: for every item the limit value complies and one step beyond it violates (0 failures) |
| Round-trip gate | Gold condition → compiler → engine reproduces the gold verdict for all 540 pairs (0 unsafe passes, 0 false alarms, 0 holds, 90/90 exact values). reports/p3_tests.json |
| Data hashes | items_public.jsonl 2f6f1ce66a942d23a4df348b52e52fdfa2b5919c21c6e0e9152fd2b48b3c8474. gold.jsonl is in data/ and is not put into the run package (the run package holds prompts only) |

The amending texts, the amended values, the notice numbers and the addenda are all synthetic (marked "(가상)", "synthetic", in the documents). The sentences of the current clauses and the table values are the same as in the clause register of the engine profile. The tables were written out as text from the engine values.

## 3. Models and generation

| Model | Revision | Variants |
|---|---|---|
| Qwen/Qwen3-8B (primary model, Apache-2.0, thinking off) | b968826d9c46dd6066d109eabc6255188de91218 | V0, V1, V1SC |
| Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0) | cdbee75f17c01a7cc42f958dc650907174af0554 | V0, V1 |
| Qwen/Qwen3.5-2B (Apache-2.0) | 15852e8c16360a2fea060d615a32b45270f8a8fc | V0, V1 |

- Quantisation: bitsandbytes NF4, double quantisation, bfloat16 compute (as in P2). transformers 5.22.2, torch 2.14.1+cu126.
- V0: system instruction and JSON format only. V1: seven writing rules, the list of conditions, and three worked examples on clauses that are not in the test set (the table of Article 8(1), an example amendment of the table of Article 14(1), an example addendum for Article 10(3)).
- Generation: V0 and V1 use greedy decoding, at most 256 tokens. V1SC samples five answers with the V1 prompt (temperature 0.7, top-p 0.9, seeds 1000–1004).
- Format-constrained decoding is not used (deviation 2).

## 4. Scoring (scorer code/p3gen/score_p3.py, fixed)

1. The first JSON object is taken from the output. If abstain is true, the item is a hold.
2. Compilation: the value (commas allowed), the unit conversion (m, km, mm; ‰, per mille, %) and the inequality (>= or "이상", <= or "이하") must agree with the direction of the rule. If the design speed of the scope is given, it must equal the project's; for rules whose value depends on the track type, the track type must equal the project's. Otherwise the answer is "invalid".
3. With a profile in which the single compiled value is put into the project's cell, the engine judges the six candidates.
4. Primary analysis: invalid and abstaining answers are counted as "hold" (handed to the designer). Unsafe pass = a pair that violates the gold condition and is judged "complies" with the model's condition.
5. Sensitivity analysis (omission): for invalid and abstaining answers, C items are judged as if the rule did not exist (everything complies), and A and T items with the current official value.
6. V1SC: each of the five outputs is compiled; if four or more give the same value, that value is used, otherwise hold. The whole curve for τ = 1–5 is also reported.
7. Measures: unsafe-pass rate (primary), false-alarm rate, item hold rate, false alarm + hold rate on complying pairs, exact-value rate. They are also reported per set (C, A_target, A_neighbour, T) and per subset.
8. 95% confidence interval: bootstrap with the seven rules as clusters, 2,000 draws (seed 20261121), percentile interval.

## 5. Choice of the primary result (fixed in advance)

The primary result is, of V1 and V1SC of Qwen3-8B, the one with the lower unsafe-pass rate among those whose item hold rate is at most 20%. Because the lower one is chosen, the choice works against the claim "prompting alone is not enough" (conservative). If both exceed a hold rate of 20%, no conclusion is drawn. The results of all models and variants are reported regardless of this choice.

## 6. Reporting wording (fixed in advance)

| Condition (primary result) | Wording |
|---|---|
| Hold rate > 20% | Inconclusive (too many holds) |
| Lower bound of the interval > 5% | "Even with best-effort prompting, k of n violating candidates pass (hold a%)" |
| Upper bound of the interval < 5% | "Prompting is enough; the training effect is to be tested on addenda and consultation opinions; 'no additional training needed' is also a conclusion" |
| Otherwise | Inconclusive (the interval contains 5%) |

If the second case comes out, it is written in the research plan as it is. In that case the test material of the main study is moved to harder documents such as addenda and consultation opinions, and "no additional training is needed" stays in the plan as a possible conclusion.

## 7. Deviations (from section 1 of C_ML_engineer_plan.md)

1. 117 items → 90: only the seven rules (no additional rule passed the gate), and set R (seven real amendments from 2020 to 2022) is left out. That set was to be reported separately because of the risk of contamination and is not part of the primary result.
2. Instead of format-constrained decoding and llama.cpp GGUF, transformers + bitsandbytes NF4 is used, which was verified on this notebook in P2. Format errors are counted as holds, which lowers the unsafe-pass rate (conservative).
3. Set A has, per context, two amendments of the project's cell and one amendment of a neighbouring cell (plan: half and half).
4. The step that checks memorisation of Qwen3-8B with the 44 closed questions of P2, and the optional model kanana-1.5-8B, are left out.
5. Article 2 of the addenda was not copied from a real notice; it was written synthetically in the usual form of transitional provisions in notices of the Ministry of Land, Infrastructure and Transport.

## 8. Limits of interpretation

- The documents are short and each item has one condition. This is easier than real task orders and consultation opinions. The unsafe-pass rate measured here is therefore likely to be lower than the rate on real documents (it has the character of a lower bound).
- The clusters are only seven rules, so the confidence interval is wide and may be unstable.
- Because the amendments are synthetic, there is almost no effect of a model answering from a memorised real amendment history. On the other hand, the style may differ from that of real notices.
- This experiment says nothing about the training effect (H1).

## 9. Files

- code/p3gen/p3core.py (rules, cells, compiler, engine verdict), gen_p3.py (item generation), prompts_p3.py (V0 and V1 prompts), score_p3.py (scoring), test_p3.py (the gate and 10 scoring scenarios)
- data/gold.jsonl, data/items_public.jsonl, data/gen_report.json, reports/p3_tests.json
- run_package/: the run package for the notebook (README_RU.md)
