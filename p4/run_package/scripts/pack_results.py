"""Last step: zips results/ (logs and raw outputs; no adapter weights, no model cache) into p4_results_<time>.zip in the
package folder and prints, in Russian, which stages finished and which file to send back.
Usage: python scripts/pack_results.py [--backend hf|mock|tiny]"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # works also under PYTHONSAFEPATH

import argparse, json, time, zipfile
from common import ROOT, data_dir, now, out_root, read_json, results_dir, run_main, sha256_file, write_json

SKIP_SUFFIXES = (".safetensors", ".pt", ".bin", ".tmp")
SMOKE_NAMES = [("infer_8b", "пробный вывод Qwen3-8B"), ("infer_4b", "пробный вывод Qwen3-4B"),
               ("sft", "пробное дообучение (несколько шагов)"), ("sft_infer", "пробный вывод с адаптером"),
               ("grpo", "пробная проверка обучения с наградой (2 шага)")]
FULL_NAMES = [("closedbook_8b", "вопросы без документа, Qwen3-8B"), ("base_8b_v1", "базовый прогон Qwen3-8B (V1)"),
              ("base_4b_v1v2", "базовый прогон Qwen3-4B (V1 и V2)"),
              ("sft_s1", "дообучение, зерно 1"), ("eval_sft_s1", "прогон после дообучения, зерно 1"),
              ("sft_s2", "дообучение, зерно 2"), ("eval_sft_s2", "прогон после дообучения, зерно 2"),
              ("sft_s3", "дообучение, зерно 3"), ("eval_sft_s3", "прогон после дообучения, зерно 3"),
              ("grpo", "проверка обучения с наградой (150 шагов)")]
WORDS = {"ok": "ГОТОВО", "failed": "СБОЙ", "skipped_smoke_failed": "ПРОПУЩЕН (пробный запуск не прошёл)",
         "skipped_no_adapter": "ПРОПУЩЕН (нет адаптера: дообучение не закончилось)",
         "skipped_no_commit_file": "ПРОПУЩЕН (нет файла P4_COMMIT.txt)", None: "НЕ ЗАПУСКАЛСЯ"}


def count_lines(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return sum(1 for l in fh if l.strip())


def item_ids(p):
    """item_id of every readable row of a jsonl file (a row that cannot be read is skipped)."""
    ids = set()
    with open(p, encoding="utf-8", errors="replace") as fh:
        for l in fh:
            try:
                d = json.loads(l)
            except Exception:
                continue
            if isinstance(d, dict) and "item_id" in d:
                ids.add(d["item_id"])
    return ids


def overview(base):
    """What is in results/: saved outputs per run and variant, training logs, reward-check logs (counts only).
    For <variant>.errors.jsonl the count is the number of prompts that still have NO saved answer: the errors file is
    append-only, so a prompt that failed once and was answered by a later run stays in it and is not counted."""
    out = {"raw": {}, "train": {}, "grpo": {}}
    for f in sorted(base.glob("raw/*/*.jsonl")):
        if f.name.endswith(".errors.jsonl"):
            answers = f.with_name(f.name[: -len(".errors.jsonl")] + ".jsonl")
            n = len(item_ids(f) - (item_ids(answers) if answers.is_file() else set()))
        else:
            n = count_lines(f)
        out["raw"].setdefault(f.parent.name, {})[f.stem] = n
    for f in sorted(base.glob("train/*.json")):
        try:
            d = read_json(f)
            out["train"][f.stem] = {"finished": d.get("finished"), "steps_done": len(d.get("steps", [])), "steps_planned": d.get("n_optimizer_steps")}
        except Exception as e:
            out["train"][f.stem] = {"unreadable": repr(e)}
    for f in sorted(base.glob("grpo/grpo_check_*.json")):
        try:
            d = read_json(f)
            out["grpo"][f.stem] = {"finished": d.get("finished"), "summary": d.get("summary")}
        except Exception as e:
            out["grpo"][f.stem] = {"unreadable": repr(e)}
    return out


def adapters_index():
    """Names, sizes and hashes of the adapter files (the weights themselves stay on the notebook)."""
    idx = []
    for f in sorted((out_root() / "adapters").glob("**/*")):
        if f.is_file():
            rel = str(f.relative_to(out_root())).replace("\\", "/")
            idx.append({"file": rel, "bytes": f.stat().st_size,
                        "sha256": sha256_file(f) if f.name == "adapter_model.safetensors" else None})
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hf", choices=["hf", "mock", "tiny"])
    a = ap.parse_args()
    res = out_root() / "results"
    base = results_dir(a.backend, False)
    base.mkdir(parents=True, exist_ok=True)
    counts = overview(base)
    write_json(base / "summary_counts.json", {"time": now(), "backend": a.backend, "counts": counts, "smoke_counts": overview(base / "smoke")})
    write_json(base / "adapters_index.json", adapters_index())

    name = out_root() / (f"p4_results_{'' if a.backend == 'hf' else a.backend + '_'}{time.strftime('%Y%m%d_%H%M')}.zip")
    added, skipped = 0, []
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        files = [f for f in sorted(res.glob("**/*")) if f.is_file() and not f.name.endswith(SKIP_SUFFIXES)]
        files += [f for f in sorted((out_root() / "adapters").glob("**/adapter_config.json"))]
        for f in files:
            try:
                z.write(f, str(f.relative_to(out_root())).replace("\\", "/")); added += 1
            except Exception as e:
                skipped.append(f"{f}: {e!r}")
        for f in (ROOT / "PACKAGE_INFO.txt", ROOT / "P4_COMMIT.txt", ROOT / "config" / "p4_config.json", data_dir() / "MANIFEST.json"):
            if f.is_file():
                z.write(f, "package_meta/" + f.name); added += 1
        if skipped:
            z.writestr("package_meta/SKIPPED_FILES.txt", "\n".join(skipped))

    def status(path):
        try:
            return read_json(path)
        except Exception:
            return {}

    smoke, full = status(base / "smoke" / "status.json"), status(base / "status.json")
    print("")
    print("=" * 78)
    print("ИТОГ ПРОГОНА P4" + ("" if a.backend == "hf" else f"  (тестовый режим: {a.backend})"))
    print("Пробные запуски:")
    for key, title in SMOKE_NAMES:
        word = "НЕ ЗАПУСКАЛСЯ" if key not in smoke else ("ГОТОВО" if smoke[key] is True else "СБОЙ")
        print(f"  {title:<48} {word}")
    print("Основные этапы:")
    for key, title in FULL_NAMES:
        print(f"  {title:<48} {WORDS.get(full.get(key), str(full.get(key)))}")
    print("Сохранено ответов (модель и вариант: число):")
    for run_tag, per in counts["raw"].items():
        print("  " + run_tag + ": " + ", ".join(f"{v} {n}" for v, n in per.items() if not v.endswith(".errors"))
              + "".join(f"; без ответа осталось в {v[:-7]}: {n}" for v, n in per.items() if v.endswith(".errors") and n))
    if skipped:
        print(f"Не удалось добавить в архив файлов: {len(skipped)} (список внутри архива)")
    print("")
    print(f"Архив для отправки ({added} файлов, {name.stat().st_size / 2**20:.1f} МБ):")
    print(f"  {name}")
    if a.backend == "hf":
        print("Пришлите этот файл, даже если какой-то этап не прошёл. На ноутбуке ничего не оценивается.")
    else:
        print(f"Это был тестовый режим ({a.backend}): настоящие модели не загружались. Если в двух списках выше везде «ГОТОВО»,")
        print("присылать ничего не нужно: запускайте основной прогон. Иначе пришлите этот файл и основной прогон не запускайте.")
    print("=" * 78)


if __name__ == "__main__":
    run_main(main, "pack_results")
