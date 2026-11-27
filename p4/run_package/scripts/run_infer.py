"""P4 inference: one model (optionally with a LoRA adapter) answers the prompts of one items file, greedy decoding.
This script never scores. The package holds no gold answers of the test items.

Usage:
  python scripts/run_infer.py --model qwen3_8b --items data/items_test.jsonl --variants V1
  python scripts/run_infer.py --model qwen3_4b_2507 --items data/items_test.jsonl --variants V2 --adapter adapters/sft_s1
  python scripts/run_infer.py --model qwen3_8b --items data/closedbook_items.jsonl --variants CB
  python scripts/run_infer.py --model qwen3_8b --items data/smoke_test.jsonl --variants V1,V2 --limit 3   # smoke -> results/smoke/
  ... --backend mock   (pipeline test, no model)      ... --backend tiny   (developer test, tiny random model on CPU)
Re-running the same command resumes: items with a saved output are skipped, failed items are tried again.
Output: results/raw/<run_tag>/<variant>.jsonl, <variant>.errors.jsonl, run_manifest_<time>.json
        (run_tag = <model>__base or <model>__<adapter folder name>)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # works also under PYTHONSAFEPATH

import argparse, json, time, traceback
from common import (EST, attach_adapter, check_reportable, data_file, dev_overrides, env_info, free_cuda, fresh_stamp, gen_once, load_backend,
                    load_config, minutes, model_entry, neutral_config, now, out_path, package_fingerprint, pip_freeze, read_json,
                    read_jsonl, replace_file, results_dir, run_label, run_main, sha256_file, stamp, verify_manifest, vram_peaks,
                    write_json)

MOCK_OUTPUT = '{"abstain": true}'
MOCK_ADAPTER_MARKER = "MOCK_ADAPTER.txt"
MAX_FAILED_IN_A_ROW = 5             # five failed prompts in a row stop the variant (the error is systematic)


def read_done(f):
    """Item ids that already have a saved output. Lines that cannot be read or carry an error are dropped from the file."""
    done = set()
    if f.exists():
        good, n_lines = [], 0
        with open(f, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()                    # read and close: on Windows an open file cannot be replaced
        for l in lines:
            n_lines += 1
            try:
                d = json.loads(l)
            except Exception:
                continue
            if isinstance(d, dict) and d.get("error") is None and d.get("output") is not None and "item_id" in d:
                done.add(d["item_id"]); good.append(json.dumps(d, ensure_ascii=False) + "\n")
        if len(good) != n_lines:                      # rewrite only when something has to go, and never in place
            tmp = f.with_name(f.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(good)
            replace_file(tmp, f)
    return done


def keep_or_set_aside(outdir, v, identity):
    """Outputs saved earlier under this run tag and variant are reused only if they come from the same model revision,
    the same adapter file and the same items file. Otherwise the old files are renamed (never mixed, never deleted)."""
    f, fid = outdir / f"{v}.jsonl", outdir / f"{v}.identity.json"
    old = None
    if fid.is_file():
        try:
            old = read_json(fid)
        except Exception:
            old = None
    if f.is_file() and f.stat().st_size > 0 and old != identity:
        aside = f"{v}__old_{stamp()}"
        print(f"WARNING: {f} was written for another model, adapter or items file; it is kept as {aside}.jsonl and not reused", flush=True)
        for src, dst in ((f, f"{aside}.jsonl"), (outdir / f"{v}.errors.jsonl", f"{aside}.errors.jsonl"), (fid, f"{aside}.identity.json")):
            if src.is_file():
                replace_file(src, outdir / dst)
    write_json(fid, identity)


def main():
    ap = argparse.ArgumentParser(description="P4 inference (see the text at the top of this file)")
    ap.add_argument("--model", required=True, help="model tag of config/p4_config.json")
    ap.add_argument("--items", required=True, help="data/items_test.jsonl, data/smoke_test.jsonl or data/closedbook_items.jsonl")
    ap.add_argument("--variants", default="", help="comma list, for example V1,V2 ; rows without a variant field are variant CB")
    ap.add_argument("--adapter", default="", help="folder of a LoRA adapter saved by train_sft.py")
    ap.add_argument("--run_tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N items of every variant; writes under results/smoke/")
    ap.add_argument("--backend", default="hf", choices=["hf", "mock", "tiny"])
    a = ap.parse_args()
    cfg = load_config(); man = verify_manifest()
    m = model_entry(cfg, a.model)
    partial = bool(a.limit)
    commit, reportable = check_reportable(a.backend, partial)

    items_path = data_file(a.items, man)
    by_var = {}
    for it in read_jsonl(items_path):
        by_var.setdefault(it.get("variant", "CB"), []).append(it)
    if a.variants:
        variants = [v for v in a.variants.split(",") if v]
    else:
        variants = [v for v in m["variants"] if v in by_var] or list(by_var)
    for v in variants:
        if v not in by_var:
            raise SystemExit(f"VARIANT {v!r} NOT IN {items_path.name}. Variants in this file: {sorted(by_var)}")
        ids = [it["item_id"] for it in by_var[v]]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"DUPLICATE item_id in {items_path.name}, variant {v}")

    adapter_dir = out_path(a.adapter) if a.adapter else None
    adapter_sha = None
    if adapter_dir is not None:
        weights = adapter_dir / (MOCK_ADAPTER_MARKER if a.backend == "mock" else "adapter_model.safetensors")
        if not weights.is_file():
            raise SystemExit(f"ADAPTER NOT FOUND: {weights} (train it first with scripts/train_sft.py)")
        adapter_sha = sha256_file(adapter_dir / "adapter_config.json") if a.backend == "mock" else sha256_file(weights)
    run_tag = a.run_tag or f"{a.model}__{adapter_dir.name if adapter_dir else 'base'}"
    outdir = results_dir(a.backend, partial) / "raw" / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    items_sha = sha256_file(items_path)
    identity = {"model_tag": a.model, "model_revision": m["revision"], "backend": a.backend, "adapter_sha256": adapter_sha,
                "items_file": items_path.name, "items_sha256": items_sha}

    # What is left to do (resume). Decided before the model is loaded, so a finished stage returns at once.
    plan, n_todo = {}, 0
    for v in variants:
        src = by_var[v][: a.limit] if a.limit else by_var[v]
        keep_or_set_aside(outdir, v, identity)
        done = read_done(outdir / f"{v}.jsonl")
        plan[v] = (src, done)
        n_todo += sum(1 for it in src if it["item_id"] not in done)
    print(f"[{run_tag}] items file {items_path.name}, variants {variants}, backend {a.backend}, to do: {n_todo} prompts", flush=True)
    if n_todo == 0:
        print(f"NOTHING TO DO: every output of {run_tag} {variants} is already saved in {outdir}")
        print("FINISHED", run_tag, outdir)
        return
    per_item = EST["closedbook_s_per_item"] if variants == ["CB"] else EST["infer_s_per_item"]
    if adapter_dir is not None:
        per_item *= EST["adapter_infer_factor"]       # the adapter is not merged: more work per generated token (estimate)
    print(f"[{run_tag}] expected time: about {minutes(EST['load_s'] + per_item * n_todo)} (rough; a measured ETA follows)", flush=True)

    g, tmpl = cfg["generation"]["greedy"], cfg["generation"]["chat_template_kwargs"]
    if (g["do_sample"], g["num_beams"], g["repetition_penalty"]) != (False, 1, 1.0):
        raise SystemExit(f"CONFIG generation.greedy is not greedy decoding as in P3: {g}")
    run = {"label": run_label(a.backend, partial), "backend": a.backend, "run_tag": run_tag, "model": m, "prereg_commit": commit,
           "reportable": reportable, "dev_overrides": dev_overrides(), "argv": _s.argv[1:], "start": now(),
           "items_file": items_path.name, "items_sha256": items_sha, "identity": identity, "data_manifest": man,
           "quantization": cfg["quantization"], "chat_template_kwargs": tmpl,
           "package_files": package_fingerprint(), "adapter": None, "variants": {}, "rendered_prompt_example": {}}
    t_run = time.perf_counter()

    torch = tok = model = greedy_kw = None
    mock_fail = _o.environ.get("P4_DEV_MOCK_FAIL", "") if a.backend == "mock" else ""      # tests of the error path only
    if a.backend == "mock":
        if adapter_dir is not None:
            run["adapter"] = {"path": str(adapter_dir), "mock": True, "adapter_model_sha256": adapter_sha}
    else:
        torch, tok, model, minfo = load_backend(a.backend, m, cfg, seed=cfg["generation"]["seed"])
        orig_gc = neutral_config(model, tok, g["max_new_tokens"])       # on the base model, before an adapter wraps it
        if adapter_dir is not None:
            model, run["adapter"] = attach_adapter(model, adapter_dir)
        greedy_kw = dict(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=g["max_new_tokens"], pad_token_id=tok.pad_token_id,
                         temperature=None, top_p=None, top_k=None)
        run.update({"load_seconds": round(time.perf_counter() - t_run, 1), "model_info": minfo,
                    "generation_config_repository": orig_gc, "greedy_kwargs": greedy_kw})
        print(f"[{run_tag}] model loaded in {run['load_seconds']} s ({minfo['loader']})", flush=True)
    run["env"] = env_info(); run["pip_freeze"] = pip_freeze()

    abort = None
    for v in variants:
        src, done = plan[v]
        f, fe = outdir / f"{v}.jsonl", outdir / f"{v}.errors.jsonl"
        n_left = sum(1 for it in src if it["item_id"] not in done)
        tv = time.perf_counter(); n_new = n_err = failed_in_a_row = 0
        with open(f, "a", encoding="utf-8") as fh, open(fe, "a", encoding="utf-8") as fhe:
            for it in src:
                if it["item_id"] in done:
                    continue
                rec = {"item_id": it["item_id"], "variant": v, "model_tag": a.model, "run_tag": run_tag}
                failed = False
                try:
                    if a.backend == "mock":
                        if mock_fail == "ALL" or it["item_id"] in mock_fail.split(","):
                            raise RuntimeError("mock failure requested by P4_DEV_MOCK_FAIL (test of the error path)")
                        o, n_in, n_out, s, mode = MOCK_OUTPUT, 0, 0, 0.0, "mock"
                    else:
                        o, n_in, n_out, s, mode = gen_once(tok, model, it["messages"], tmpl, greedy_kw, torch)
                    rec.update({"output": o, "n_input_tokens": n_in, "n_output_tokens": n_out, "seconds": s, "template_mode": mode})
                    if v not in run["rendered_prompt_example"]:
                        run["rendered_prompt_example"][v] = {"item_id": it["item_id"], "n_input_tokens": n_in}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush(); n_new += 1
                except Exception as e:
                    failed = True; n_err += 1
                    rec.update({"output": None, "error": repr(e), "traceback": traceback.format_exc()[-1500:], "time": now()})
                    fhe.write(json.dumps(rec, ensure_ascii=False) + "\n"); fhe.flush()
                    if n_err <= 3:
                        print(f"[{run_tag}] {v}: item {it['item_id']} failed: {e!r}", flush=True)
                if failed and torch is not None:
                    free_cuda(torch)                  # after an out-of-memory error the next item starts with a clean cache
                failed_in_a_row = failed_in_a_row + 1 if failed else 0
                if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                    abort = f"{failed_in_a_row} prompts of variant {v} failed in a row; the run was stopped"
                    break
                k = n_new + n_err
                if k % 10 == 0:
                    el = time.perf_counter() - tv
                    print(f"[{run_tag}] {v}: {k}/{n_left}  {el:.0f}s  ETA {minutes(el / k * (n_left - k))}  errors {n_err}", flush=True)
        run["variants"][v] = {"n_items": len(src), "n_already_done": len(src) - n_left, "n_new": n_new, "n_errors": n_err,
                              "wall_seconds": round(time.perf_counter() - tv, 1)}
        print(f"[{run_tag}] {v} done: {run['variants'][v]}", flush=True)
        if abort:
            break

    if torch is not None:
        run["peak_vram_allocated_gb"], run["peak_vram_reserved_gb"] = vram_peaks(torch)
    run["aborted"] = abort
    run["wall_seconds"] = round(time.perf_counter() - t_run, 1)
    run["end"] = now()
    write_json(outdir / f"run_manifest_{fresh_stamp(outdir, 'run_manifest_', '.json')}.json", run)
    n_err_total = sum(x["n_errors"] for x in run["variants"].values())
    if abort or n_err_total:
        raise SystemExit(f"{n_err_total} prompts of {run_tag} failed ({abort or 'the others are saved'}). Details: {outdir} "
                         "*.errors.jsonl. Running the same command again retries only the failed ones.")
    print("FINISHED", run_tag, outdir)


if __name__ == "__main__":
    run_main(main, "run_infer")
