"""P4 supervised QLoRA fine-tuning, one seed per call. Settings: config/p4_config.json -> "sft".

Usage:
  python scripts/train_sft.py --seed 1                                   # full run -> adapters/sft_s1, results/train/sft_s1.json
  python scripts/train_sft.py --seed 0 --train data/smoke_train.jsonl --limit 8 --max_steps 4 --out adapters/smoke \
                              --probe_file data/train.jsonl               # smoke -> results/smoke/train/smoke.json
  ... --backend mock   (pipeline test, no model)      ... --backend tiny   (developer test, tiny random model on CPU)

One training sequence = chat template of `messages` with the generation prompt, then `answer` + EOS; the loss is taken on
the answer tokens only. One optimizer step = `grad_accum` sequences (micro batch 1). A sequence longer than max_length
stops the run; nothing is truncated. Every 50 optimizer steps a checkpoint is written to <out>/checkpoint/; running the
same command again continues from it. A finished run (adapter + finished log) is not repeated."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # works also under PYTHONSAFEPATH

import argparse, copy, json, math, shutil, time
from common import (EST, add_lora, answer_nll, build_sequence, cap_vram, check_grad_flow, check_lengths, check_reportable, chunks,
                    cuda_sync, data_file, dev_overrides, env_info, epoch_order, grad_flow, load_backend, load_config, loss_path_check,
                    lr_at, minutes, model_entry, now, out_path, out_root, package_fingerprint, prepare_for_training, read_json,
                    read_jsonl, replace_file, reset_vram_peaks, results_dir, run_label, run_main, sha256_file, stamp,
                    stop_if_full_logits, trainable, verify_manifest, vram_free_gb, vram_note, vram_peaks, write_json)

MOCK_ADAPTER_MARKER = "MOCK_ADAPTER.txt"
LR_SCHEDULE = "linear warm-up: lr*(step+1)/warmup_steps for step < warmup_steps (step counted from 0); then lr*(total-step)/(total-warmup_steps)"


def save_checkpoint(ck_path, named, opt, meta, torch):
    """Adapter weights, optimizer state, random-generator state and the step records in ONE file (written through a
    temporary file, so the file on disk is always complete)."""
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"meta": meta, "trainable": {n: p.detach().cpu() for n, p in named}, "optimizer": opt.state_dict(),
             "rng_cpu": torch.get_rng_state(),
             "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = ck_path.with_name(ck_path.name + ".tmp")
    torch.save(state, str(tmp))
    replace_file(tmp, ck_path)


def try_resume(ck_path, fp, named, opt, torch):
    """Continue from the checkpoint if it belongs to exactly this run. Returns (start_step, records, note).
    If applying it fails half-way, the freshly initialised adapter and a clean optimizer state are put back."""
    if not ck_path.is_file():
        return 0, [], None
    try:
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    except Exception as e:
        return 0, [], f"checkpoint could not be read ({e!r}); training starts at step 0"
    if ck.get("meta", {}).get("fingerprint") != fp:
        return 0, [], "checkpoint of another run configuration ignored; training starts at step 0"
    names = [n for n, _ in named]
    if sorted(ck["trainable"]) != sorted(names) or any(ck["trainable"][n].shape != p.shape for n, p in named):
        return 0, [], "checkpoint does not match the adapter layout; training starts at step 0"
    init = {n: p.detach().cpu().clone() for n, p in named}
    opt_init = copy.deepcopy(opt.state_dict())
    try:
        with torch.no_grad():
            for n, p in named:
                p.copy_(ck["trainable"][n].to(device=p.device, dtype=p.dtype))
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng_cpu"])
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
    except Exception as e:
        with torch.no_grad():
            for n, p in named:
                p.copy_(init[n].to(device=p.device, dtype=p.dtype))
        opt.load_state_dict(opt_init)
        return 0, [], f"checkpoint could not be applied ({e!r}); training starts at step 0"
    step = int(ck["meta"]["step"])
    return step, list(ck["meta"]["records"]), f"resumed from the checkpoint of step {step} written at {ck['meta'].get('time')}"


def main():
    ap = argparse.ArgumentParser(description="P4 supervised QLoRA fine-tuning (see the text at the top of this file)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", default="", help="default: config sft.train_file")
    ap.add_argument("--out", default="", help="adapter folder; default adapters/sft_s<seed>")
    ap.add_argument("--limit", type=int, default=0, help="first N training rows (smoke)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after K optimizer steps (smoke)")
    ap.add_argument("--backend", default="hf", choices=["hf", "mock", "tiny"])
    ap.add_argument("--prep", default="lean", choices=["lean", "kbit_helper"], help="see common.prepare_for_training")
    ap.add_argument("--probe_file", default="", help="also check the sequence lengths of this file and test memory on its longest row")
    ap.add_argument("--ckpt_every", type=int, default=50, help="checkpoint every N optimizer steps (0 = never)")
    ap.add_argument("--dev_stop_after", type=int, default=0, help="developer test: stop with an error after K steps (mock/tiny only)")
    a = ap.parse_args()
    cfg = load_config(); S = cfg["sft"]; man = verify_manifest()
    m = model_entry(cfg, S["model_tag"])
    partial = bool(a.limit or a.max_steps)
    commit, reportable = check_reportable(a.backend, partial)
    if a.dev_stop_after and a.backend == "hf":
        raise SystemExit("--dev_stop_after is a developer option for --backend mock or tiny")
    if S["micro_batch"] != 1:
        raise SystemExit(f"CONFIG sft.micro_batch is {S['micro_batch']}: only micro batch 1 is implemented")
    if reportable and a.seed not in S["seeds"]:
        print(f"WARNING: seed {a.seed} is not one of the registered seeds {S['seeds']}", flush=True)

    train_path = data_file(a.train or S["train_file"], man)
    rows = read_jsonl(train_path)
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        raise SystemExit(f"NO TRAINING ROWS in {train_path.name}")
    if a.out:
        out = out_path(a.out)
    else:
        out = out_root() / "adapters" / ("" if a.backend == "hf" else a.backend) / ("smoke" if partial else f"sft_s{a.seed}")
    log_path = results_dir(a.backend, partial) / "train" / f"{out.name}.json"
    adapter_file = out / (MOCK_ADAPTER_MARKER if a.backend == "mock" else "adapter_model.safetensors")
    tag = f"sft {out.name}"

    order = epoch_order(len(rows), S["epochs"], a.seed)
    steps = chunks(order, S["grad_accum"])
    if a.max_steps:
        steps = steps[: a.max_steps]
    n_seq = sum(len(s) for s in steps)
    train_sha = sha256_file(train_path)
    fp = json.loads(json.dumps({"seed": a.seed, "train_file": train_path.name, "train_sha256": train_sha, "n_rows": len(rows),
                                "limit": a.limit, "max_steps": a.max_steps, "settings": S, "model_revision": m["revision"],
                                "prep": a.prep, "backend": a.backend, "n_optimizer_steps": len(steps),
                                "probe_file": a.probe_file.replace("\\", "/")}))

    # Run-level resume: a finished run with the same configuration is not repeated.
    old = None
    if log_path.is_file():
        try:
            old = read_json(log_path)
        except Exception:
            old = None
    if adapter_file.is_file() and old is not None and old.get("finished") is True:
        if old.get("fingerprint") == fp:
            print(f"ALREADY FINISHED: {out} and {log_path} exist for exactly this configuration; nothing to do")
            print("FINISHED", tag, out)
            return
        aside = f"__old_{stamp()}"
        print(f"WARNING: {out} holds a finished run with another configuration; it is moved to {out.name}{aside}", flush=True)
        try:
            shutil.move(str(out), str(out) + aside)
            shutil.move(str(log_path), str(log_path.with_name(out.name + aside + ".json")))
        except Exception as e:
            print(f"WARNING: could not move it ({e!r}); it will be overwritten", flush=True)
    out.mkdir(parents=True, exist_ok=True)

    log = {"label": run_label(a.backend, partial), "finished": False, "backend": a.backend, "seed": a.seed, "settings": S, "model": m,
           "prep": a.prep, "train_file": train_path.name, "train_sha256": train_sha, "n_rows": len(rows), "limit": a.limit,
           "max_steps": a.max_steps, "n_optimizer_steps": len(steps), "n_sequences": n_seq, "fingerprint": fp,
           "lr_schedule": LR_SCHEDULE, "optimizer": "torch.optim.AdamW, betas (0.9, 0.999), eps 1e-8",
           "loss": "mean cross-entropy over the answer tokens of one sequence; one step = mean over its sequences",
           "row_item_ids": [r["item_id"] for r in rows], "steps_rows": steps, "prereg_commit": commit, "reportable": reportable,
           "dev_overrides": dev_overrides(), "argv": _s.argv[1:], "adapter_dir": str(out), "start": now(), "end": None,
           "data_manifest": man, "package_files": package_fingerprint(), "steps": [], "error": None}
    print(f"[{tag}] {len(rows)} rows, {S['epochs']} epochs, {len(steps)} optimizer steps of {S['grad_accum']} sequences, seed {a.seed}, "
          f"backend {a.backend}", flush=True)
    print(f"[{tag}] expected time: about {minutes(EST['load_s'] + EST['sft_s_per_sequence'] * n_seq)} (rough; a measured ETA follows)", flush=True)

    if a.backend == "mock":
        # No model: fake token counts and losses. Exercises data order, schedule, logging and the run-level resume.
        lens = [(sum(len(x["content"]) for x in r["messages"]) + len(r["answer"])) // 2 for r in rows]
        log["tokens_per_sequence"] = {"n": len(lens), "mean": round(sum(lens) / len(lens), 1), "max": max(lens),
                                      "note": "mock: character count / 2, not tokens"}
        log["env"] = env_info()
        for si, idx in enumerate(steps):
            log["steps"].append({"step": si + 1, "loss": round(2.5 * math.exp(-0.03 * si) + 0.01 * ((si * 7919) % 13) / 13, 4),
                                 "lr": lr_at(si, len(steps), S["warmup_steps"], S["learning_rate"]), "grad_norm": 1.0,
                                 "seconds": 0.0, "n_sequences": len(idx), "tokens": sum(lens[i] for i in idx)})
            if a.dev_stop_after and si + 1 == a.dev_stop_after:
                write_json(log_path, log)
                raise RuntimeError(f"developer stop after step {si + 1} (--dev_stop_after)")
        (out / MOCK_ADAPTER_MARKER).write_text("mock adapter: no weights (pipeline test)\n", encoding="utf-8")
        write_json(out / "adapter_config.json", {"mock": True, "seed": a.seed})
        log.update({"finished": True, "end": now(), "trainable_params": 0, "loss_path": "mock", "peak_vram_allocated_gb": None,
                    "peak_vram_reserved_gb": None, "adapter_sha256": None})
        write_json(log_path, log)
        print("FINISHED", tag, out)
        return

    t_run = time.perf_counter()
    torch, tok, base, minfo = load_backend(a.backend, m, cfg, seed=a.seed)
    log["load_seconds"] = round(time.perf_counter() - t_run, 1)
    log["model_info"] = minfo
    log["vram_cap"] = cap_vram(torch)                           # a memory shortage becomes an error, not a crawl (see common.cap_vram)
    log["vram_free_gb_at_progress"] = []
    log["prepare"] = prepare_for_training(base, a.prep)
    torch.manual_seed(a.seed)                                   # the LoRA A matrices are drawn from this seed
    model = add_lora(base, S)
    named = trainable(model)
    params = [p for _, p in named]
    dev = params[0].device
    opt = torch.optim.AdamW(params, lr=S["learning_rate"], weight_decay=S["weight_decay"])
    log["trainable_params"] = sum(p.numel() for p in params)
    log["trainable_dtype"] = str(params[0].dtype)
    log["env"] = env_info()
    print(f"[{tag}] model loaded in {log['load_seconds']} s; trainable parameters {log['trainable_params']:,}; prepare {log['prepare']}; "
          f"GPU memory limit {log['vram_cap']}", flush=True)

    # Sequences. Every row is checked before any training (a too long sequence stops the run here).
    tmpl = cfg["generation"]["chat_template_kwargs"]
    seqs = [build_sequence(tok, r, tmpl) for r in rows]
    log["tokens_per_sequence"] = check_lengths(seqs, rows, S["max_length"], train_path.name)
    longest = max(seqs, key=lambda s: len(s[0]) + len(s[1]))
    shortest = min(seqs, key=lambda s: len(s[0]) + len(s[1]))
    if a.probe_file:
        prows = read_jsonl(data_file(a.probe_file, man))
        pseqs = [build_sequence(tok, r, tmpl) for r in prows]
        log["probe_file"] = dict(check_lengths(pseqs, prows, S["max_length"], a.probe_file), file=a.probe_file)
        longest = max(pseqs + [longest], key=lambda s: len(s[0]) + len(s[1]))
    p0, a0, _ = seqs[steps[0][0]]
    log["rendered_example"] = {"item_id": rows[steps[0][0]]["item_id"], "prompt_tail": tok.decode(p0[-12:]), "answer": tok.decode(a0),
                               "answer_first_tokens": tok.convert_ids_to_tokens(a0[:4]), "eos_token": tok.eos_token,
                               "eos_token_id": tok.eos_token_id, "n_prompt_tokens": len(p0), "n_answer_tokens": len(a0)}
    print(f"[{tag}] tokens per sequence {log['tokens_per_sequence']}", flush=True)

    # Check 1: the answer-only loss equals the plain loss (one short sequence, no gradient).
    lp = loss_path_check(model, shortest[0], shortest[1], dev, torch)
    force_full = lp["force_full"]
    lp["peak_vram_allocated_gb"], lp["peak_vram_reserved_gb"] = vram_peaks(torch)      # of this check alone (full logits, once)
    reset_vram_peaks(torch)                                     # the peaks reported below are those of training
    log["loss_path_check"] = lp
    print(f"[{tag}] loss path check: {lp}", flush=True)
    stop_if_full_logits(lp, a.backend)                          # real backend: the full-logits path does not fit this GPU
    if not lp["agree"]:
        print(f"WARNING [{tag}]: the two loss paths disagree; training uses the plain full-logits path", flush=True)

    # Check 2: forward and backward on the longest sequence. Shows in the first minute whether the memory is enough
    # and whether the gradient reaches the adapter through the checkpointed layers.
    model.train()
    x = torch.tensor([longest[0] + longest[1]], device=dev)
    t = time.perf_counter()
    loss, loss_path = answer_nll(model, x, torch.ones_like(x), len(longest[0]), torch, force_full)
    loss.backward()
    cuda_sync(torch)
    gf = grad_flow(named)
    alloc, reserved = vram_peaks(torch)
    log["preflight_longest_sequence"] = {"tokens": int(x.shape[1]), "loss": float(loss.item()), "seconds": round(time.perf_counter() - t, 3),
                                         "peak_vram_allocated_gb": alloc, "peak_vram_reserved_gb": reserved, "grad_flow": gf}
    log["loss_path"] = loss_path
    del loss, x
    opt.zero_grad(set_to_none=True)
    print(f"[{tag}] preflight on the longest sequence: {log['preflight_longest_sequence']}", flush=True)
    check_grad_flow(gf)

    # Checkpoint resume, or a fresh start from the seed.
    ck_path = out / "checkpoint" / "state.pt"
    start_step, records, note = try_resume(ck_path, fp, named, opt, torch)
    if note:
        print(f"[{tag}] {note}", flush=True)
    if start_step == 0:
        torch.manual_seed(a.seed)                               # dropout stream of the training loop
    log["resume"] = {"start_step": start_step, "note": note}
    log["steps"] = records
    log["checkpoints"] = []

    t_loop = time.perf_counter()
    try:
        for si in range(start_step, len(steps)):
            cuda_sync(torch); t = time.perf_counter()
            lr = lr_at(si, len(steps), S["warmup_steps"], S["learning_rate"])
            for group in opt.param_groups:
                group["lr"] = lr
            total, n_tok = 0.0, 0
            for ridx in steps[si]:
                p_ids, a_ids, _ = seqs[ridx]
                x = torch.tensor([p_ids + a_ids], device=dev)
                loss, _ = answer_nll(model, x, torch.ones_like(x), len(p_ids), torch, force_full)
                value = float(loss.item())
                if not math.isfinite(value):
                    raise RuntimeError(f"NON-FINITE LOSS ({value}) at step {si + 1}, item {rows[ridx]['item_id']}; training stopped")
                (loss / len(steps[si])).backward()
                total += value / len(steps[si]); n_tok += len(p_ids) + len(a_ids)
            gn = float(torch.nn.utils.clip_grad_norm_(params, S["max_grad_norm"]))
            if not math.isfinite(gn):
                raise RuntimeError(f"NON-FINITE GRADIENT NORM ({gn}) at step {si + 1}; training stopped")
            opt.step()
            opt.zero_grad(set_to_none=True)
            cuda_sync(torch)
            records.append({"step": si + 1, "loss": round(total, 5), "lr": lr, "grad_norm": round(gn, 4),
                            "seconds": round(time.perf_counter() - t, 3), "n_sequences": len(steps[si]), "tokens": n_tok})
            last = si + 1 == len(steps)
            if (si + 1) % 10 == 0 or last:
                el = time.perf_counter() - t_loop
                per = el / (si + 1 - start_step)
                alloc, reserved = vram_peaks(torch)
                free_now = vram_free_gb(torch)
                print(f"[{tag}] step {si + 1}/{len(steps)}  loss {total:.4f}  lr {lr:.2e}  {per:.1f} s/step  "
                      f"ETA {minutes(per * (len(steps) - si - 1))}  peak VRAM {alloc} GB  free {free_now} GB{vram_note(free_now)}", flush=True)
                log["peak_vram_allocated_gb"], log["peak_vram_reserved_gb"] = alloc, reserved
                log["vram_free_gb_at_progress"].append({"step": si + 1, "free_gb": free_now})
                write_json(log_path, log, must=False)           # partial log: survives a later crash
            if a.ckpt_every and (si + 1) % a.ckpt_every == 0 and not last:
                tc = time.perf_counter()
                try:
                    save_checkpoint(ck_path, named, opt, {"fingerprint": fp, "step": si + 1, "records": records, "time": now()}, torch)
                    log["checkpoints"].append({"step": si + 1, "seconds": round(time.perf_counter() - tc, 2)})
                except Exception as e:
                    print(f"WARNING [{tag}]: checkpoint at step {si + 1} not written ({e!r}); training goes on", flush=True)
            if a.dev_stop_after and si + 1 == a.dev_stop_after:
                raise RuntimeError(f"developer stop after step {si + 1} (--dev_stop_after)")
    except BaseException as e:
        log["error"] = {"type": type(e).__name__, "message": str(e)[:2000], "time": now(), "steps_done": len(records)}
        log["peak_vram_allocated_gb"], log["peak_vram_reserved_gb"] = vram_peaks(torch)
        write_json(log_path, log)
        raise

    model.save_pretrained(str(out), save_embedding_layers=False)
    if not adapter_file.is_file():
        raise SystemExit(f"ADAPTER FILE MISSING AFTER SAVE: {adapter_file}")
    log["peak_vram_allocated_gb"], log["peak_vram_reserved_gb"] = vram_peaks(torch)
    log.update({"finished": True, "end": now(), "wall_seconds": round(time.perf_counter() - t_run, 1),
                "train_seconds": round(time.perf_counter() - t_loop, 1), "adapter_sha256": sha256_file(adapter_file),
                "adapter_bytes": adapter_file.stat().st_size,
                "loss_first_5_mean": round(sum(r["loss"] for r in records[:5]) / len(records[:5]), 5),
                "loss_last_5_mean": round(sum(r["loss"] for r in records[-5:]) / len(records[-5:]), 5)})
    write_json(log_path, log)
    shutil.rmtree(ck_path.parent, ignore_errors=True)           # the checkpoint is not needed after a finished run
    print(f"[{tag}] loss first 5 steps {log['loss_first_5_mean']}, last 5 steps {log['loss_last_5_mean']}; "
          f"peak VRAM {log['peak_vram_allocated_gb']} GB allocated, {log['peak_vram_reserved_gb']} GB reserved", flush=True)
    print("FINISHED", tag, out)


if __name__ == "__main__":
    run_main(main, "train_sft")
