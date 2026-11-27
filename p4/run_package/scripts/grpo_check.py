"""P4 reward-learning PROCEDURE CHECK: does a reward-learning loop with the checker's verdict as reward run on this GPU?
No effect is claimed from this script; the adapter is not saved. Settings: config/p4_config.json -> "reward_check".

Usage:
  python scripts/grpo_check.py                     # 150 steps -> results/grpo/grpo_check_<time>.json, results/grpo/samples.jsonl
  python scripts/grpo_check.py --steps 2 --limit 4 # smoke -> results/smoke/grpo/
  ... --backend mock   (pipeline test, no model)      ... --backend tiny   (developer test, tiny random model on CPU)

One step = one training row:
  1. sample `group` completions of its prompt in one generate call (no gradient, eval mode, full softmax);
  2. reward of each completion from railcheck: mean over the six candidate design values of +1 (same verdict as gold),
     `unsafe` (gold violates, answer lets it pass) or `false_alarm` (gold complies, answer rejects it); `abstain` for an
     abstention; `unusable` for an answer the checker cannot use;
  3. advantage = (reward - group mean) / (group standard deviation + 1e-4); all rewards equal -> no update (zero-variance group);
  4. policy-gradient update: loss_i = -advantage_i * mean log-probability of the sampled tokens of completion i, one
     forward/backward per completion (micro batch 1), mean over the group, gradient norm clipped, one AdamW step.
     (On-policy first step of GRPO without a KL term.)
Before the first step the update path is tried once on a real sequence (forward and backward, no optimizer step), so
that also a short run in which every group has equal rewards has executed it.
An error in one step (for example out of memory) is logged and the loop goes on; five failed steps in a row stop it.
A run in which no step succeeded, and a smoke run on the real backend with any failed step, end with a non-zero exit code."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # works also under PYTHONSAFEPATH

import argparse, json, math, random, statistics, time, traceback
import railcheck
from common import (EST, add_lora, answer_nll, build_sequence, cap_vram, check_grad_flow, check_reportable, cuda_sync, data_file,
                    dev_overrides, encode, env_info, free_cuda, fresh_stamp, grad_flow, load_backend, load_config, loss_path_check,
                    minutes, model_device, model_entry, neutral_config, now, package_fingerprint, prepare_for_training, read_json,
                    read_jsonl, reset_vram_peaks, results_dir, run_label, run_main, sha256_file, stop_if_full_logits, trainable,
                    verify_manifest, vram_free_gb, vram_note, vram_peaks, write_json)

MAX_FAILED_IN_A_ROW = 5
STD_KIND = "sample standard deviation (n-1) of the group rewards"


def reward_of(text, row, w):
    """Reward of one completion for one training row. Returns (reward, status, reason); status is ok, abstain or unusable."""
    try:
        status, value, reason = railcheck.compile_condition(row, railcheck.extract_json(text))
    except Exception as e:                       # the checker is never allowed to stop the loop
        return w["unusable"], "unusable", f"checker_exception:{type(e).__name__}"
    if status == "abstain":
        return w["abstain"], "abstain", reason
    if status != "ok":
        return w["unusable"], "unusable", reason
    scores = []
    for c in row["candidates"]:
        v = railcheck.verdict(row["dir"], c["x"], value)
        if v == c["gold"]:
            scores.append(w["match"])
        elif c["gold"] == railcheck.VIOLATE and v == railcheck.COMPLY:
            scores.append(w["unsafe"])
        elif c["gold"] == railcheck.COMPLY and v == railcheck.VIOLATE:
            scores.append(w["false_alarm"])
        else:
            raise ValueError(f"unexpected gold verdict {c['gold']!r} in item {row['item_id']}")
    return sum(scores) / len(scores), "ok", ""


def group_advantages(rewards):
    """(mean, std, advantages). advantages is None when all rewards are equal (zero-variance group: no update)."""
    mean = sum(rewards) / len(rewards)
    if max(rewards) == min(rewards):
        return mean, 0.0, None
    std = statistics.stdev(rewards)
    return mean, std, [(r - mean) / (std + 1e-4) for r in rewards]


def split_completions(rows_of_ids, eos_ids):
    """Sampled token ids of every sequence (padded after its end) -> the completion of each, cut after its first EOS."""
    comps = []
    for ids in rows_of_ids:
        cut = len(ids)
        for j, t in enumerate(ids):
            if t in eos_ids:
                cut = j + 1
                break
        comps.append(ids[:cut])
    return comps


def mock_completions(row, group, rng):
    """Stand-ins for sampled answers (mock backend): the gold answer, a wrong value, an abstention or broken JSON."""
    gold = json.loads(row["answer"])
    texts = []
    for _ in range(group):
        k = rng.random()
        if k < 0.35:
            texts.append(row["answer"])
        elif k < 0.70:
            wrong = dict(gold)
            wrong["value"] = round(float(gold["value"]) * rng.choice([0.5, 0.8, 1.25, 2.0]), 3)
            texts.append(json.dumps(wrong, ensure_ascii=False))
        elif k < 0.85:
            texts.append('{"abstain": true}')
        else:
            texts.append(row["answer"][: max(5, len(row["answer"]) // 2)])       # cut in the middle: no JSON object
    return texts


def summarise(recs, steps_requested):
    ok = [r for r in recs if not r["failed"]]
    means = [r["reward_mean"] for r in ok]
    flat = [s for r in ok for s in r["statuses"]]

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    def share(name):
        return round(flat.count(name) / len(flat), 4) if flat else None

    return {"steps_requested": steps_requested, "steps_done": len(ok), "steps_failed": len(recs) - len(ok),
            "zero_variance_groups": sum(1 for r in ok if r["zero_variance"]),
            "share_zero_variance": round(sum(1 for r in ok if r["zero_variance"]) / len(ok), 4) if ok else None,
            "updates": sum(1 for r in ok if r["updated"]),
            "mean_reward_first_30": mean(means[:30]), "n_steps_in_first": len(means[:30]),
            "mean_reward_last_30": mean(means[-30:]), "n_steps_in_last": len(means[-30:]),
            "seconds_per_step_mean": mean([r["seconds"] for r in ok]),
            "seconds_generate_mean": mean([r["seconds_generate"] for r in ok]),
            "seconds_update_mean": mean([r["seconds_update"] for r in ok if r["updated"]]),
            "share_ok": share("ok"), "share_abstain": share("abstain"), "share_unusable": share("unusable"),
            "note": "procedure check: no effect is claimed from these numbers"}


def end_stop_reason(summary, steps, partial, backend):
    """Why a run whose loop reached its last step still counts as stopped (None = it finished).
    - No step succeeded: nothing was sampled, so nothing was checked (fewer than five steps can all fail without
      reaching the five-in-a-row stop; the two-step smoke test is such a run).
    - Smoke run on the real backend (--limit / --steps): any failed step fails the smoke test, because
      run_all_windows.ps1 starts the full 150-step run only after a clean smoke test.
    A full run with single failed steps still finishes; its log counts them (summary.steps_failed)."""
    if summary["steps_done"] == 0:
        return f"every one of the {steps} steps failed"
    if partial and backend == "hf" and summary["steps_failed"] > 0:
        return f"{summary['steps_failed']} of {steps} steps failed in a smoke run"
    return None


def main():
    cfg = load_config(); RC = cfg["reward_check"]
    ap = argparse.ArgumentParser(description="P4 reward-learning procedure check (see the text at the top of this file)")
    ap.add_argument("--steps", type=int, default=RC["steps"])
    ap.add_argument("--train", default="", help="default: config reward_check.train_file")
    ap.add_argument("--limit", type=int, default=0, help="first N training rows (smoke)")
    ap.add_argument("--backend", default="hf", choices=["hf", "mock", "tiny"])
    ap.add_argument("--prep", default="lean", choices=["lean", "kbit_helper"], help="see common.prepare_for_training")
    ap.add_argument("--dev_reward_jitter", action="store_true",
                    help="developer test (mock/tiny only): add a small fixed pattern to the rewards so that every group has spread")
    ap.add_argument("--dev_check_update", action="store_true",
                    help="developer test (tiny only): after every update compute the group loss again (field loss_after_update)")
    a = ap.parse_args()
    man = verify_manifest()
    m = model_entry(cfg, RC["model_tag"])
    partial = bool(a.limit) or a.steps != RC["steps"]
    commit, reportable = check_reportable(a.backend, partial)
    if (a.dev_reward_jitter or a.dev_check_update) and a.backend == "hf":
        raise SystemExit("--dev_reward_jitter and --dev_check_update are developer options for --backend mock or tiny")
    G, W = RC["group"], RC["weights"]
    if G < 2 or a.steps < 1:
        raise SystemExit(f"CONFIG reward_check.group must be at least 2 (is {G}) and --steps at least 1 (is {a.steps})")

    train_path = data_file(a.train or RC["train_file"], man)
    rows = read_jsonl(train_path)
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        raise SystemExit(f"NO TRAINING ROWS in {train_path.name}")
    order = list(range(len(rows)))
    random.Random(RC["seed"]).shuffle(order)
    outdir = results_dir(a.backend, partial) / "grpo"
    outdir.mkdir(parents=True, exist_ok=True)
    train_sha = sha256_file(train_path)
    fp = json.loads(json.dumps({"steps": a.steps, "limit": a.limit, "train_file": train_path.name, "train_sha256": train_sha,
                                "settings": RC, "model_revision": m["revision"], "prep": a.prep, "backend": a.backend,
                                "dev_reward_jitter": a.dev_reward_jitter, "dev_check_update": a.dev_check_update}))
    for old_path in sorted(outdir.glob("grpo_check_*.json")):        # run-level resume: a finished check is not repeated
        try:
            old = read_json(old_path)
        except Exception:
            continue
        if old.get("finished") is True and old.get("fingerprint") == fp:
            print(f"ALREADY FINISHED: {old_path} holds a finished check with exactly this configuration; nothing to do")
            print("FINISHED grpo_check", old_path)
            return

    run_stamp = fresh_stamp(outdir, "grpo_check_", ".json")
    log_path, samples_path = outdir / f"grpo_check_{run_stamp}.json", outdir / "samples.jsonl"
    sample_kw = dict(do_sample=True, temperature=RC["temperature"], top_p=RC["top_p"], top_k=RC.get("top_k", 0), num_beams=1,
                     max_new_tokens=RC["max_new_tokens"], num_return_sequences=G)
    log = {"label": "PROCEDURE CHECK - no effect is claimed. " + run_label(a.backend, partial), "finished": False, "backend": a.backend,
           "stamp": run_stamp, "settings": RC, "model": m, "prep": a.prep, "steps_requested": a.steps, "limit": a.limit,
           "train_file": train_path.name, "train_sha256": train_sha, "n_rows": len(rows), "fingerprint": fp,
           "item_order": [rows[i]["item_id"] for i in order], "sampling_kwargs": sample_kw, "std_kind": STD_KIND,
           "group_loss": "mean over the group of -advantage_i * mean log-probability of the sampled tokens (EOS included)",
           "optimizer": "torch.optim.AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0", "adapter_saved": False,
           "prereg_commit": commit, "reportable": reportable, "dev_overrides": dev_overrides(), "argv": _s.argv[1:],
           "start": now(), "end": None, "data_manifest": man, "package_files": package_fingerprint(), "steps": [],
           "summary": None, "error": None}
    print(f"[grpo] {a.steps} steps, group {G}, {len(rows)} rows, backend {a.backend}", flush=True)
    print(f"[grpo] expected time: about {minutes(EST['load_s'] + EST['grpo_s_per_step'] * a.steps)} (rough; a measured ETA follows)", flush=True)

    t_run = time.perf_counter()
    torch = tok = model = opt = None
    named, params, force_full, eos_ids, tmpl = [], [], False, set(), cfg["generation"]["chat_template_kwargs"]
    rng = random.Random(RC["seed"])                       # mock completions only
    dev_fail_steps = _o.environ.get("P4_DEV_FAIL_STEPS", "").split(",") if a.backend != "hf" else []   # tests of the error path only
    if a.backend != "mock":
        torch, tok, base, minfo = load_backend(a.backend, m, cfg, seed=RC["seed"])
        log["load_seconds"] = round(time.perf_counter() - t_run, 1)
        log["model_info"] = minfo
        log["vram_cap"] = cap_vram(torch)                 # a memory shortage becomes an error, not a crawl (see common.cap_vram)
        log["vram_free_gb_at_progress"] = []
        log["prepare"] = prepare_for_training(base, a.prep)
        log["generation_config_repository"] = neutral_config(base, tok, RC["max_new_tokens"])
        eos = base.generation_config.eos_token_id
        eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos if eos is not None else tok.eos_token_id}
        sample_kw["pad_token_id"] = tok.pad_token_id
        torch.manual_seed(RC["seed"])                     # LoRA initialisation and the sampling stream
        model = add_lora(base, RC)
        named = trainable(model)
        params = [p for _, p in named]
        opt = torch.optim.AdamW(params, lr=RC["learning_rate"], weight_decay=0.0)
        log["trainable_params"] = sum(p.numel() for p in params)
        log["eos_ids"] = sorted(eos_ids)
        p_ids, a_ids, _ = build_sequence(tok, rows[order[0]], tmpl)
        lp = loss_path_check(model, p_ids, a_ids, params[0].device, torch)
        force_full = lp["force_full"]
        lp["peak_vram_allocated_gb"], lp["peak_vram_reserved_gb"] = vram_peaks(torch)   # of this check alone (full logits, once)
        reset_vram_peaks(torch)                           # the peaks reported below are those of sampling and updating
        log["loss_path_check"] = lp
        print(f"[grpo] model loaded in {log['load_seconds']} s; trainable parameters {log['trainable_params']:,}; "
              f"GPU memory limit {log['vram_cap']}; loss path check {lp}", flush=True)
        stop_if_full_logits(lp, a.backend)                # real backend: the full-logits path does not fit this GPU
        # Preflight of the update path: one forward and backward pass on the first prompt followed by its supervised
        # answer. Groups with equal rewards make no update, so without this a short run (the smoke test) may never
        # execute the update code. No optimizer step is made and the gradient is dropped again.
        model.train()
        x = torch.tensor([p_ids + a_ids], device=params[0].device)
        t = time.perf_counter()
        nll, log["loss_path"] = answer_nll(model, x, torch.ones_like(x), len(p_ids), torch, force_full)
        nll.backward()
        cuda_sync(torch)
        gf = grad_flow(named)
        alloc, reserved = vram_peaks(torch)
        log["preflight_update"] = {"item_id": rows[order[0]]["item_id"], "tokens": int(x.shape[1]), "nll": float(nll.item()),
                                   "seconds": round(time.perf_counter() - t, 3), "peak_vram_allocated_gb": alloc,
                                   "peak_vram_reserved_gb": reserved, "grad_flow": gf}
        del nll, x
        opt.zero_grad(set_to_none=True)
        reset_vram_peaks(torch)                           # the preflight has its own peak fields; the loop starts a new measurement
        print(f"[grpo] preflight of the update path: {log['preflight_update']}", flush=True)
        check_grad_flow(gf)
    log["env"] = env_info()

    failed_in_a_row, stop_reason = 0, None
    t_loop = time.perf_counter()
    try:
        with open(samples_path, "a", encoding="utf-8") as fs:
            for s in range(a.steps):
                row = rows[order[s % len(order)]]
                rec = {"step": s + 1, "item_id": row["item_id"], "failed": False, "error": None, "rewards": None, "statuses": [],
                       "reward_mean": None, "reward_std": None, "zero_variance": None, "updated": False, "loss": None,
                       "grad_norm": None, "seconds": None, "seconds_generate": None, "seconds_update": None,
                       "n_prompt_tokens": None, "n_completion_tokens": None, "share_ok": None, "share_abstain": None, "share_unusable": None}
                t0 = time.perf_counter()
                try:
                    # 1. sample the group
                    if a.backend == "mock":
                        if str(s + 1) in dev_fail_steps:
                            raise RuntimeError("CUDA out of memory (failure requested by P4_DEV_FAIL_STEPS, test of the error path)")
                        texts = mock_completions(row, G, rng)
                        p_ids, comps = [], [[0]] * G
                    else:
                        enc, _ = encode(tok, row["messages"], tmpl)
                        dev = model_device(model)
                        enc = {k: v.to(dev) for k, v in enc.items()}
                        p_ids = enc["input_ids"][0].tolist()
                        free_cuda(torch)                   # blocks cached by the last update are released before the key/value cache grows
                        model.eval()
                        with torch.no_grad():
                            out = model.generate(**enc, **sample_kw)
                        cuda_sync(torch)
                        comps = split_completions(out[:, len(p_ids):].tolist(), eos_ids)
                        texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
                        del out, enc
                    rec["seconds_generate"] = round(time.perf_counter() - t0, 3)
                    rec["n_prompt_tokens"], rec["n_completion_tokens"] = len(p_ids), [len(c) for c in comps]
                    # 2. rewards
                    scored = [reward_of(t, row, W) for t in texts]
                    rewards = [r for r, _, _ in scored]
                    rec["statuses"] = [st for _, st, _ in scored]
                    reasons = [why for _, _, why in scored]
                    if a.dev_reward_jitter:
                        rec["rewards_before_jitter"] = rewards
                        rewards = [r + 0.01 * ((7 * i + 3 * s) % 5) for i, r in enumerate(rewards)]
                    # 3. advantages
                    mean, std, advs = group_advantages(rewards)
                    rec.update({"rewards": rewards, "reward_mean": round(mean, 5), "reward_std": round(std, 5), "zero_variance": advs is None})
                    for name in ("ok", "abstain", "unusable"):
                        rec["share_" + name] = round(rec["statuses"].count(name) / G, 4)
                    # 4. policy-gradient update
                    tu = time.perf_counter()
                    if advs is not None and a.backend == "mock":
                        fake_logp = [-(1.0 + rng.random()) for _ in advs]
                        rec.update({"loss": round(sum(-adv * lp_ for adv, lp_ in zip(advs, fake_logp)) / G, 5), "grad_norm": 0.0, "updated": True})
                    elif advs is not None:
                        model.train()
                        total = 0.0
                        for adv, comp in zip(advs, comps):
                            x = torch.tensor([p_ids + comp], device=dev)
                            nll, log["loss_path"] = answer_nll(model, x, torch.ones_like(x), len(p_ids), torch, force_full)
                            loss_i = adv * nll                 # = -advantage * mean log-probability of the sampled tokens
                            value = float(loss_i.item())
                            if not math.isfinite(value):
                                raise RuntimeError(f"non-finite loss ({value}) in step {s + 1}")
                            (loss_i / G).backward()
                            total += value / G
                            if str(s + 1) in dev_fail_steps:   # in the middle of an update, with gradients already accumulated
                                raise RuntimeError("CUDA out of memory (failure requested by P4_DEV_FAIL_STEPS, test of the error path)")
                            del x, nll, loss_i
                        if "grad_flow_first_update" not in log:
                            log["grad_flow_first_update"] = grad_flow(named)
                            check_grad_flow(log["grad_flow_first_update"])
                        gn = float(torch.nn.utils.clip_grad_norm_(params, RC["max_grad_norm"]))
                        if not math.isfinite(gn):
                            raise RuntimeError(f"non-finite gradient norm ({gn}) in step {s + 1}")
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        cuda_sync(torch)
                        rec.update({"loss": total, "grad_norm": round(gn, 4), "updated": True})
                        if a.dev_check_update:                 # the same completions after the step: the loss must have gone down
                            after = 0.0
                            with torch.no_grad():
                                for adv, comp in zip(advs, comps):
                                    x = torch.tensor([p_ids + comp], device=dev)
                                    after += adv * float(answer_nll(model, x, torch.ones_like(x), len(p_ids), torch, force_full)[0].item()) / G
                            rec["loss_after_update"] = after
                    rec["seconds_update"] = round(time.perf_counter() - tu, 3)
                    fs.write(json.dumps({"stamp": run_stamp, "step": s + 1, "item_id": row["item_id"], "completions": texts,
                                         "rewards": rewards, "statuses": rec["statuses"], "reasons": reasons,
                                         "n_completion_tokens": rec["n_completion_tokens"]}, ensure_ascii=False) + "\n")
                    fs.flush()
                    failed_in_a_row = 0
                except Exception as e:
                    rec.update({"failed": True, "error": {"type": type(e).__name__, "message": str(e)[:1000],
                                                          "out_of_memory": "out of memory" in str(e).lower(),
                                                          "traceback": traceback.format_exc()[-1500:]}})
                    failed_in_a_row += 1
                    print(f"[grpo] step {s + 1} FAILED ({type(e).__name__}: {str(e)[:200]}); the loop goes on", flush=True)
                if rec["failed"] and torch is not None:        # outside the except block, so the failed tensors can be freed
                    enc = out = x = nll = loss_i = None
                    opt.zero_grad(set_to_none=True)
                    free_cuda(torch)
                rec["seconds"] = round(time.perf_counter() - t0, 3)
                log["steps"].append(rec)
                if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                    stop_reason = f"{MAX_FAILED_IN_A_ROW} failed steps in a row; stopped at step {s + 1}"
                    break
                if (s + 1) % 10 == 0 or s + 1 == a.steps:
                    el = time.perf_counter() - t_loop
                    last = [r for r in log["steps"][-10:] if not r["failed"]]
                    alloc = vram_peaks(torch)[0] if torch is not None else None
                    free_now = vram_free_gb(torch) if torch is not None else None
                    print(f"[grpo] step {s + 1}/{a.steps}  mean reward of the last {len(last)} steps "
                          f"{(sum(r['reward_mean'] for r in last) / len(last)) if last else float('nan'):.3f}  "
                          f"zero-variance {sum(1 for r in last if r['zero_variance'])}  updates {sum(1 for r in last if r['updated'])}  "
                          f"failed {sum(1 for r in log['steps'][-10:] if r['failed'])}  "
                          f"{el / (s + 1):.1f} s/step  ETA {minutes(el / (s + 1) * (a.steps - s - 1))}  peak VRAM {alloc} GB  "
                          f"free {free_now} GB{vram_note(free_now)}", flush=True)
                    log["summary"] = summarise(log["steps"], a.steps)
                    if torch is not None:
                        log["peak_vram_allocated_gb"], log["peak_vram_reserved_gb"] = vram_peaks(torch)
                        log["vram_free_gb_at_progress"].append({"step": s + 1, "free_gb": free_now})
                    write_json(log_path, log, must=False)       # partial log: survives a later crash
    except BaseException as e:
        log["error"] = {"type": type(e).__name__, "message": str(e)[:2000], "time": now()}
        log["summary"] = summarise(log["steps"], a.steps)
        write_json(log_path, log)
        raise

    log["summary"] = summarise(log["steps"], a.steps)
    log["peak_vram_allocated_gb"], log["peak_vram_reserved_gb"] = vram_peaks(torch) if torch is not None else (None, None)
    log["summary"].update({"peak_vram_allocated_gb": log["peak_vram_allocated_gb"], "peak_vram_reserved_gb": log["peak_vram_reserved_gb"]})
    if stop_reason is None:
        stop_reason = end_stop_reason(log["summary"], a.steps, partial, a.backend)
    log.update({"end": now(), "wall_seconds": round(time.perf_counter() - t_run, 1), "stop_reason": stop_reason,
                "finished": stop_reason is None and log["summary"]["steps_done"] + log["summary"]["steps_failed"] == a.steps})
    write_json(log_path, log)
    print(f"[grpo] summary: {log['summary']}", flush=True)
    if stop_reason:
        raise SystemExit(f"REWARD CHECK STOPPED: {stop_reason}. The log up to this step is in {log_path}")
    print("FINISHED grpo_check", log_path)


if __name__ == "__main__":
    run_main(main, "grpo_check")
