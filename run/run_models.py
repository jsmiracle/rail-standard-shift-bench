"""Runner: an open language model turns each benchmark document into one JSON condition.
This script only generates outputs; scoring is done by checker/score.py. It was used for the published run
(Windows 11, NVIDIA RTX 4070 Laptop 8 GB, 4-bit NF4 quantisation; see results/env_run.json).

Usage (from the repository root, in an environment with the packages of run/requirements.txt and a CUDA GPU):
  python run/run_models.py --model qwen3_8b                 # all variants of this model (run/config/models.json)
  python run/run_models.py --model qwen3_8b --limit 3       # smoke test -> outputs/smoke/
  python run/run_models.py --model qwen3_8b --backend mock  # pipeline test without a model -> outputs/mock/
Re-running the same command resumes: items with a saved output are skipped; failed items are retried.
Output: outputs/raw/<model>/<variant>.jsonl  (V0, V1: {item_id, output}; V1SC: {item_id, outputs:[5]}),
        <variant>.errors.jsonl, run_manifest_<time>.json
Then:   python checker/score.py --raw outputs/raw --out my_scores.json"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # works also under PYTHONSAFEPATH

import argparse, hashlib, json, subprocess, time, traceback
from common import ROOT, load_config, verify_manifest, read_jsonl, sha256_file, env_info, now


def pip_freeze():
    try:
        return subprocess.run([_s.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout.splitlines()
    except Exception as e:
        return [f"pip freeze failed: {e!r}"]


def load_model(m, cfg):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    local = snapshot_download(m["repo"], revision=m["revision"],
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
    tok = AutoTokenizer.from_pretrained(local)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = cfg["quantization"]
    kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
                                                    bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
                                                    bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"])),
          "device_map": {"": 0}, "dtype": torch.bfloat16}
    first_error = None
    try:
        model = AutoModelForCausalLM.from_pretrained(local, **kw); loader = "AutoModelForCausalLM"
    except Exception as e1:
        first_error = {"type": type(e1).__name__, "message": str(e1)[:4000]}
        if "out of memory" in str(e1).lower():
            raise
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(local, **kw); loader = "AutoModelForImageTextToText (fallback)"
    model.eval()
    ct = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template, sort_keys=True, ensure_ascii=False)
    return tok, model, {"local_path": str(local), "pinned_revision": m["revision"], "loader": loader, "loader_first_error": first_error,
                        "chat_template_sha256": hashlib.sha256((ct or "").encode("utf-8")).hexdigest(), "tokenizer_class": type(tok).__name__}


def neutral_config(model, tok, max_new):
    from transformers import GenerationConfig
    orig = model.generation_config.to_dict() if getattr(model, "generation_config", None) is not None else {}
    model.generation_config = GenerationConfig(eos_token_id=orig.get("eos_token_id", tok.eos_token_id), bos_token_id=orig.get("bos_token_id"),
                                               pad_token_id=tok.pad_token_id, do_sample=False, max_new_tokens=max_new)
    return orig


def encode(tok, messages, tmpl_kwargs):
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **tmpl_kwargs), "as_is"
    except Exception:
        sysmsg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        rest = messages[1:] if sysmsg else messages
        merged = [dict(rest[0], content=sysmsg + "\n\n" + rest[0]["content"])] + rest[1:]
        return tok.apply_chat_template(merged, add_generation_prompt=True, return_tensors="pt", return_dict=True, **tmpl_kwargs), "system_merged"


def gen_once(tok, model, messages, tmpl, kwargs, torch, seed=None):
    enc, mode = encode(tok, messages, tmpl)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = int(enc["input_ids"].shape[1])
    if seed is not None:
        torch.manual_seed(seed)
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    torch.cuda.synchronize()
    new = out[0, n_in:]
    return tok.decode(new, skip_special_tokens=True), n_in, int(new.shape[0]), round(time.perf_counter() - t, 3), mode


def read_done(f):
    done = set()
    if f.exists():
        good = []
        for l in open(f, encoding="utf-8"):
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get("error") is None and (d.get("output") is not None or d.get("outputs")):
                done.add(d["item_id"]); good.append(json.dumps(d, ensure_ascii=False) + "\n")
        open(f, "w", encoding="utf-8").writelines(good)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"])
    a = ap.parse_args()
    cfg = load_config(); man = verify_manifest()
    m = next(x for x in cfg["models"] if x["tag"] == a.model)
    variants = a.variants.split(",") if a.variants else m["variants"]
    items = read_jsonl(ROOT / cfg["items_file"])
    by_var = {}
    for it in items:
        by_var.setdefault(it["variant"], []).append(it)
    label = "mock" if a.backend == "mock" else ("smoke" if a.limit else "raw")
    outdir = ROOT / "outputs" / label / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    if a.backend == "mock":
        for v in variants:
            src = by_var["V1" if v == "V1SC" else v][: a.limit or None]
            with open(outdir / f"{v}.jsonl", "w", encoding="utf-8") as fh:
                for it in src:
                    o = '{"abstain": true}'
                    fh.write(json.dumps({"item_id": it["item_id"], ("outputs" if v == "V1SC" else "output"): ([o] * 5 if v == "V1SC" else o)}, ensure_ascii=False) + "\n")
        print("mock written", outdir); return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. The published run used a CUDA GPU; this runner does not run on CPU.")
    torch.manual_seed(cfg["generation"]["seed"])
    t0 = time.perf_counter()
    tok, model, minfo = load_model(m, cfg)
    load_s = round(time.perf_counter() - t0, 1)
    g, sc, tmpl = cfg["generation"]["greedy"], cfg["generation"]["self_consistency"], cfg["generation"]["chat_template_kwargs"]
    orig_gc = neutral_config(model, tok, g["max_new_tokens"])
    greedy_kw = dict(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=g["max_new_tokens"], pad_token_id=tok.pad_token_id,
                     temperature=None, top_p=None, top_k=None)
    sc_kw = dict(do_sample=True, temperature=sc["temperature"], top_p=sc["top_p"], top_k=sc["top_k"], num_beams=1,
                 max_new_tokens=sc["max_new_tokens"], pad_token_id=tok.pad_token_id)
    torch.cuda.reset_peak_memory_stats()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = {"label": label, "model": m, "start": now(), "load_seconds": load_s, "model_info": minfo,
           "generation_config_repository": orig_gc, "greedy_kwargs": greedy_kw, "sc_kwargs": sc_kw, "sc_n": sc["n"], "sc_seed_base": sc["seed_base"],
           "chat_template_kwargs": tmpl, "env": env_info(), "pip_freeze": pip_freeze(), "data_manifest": man, "quantization": cfg["quantization"],
           "variants": {}, "rendered_prompt_example": {}}
    for v in variants:
        src = by_var["V1" if v == "V1SC" else v]
        src = src[: a.limit] if a.limit else src
        f, fe = outdir / f"{v}.jsonl", outdir / f"{v}.errors.jsonl"
        done = read_done(f)
        tv = time.perf_counter(); n_new = n_err = 0
        with open(f, "a", encoding="utf-8") as fh, open(fe, "a", encoding="utf-8") as fhe:
            for i, it in enumerate(src):
                if it["item_id"] in done:
                    continue
                rec = {"item_id": it["item_id"], "variant": v, "model_tag": a.model}
                try:
                    if v == "V1SC":
                        outs, secs, nout = [], 0.0, []
                        for k in range(sc["n"]):
                            o, n_in, n_out, s, mode = gen_once(tok, model, it["messages"], tmpl, sc_kw, torch, seed=sc["seed_base"] + k)
                            outs.append(o); secs += s; nout.append(n_out)
                        rec.update({"outputs": outs, "n_input_tokens": n_in, "n_output_tokens": nout, "seconds": round(secs, 3), "template_mode": mode})
                    else:
                        o, n_in, n_out, s, mode = gen_once(tok, model, it["messages"], tmpl, greedy_kw, torch)
                        rec.update({"output": o, "n_input_tokens": n_in, "n_output_tokens": n_out, "seconds": s, "template_mode": mode})
                    if v not in run["rendered_prompt_example"]:
                        run["rendered_prompt_example"][v] = {"item_id": it["item_id"], "n_input_tokens": rec["n_input_tokens"]}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush(); n_new += 1
                except Exception as e:
                    n_err += 1
                    rec.update({"output": None, "error": repr(e), "traceback": traceback.format_exc()[-1500:], "time": now()})
                    fhe.write(json.dumps(rec, ensure_ascii=False) + "\n"); fhe.flush()
                if (i + 1) % 10 == 0:
                    print(f"[{a.model}] {v}: {i + 1}/{len(src)}  {time.perf_counter() - tv:.0f}s", flush=True)
        run["variants"][v] = {"n_items": len(src), "n_new": n_new, "n_errors": n_err, "wall_seconds": round(time.perf_counter() - tv, 1)}
        print(f"[{a.model}] {v} done: {run['variants'][v]}", flush=True)
    run["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    run["end"] = now()
    json.dump(run, open(outdir / f"run_manifest_{stamp}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("FINISHED", a.model, outdir)


if __name__ == "__main__":
    main()
