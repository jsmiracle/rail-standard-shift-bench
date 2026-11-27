"""Step 0: writes results/env.json, prints a short verdict and the plan with expected times.
Usage: python scripts/env_report.py [--backend hf|mock|tiny]
Exit code 0 = GPU, bitsandbytes and peft are ready; 2 = not ready (the stages will stop with a clear message)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # works also under PYTHONSAFEPATH

import argparse, json, os, shutil
from pathlib import Path
from common import (EST, ROOT, data_dir, dev_overrides, env_info, load_config, out_root, package_fingerprint, prereg_commit,
                    read_jsonl, results_dir, run_main, verify_manifest, write_json)


def model_cache(cfg):
    """Is the pinned snapshot of every model already in the Hugging Face cache? (No download is needed then.)"""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ["HF_HOME"]) / "hub")
    out = {}
    for m in cfg["models"]:
        snap = hub / ("models--" + m["repo"].replace("/", "--")) / "snapshots" / m["revision"]
        out[m["tag"]] = {"snapshot": str(snap), "present": snap.is_dir() and any(snap.glob("*.safetensors"))}
    return out


def plan(cfg):
    """Expected time of every stage of run_all_windows.ps1, from the row counts of the data files (rough)."""
    d = data_dir()
    test = read_jsonl(d / Path(cfg["files"]["items_test"]).name)
    n_v1 = sum(1 for r in test if r.get("variant") == "V1"); n_v2 = sum(1 for r in test if r.get("variant") == "V2")
    n_cb = len(read_jsonl(d / Path(cfg["files"]["closedbook"]).name))
    n_train = len(read_jsonl(d / Path(cfg["sft"]["train_file"]).name))
    S, RC = cfg["sft"], cfg["reward_check"]
    n_seq = n_train * S["epochs"]
    L, inf = EST["load_s"], EST["infer_s_per_item"]
    rows = [("smoke tests of every stage", 12 * 60),
            (f"closed-book questions, 8B ({n_cb} prompts)", L + EST["closedbook_s_per_item"] * n_cb),
            (f"baseline 8B, V1 ({n_v1} prompts)", L + inf * n_v1),
            (f"baseline 4B, V1 and V2 ({n_v1 + n_v2} prompts)", L + inf * (n_v1 + n_v2))]
    for seed in S["seeds"]:
        rows.append((f"fine-tuning seed {seed} ({-(-n_seq // S['grad_accum'])} steps, {n_seq} sequences)", L + EST["sft_s_per_sequence"] * n_seq))
        # with the adapter every generated token is slower (estimate, see EST in common.py)
        rows.append((f"evaluation of seed {seed}, V2 ({n_v2} prompts)", L + EST["adapter_infer_factor"] * inf * n_v2))
    rows.append((f"reward-learning check ({RC['steps']} steps, group {RC['group']})", L + EST["grpo_s_per_step"] * RC["steps"]))
    return [{"stage": name, "expected_minutes": round(sec / 60)} for name, sec in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hf", choices=["hf", "mock", "tiny"])
    a = ap.parse_args()
    cfg = load_config()
    out = results_dir(a.backend, False)
    out.mkdir(parents=True, exist_ok=True)
    info = env_info()
    info["backend"] = a.backend
    info["package_root"] = str(ROOT)
    info["hf_home"] = os.environ.get("HF_HOME")
    info["dev_overrides"] = dev_overrides()
    info["prereg_commit"] = prereg_commit()
    info["package_files"] = package_fingerprint()
    info["disk_free_gb"] = round(shutil.disk_usage(out_root()).free / 2**30, 1)
    info["model_cache"] = model_cache(cfg)
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["gpu_memory_free_gb_now"] = round(free / 2**30, 2)
    except Exception as e:
        info["gpu_memory_free_gb_now"] = repr(e)
    write_json(out / "env.json", info)                       # written first: it exists even if the data check stops below
    man = verify_manifest()
    info["data_manifest"] = man["sha256"]
    info["plan"] = plan(cfg)
    write_json(out / "env.json", info)

    print(json.dumps({k: info.get(k) for k in ("python", "torch", "torch_cuda_build", "cuda_available", "gpu", "transformers",
                                               "bitsandbytes", "peft", "gpu_memory_free_gb_now", "disk_free_gb", "prereg_commit")},
                     ensure_ascii=False, indent=1))
    for tag, c in info["model_cache"].items():
        print(f"model {tag}: " + ("in the cache" if c["present"] else "NOT in the cache - it will be downloaded first (several GB, needs internet)"))
    if info["prereg_commit"] is None:
        print("WARNING: P4_COMMIT.txt missing or invalid - the full runs will refuse to start (smoke tests still run)")
    if info["disk_free_gb"] < 3:
        print(f"WARNING: only {info['disk_free_gb']} GB free on this drive; the adapters and checkpoints need about 2 GB (README_RU.md asks for 3 GB)")
    if isinstance(info.get("gpu_memory_free_gb_now"), float) and info["gpu_memory_free_gb_now"] < 6.8:
        print(f"WARNING: only {info['gpu_memory_free_gb_now']} GB of GPU memory are free now - close the browser and other programs")
    print("PLAN (expected times are rough; every stage prints a measured ETA):")
    for row in info["plan"]:
        print(f"  {row['expected_minutes']:>4} min  {row['stage']}")
    print(f"  {sum(r['expected_minutes'] for r in info['plan']):>4} min  TOTAL (about {sum(r['expected_minutes'] for r in info['plan']) / 60:.1f} hours)")
    ok_gpu = info.get("cuda_available") is True
    libs_ok = not any(str(info.get(k, "")).startswith("NOT") for k in ("bitsandbytes", "peft", "transformers", "torch"))
    print("GPU OK" if ok_gpu and libs_ok else "GPU NOT READY - do not run on the CPU; send the zip (README_RU.md, section 5)")
    _s.exit(0 if (ok_gpu and libs_ok) else 2)


if __name__ == "__main__":
    run_main(main, "env_report")
