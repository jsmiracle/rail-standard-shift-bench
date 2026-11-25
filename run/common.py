"""Shared helpers of the runner (no verification engine is needed to generate model outputs)."""
import hashlib, json, os, re, sys, platform, subprocess, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))          # model cache; set HF_HOME yourself to use another folder
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_config():
    return json.load(open(ROOT / "run" / "config" / "models.json", encoding="utf-8"))

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def verify_manifest():
    man = json.load(open(ROOT / "data" / "MANIFEST.json", encoding="utf-8"))
    bad = {f: (sha256_file(ROOT / "data" / f), h) for f, h in man["sha256"].items() if sha256_file(ROOT / "data" / f) != h}
    if bad:
        raise SystemExit(f"DATA HASH MISMATCH - do not run: {bad}")
    return man

def read_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

def nvidia_smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        full = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        return {"query": out.stdout.strip(), "banner": full.stdout.splitlines()[:12]}
    except Exception as e:
        return {"error": repr(e)}

def env_info():
    info = {"time": now(), "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(), "nvidia_smi": nvidia_smi()}
    try:
        import psutil; info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception as e: info["ram_gb"] = repr(e)
    for mod in ("torch", "transformers", "bitsandbytes", "accelerate", "peft", "huggingface_hub", "safetensors", "tokenizers"):
        try:
            m = __import__(mod); info[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            info[mod] = f"NOT IMPORTABLE: {e!r}"
    try:
        import torch
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "total_memory_gb": round(p.total_memory / 2**30, 2), "capability": f"{p.major}.{p.minor}"}
            info["cudnn"] = torch.backends.cudnn.version()
    except Exception as e:
        info["torch_error"] = repr(e)
    return info
