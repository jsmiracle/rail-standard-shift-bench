"""Shared helpers of the P4 run package.

No railway engine inside. The package holds prompts, training rows (with the gold of the TRAINING tables only) and a
small checker (railcheck.py). Nothing here imports torch at import time: the model helpers import it inside the
functions, so `--backend mock` needs the Python standard library only.

Sections: 1 paths and developer overrides, 2 small io helpers, 3 data manifest and commit guard, 4 environment record,
5 log file and main wrapper, 6 model loading and generation (as in P3 run_p3.py), 7 training helpers."""
import datetime, hashlib, json, math, os, platform, random, re, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))          # run_all_windows.ps1 points this to C:\p2\hf_cache
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CONFIG_FILE = ROOT / "config" / "p4_config.json"
COMMIT_FILE = ROOT / "P4_COMMIT.txt"
# Rough speeds on the target notebook. Inference: measured in P3 on 2026-11-23 (4.9 to 6.6 s per prompt, 16 to 47 s
# to load a cached model). Training and reward check: NOT measured yet, derived from the P3 prefill speed. Only used
# to PRINT expected times; every stage also prints a measured ETA while it runs.
# adapter_infer_factor: inference with a LoRA adapter is slower than without, because the adapter is not merged into the
# 4-bit weights (each of the 252 adapted linear modules does about seven more small GPU operations per generated token).
# The factor 1.7 is an estimate, not a measurement; smoke stage s4 records the real seconds per prompt.
EST = {"load_s": 45.0, "infer_s_per_item": 5.5, "closedbook_s_per_item": 2.5, "sft_s_per_sequence": 1.8, "grpo_s_per_step": 20.0,
       "adapter_infer_factor": 1.7}
LOSS_PATH_ANSWER = "decoder hidden states + LM head on the answer positions only"
LOSS_PATH_FULL = "full logits with labels (plain model call)"
TINY_INIT_SEED = 20261126


# ----------------------------------------------------------------------------------------------------------------------
# 1. Paths and developer overrides
# ----------------------------------------------------------------------------------------------------------------------
def dev_overrides():
    """Developer overrides that are set in the environment. A reportable run refuses to start when any is set."""
    return {k: os.environ[k] for k in ("P4_DATA_DIR", "P4_OUT_DIR") if os.environ.get(k, "").strip()}


def data_dir():
    """Folder of the data files and their MANIFEST.json. Tests point P4_DATA_DIR at tests/fixtures."""
    v = os.environ.get("P4_DATA_DIR", "").strip()
    return Path(v).resolve() if v else ROOT / "data"


def out_root():
    """Folder that receives results/ and adapters/ (the package folder unless P4_OUT_DIR is set)."""
    v = os.environ.get("P4_OUT_DIR", "").strip()
    return Path(v).resolve() if v else ROOT


def results_dir(backend, partial):
    """results/ (real, full run), results/smoke/ (real, --limit), results/<backend>/ and results/<backend>/smoke/."""
    d = out_root() / "results"
    if backend != "hf":
        d = d / backend
    return d / "smoke" if partial else d


def run_label(backend, partial):
    return ("" if backend == "hf" else backend + " ") + ("smoke" if partial else "full")


def out_path(arg):
    """A path argument such as adapters/sft_s1: absolute as given, relative below out_root()."""
    p = Path(str(arg).replace("\\", "/"))
    return p if p.is_absolute() else out_root() / p


# ----------------------------------------------------------------------------------------------------------------------
# 2. Small io helpers
# ----------------------------------------------------------------------------------------------------------------------
def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def fresh_stamp(folder, prefix, suffix):
    """A time stamp for which <folder>/<prefix><stamp><suffix> does not exist yet (two runs in one second get _2, _3 ...)."""
    base = stamp()
    s, k = base, 1
    while (Path(folder) / f"{prefix}{s}{suffix}").exists():
        k += 1
        s = f"{base}_{k}"
    return s


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_config():
    return json.load(open(CONFIG_FILE, encoding="utf-8"))


def model_entry(cfg, tag):
    for m in cfg["models"]:
        if m["tag"] == tag:
            return m
    raise SystemExit(f"UNKNOWN MODEL TAG {tag!r}. Known tags: " + ", ".join(m["tag"] for m in cfg["models"]))


def read_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def read_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def replace_file(tmp, dst):
    """os.replace with a few retries (on Windows a virus scanner or the indexer may hold the target for a moment)."""
    for k in range(6):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if k == 5:
                raise
            time.sleep(0.5)


def write_json(p, obj, must=True):
    """Write a JSON file through a temporary file, so that a crash never leaves a half-written file.
    must=False (progress logs written during a long run): a failure prints a warning instead of stopping the run."""
    p = Path(p)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)
        replace_file(tmp, p)
    except Exception as e:
        if must:
            raise
        print(f"WARNING: could not write {p} ({e!r}); the run goes on", flush=True)


def minutes(seconds):
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


# ----------------------------------------------------------------------------------------------------------------------
# 3. Data manifest and commit guard
# ----------------------------------------------------------------------------------------------------------------------
def verify_manifest():
    """Every file listed in MANIFEST.json of the data folder must exist with the recorded sha256. Runs before every stage."""
    d = data_dir()
    mp = d / "MANIFEST.json"
    if not mp.exists():
        raise SystemExit(f"DATA MANIFEST MISSING - do not run: {mp}")
    man = read_json(mp)
    bad = {}
    for f, h in man["sha256"].items():
        got = sha256_file(d / f) if (d / f).is_file() else "FILE MISSING"
        if got != h:
            bad[f] = (got, h)
    if bad:
        raise SystemExit(f"DATA HASH MISMATCH - do not run: {bad}")
    return man


def data_file(arg, man):
    """Path of a data file given as data/<name> (or as a bare name). Only files listed in the verified manifest are accepted."""
    parts = Path(str(arg).replace("\\", "/")).parts
    ok_form = len(parts) == 1 or (len(parts) == 2 and parts[0] == "data")
    if not ok_form or parts[-1] not in man["sha256"]:
        raise SystemExit(f"DATA FILE NOT IN MANIFEST - do not run: {arg!r}. Allowed: "
                         + ", ".join("data/" + k for k in sorted(man["sha256"])))
    return data_dir() / parts[-1]


def prereg_commit():
    """Reportable runs need the hash of the commit that froze the P4 pre-registration (40 hex chars in P4_COMMIT.txt)."""
    if not COMMIT_FILE.exists():
        return None
    t = COMMIT_FILE.read_text(encoding="utf-8-sig").strip()         # utf-8-sig: a BOM written by a Windows editor is ignored
    h = t.split()[0] if t else ""
    return h if re.fullmatch(r"[0-9a-f]{40}", h) else None


def check_reportable(backend, partial):
    """Returns (commit, reportable). Reportable = real backend and no --limit / --max_steps. A reportable run needs
    P4_COMMIT.txt and must use the package's own data and output folders."""
    commit = prereg_commit()
    reportable = backend == "hf" and not partial
    if reportable and commit is None:
        raise SystemExit("P4_COMMIT.txt missing or invalid: the P4 pre-registration commit hash must be in this file "
                         "(40 hex characters). Smoke tests (--limit) are allowed without it.")
    if reportable and dev_overrides():
        raise SystemExit(f"DEVELOPER OVERRIDE SET - a full run must use the package folders: {dev_overrides()}. "
                         "Remove these environment variables.")
    return commit, reportable


# ----------------------------------------------------------------------------------------------------------------------
# 4. Environment record
# ----------------------------------------------------------------------------------------------------------------------
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


def pip_freeze():
    try:
        return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout.splitlines()
    except Exception as e:
        return [f"pip freeze failed: {e!r}"]


def package_fingerprint():
    """sha256 of the code and settings of this package; recorded in every log to show which code produced it."""
    files = sorted((ROOT / "scripts").glob("*.py")) + [CONFIG_FILE, ROOT / "run_all_windows.ps1"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in files if p.is_file()}


def keep_awake():
    """Windows only, never raises. (1) Ask the system not to go to sleep on idle while this process runs (no setting is
    changed). (2) Switch off QuickEdit of the console window while this process runs: with QuickEdit a mouse click into
    the window freezes the run. The console mode is put back when the process ends."""
    if os.name != "nt":
        return False
    try:
        import atexit, ctypes
        k = ctypes.windll.kernel32
        k.SetThreadExecutionState(ctypes.c_uint32(0x80000000 | 0x00000001))      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        k.GetStdHandle.restype = ctypes.c_void_p
        h = ctypes.c_void_p(k.GetStdHandle(ctypes.c_uint32(0xFFFFFFF6)))          # STD_INPUT_HANDLE (-10)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            original = mode.value
            if k.SetConsoleMode(h, ctypes.c_uint32((original & ~0x0040) | 0x0080)):   # clear QUICK_EDIT, set EXTENDED_FLAGS
                atexit.register(k.SetConsoleMode, h, ctypes.c_uint32(original))
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------------------------------------------------
# 5. Log file and main wrapper
# ----------------------------------------------------------------------------------------------------------------------
class _Tee:
    """Writes to the console stream and to the stage log file. Progress-bar refreshes (carriage return, no newline)
    stay on the console only."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        try:
            self._stream.write(s)
        except Exception:
            pass
        try:
            if not ("\r" in s and not s.endswith("\n")):
                self._fh.write(s); self._fh.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        for x in (self._stream, self._fh):
            try:
                x.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_log(name):
    """Copy everything this process prints to a UTF-8 log file. run_all_windows.ps1 names the file per stage through
    P4_LOG_FILE; without it the file is results/logs/<name>_<time>.log."""
    p = os.environ.get("P4_LOG_FILE", "").strip()
    path = Path(p) if p else out_root() / "results" / "logs" / f"{name}_{stamp()}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a", encoding="utf-8", errors="replace")
        fh.write(f"\n===== {now()} START {name} argv={sys.argv[1:]} =====\n"); fh.flush()
        sys.stdout, sys.stderr = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)
        try:
            import faulthandler
            faulthandler.enable(file=fh)        # a crash inside a native library leaves a Python traceback in the log
        except Exception:
            pass
    except Exception as e:
        print(f"WARNING: log file not started ({e!r}); output goes to the console only", flush=True)
    return path


def log_error(name, kind, message, tb):
    """One row per failure in results/logs/errors.jsonl (never raises)."""
    try:
        p = out_root() / "results" / "logs" / "errors.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": now(), "script": name, "stage": os.environ.get("P4_STAGE", ""), "argv": sys.argv[1:],
                                 "type": kind, "message": message[:2000], "traceback": tb[-4000:]}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_main(main, name):
    """Runs main() with the log file switched on. Every failure ends with one clear line (STOPPED / ERROR) in the log,
    a row in errors.jsonl and a non-zero exit code."""
    start_log(name)
    keep_awake()
    try:
        main()
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):      # a plain exit code, no message
            raise
        print(f"STOPPED [{name}]: {e.code}", flush=True)
        log_error(name, "SystemExit", str(e.code), "")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"INTERRUPTED [{name}]: stopped by the user. Run the same command again to continue.", flush=True)
        log_error(name, "KeyboardInterrupt", "stopped by the user", "")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR [{name}]: {type(e).__name__}: {str(e)[:1000]}", flush=True)
        traceback.print_exc()
        log_error(name, type(e).__name__, str(e), traceback.format_exc())
        sys.exit(1)


# ----------------------------------------------------------------------------------------------------------------------
# 6. Model loading and generation (same code path as P3 run_p3.py, which ran on the target notebook)
# ----------------------------------------------------------------------------------------------------------------------
def chat_template_sha256(tok):
    ct = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((ct or "").encode("utf-8")).hexdigest()


def load_model(m, cfg):
    """Real backend: 4-bit NF4 model on GPU 0 (as run_p3.py, including its fallback loader)."""
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
    return tok, model, {"local_path": str(local), "pinned_revision": m["revision"], "loader": loader, "loader_first_error": first_error,
                        "chat_template_sha256": chat_template_sha256(tok), "tokenizer_class": type(tok).__name__}


def load_tiny(m, cfg):
    """Developer backend for a CPU-only machine: a tiny randomly initialised causal LM (2 layers, hidden size 64) with the
    real chat tokenizer of model m. No quantisation, no bitsandbytes, no CUDA. The weights are the same in every process
    (fixed seed), so an adapter trained on it can be loaded onto it again."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")       # no fork warning on Linux/macOS developer machines
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    local = os.environ.get("P4_TINY_TOKENIZER", "").strip()       # optional: folder with the tokenizer files (offline use)
    if not local:
        from huggingface_hub import snapshot_download
        local = snapshot_download(m["repo"], revision=m["revision"],
                                  allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "*.jinja",
                                                  "special_tokens_map.json", "added_tokens.json"])
    try:
        tok = AutoTokenizer.from_pretrained(local); tok_note = "fast"
    except Exception as e:        # an old `tokenizers` library cannot read a newer tokenizer.json: use the slow tokenizer
        tok = AutoTokenizer.from_pretrained(local, use_fast=False); tok_note = f"slow (fast tokenizer failed: {type(e).__name__})"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        from transformers import Qwen3Config as TinyConfig; arch = "Qwen3"
    except Exception:             # transformers before 4.51 has no Qwen3 classes; Qwen2 has the same layout (.model, .lm_head)
        from transformers import Qwen2Config as TinyConfig; arch = "Qwen2"
    conf = TinyConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=8192, tie_word_embeddings=True,
                      bos_token_id=None, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(TINY_INIT_SEED)
    model = AutoModelForCausalLM.from_config(conf)
    model.eval()
    return tok, model, {"local_path": str(local), "pinned_revision": m["revision"], "loader": f"tiny random {arch} causal LM (CPU)",
                        "loader_first_error": None, "chat_template_sha256": chat_template_sha256(tok),
                        "tokenizer_class": type(tok).__name__, "tokenizer_kind": tok_note, "tiny_init_seed": TINY_INIT_SEED}


def load_backend(backend, m, cfg, seed):
    """Returns (torch, tok, model, model_info) for backend 'hf' (GPU, 4-bit) or 'tiny' (CPU). Seeds torch before loading,
    as run_p3.py does, and resets the peak-memory counters after loading."""
    import torch
    if backend == "hf" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Do not run on CPU; send results/env.json back (README).")
    torch.manual_seed(seed)
    tok, model, minfo = load_model(m, cfg) if backend == "hf" else load_tiny(m, cfg)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return torch, tok, model, minfo


def neutral_config(model, tok, max_new):
    """Replace the repository's generation defaults (sampling settings) by a neutral config; returns the original."""
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


def model_device(model):
    dev = getattr(model, "device", None)
    return dev if dev is not None else next(model.parameters()).device


def cuda_sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gen_once(tok, model, messages, tmpl, kwargs, torch, seed=None):
    """One prompt, one generate call (as run_p3.py). Returns (text, n_input_tokens, n_output_tokens, seconds, template_mode)."""
    enc, mode = encode(tok, messages, tmpl)
    dev = model_device(model)
    enc = {k: v.to(dev) for k, v in enc.items()}
    n_in = int(enc["input_ids"].shape[1])
    if seed is not None:
        torch.manual_seed(seed)
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    cuda_sync(torch)
    new = out[0, n_in:]
    return tok.decode(new, skip_special_tokens=True), n_in, int(new.shape[0]), round(time.perf_counter() - t, 3), mode


def vram_peaks(torch):
    """(peak allocated GB, peak reserved GB) since the last reset; (None, None) without CUDA."""
    if not torch.cuda.is_available():
        return None, None
    return round(torch.cuda.max_memory_allocated() / 2**30, 3), round(torch.cuda.max_memory_reserved() / 2**30, 3)


def vram_free_gb(torch):
    """Memory of the GPU that is free right now, in GB, as the driver reports it (every program counted).
    None without CUDA or if the driver cannot be asked."""
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.mem_get_info()[0] / 2**30, 2)
    except Exception:
        return None


def vram_note(free_gb):
    """A warning text when less than 0.3 GB of the GPU's memory is free (free_gb from vram_free_gb). On Windows the
    driver then moves memory into shared system memory, which makes a run very slow instead of stopping it: a value near
    zero together with a jump in the seconds per step means exactly that. Empty text otherwise."""
    if free_gb is None or free_gb >= 0.3:
        return ""
    return f"  WARNING: only {free_gb} GB of GPU memory free; if the steps become very slow, close other programs"


def cap_vram(torch, keep_free_gb=0.3, floor=0.78):
    """Training stages only (fine-tuning and reward check; NOT inference, which ran in P3 as it is).
    Tells PyTorch not to take more GPU memory than the card can give: limit = what PyTorch holds now + what is free on
    the card now - keep_free_gb, as a fraction of the card, never below `floor` (P3 ran Qwen3-8B on the target notebook
    at 6.33 of 8.0 GB = 0.79 without trouble, so a limit of 0.78 or more cannot refuse what is known to work).

    Why: on Windows the NVIDIA driver does not refuse memory beyond the card; by default it moves the excess into
    shared system memory and the run becomes many times slower instead of stopping. With the limit a shortage is an
    ordinary 'CUDA out of memory' error, which the smoke tests and the per-step handler of the reward check see. Before
    PyTorch raises it, it releases its cached blocks, so memory that is only cached cannot push a run over the edge.
    (torch.cuda.set_per_process_memory_fraction; documented for torch 2.14, NOT yet run on the notebook.)
    Never raises. Returns a dict for the log; None without CUDA."""
    if not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved(0)
        fraction = max(floor, min(1.0, (reserved + free - keep_free_gb * 2**30) / total))
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        return {"fraction": round(fraction, 3), "limit_gb": round(fraction * total / 2**30, 2), "floor": floor,
                "keep_free_gb": keep_free_gb, "free_gb_before": round(free / 2**30, 2),
                "reserved_gb_before": round(reserved / 2**30, 2), "total_gb": round(total / 2**30, 2), "error": None}
    except Exception as e:
        return {"fraction": None, "limit_gb": None, "error": repr(e)}


def attach_adapter(model, adapter_dir):
    """PeftModel.from_pretrained on a loaded base model, eval mode. Stops if the adapter weights did not arrive
    (all LoRA B matrices zero would mean the outputs are the base model's)."""
    import warnings
    from peft import PeftModel
    adapter_dir = Path(adapter_dir)
    f = adapter_dir / "adapter_model.safetensors"
    if not f.is_file():
        raise SystemExit(f"ADAPTER NOT FOUND: {f} (train it first with scripts/train_sft.py)")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pm = PeftModel.from_pretrained(model, str(adapter_dir))
    pm.eval()
    n_b, b_sum = 0, 0.0
    for name, p in pm.named_parameters():
        if "lora_B" in name:
            n_b += 1
            b_sum += float(p.detach().float().abs().sum().item())
    if n_b == 0 or not (b_sum > 0 and math.isfinite(b_sum)):
        raise SystemExit(f"ADAPTER NOT APPLIED: {n_b} LoRA B tensors, sum of absolute values {b_sum}. "
                         "The outputs would be those of the base model; nothing was generated.")
    info = {"path": str(adapter_dir), "adapter_model_sha256": sha256_file(f), "adapter_config": read_json(adapter_dir / "adapter_config.json"),
            "n_lora_B_tensors": n_b, "lora_B_abs_sum": b_sum, "load_warnings": [str(w.message)[:500] for w in caught]}
    return pm, info


# ----------------------------------------------------------------------------------------------------------------------
# 7. Training helpers (supervised fine-tuning and reward check)
# ----------------------------------------------------------------------------------------------------------------------
def epoch_order(n_rows, epochs, seed):
    """Row indices for all epochs: every epoch is a new shuffle drawn from one random.Random(seed) stream."""
    rng = random.Random(seed)
    order = []
    for _ in range(epochs):
        idx = list(range(n_rows))
        rng.shuffle(idx)
        order.extend(idx)
    return order


def chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def lr_at(step, total_steps, warmup_steps, base_lr):
    """Learning rate of optimizer step `step` (0-based): linear warm-up over `warmup_steps` steps, then linear decay
    that would reach zero at step `total_steps`."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def prepare_for_training(model, prep):
    """Freeze the base model and switch gradient checkpointing on (non-reentrant), with input embeddings that require grad.

    prep 'lean' (default): explicit calls. The non-quantised parameters (token embedding tied to the LM head, norm
        weights) keep their loading dtype; the float32 copy that peft's helper makes (peft 0.21.2, utils/other.py:
        every float16/bfloat16 parameter that is not a Params4bit) is avoided. The LoRA matrices are float32 either way.
    prep 'kbit_helper': peft.prepare_model_for_kbit_training (casts those parameters to float32); kept as a fallback."""
    if prep == "kbit_helper":
        from peft import prepare_model_for_kbit_training
        prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    if not getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # transformers 5 registers the embedding hook inside gradient_checkpointing_enable; older versions do not
    if not (getattr(model, "_require_grads_hooks", None) or getattr(model, "_require_grads_hook", None)):
        model.enable_input_require_grads()
    emb = model.get_input_embeddings().weight
    return {"prep": prep, "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
            "use_reentrant": False, "input_embedding_dtype": str(emb.dtype),
            "tied_lm_head": bool(getattr(model, "lm_head", None) is not None and model.lm_head.weight is emb)}


def add_lora(model, s):
    """get_peft_model with the LoRA settings of a config section (keys lora_r, lora_alpha, lora_dropout, target_modules)."""
    from peft import LoraConfig, get_peft_model
    conf = LoraConfig(r=s["lora_r"], lora_alpha=s["lora_alpha"], lora_dropout=s["lora_dropout"],
                      target_modules=list(s["target_modules"]), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, conf)


def trainable(model):
    """[(name, parameter)] of the parameters that are trained (the LoRA matrices)."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def build_sequence(tok, row, tmpl_kwargs):
    """Token ids of one training row: (prompt ids, answer ids). Prompt = chat template of `messages` with the generation
    prompt (the same encode() as inference); answer = the `answer` string followed by the EOS token."""
    enc, mode = encode(tok, row["messages"], tmpl_kwargs)
    p_ids = enc["input_ids"][0].tolist()
    if tok.eos_token_id is None:
        raise SystemExit("TOKENIZER HAS NO EOS TOKEN - the supervised target cannot be built")
    a_ids = list(tok(row["answer"], add_special_tokens=False)["input_ids"]) + [tok.eos_token_id]
    return p_ids, a_ids, mode


def check_lengths(seqs, rows, max_length, what):
    """Stops if a sequence is longer than max_length (nothing is ever truncated). Returns token statistics."""
    lens = [len(p) + len(a) for p, a, _ in seqs]
    for n, r in zip(lens, rows):
        if n > max_length:
            raise SystemExit(f"SEQUENCE TOO LONG in {what}: item {r['item_id']} has {n} tokens > max_length {max_length}. "
                             "Nothing was truncated and nothing was trained; fix the data or the registered max_length.")
    return {"n": len(lens), "mean": round(sum(lens) / len(lens), 1), "max": max(lens), "min": min(lens),
            "answer_tokens_mean": round(sum(len(a) for _, a, _ in seqs) / len(seqs), 1), "answer_tokens_max": max(len(a) for _, a, _ in seqs),
            "template_modes": sorted({mode for _, _, mode in seqs})}


def answer_nll(model, input_ids, attention_mask, start, torch, force_full=False):
    """Mean negative log-likelihood of the tokens input_ids[0, start:] given everything before them (scalar with grad).

    Preferred path: run the decoder for the hidden states and apply the LM head to the answer positions only, so that
    logits over the whole vocabulary are never built for the prompt. Fallback (attributes missing or force_full):
    the plain model call with labels. Returns (loss, name of the path used)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not force_full and hasattr(base, "model") and hasattr(base, "lm_head"):
        out = base.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        logits = base.lm_head(h[:, start - 1:-1, :])                 # position t-1 predicts token t
        targets = input_ids[:, start:]
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, LOSS_PATH_ANSWER
    labels = input_ids.clone()
    labels[:, :start] = -100
    loss = base(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False).loss
    return loss, LOSS_PATH_FULL


def loss_path_check(model, p_ids, a_ids, dev, torch, tolerance=0.02):
    """Before any training: the answer-only loss must equal the plain full-logits loss on one sequence (eval mode, no
    gradient). If the two differ by more than `tolerance`, training uses the plain path (force_full)."""
    was_training = model.training
    model.eval()
    x = torch.tensor([p_ids + a_ids], device=dev)
    mask = torch.ones_like(x)
    with torch.no_grad():
        a, path = answer_nll(model, x, mask, len(p_ids), torch)
        b, _ = answer_nll(model, x, mask, len(p_ids), torch, force_full=True)
    a, b = float(a.item()), float(b.item())
    model.train(was_training)
    ok = math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance
    return {"tokens": int(x.shape[1]), "preferred_path": path, "answer_only_loss": a, "full_logits_loss": b,
            "abs_diff": abs(a - b), "tolerance": tolerance, "agree": ok, "force_full": (not ok) or path == LOSS_PATH_FULL}


def stop_if_full_logits(lp, backend):
    """Real backend only: stop when training would have to use the plain full-logits loss path (lp from loss_path_check).

    That path builds logits over the whole vocabulary for every position: about 3 GB more than the answer-only path at
    1,200 tokens, which this 8 GB GPU does not have (on Windows the run would crawl in shared system memory instead of
    stopping). For the pinned Qwen3 models the two paths cannot legitimately differ (the model's forward applies only
    lm_head to the decoder's hidden states), so a disagreement is a fault that must be looked at, not trained through.
    The developer backends (tiny) keep the fallback."""
    if backend != "hf" or not lp["force_full"]:
        return
    if not lp["agree"]:
        why = f"LOSS PATHS DISAGREE (difference {lp['abs_diff']:.4f}, tolerance {lp['tolerance']})"
    else:
        why = "ANSWER-ONLY LOSS PATH NOT AVAILABLE (the loaded model has no .model / .lm_head)"
    raise SystemExit(f"{why}: the plain full-logits path needs about 3 GB more than this GPU has. "
                     "Nothing was trained; send the log.")


def grad_flow(named):
    """After a backward pass: how many trainable tensors received a non-zero gradient, and whether the first decoder
    layer did (that proves the gradient passes through every checkpointed layer)."""
    nonzero = [n for n, p in named if p.grad is not None and float(p.grad.detach().float().abs().sum().item()) > 0]
    has_layer0 = any(".layers.0." in n for n, _ in named)
    return {"n_trainable_tensors": len(named), "n_with_nonzero_grad": len(nonzero),
            "first_layer_has_grad": any(".layers.0." in n for n in nonzero) if has_layer0 else None,
            "example": nonzero[0] if nonzero else None}


def check_grad_flow(info):
    if info["n_with_nonzero_grad"] == 0 or info["first_layer_has_grad"] is False:
        raise SystemExit(f"NO GRADIENT REACHED THE ADAPTER: {info}. Gradient checkpointing or the frozen base blocks the "
                         "gradient; nothing was trained.")


def free_cuda(torch):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_vram_peaks(torch):
    """Start a new measurement of the peak GPU memory: release what is not in use, then reset the peak counters.
    Called after the one-off loss-path check (it builds logits for a whole sequence once), so that the peaks reported
    for training are those of training. Does nothing without CUDA."""
    free_cuda(torch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
