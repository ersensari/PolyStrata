"""Fresh processes, restored expert profile, alternating GPU/host KV; GPU must be otherwise idle.

python compare.py --model FIRST.gguf --engine NEW --baseline GQA16 --tokenizer PACK/tokenizer \
    --profile SEED --out RESULTS
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from strata_tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    for name in ("model", "engine", "baseline", "tokenizer", "profile", "out"):
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_dir(a.tokenizer)
    short = "Write a complete Python LRU cache class with get and put methods, type hints, docstrings and five unit tests. Explain its time complexity."
    long = "Reference words:\n" + "river stone market lantern copper winter garden signal harbor needle orbit velvet " * 600 + "\n\n" + short
    rows = []

    def run(tag, host=False, text=short, exe=a.engine, draft=True, freeze=False, fmt="q8_0"):
        ids = tok.encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
        profile = out / "active-profile.txt"
        shutil.copyfile(a.profile, profile)
        env = dict(os.environ, STRATA_FATTN="1", STRATA_FDEC="1", STRATA_POLY_KV_DIAGNOSTICS="1")
        env.pop("STRATA_POLY_HOST_KV", None)
        env.pop("STRATA_FREEZE_EXPERTS", None)
        if host: env["STRATA_POLY_HOST_KV"] = "1"
        if freeze: env["STRATA_FREEZE_EXPERTS"] = "1"
        cmd = [exe, "--model", a.model, "--tokens", ",".join(map(str, ids)), "--max-new", "256",
               "--max-context", "262144", "--kv", fmt, "--threads", "12", "--vram-reserve-mib", "1024",
               "--expert-profile", str(profile), "--expert-swaps", "0", "--tail-skip", "0",
               "--temperature", "0", "--window", "8", "--lookup", "3"]
        if not draft: cmd += ["--no-draft", "--window", "1", "--lookup", "0"]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        log = p.stderr.replace(a.model, "<MODEL>").replace(str(profile), "<PROFILE>")
        (out / (tag + ".txt")).write_text(log)
        (out / (tag + ".ids")).write_text(p.stdout)
        match = re.search(r"prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s.*?experts from VRAM ([\d.]+)%, CPU experts ([\d.]+) ms/token, router wait ([\d.]+) ms/token", log)
        loaded = re.search(r"loaded in ([\d.]+) s; (\d+) of (\d+) routed experts in VRAM \(([\d.]+) GiB\)", log)
        row = dict(tag=tag, host_kv=host, draft=draft, freeze=freeze, kv_format=fmt,
                   exit=p.returncode, input_tokens=len(ids), output_sha256=hashlib.sha256(p.stdout.encode()).hexdigest())
        if match:
            row.update(prompt_n=int(match[1]), prompt_tps=float(match[2]), generated_n=int(match[3]),
                       decode_tps=float(match[4]), expert_hit_pct=float(match[5]), cpu_ms_per_token=float(match[6]),
                       router_ms_per_token=float(match[7]))
        if loaded:
            row.update(load_s=float(loaded[1]), gpu_experts=int(loaded[2]), total_experts=int(loaded[3]), expert_cache_gib=float(loaded[4]))
        rows.append(row)
        (out / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
        if p.returncode or not match: raise RuntimeError("engine failed: " + tag)
        return p.stdout

    baseline = run("default-reference", exe=a.baseline, draft=False, freeze=True)
    patched = run("default-patched", draft=False, freeze=True)
    assert baseline == patched, "default path changed"
    for round in range(3):
        for host in ([False, True] if round % 2 == 0 else [True, False]):
            run(f"short-r{round}-{'host' if host else 'gpu'}", host=host)
    for host in (False, True):
        run("long-" + ("host" if host else "gpu"), host=host, text=long)
    # FP16 writes and tensor-core reads use a different cache layout.
    run("fp16-host", host=True, fmt="fp16")


if __name__ == "__main__":
    main()
