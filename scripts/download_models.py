"""Download all models required by Chain-of-Zoom into the local HF cache and ckpt/ tree.

Run from workspace root after `source ./activate.sh`.
"""
from __future__ import annotations

import os
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "ckpt"
RAM_DEST = CKPT_DIR / "RAM" / "ram_swin_large_14m.pth"


def fetch_sd3() -> str:
    path = snapshot_download(
        repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "text_encoder_2/*",
            "text_encoder_3/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "tokenizer_3/*",
            "transformer/*",
            "vae/*",
        ],
        max_workers=8,
    )
    return f"SD3 -> {path}"


def fetch_qwen() -> str:
    path = snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        max_workers=8,
    )
    return f"Qwen2.5-VL -> {path}"


def fetch_ram() -> str:
    cached = hf_hub_download(
        repo_id="xinyu1205/recognize_anything_model",
        filename="ram_swin_large_14m.pth",
    )
    RAM_DEST.parent.mkdir(parents=True, exist_ok=True)
    if RAM_DEST.exists() or RAM_DEST.is_symlink():
        RAM_DEST.unlink()
    os.symlink(cached, RAM_DEST)
    return f"RAM -> {RAM_DEST} (-> {cached})"


def main() -> int:
    jobs = {
        "ram": fetch_ram,
        "sd3": fetch_sd3,
        "qwen": fetch_qwen,
    }
    results: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {ex.submit(fn): name for name, fn in jobs.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[name] = exc

    failed = False
    for name, res in results.items():
        if isinstance(res, BaseException):
            print(f"[FAIL] {name}: {type(res).__name__}: {res}", file=sys.stderr)
            failed = True
        else:
            print(f"[ OK ] {res}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
