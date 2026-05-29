#!/usr/bin/env python3
"""Download and preprocess DIV2K HR into a fixed 512x512 train/valid set (W0.T6).

Steps:
  1. Download official DIV2K HR zips into ``data/div2k/_raw/`` (gitignored):
     DIV2K_train_HR.zip (ids 0001-0800) and DIV2K_valid_HR.zip (ids 0801-0900).
     Downloads are resumable and skipped when a valid zip already exists.
  2. Apply the exact inference preprocessing ``resize_and_center_crop(img, 512)``
     (LANCZOS resize to min-side==512, then a centred 512x512 crop) and write
     ``data/div2k/{train,valid}/<id>.png``.
  3. Write absolute-path manifests ``data/manifests/{train,valid}.txt``.
  4. Assert ``set(train_ids) & set(valid_ids) == empty`` (R8 - no eval leak) and,
     on success, write ``data/manifests/disjoint_ok.txt`` with the counts.

Idempotent: valid zips are not re-downloaded and correct 512x512 PNGs are not
reprocessed. The smaller ``valid`` split is processed first to unblock eval.

Run from the repo root after ``source ./activate.sh``::

    python scripts/data/prepare_div2k.py

resize_and_center_crop import: ``inference_coz.py`` can temporarily hold git
merge-conflict markers, but they live only inside its ``__main__`` block while
resize_and_center_crop (and everything above it) is identical on both branches.
We import normally first and, only if the whole module raises ``SyntaxError``
from those markers, fall back to executing just that one clean function from
source - byte-for-byte the same preprocessing either way.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypedDict, cast

from PIL import Image

ResizeFn = Callable[[Image.Image, int], Image.Image]


class SplitCfg(TypedDict):
    zip: str
    out: Path
    ids: range
    expect: int


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DIV2K = DATA / "div2k"
RAW = DIV2K / "_raw"
MANIFESTS = DATA / "manifests"

SIZE = 512
BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"

SPLITS: dict[str, SplitCfg] = {
    "valid": {"zip": "DIV2K_valid_HR.zip", "out": DIV2K / "valid", "ids": range(801, 901), "expect": 100},
    "train": {"zip": "DIV2K_train_HR.zip", "out": DIV2K / "train", "ids": range(1, 801), "expect": 800},
}


def _load_resize_and_center_crop() -> ResizeFn:
    try:
        from inference_coz import resize_and_center_crop  # type: ignore

        print("[init] imported resize_and_center_crop from inference_coz", flush=True)
        return cast(ResizeFn, resize_and_center_crop)
    except SyntaxError as exc:
        print(
            f"[init] inference_coz.py not importable (SyntaxError @ line {exc.lineno}); repo is mid-merge - extracting the unconflicted resize_and_center_crop from source.",
            flush=True,
        )

    src_path = ROOT / "inference_coz.py"
    captured: list[str] = []
    grabbing = False
    for ln in src_path.read_text().splitlines(keepends=True):
        if not grabbing:
            if re.match(r"^def\s+resize_and_center_crop\b", ln):
                grabbing = True
                captured.append(ln)
            continue
        if ln.strip() and not ln[:1].isspace():
            break
        captured.append(ln)
    func_src = "".join(captured).rstrip() + "\n"
    has_conflict = re.search(r"^(<<<<<<<|=======|>>>>>>>)", func_src, flags=re.M) is not None
    if "def resize_and_center_crop" not in func_src or has_conflict:
        raise RuntimeError(
            f"Could not cleanly extract resize_and_center_crop from {src_path}; refusing to guess preprocessing."
        )
    namespace: dict[str, object] = {"Image": Image}
    exec(compile(func_src, str(src_path), "exec"), namespace)  # noqa: S102
    print("[init] loaded resize_and_center_crop via source-extraction fallback", flush=True)
    return cast(ResizeFn, namespace["resize_and_center_crop"])


def _zip_png_count(zip_path: Path) -> int:
    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        return -1
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return sum(1 for n in zf.namelist() if n.lower().endswith(".png"))
    except (zipfile.BadZipFile, OSError):
        return -1


def _have_valid_zip(zip_path: Path, min_pngs: int) -> bool:
    return _zip_png_count(zip_path) >= min_pngs


def _human(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024 or unit == "TiB":
            return f"{val:.2f} {unit}"
        val /= 1024
    return f"{nbytes} B"


def _run_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "--fail", "--retry", "5", "--retry-delay",
               "3", "--connect-timeout", "30", "-o", str(dest), url]
    elif shutil.which("wget"):
        cmd = ["wget", "-c", "--tries=5", "-O", str(dest), url]
    else:
        raise RuntimeError("Neither curl nor wget is available for downloading.")
    print(f"[dl] $ {' '.join(cmd)}", flush=True)
    _ = subprocess.run(cmd, check=True)


def ensure_zip(name: str, cfg: SplitCfg) -> tuple[Path, int, float]:
    """Ensure a valid zip is present; returns (path, bytes_downloaded, seconds)."""
    zip_path = RAW / cfg["zip"]
    url = BASE_URL + cfg["zip"]
    expect = cfg["expect"]

    if _have_valid_zip(zip_path, expect):
        print(f"[dl] {name}: valid zip cached, skipping ({_human(zip_path.stat().st_size)})", flush=True)
        return zip_path, 0, 0.0

    print(f"[dl] {name}: downloading {url}", flush=True)
    start = time.perf_counter()
    try:
        _run_download(url, zip_path)
    except subprocess.CalledProcessError as exc:
        # A stale partial file wedges resume with HTTP 416; retry once from scratch.
        print(f"[dl] {name}: resume failed ({exc}); retrying clean", flush=True)
        if zip_path.exists():
            zip_path.unlink()
        _run_download(url, zip_path)
    elapsed = time.perf_counter() - start

    pngs = _zip_png_count(zip_path)
    if pngs < expect:
        raise RuntimeError(
            f"[dl] {name}: zip {zip_path} incomplete ({pngs} png members, expected >= {expect})."
        )
    size = zip_path.stat().st_size
    rate = (size / elapsed / 1024 / 1024) if elapsed > 0 else float("nan")
    print(f"[dl] {name}: OK {_human(size)} in {elapsed:.1f}s ({rate:.1f} MiB/s), {pngs} members", flush=True)
    return zip_path, size, elapsed


def _already_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            return im.size == (SIZE, SIZE)
    except (OSError, Image.DecompressionBombError):
        return False


def process_split(name: str, cfg: SplitCfg, resize_fn: ResizeFn) -> list[str]:
    """Preprocess every in-range HR image; returns the sorted id stems written."""
    out_dir = cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    id_range = cfg["ids"]
    zip_path = RAW / cfg["zip"]

    done_ids: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
        total = len(members)
        processed = skipped = 0
        for i, member in enumerate(members, start=1):
            stem = Path(member).stem
            try:
                idn = int(stem)
            except ValueError:
                print(f"[{name}] skip non-numeric member: {member}", flush=True)
                continue
            if idn not in id_range:
                print(f"[{name}] skip out-of-range id {stem}", flush=True)
                continue

            dst = out_dir / f"{stem}.png"
            if _already_done(dst):
                done_ids.add(stem)
                skipped += 1
            else:
                with Image.open(io.BytesIO(zf.read(member))) as img:
                    cropped = resize_fn(img.convert("RGB"), SIZE)
                if cropped.size != (SIZE, SIZE):
                    raise RuntimeError(f"[{name}] {member} -> {cropped.size}, expected ({SIZE}, {SIZE}).")
                tmp = dst.with_suffix(".png.tmp")
                cropped.save(tmp, format="PNG")
                os.replace(tmp, dst)
                done_ids.add(stem)
                processed += 1

            if i % 50 == 0 or i == total:
                print(f"[{name}] {i}/{total} (new={processed} skipped={skipped})", flush=True)

    out = sorted(done_ids)
    print(f"[{name}] complete: {len(out)} images (expected {cfg['expect']}) -> {out_dir}", flush=True)
    return out


def write_manifest(path: Path, out_dir: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str((out_dir / f"{i}.png").resolve()) for i in ids]
    _ = path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"[manifest] wrote {len(lines)} paths -> {path}", flush=True)


def main() -> int:
    print(f"[init] repo root: {ROOT}", flush=True)
    for d in (RAW, MANIFESTS):
        d.mkdir(parents=True, exist_ok=True)

    resize_fn = _load_resize_and_center_crop()

    split_ids: dict[str, list[str]] = {}
    total_bytes = 0
    total_dl_secs = 0.0

    for name in ("valid", "train"):
        cfg = SPLITS[name]
        _, dl_bytes, dl_secs = ensure_zip(name, cfg)
        total_bytes += dl_bytes
        total_dl_secs += dl_secs

        ids = process_split(name, cfg, resize_fn)
        split_ids[name] = ids
        write_manifest(MANIFESTS / f"{name}.txt", cfg["out"], ids)

        if len(ids) != cfg["expect"]:
            # Never silently shrink a split; flag loudly without faking the count.
            print(
                f"[BLOCKER] {name}: produced {len(ids)} images, expected {cfg['expect']}.",
                file=sys.stderr, flush=True,
            )

    train_ids = set(split_ids["train"])
    valid_ids = set(split_ids["valid"])

    overlap = train_ids & valid_ids  # R8: must be empty (no eval leak)
    if overlap:
        raise RuntimeError(f"DISJOINT ASSERTION FAILED (R8): leaked ids {sorted(overlap)[:10]}")
    bad_train = [i for i in train_ids if not 1 <= int(i) <= 800]
    bad_valid = [i for i in valid_ids if not 801 <= int(i) <= 900]
    if bad_train or bad_valid:
        raise RuntimeError(f"id range violation: bad_train={bad_train[:5]} bad_valid={bad_valid[:5]}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    disjoint_path = MANIFESTS / "disjoint_ok.txt"
    disjoint_lines = [
        "DIV2K train/valid id split is DISJOINT (R8 verified)",
        f"verified_utc={stamp}",
        f"train_count={len(train_ids)}",
        f"valid_count={len(valid_ids)}",
        "train_id_range=0001-0800",
        "valid_id_range=0801-0900",
        f"intersection_count={len(overlap)}",
        f"train_manifest={(MANIFESTS / 'train.txt').resolve()}",
        f"valid_manifest={(MANIFESTS / 'valid.txt').resolve()}",
    ]
    _ = disjoint_path.write_text("\n".join(disjoint_lines) + "\n")
    print(f"[disjoint] OK - wrote {disjoint_path}", flush=True)

    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print(f"  train images : {len(train_ids)} (512x512) -> {SPLITS['train']['out']}", flush=True)
    print(f"  valid images : {len(valid_ids)} (512x512) -> {SPLITS['valid']['out']}", flush=True)
    print(f"  disjoint     : train & valid = {len(overlap)} (must be 0)", flush=True)
    print(f"  downloaded   : {_human(total_bytes)} in {total_dl_secs:.1f}s (0 if cached)", flush=True)
    print("ALL DONE", flush=True)

    ok = len(train_ids) == SPLITS["train"]["expect"] and len(valid_ids) >= SPLITS["valid"]["expect"]
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
