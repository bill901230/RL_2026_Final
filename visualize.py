#!/usr/bin/env python3
"""
Visualize a single CoZ inference result folder.

Layout:
  Row 1  — SR output images: scale 0 (original) .. scale N
  Row 2  — Prompts used at each scale transition (col 0 = label, col 1..N = txt)
  Row 3  — No-reference IQA metrics (col 0 = metric names, col 1..N = scores)

Usage:
    python visualize.py <result_dir> [output_path]

Example:
    python visualize.py inference_results/failcases/0086
    python visualize.py inference_results/failcases/0086 out.png
"""

import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── layout constants ────────────────────────────────────────────────────────
CELL_W    = 512
IMG_H     = 512
PROMPT_H  = 380   # row 2 height
METRIC_H  = 260   # row 3 height
PAD       = 22
LINE_GAP  = 8

# colours
BG        = (28,  28,  28)
PANEL_BG  = (18,  18,  18)
FG        = (220, 220, 220)
MUTED     = (130, 130, 130)
ACCENT    = (100, 180, 255)
UP_COL    = (90,  210,  90)   # ↑ higher is better
DOWN_COL  = (220, 100, 100)   # ↓ lower is better
DIVIDER   = (55,  55,  55)

FONT_MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SANS  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _wrap(draw, text, max_px, font):
    """Wrap text to fit within max_px width; return list of lines."""
    words = text.split()
    lines, cur = [], ""
    for word in words:
        test = f"{cur} {word}".strip()
        w = draw.textlength(test, font=font)
        if w <= max_px:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_text_block(draw, lines, x, y, font, fill, max_h=None):
    """Draw wrapped lines; return final y. Stops if max_h is exceeded."""
    cy = y
    lh = draw.textbbox((0, 0), "Ag", font=font)[3] + LINE_GAP
    for line in lines:
        if max_h and (cy + lh > y + max_h):
            draw.text((x, cy), "…", font=font, fill=MUTED)
            break
        draw.text((x, cy), line, font=font, fill=fill)
        cy += lh
    return cy


# ── metrics ─────────────────────────────────────────────────────────────────
METRIC_DEFS = [
    ("NIQE",    False),   # lower is better
    ("MUSIQ",   True),
    ("MANIQA",  True),
    ("CLIPIQA", True),
]


def compute_metrics(image_paths):
    """Return list-of-dicts of scores. Missing metrics appear as float('nan')."""
    try:
        import pyiqa
    except ImportError:
        print("[warn] pyiqa not installed — skipping metrics (pip install pyiqa)")
        return None

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load each metric independently so one failure doesn't block the others
    models = {}
    for name, _ in METRIC_DEFS:
        try:
            models[name] = pyiqa.create_metric(name.lower(), device=device)
        except Exception as e:
            print(f"[warn] could not load metric {name}: {e}")

    results = []
    for path in image_paths:
        scores = {}
        for name in [n for n, _ in METRIC_DEFS]:
            if name not in models:
                scores[name] = float("nan")
                continue
            try:
                scores[name] = models[name](str(path)).item()
            except Exception as e:
                print(f"[warn] {name} failed on {path.name}: {e}")
                scores[name] = float("nan")
        results.append(scores)
    return results


# ── main ─────────────────────────────────────────────────────────────────────
def make_visualization(result_dir, output_path=None):
    result_dir = Path(result_dir)

    # Auto-resolve path: accept either the per-sample leaf dir (contains 0.png
    # directly) or the top-level output dir produced by inference_coz.py
    # (contains per-sample/<stem>/0.png).
    if not (result_dir / "0.png").exists():
        per_sample = result_dir / "per-sample"
        if per_sample.is_dir():
            candidates = sorted(per_sample.iterdir())
            if candidates:
                result_dir = candidates[0]   # take the first (usually only) sub-folder
        if not (result_dir / "0.png").exists():
            raise FileNotFoundError(
                f"Could not find 0.png under {result_dir}. "
                "Pass either the per-sample leaf folder or the top-level output dir."
            )

    # discover scale images  (0.png … N.png)
    img_paths = []
    for i in range(10):
        p = result_dir / f"{i}.png"
        if not p.exists():
            break
        img_paths.append(p)
    if len(img_paths) < 2:
        raise FileNotFoundError(f"Need at least 0.png and 1.png in {result_dir}")

    n_cols   = len(img_paths)          # one column per scale
    n_scales = n_cols - 1              # number of SR steps

    # prompts: txt/0.txt … txt/{n_scales-1}.txt
    txt_dir = result_dir / "txt"
    prompts = []
    for i in range(n_scales):
        p = txt_dir / f"{i}.txt"
        prompts.append(p.read_text(encoding="utf-8").strip() if p.exists() else "")

    # metrics for scales 1..N (skip scale 0 = original)
    metric_results = compute_metrics(img_paths[1:])

    # ── canvas ───────────────────────────────────────────────────────────────
    total_w = n_cols * CELL_W
    total_h = IMG_H + PROMPT_H + METRIC_H
    canvas  = Image.new("RGB", (total_w, total_h), BG)
    draw    = ImageDraw.Draw(canvas)

    f_label   = _font(FONT_SANS,  56)
    f_prompt  = _font(FONT_MONO,  44)
    f_metric  = _font(FONT_MONO,  52)
    f_mname   = _font(FONT_SANS,  52)

    # ── Row 1: images ─────────────────────────────────────────────────────────
    for col, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert("RGB").resize((CELL_W, IMG_H), Image.LANCZOS)
        canvas.paste(img, (col * CELL_W, 0))
        lbl = "Original" if col == 0 else f"Scale ×{4**col}"
        draw.text((col * CELL_W + PAD, PAD), lbl, font=f_label, fill=ACCENT)

    # ── Row 2: prompts ────────────────────────────────────────────────────────
    r2y = IMG_H
    draw.rectangle([(0, r2y), (total_w - 1, r2y + PROMPT_H - 1)], fill=PANEL_BG)

    # col 0: label
    draw.text((PAD, r2y + PAD), "Prompt", font=f_label, fill=ACCENT)
    draw.text((PAD, r2y + PAD + 72), "(used for SR →)", font=f_prompt, fill=MUTED)

    # cols 1…N: prompt text
    for col in range(1, n_cols):
        prompt = prompts[col - 1] if (col - 1) < len(prompts) else ""
        x0 = col * CELL_W
        lines = _wrap(draw, prompt if prompt else "(no prompt saved)", CELL_W - 2 * PAD, f_prompt)
        _draw_text_block(draw, lines, x0 + PAD, r2y + PAD, f_prompt, FG,
                         max_h=PROMPT_H - 2 * PAD)

    # ── Row 3: metrics ────────────────────────────────────────────────────────
    r3y = IMG_H + PROMPT_H
    draw.rectangle([(0, r3y), (total_w - 1, r3y + METRIC_H - 1)], fill=PANEL_BG)

    row_h = (METRIC_H - 2 * PAD) // len(METRIC_DEFS)

    # col 0: metric names + arrows
    for i, (name, higher) in enumerate(METRIC_DEFS):
        y   = r3y + PAD + i * row_h
        arr = "↑" if higher else "↓"
        col_arr = UP_COL if higher else DOWN_COL
        draw.text((PAD, y), name, font=f_mname, fill=FG)
        draw.text((PAD + draw.textlength(name, font=f_mname) + 4, y), arr,
                  font=f_mname, fill=col_arr)

    # cols 1…N: scores
    for col in range(1, n_cols):
        x0 = col * CELL_W
        if metric_results and (col - 1) < len(metric_results):
            scores = metric_results[col - 1]
            for i, (name, _) in enumerate(METRIC_DEFS):
                y   = r3y + PAD + i * row_h
                val = scores.get(name, float("nan"))
                draw.text((x0 + PAD, y), f"{val:.4f}", font=f_metric, fill=FG)
        else:
            draw.text((x0 + PAD, r3y + PAD), "N/A", font=f_metric, fill=MUTED)

    # ── grid lines ────────────────────────────────────────────────────────────
    for col in range(1, n_cols):
        x = col * CELL_W
        draw.line([(x, 0), (x, total_h)], fill=DIVIDER, width=1)
    draw.line([(0, r2y), (total_w, r2y)], fill=DIVIDER, width=2)
    draw.line([(0, r3y), (total_w, r3y)], fill=DIVIDER, width=2)

    # ── save ──────────────────────────────────────────────────────────────────
    if output_path is None:
        output_path = result_dir / "summary.png"
    canvas.save(output_path)
    print(f"Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    make_visualization(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
