#!/usr/bin/env python3
"""Build veRL-ready CoZ state parquet files.

Rows are one (image, scale-step) state each.  The visual input is the current
single crop image, while the global context is carried as a cached text caption
of scale-0 to avoid unvalidated multi-image vLLM rollout behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ANCHOR_MSG = "What is in this image? Give me a set of words."
SYSTEM_TEMPLATE = (
    "You are the Chain-of-Zoom prompt extractor for extreme super-resolution. "
    "The original image has this global caption: {x0_caption!r}. "
    "You are now inspecting a single current crop at zoom factor={zoom_factor}x "
    "(scale step {scale}). Stay semantically consistent with the global caption, "
    "avoid hallucinating unrelated objects, avoid repeating the previous scale, "
    "and answer with a concise set of words describing new fine details visible "
    "in the current crop."
)


def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def read_manifest(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def list_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.png"))
    return read_manifest(path)


def image_id(path: Path) -> str:
    return path.stem


def rec_dir(baseline_dir: Path, img: Path) -> Path:
    return baseline_dir / "per-sample" / image_id(img)


def baseline_complete(baseline_dir: Path, img: Path) -> bool:
    sample_dir = rec_dir(baseline_dir, img)
    if not (sample_dir / "0.png").is_file():
        return False
    if not (sample_dir / "txt").is_dir():
        return False
    for scale in range(1, 5):
        if not (sample_dir / f"{scale}_input.png").is_file():
            return False
        if not (sample_dir / "txt" / f"{scale - 1}.txt").is_file():
            return False
    return True


def make_symlink_dir(images: list[Path], split: str) -> Path:
    link_dir = ROOT / "data" / "parquet" / "coz_baseline_inputs" / split
    link_dir.mkdir(parents=True, exist_ok=True)
    wanted = {img.name for img in images}
    for stale in link_dir.glob("*.png"):
        if stale.name not in wanted:
            stale.unlink()
    for img in images:
        target = link_dir / img.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(img.resolve())
    return link_dir


def run_baseline(images: list[Path], baseline_dir: Path, split: str) -> None:
    input_dir = make_symlink_dir(images, split)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.environ.get("PYTHON", "python"),
        str(ROOT / "inference_coz.py"),
        "-i",
        str(input_dir),
        "-o",
        str(baseline_dir),
        "--rec_type",
        "recursive_multiscale",
        "--prompt_type",
        "vlm",
        "--lora_path",
        "ckpt/SR_LoRA/model_20001.pkl",
        "--vae_path",
        "ckpt/SR_VAE/vae_encoder_20001.pt",
        "--vlm_lora_path",
        "ckpt/VLM_LoRA/checkpoint-10000",
        "--pretrained_model_name_or_path",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "--ram_ft_path",
        "ckpt/DAPE/DAPE.pth",
        "--ram_path",
        "ckpt/RAM/ram_swin_large_14m.pth",
        "--save_prompts",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def ensure_baseline(images: list[Path], baseline_dir: Path, split: str, allow_run: bool) -> None:
    missing = [img for img in images if not baseline_complete(baseline_dir, img)]
    if not missing:
        return
    if not allow_run:
        preview = ", ".join(image_id(img) for img in missing[:8])
        raise FileNotFoundError(
            f"missing baseline CoZ outputs for {len(missing)} {split} images in {baseline_dir}: {preview}; "
            "rerun with --run-baseline-missing"
        )
    run_baseline(images, baseline_dir, split)
    still_missing = [img for img in images if not baseline_complete(baseline_dir, img)]
    if still_missing:
        preview = ", ".join(image_id(img) for img in still_missing[:8])
        raise RuntimeError(f"baseline generation incomplete for {split}: {preview}")


def load_caption_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_caption_cache(path: Path, captions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(captions, indent=2, sort_keys=True) + "\n")


def caption_x0s(
    images: list[Path],
    cache_path: Path,
    model_id: str,
    adapter_path: Path,
    process_size: int,
    refresh: bool = False,
) -> dict[str, str]:
    captions = {} if refresh else load_caption_cache(cache_path)
    needed = [img for img in images if image_id(img) not in captions]
    if not needed:
        return captions

    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload().eval()
    processor = AutoProcessor.from_pretrained(model_id)

    for img in needed:
        x0 = resize_and_center_crop(Image.open(img).convert("RGB"), process_size)
        messages = [
            {"role": "system", "content": ANCHOR_MSG},
            {"role": "user", "content": [{"type": "image", "image": x0}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        trimmed = generated[0][inputs.input_ids.shape[1] :]
        captions[image_id(img)] = processor.decode(trimmed, skip_special_tokens=True).strip()
        save_caption_cache(cache_path, captions)

    del model
    if use_cuda:
        torch.cuda.empty_cache()
    return captions


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_rows(
    images: list[Path],
    baseline_dir: Path,
    captions: dict[str, str],
    start_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = start_index
    for img in images:
        iid = image_id(img)
        sample_dir = rec_dir(baseline_dir, img)
        x0_caption = captions[iid]
        for scale in range(1, 5):
            crop_path = (sample_dir / f"{scale}_input.png").resolve()
            prev_prompt = ""
            if scale > 1:
                prev_prompt = read_prompt(sample_dir / "txt" / f"{scale - 2}.txt")
            zoom_factor = 4**scale
            system = SYSTEM_TEMPLATE.format(
                x0_caption=x0_caption,
                zoom_factor=zoom_factor,
                scale=scale,
            )
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": "<image>"},
                    ],
                    "images": [{"image": f"file://{crop_path}"}],
                    "data_source": "coz",
                    "reward_model": {"style": "model", "ground_truth": ""},
                    "extra_info": {
                        "x0_caption": x0_caption,
                        "prev_prompt": prev_prompt,
                        "scale": scale,
                        "zoom_factor": zoom_factor,
                        "image_id": iid,
                        "index": idx,
                    },
                }
            )
            idx += 1
    return rows


def write_evidence(path: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], train_out: Path, val_out: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = train_rows[0] if train_rows else {}
    evidence = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_parquet": str(train_out),
        "val_parquet": str(val_out),
        "train_image_ids": len({row["extra_info"]["image_id"] for row in train_rows}),
        "val_image_ids": len({row["extra_info"]["image_id"] for row in val_rows}),
        "image_ids_disjoint": {row["extra_info"]["image_id"] for row in train_rows}.isdisjoint(
            {row["extra_info"]["image_id"] for row in val_rows}
        ),
        "sample_row": sample,
    }
    path.write_text(json.dumps(evidence, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/manifests/train50.txt")
    parser.add_argument("--val-input", type=Path, default=ROOT / "data/eval_subset")
    parser.add_argument("--train-baseline-dir", type=Path, default=ROOT / "results/A3_train50_sr")
    parser.add_argument("--val-baseline-dir", type=Path, default=ROOT / "results/A3_sr")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/parquet")
    parser.add_argument("--caption-cache", type=Path, default=ROOT / "data/parquet/coz_x0_captions.json")
    parser.add_argument("--evidence", type=Path, default=ROOT / ".sisyphus/evidence/task-w3wire-parquet.txt")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=ROOT / "ckpt/VLM_LoRA/checkpoint-10000")
    parser.add_argument("--process-size", type=int, default=512)
    parser.add_argument("--run-baseline-missing", action="store_true")
    parser.add_argument("--refresh-captions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_images = read_manifest(args.train_manifest)
    val_images = list_images(args.val_input)
    train_ids = {image_id(path) for path in train_images}
    val_ids = {image_id(path) for path in val_images}
    if not train_ids.isdisjoint(val_ids):
        overlap = ", ".join(sorted(train_ids & val_ids))
        raise ValueError(f"train/val image_ids overlap: {overlap}")

    ensure_baseline(train_images, args.train_baseline_dir, "train", args.run_baseline_missing)
    ensure_baseline(val_images, args.val_baseline_dir, "val", args.run_baseline_missing)

    captions = caption_x0s(
        train_images + val_images,
        args.caption_cache,
        args.model_id,
        args.adapter_path,
        args.process_size,
        refresh=args.refresh_captions,
    )

    train_rows = build_rows(train_images, args.train_baseline_dir, captions, start_index=0)
    val_rows = build_rows(val_images, args.val_baseline_dir, captions, start_index=len(train_rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.out_dir / "coz_states_train.parquet"
    val_out = args.out_dir / "coz_states_val.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_out))
    Dataset.from_list(val_rows).to_parquet(str(val_out))
    write_evidence(args.evidence, train_rows, val_rows, train_out, val_out)
    print(f"wrote {train_out} rows={len(train_rows)}")
    print(f"wrote {val_out} rows={len(val_rows)}")
    print(f"evidence {args.evidence}")


if __name__ == "__main__":
    main()
