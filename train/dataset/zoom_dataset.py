"""Dataset that yields 512x512 base images (x_0) for GRPO zoom rollouts.

Each item is just the anchor image; the recursive zoom trajectory (crops + SR
outputs) is produced online by the rollout module, because the trajectory is
policy-dependent.
"""
from __future__ import annotations

import os
from typing import TypedDict

from PIL import Image

from torch.utils.data import Dataset
from typing_extensions import override


class ZoomItem(TypedDict):
    image: Image.Image
    path: str


def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Match inference_coz preprocessing without importing inference-only dependencies."""
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _read_path_list(txt_path: str) -> list[str]:
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(
            f"Dataset list not found: {txt_path}. Run `bash scripts/download_div2k.sh` first."
        )
    with open(txt_path, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    if not paths:
        raise ValueError(f"Dataset list is empty: {txt_path}")
    return paths


class ZoomBaseImageDataset(Dataset[ZoomItem]):
    """Yields dicts: {"image": PIL.Image (RGB, process_size^2), "path": str}."""

    paths: list[str]
    process_size: int

    def __init__(self, txt_path: str, process_size: int = 512, limit: int | None = None) -> None:
        self.paths = _read_path_list(txt_path)
        if limit is not None:
            self.paths = self.paths[:limit]
        self.process_size = process_size

    def __len__(self) -> int:
        return len(self.paths)

    @override
    def __getitem__(self, idx: int) -> ZoomItem:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = resize_and_center_crop(img, self.process_size)
        return {"image": img, "path": path}


def collate_identity(batch: list[ZoomItem]) -> list[ZoomItem]:
    """DataLoader collate that keeps the list of dicts intact (PIL images)."""
    return batch
