"""Dataset that yields 512x512 base images (x_0) for GRPO zoom rollouts.

Each item is just the anchor image; the recursive zoom trajectory (crops + SR
outputs) is produced online by the rollout module, because the trajectory is
policy-dependent.
"""
import os
from PIL import Image

from torch.utils.data import Dataset

# reuse the exact preprocessing inference uses to make x_0
from inference_coz import resize_and_center_crop


def _read_path_list(txt_path):
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(
            f"Dataset list not found: {txt_path}. "
            f"Run `bash scripts/download_div2k.sh` first."
        )
    with open(txt_path, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    if not paths:
        raise ValueError(f"Dataset list is empty: {txt_path}")
    return paths


class ZoomBaseImageDataset(Dataset):
    """Yields dicts: {"image": PIL.Image (RGB, process_size^2), "path": str}."""

    def __init__(self, txt_path, process_size=512, limit=None):
        self.paths = _read_path_list(txt_path)
        if limit is not None:
            self.paths = self.paths[:limit]
        self.process_size = process_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = resize_and_center_crop(img, self.process_size)
        return {"image": img, "path": path}


def collate_identity(batch):
    """DataLoader collate that keeps the list of dicts intact (PIL images)."""
    return batch
