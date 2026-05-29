#!/usr/bin/env python3
"""Single-source Chain-of-Zoom evaluator for saved per-sample outputs.

Consumes a CoZ ``per-sample`` tree whose leaves contain numeric scale images
(``0.png`` ... ``N.png``) and saved prompts under ``txt/*.txt``.  The deepest
scale image is scored on the registered IQA axes, CLIP consistency to scale 0,
and prompt-token diversity.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.grpo.metrics import IQAMetrics  # noqa: E402
from train.grpo.text_sim import ClipEmbedder  # noqa: E402


IQA_METRICS: tuple[str, ...] = ("niqe", "musiq", "maniqa", "clipiqa")
CSV_COLUMNS: list[str] = [
    "arm",
    "seed",
    "image",
    "niqe",
    "musiq",
    "maniqa",
    "clipiqa",
    "consistency",
    "unique_token_ratio",
]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]

_NUMERIC_PNG_RE: re.Pattern[str] = re.compile(r"^(\d+)\.png$")
_NUMERIC_TXT_RE: re.Pattern[str] = re.compile(r"^(\d+)\.txt$")
_INPUT_CROP_RE: re.Pattern[str] = re.compile(r".*_input\.png$", re.IGNORECASE)
_TOKEN_RE: re.Pattern[str] = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class Sample:
    name: str
    directory: Path
    image_paths: tuple[Path, ...]
    prompt_paths: tuple[Path, ...]

    @property
    def scale0_path(self) -> Path:
        return self.image_paths[0]

    @property
    def deepest_path(self) -> Path:
        return self.image_paths[-1]


@dataclass(frozen=True)
class Args:
    arm: str
    images: Path
    seed: int
    out: Path | None
    limit: int | None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _numeric_files(directory: Path, pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return pairs
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            pairs.append((int(match.group(1)), path))
    return sorted(pairs, key=lambda item: item[0])


def _contiguous_scale_images(sample_dir: Path) -> tuple[Path, ...]:
    by_index: dict[int, Path] = {}
    for idx, path in _numeric_files(sample_dir, _NUMERIC_PNG_RE):
        # Exclude the concat strip (<stem>.png inside <stem>/, e.g. 0801.png->801)
        # and zoom-crop inputs (*_input.png) so they cannot poison scale indices.
        if path.stem == sample_dir.name:
            continue
        if _INPUT_CROP_RE.fullmatch(path.name):
            continue
        by_index[idx] = path
    if 0 not in by_index:
        raise FileNotFoundError(f"{sample_dir} does not contain 0.png")
    ordered: list[Path] = []
    index = 0
    while index in by_index:  # deepest scale = max contiguous index from 0
        ordered.append(by_index[index])
        index += 1
    if len(ordered) < 2:
        raise FileNotFoundError(f"{sample_dir} needs at least 0.png and 1.png")
    return tuple(ordered)


def _prompt_files(sample_dir: Path) -> tuple[Path, ...]:
    txt_dir = sample_dir / "txt"
    return tuple(path for _, path in _numeric_files(txt_dir, _NUMERIC_TXT_RE))


def _resolve_per_sample_root(images: Path) -> Path:
    if (images / "per-sample").is_dir():
        return images / "per-sample"
    return images


def discover_samples(images: Path, limit: int | None = None) -> list[Sample]:
    root = _resolve_per_sample_root(images)
    if not root.exists():
        raise FileNotFoundError(f"images path does not exist: {root}")

    if (root / "0.png").is_file():
        sample_dirs = [root]
    else:
        sample_dirs = sorted(
            (path for path in root.iterdir() if path.is_dir() and (path / "0.png").is_file()),
            key=lambda path: path.name,
        )

    if limit is not None:
        sample_dirs = sample_dirs[:limit]
    if not sample_dirs:
        raise FileNotFoundError(
            f"no per-sample leaf directories with numeric scale images under {root}"
        )

    return [
        Sample(
            name=sample_dir.name,
            directory=sample_dir,
            image_paths=_contiguous_scale_images(sample_dir),
            prompt_paths=_prompt_files(sample_dir),
        )
        for sample_dir in sample_dirs
    ]


def load_image_tensor(path: Path) -> torch.Tensor:
    """Load an RGB image as NCHW float tensor in [0, 1]."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        raw_array: NDArray[np.float32] = np.asarray(rgb, dtype=np.float32)
        array = raw_array / np.float32(255.0)
    return torch.tensor(array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).contiguous()


def load_pil_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _float_item(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    return float(value)


def unique_token_ratio(prompt_paths: tuple[Path, ...]) -> float:
    tokens: list[str] = []
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        found_tokens: list[str] = _TOKEN_RE.findall(text)
        tokens.extend(token.lower() for token in found_tokens)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


class CoZEvaluator:
    def __init__(self, device: str | None = None) -> None:
        self.device: str = device or _device()
        self.iqa: IQAMetrics = IQAMetrics(IQA_METRICS, device=self.device)
        self.clip: ClipEmbedder = ClipEmbedder(device=self.device)
        self._embed_image: Callable[[list[Image.Image]], torch.Tensor] = cast(
            Callable[[list[Image.Image]], torch.Tensor],
            self.clip.embed_image,
        )
        self._cosine: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cast(
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            self.clip.cosine,
        )

    def score_sample(self, sample: Sample) -> dict[str, float]:
        deepest_tensor = load_image_tensor(sample.deepest_path)
        iqa_scores = self.iqa.score_all(deepest_tensor)

        sr_image = load_pil_rgb(sample.deepest_path)
        scale0_image = load_pil_rgb(sample.scale0_path)
        image_features = self._embed_image([sr_image, scale0_image])
        consistency = _float_item(self._cosine(image_features[:1], image_features[1:2]))
        consistency = min(1.0, max(0.0, consistency))

        return {
            "niqe": _float_item(iqa_scores["niqe"]),
            "musiq": _float_item(iqa_scores["musiq"]),
            "maniqa": _float_item(iqa_scores["maniqa"]),
            "clipiqa": _float_item(iqa_scores["clipiqa"]),
            "consistency": consistency,
            "unique_token_ratio": unique_token_ratio(sample.prompt_paths),
        }


def write_rows(rows: list[CsvRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Evaluate a Chain-of-Zoom per-sample output tree."
    )
    _ = parser.add_argument("--arm", required=True, help="training/evaluation arm name")
    _ = parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="CoZ output dir, per-sample dir, or a single per-sample leaf dir",
    )
    _ = parser.add_argument("--seed", type=int, default=0, help="seed label for CSV rows")
    _ = parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output CSV path; defaults to results/<arm>.csv",
    )
    _ = parser.add_argument("--limit", type=int, default=None, help="max samples to score")
    namespace = parser.parse_args()
    return Args(
        arm=cast(str, namespace.arm),
        images=cast(Path, namespace.images),
        seed=cast(int, namespace.seed),
        out=cast(Path | None, namespace.out),
        limit=cast(int | None, namespace.limit),
    )


def main() -> None:
    args = parse_args()
    out_path = args.out if args.out is not None else Path("results") / f"{args.arm}.csv"
    samples = discover_samples(args.images, limit=args.limit)
    evaluator = CoZEvaluator()

    rows: list[CsvRow] = []
    for sample in samples:
        scores = evaluator.score_sample(sample)
        row: CsvRow = {"arm": args.arm, "seed": args.seed, "image": sample.name}
        row.update(scores)
        rows.append(row)

    write_rows(rows, out_path)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
