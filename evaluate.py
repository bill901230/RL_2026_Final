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
import math
import os
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
from train.grpo.vlm_judge import VlmJudge  # noqa: E402


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
    "vlm_grounding_all",
    "vlm_grounding_deep",
    "vlm_quality",
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
    vlm_judge: bool
    vlm_model: str | None
    vlm_base_url: str | None
    vlm_api_key: str | None
    vlm_only: bool


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


def _mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _valid_judge_score(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


class CoZEvaluator:
    def __init__(
        self,
        device: str | None = None,
        vlm_judge: bool = False,
        vlm_model: str | None = None,
        vlm_base_url: str | None = None,
        vlm_api_key: str | None = None,
        vlm_only: bool = False,
    ) -> None:
        self.device: str = device or _device()
        self.vlm_only: bool = vlm_only
        self.iqa: IQAMetrics | None = None
        self.clip: ClipEmbedder | None = None
        self._embed_image: Callable[[list[Image.Image]], torch.Tensor] | None = None
        self._cosine: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        if not vlm_only:
            self.iqa = IQAMetrics(IQA_METRICS, device=self.device)
            self.clip = ClipEmbedder(device=self.device)
            self._embed_image = cast(
                Callable[[list[Image.Image]], torch.Tensor],
                self.clip.embed_image,
            )
            self._cosine = cast(
                Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                self.clip.cosine,
            )
        self.judge: VlmJudge | None = None
        if vlm_judge:
            self.judge = VlmJudge.from_env(
                base_url=vlm_base_url,
                api_key=vlm_api_key,
                model=vlm_model,
            )

    def _vlm_scores(self, sample: Sample) -> dict[str, float]:
        judge = self.judge
        if judge is None:
            return {}

        image_by_index: dict[int, Path] = {}
        for path in sample.image_paths:
            match = _NUMERIC_PNG_RE.fullmatch(path.name)
            if match:
                image_by_index[int(match.group(1))] = path

        prompt_image_pairs: list[tuple[int, Path, Path]] = []
        for prompt_path in sample.prompt_paths:
            match = _NUMERIC_TXT_RE.fullmatch(prompt_path.name)
            if match is None:
                continue
            prompt_index = int(match.group(1))
            image_path = image_by_index.get(prompt_index + 1)
            if image_path is not None:
                prompt_image_pairs.append((prompt_index, prompt_path, image_path))

        deep_prompt_indices = {index for index, _, _ in prompt_image_pairs[-2:]}
        grounding_all: list[float] = []
        grounding_deep: list[float] = []
        for prompt_index, prompt_path, image_path in prompt_image_pairs:
            try:
                prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            score = _valid_judge_score(judge.grounding(prompt_text, image_path))
            if score is None:
                continue
            grounding_all.append(score)
            if prompt_index in deep_prompt_indices:
                grounding_deep.append(score)

        quality = _valid_judge_score(judge.quality(sample.deepest_path))
        return {
            "vlm_grounding_all": _mean_or_nan(grounding_all),
            "vlm_grounding_deep": _mean_or_nan(grounding_deep),
            "vlm_quality": math.nan if quality is None else quality,
        }

    def score_sample(self, sample: Sample) -> dict[str, float]:
        scores = {"unique_token_ratio": unique_token_ratio(sample.prompt_paths)}
        if self.vlm_only:
            scores.update(
                {
                    "niqe": math.nan,
                    "musiq": math.nan,
                    "maniqa": math.nan,
                    "clipiqa": math.nan,
                    "consistency": math.nan,
                }
            )
        else:
            if self.iqa is None or self._embed_image is None or self._cosine is None:
                raise RuntimeError("legacy evaluator metrics were not initialised")
            deepest_tensor = load_image_tensor(sample.deepest_path)
            iqa_scores = self.iqa.score_all(deepest_tensor)

            sr_image = load_pil_rgb(sample.deepest_path)
            scale0_image = load_pil_rgb(sample.scale0_path)
            image_features = self._embed_image([sr_image, scale0_image])
            consistency = _float_item(self._cosine(image_features[:1], image_features[1:2]))
            consistency = min(1.0, max(0.0, consistency))

            scores.update(
                {
                    "niqe": _float_item(iqa_scores["niqe"]),
                    "musiq": _float_item(iqa_scores["musiq"]),
                    "maniqa": _float_item(iqa_scores["maniqa"]),
                    "clipiqa": _float_item(iqa_scores["clipiqa"]),
                    "consistency": consistency,
                }
            )
        scores.update(self._vlm_scores(sample))
        return scores


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
    _ = parser.add_argument(
        "--vlm-judge",
        action="store_true",
        help="enable InternVL/OpenAI-compatible prompt-grounding judge metrics",
    )
    _ = parser.add_argument(
        "--vlm-model",
        default=os.getenv("VLM_JUDGE_MODEL", "InternVL3-8B"),
        help="VLM judge model id (default: env VLM_JUDGE_MODEL or InternVL3-8B)",
    )
    _ = parser.add_argument(
        "--vlm-base-url",
        default=os.getenv("VLM_JUDGE_BASE_URL"),
        help="VLM judge OpenAI-compatible base URL (default: env VLM_JUDGE_BASE_URL)",
    )
    _ = parser.add_argument(
        "--vlm-api-key",
        default=os.getenv("VLM_JUDGE_API_KEY"),
        help="VLM judge API key (default: env VLM_JUDGE_API_KEY)",
    )
    _ = parser.add_argument(
        "--vlm-only",
        action="store_true",
        help="skip IQA/CLIP axes for GPU-free API-only VLM screens; legacy columns are NaN",
    )
    namespace = parser.parse_args()
    return Args(
        arm=cast(str, namespace.arm),
        images=cast(Path, namespace.images),
        seed=cast(int, namespace.seed),
        out=cast(Path | None, namespace.out),
        limit=cast(int | None, namespace.limit),
        vlm_judge=cast(bool, namespace.vlm_judge),
        vlm_model=cast(str | None, namespace.vlm_model),
        vlm_base_url=cast(str | None, namespace.vlm_base_url),
        vlm_api_key=cast(str | None, namespace.vlm_api_key),
        vlm_only=cast(bool, namespace.vlm_only),
    )


def main() -> None:
    args = parse_args()
    if args.vlm_only and not args.vlm_judge:
        raise SystemExit("--vlm-only requires --vlm-judge")
    out_path = args.out if args.out is not None else Path("results") / f"{args.arm}.csv"
    samples = discover_samples(args.images, limit=args.limit)
    evaluator = CoZEvaluator(
        vlm_judge=args.vlm_judge,
        vlm_model=args.vlm_model,
        vlm_base_url=args.vlm_base_url,
        vlm_api_key=args.vlm_api_key,
        vlm_only=args.vlm_only,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())

        for sample in samples:
            scores = evaluator.score_sample(sample)
            row: CsvRow = {"arm": args.arm, "seed": args.seed, "image": sample.name}
            row.update(scores)
            writer.writerow(row)
            row_count += 1
            handle.flush()
            os.fsync(handle.fileno())

    print(f"wrote {row_count} rows -> {out_path}")


if __name__ == "__main__":
    main()
