"""veRL custom reward for real CoZ VLM prompt rollouts.

The public entrypoint is ``compute_score`` with the signature expected by veRL's
custom reward hook.  For the normal veRL path we use ``reward_manager=batch`` so
all completions for the same CoZ state are visible at once; those completions are
then scored through the validated ``train.grpo.rewards.RewardOrchestrator`` using
per-group component normalization.

Enabled reward components:

* R_anc: generated prompt stays anchored to the scale-0 caption.
* R_rep: generated prompt avoids repeating the previous-scale prompt.
* R_fb: generated prompt is rendered through the frozen SR backbone and scored
  with pyiqa quality plus CLIP image consistency.
* R_phr: generated prompt avoids conversational filler phrases.

R_fb is activated only for parquet rows that carry ``extra_info.crop_path`` so
lightweight unit tests and non-CoZ smoke calls do not import the SR / pyiqa
stack.  Real CoZ parquet rows include that path.
"""
# pyright: reportConstantRedefinition=false, reportArgumentType=false

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from train.grpo.rewards import RewardContext, RewardOrchestrator
from train.grpo.text_sim import ClipEmbedder


_CLIP_EMBEDDER: Any | None = None
_ORCHESTRATOR: RewardOrchestrator | None = None  # legacy test monkeypatch hook; unused.
_TEXT_ORCHESTRATOR: RewardOrchestrator | None = None
_FB_ORCHESTRATOR: RewardOrchestrator | None = None
_SR_BACKBONE: Any | None = None
_IQA_METRICS: Any | None = None
_DUMMY_CROP = Image.new("RGB", (1, 1), (255, 255, 255))
_ROOT = Path(__file__).resolve().parent


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _cfg(enable_rfb: bool) -> dict[str, Any]:
    return {
        "rewards": {
            "normalize": True,
            "normalization_mode": "per_group",
            "r_anc": {"enabled": True, "weight": _float_env("COZ_R_ANC_WEIGHT", 0.2)},
            "r_rep": {"enabled": True, "weight": _float_env("COZ_R_REP_WEIGHT", 3.0), "ngram": 2},
            "r_fb": {
                "enabled": enable_rfb,
                "weight": _float_env("COZ_R_FB_WEIGHT", 1.25),
                "quality_metric": os.environ.get("COZ_RFB_METRIC", "musiq"),
                "consistency_weight": _float_env("COZ_RFB_CONSISTENCY_WEIGHT", 0.5),
            },
            "r_crit": {"enabled": False, "weight": 0.5},
            "r_phr": {
                "enabled": True,
                "weight": _float_env("COZ_R_PHR_WEIGHT", 0.1),
                "fillers": [
                    "the image shows",
                    "i can see",
                    "this is a",
                    "this image",
                    "in the image",
                    "there is",
                    "there are",
                    "appears to be",
                    "it looks like",
                    "a set of words",
                    "first image",
                    "second image",
                    "third image",
                    "the first image",
                    "the second image",
                ],
            },
        }
    }


def _clip_device() -> str:
    explicit = os.environ.get("COZ_REWARD_CLIP_DEVICE", os.environ.get("COZ_REWARD_DEVICE"))
    if explicit:
        return explicit

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _clip() -> Any:
    global _CLIP_EMBEDDER
    if _CLIP_EMBEDDER is None:
        model_id = os.environ.get("COZ_REWARD_CLIP_ID", "openai/clip-vit-base-patch32")
        _CLIP_EMBEDDER = ClipEmbedder(model_id=model_id, device=_clip_device())
    return _CLIP_EMBEDDER


def _sr_device() -> str:
    explicit = os.environ.get("COZ_REWARD_SR_DEVICE", os.environ.get("COZ_REWARD_DEVICE"))
    if explicit:
        return explicit

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _sr_cfg() -> dict[str, Any]:
    return {
        "model": {
            "sr_lora_path": str(_ROOT / "ckpt" / "SR_LoRA" / "model_20001.pkl"),
            "sr_vae_path": str(_ROOT / "ckpt" / "SR_VAE" / "vae_encoder_20001.pt"),
            "sr_lora_rank": 4,
        }
    }


def _sr() -> Any:
    global _SR_BACKBONE
    if _SR_BACKBONE is None:
        from train.grpo.sr_env import FrozenSRBackbone

        device = _sr_device()
        print(f"[coz_reward] loading FrozenSRBackbone on {device}", flush=True)
        _SR_BACKBONE = FrozenSRBackbone(_sr_cfg(), device=device)
    return _SR_BACKBONE


def _metrics() -> Any:
    global _IQA_METRICS
    if _IQA_METRICS is None:
        from train.grpo.metrics import IQAMetrics

        metric_name = os.environ.get("COZ_RFB_METRIC", "musiq")
        device = os.environ.get("COZ_REWARD_IQA_DEVICE", _sr_device())
        print(f"[coz_reward] loading IQAMetrics({metric_name}) on {device}", flush=True)
        _IQA_METRICS = IQAMetrics([metric_name], device=device)
    return _IQA_METRICS


def _orchestrator(enable_rfb: bool) -> RewardOrchestrator:
    global _TEXT_ORCHESTRATOR, _FB_ORCHESTRATOR
    if enable_rfb:
        if _FB_ORCHESTRATOR is None:
            _FB_ORCHESTRATOR = RewardOrchestrator(_cfg(enable_rfb=True), clip=_clip(), metrics=_metrics(), sr_backbone=_sr())
        return _FB_ORCHESTRATOR
    if _TEXT_ORCHESTRATOR is None:
        _TEXT_ORCHESTRATOR = RewardOrchestrator(_cfg(enable_rfb=False), clip=_clip())
    return _TEXT_ORCHESTRATOR


def _as_py(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _extra_dict(extra_info: Any) -> dict[str, Any]:
    if extra_info is None:
        return {}
    if isinstance(extra_info, dict):
        return {str(key): _as_py(value) for key, value in extra_info.items()}
    if hasattr(extra_info, "as_py"):
        value = extra_info.as_py()
        if isinstance(value, dict):
            return {str(key): _as_py(val) for key, val in value.items()}
    return {}


def _prev_prompts(extra: dict[str, Any]) -> list[str]:
    raw = extra.get("prev_prompts", extra.get("prev_prompt", []))
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, Iterable):
        return [str(item) for item in raw if str(item).strip()]
    text = str(raw)
    return [text] if text.strip() else []


def _crop_path(extra: dict[str, Any]) -> str:
    raw = extra.get("crop_path", extra.get("crop_uri", ""))
    if raw is None:
        return ""
    text = str(raw)
    if text.startswith("file://"):
        text = text[7:]
    return text


@lru_cache(maxsize=256)
def _load_crop(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _can_use_rfb(extra: dict[str, Any]) -> bool:
    if not _bool_env("COZ_ENABLE_RFB", True):
        return False
    path = _crop_path(extra)
    return bool(path and Path(path).is_file())


def _ctx(extra_info: Any, *, load_crop: bool = False) -> tuple[RewardContext, dict[str, Any]]:
    extra = _extra_dict(extra_info)
    crop = _DUMMY_CROP
    if load_crop:
        path = _crop_path(extra)
        if path:
            crop = _load_crop(path)
    ctx = RewardContext(
        crop_pil=crop,
        x0_caption=str(extra.get("x0_caption", "")),
        prev_prompts=_prev_prompts(extra),
    )
    return ctx, extra


def _group_key(data_source: Any, ground_truth: Any, extra: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _as_py(data_source),
        _as_py(ground_truth),
        extra.get("image_id"),
        extra.get("scale"),
        _crop_path(extra),
        extra.get("x0_caption"),
        tuple(_prev_prompts(extra)),
    )


def _finite(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score):
        return 0.0
    return score


def _record(score: float, raw: dict[str, float], extra: dict[str, Any]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"score": _finite(score)}
    for key, value in raw.items():
        out[f"raw_{key}"] = _finite(value)
    if "scale" in extra:
        try:
            out["scale"] = int(extra["scale"])
        except (TypeError, ValueError):
            pass
    return out


def _log_record(record: dict[str, float | int], extra: dict[str, Any]) -> None:
    payload: dict[str, Any] = {"event": "coz_reward", **record}
    for key in ("image_id", "scale"):
        if key in extra:
            payload[key] = extra[key]
    if "crop_path" in extra:
        payload["has_crop"] = bool(_crop_path(extra))
    print("[coz_reward] " + json.dumps(payload, sort_keys=True), flush=True)


def _compute_batch(
    data_sources: list[Any],
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: list[Any],
) -> list[dict[str, float | int]]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    extras: list[dict[str, Any]] = []
    for idx, extra_info in enumerate(extra_infos):
        extra = _extra_dict(extra_info)
        extras.append(extra)
        groups[_group_key(data_sources[idx], ground_truths[idx], extra)].append(idx)

    records: list[dict[str, float | int] | None] = [None] * len(solution_strs)
    for indices in groups.values():
        first = indices[0]
        use_rfb = _can_use_rfb(extras[first])
        ctx, _ = _ctx(extras[first], load_crop=use_rfb)
        prompts = [solution_strs[i] or "" for i in indices]
        scorer = _orchestrator(enable_rfb=use_rfb)
        scored = scorer.compute_group(prompts, ctx)
        for idx, (score, raw) in zip(indices, scored):
            record = _record(score, raw, extras[idx])
            _log_record(record, extras[idx])
            records[idx] = record

    return [record if record is not None else {"score": 0.0} for record in records]


def compute_score(
    data_source: Any | None = None,
    solution_str: str | None = None,
    ground_truth: Any | None = None,
    extra_info: Any | None = None,
    *,
    data_sources: list[Any] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[Any] | None = None,
    extra_infos: list[Any] | None = None,
) -> float | list[dict[str, float | int]]:
    """Compute a CoZ text-only reward.

    Singular arguments match veRL's naive reward-manager API and return ``float``.
    The plural keyword-only form is used by veRL's batch reward manager and returns
    one score record per completion after per-CoZ-state group normalization.
    """

    if solution_strs is not None:
        if data_sources is None or ground_truths is None:
            raise ValueError("batch reward requires data_sources and ground_truths")
        infos = extra_infos if extra_infos is not None else [None] * len(solution_strs)
        return _compute_batch(list(data_sources), list(solution_strs), list(ground_truths), list(infos))

    extra = _extra_dict(extra_info)
    use_rfb = _can_use_rfb(extra)
    ctx, extra = _ctx(extra, load_crop=use_rfb)
    score, raw = _orchestrator(enable_rfb=use_rfb).compute(solution_str or "", ctx)
    record = _record(score, raw, extra)
    _log_record(record, extra)
    return float(record["score"])


if __name__ == "__main__":
    # Small manual smoke path for quick local checks.
    sample_extra = {"x0_caption": "a bird on a branch", "prev_prompt": "bird feathers", "scale": 1}
    print(compute_score("coz", "sharp beak and wing feathers", "", sample_extra))
