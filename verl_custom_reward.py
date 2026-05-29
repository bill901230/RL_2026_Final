"""veRL custom reward for real CoZ VLM prompt rollouts.

The public entrypoint is ``compute_score`` with the signature expected by veRL's
custom reward hook.  For the normal veRL path we use ``reward_manager=batch`` so
all completions for the same CoZ state are visible at once; those completions are
then scored through the validated ``train.grpo.rewards.RewardOrchestrator`` using
per-group component normalization.

Only text-only reward components are enabled here:

* R_anc: generated prompt stays anchored to the scale-0 caption.
* R_rep: generated prompt avoids repeating the previous-scale prompt.
* R_phr: generated prompt avoids conversational filler phrases.

No SR feedback or critic reward is imported or enabled in this wrapper.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from train.grpo.rewards import RewardContext, RewardOrchestrator
from train.grpo.text_sim import ClipEmbedder


_CLIP_EMBEDDER: Any | None = None
_ORCHESTRATOR: RewardOrchestrator | None = None
_DUMMY_CROP = Image.new("RGB", (1, 1), (255, 255, 255))


def _cfg() -> dict[str, Any]:
    return {
        "rewards": {
            "normalize": True,
            "normalization_mode": "per_group",
            "r_anc": {"enabled": True, "weight": 1.0},
            "r_rep": {"enabled": True, "weight": 1.0, "ngram": 2},
            "r_fb": {"enabled": False, "weight": 1.0},
            "r_crit": {"enabled": False, "weight": 0.5},
            "r_phr": {
                "enabled": True,
                "weight": 0.2,
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
    explicit = os.environ.get("COZ_REWARD_DEVICE")
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


def _orchestrator() -> RewardOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = RewardOrchestrator(_cfg(), clip=_clip())
    return _ORCHESTRATOR


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


def _ctx(extra_info: Any) -> tuple[RewardContext, dict[str, Any]]:
    extra = _extra_dict(extra_info)
    ctx = RewardContext(
        crop_pil=_DUMMY_CROP,
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
    scorer = _orchestrator()
    for indices in groups.values():
        first = indices[0]
        ctx, _ = _ctx(extras[first])
        prompts = [solution_strs[i] or "" for i in indices]
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

    ctx, extra = _ctx(extra_info)
    score, raw = _orchestrator().compute(solution_str or "", ctx)
    record = _record(score, raw, extra)
    _log_record(record, extra)
    return float(record["score"])


if __name__ == "__main__":
    # Small manual smoke path for quick local checks.
    sample_extra = {"x0_caption": "a bird on a branch", "prev_prompt": "bird feathers", "scale": 1}
    print(compute_score("coz", "sharp beak and wing feathers", "", sample_extra))
