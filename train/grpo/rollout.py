"""Online zoom rollout + per-scale GRPO group sampling.

For each base image x_0 we roll the recursive-multiscale zoom forward (mirroring
inference_coz.py:332-346). At every *trained* scale step we sample a group of G
prompts from the current policy and attach the multi-objective reward to each;
the trajectory is then advanced using the best prompt's SR output.

A `Group` carries everything the trainer needs to recompute token log-probs:
the shared vision inputs, the prompt length, and each completion's full token ids.
"""
from dataclasses import dataclass, field
from typing import List, Dict

import torch
from PIL import Image

from .state import build_messages, process_state
from .rewards import RewardContext

ANCHOR_MSG = "What is in this image? Give me a set of words."


@dataclass
class Completion:
    seq_ids: torch.Tensor      # (1, L) full prompt+completion ids, on CPU
    prompt_len: int
    reward: float
    raw: Dict[str, float]
    text: str


@dataclass
class Group:
    pixel_values: torch.Tensor       # shared vision input (CPU)
    image_grid_thw: torch.Tensor     # shared vision input (CPU)
    prompt_len: int
    completions: List[Completion] = field(default_factory=list)
    scale: int = 0
    base_path: str = ""


def _center_zoom_crop(img: Image.Image, upscale: int) -> Image.Image:
    """Center crop by `upscale` then bicubic-resize back (== inference crop)."""
    w, h = img.size
    nw, nh = w // upscale, h // upscale
    cx, cy = w // 2, h // 2
    crop = img.crop((cx - nw // 2, cy - nh // 2, cx + nw // 2, cy + nh // 2))
    return crop.resize((w, h), Image.BICUBIC)


class Rollout:
    def __init__(self, cfg, policy, processor, sr_backbone, rewards, device):
        self.cfg = cfg
        self.policy = policy
        self.processor = processor
        self.sr = sr_backbone
        self.rewards = rewards
        self.device = device
        self.gen = cfg["generation"]
        self.roll = cfg["rollout"]
        self._anchor_cache = {}

    @torch.no_grad()
    def _anchor_caption(self, x0, base_path):
        if base_path in self._anchor_cache:
            return self._anchor_cache[base_path]
        messages = [
            {"role": "system", "content": ANCHOR_MSG},
            {"role": "user", "content": [{"type": "image", "image": x0}]},
        ]
        inputs = process_state(self.processor, messages, self.device)
        out = self.policy.generate(**inputs, max_new_tokens=32, do_sample=False)
        trimmed = out[0][inputs.input_ids.shape[1]:]
        cap = self.processor.decode(trimmed, skip_special_tokens=True)
        self._anchor_cache[base_path] = cap
        return cap

    @torch.no_grad()
    def _sample_group(self, inputs, n):
        """Sample n completions; return list of (seq_ids(1,L), text).

        Generated in micro-batches: `num_return_sequences` makes HF replicate the
        (multi-image) vision inputs, so a full group of n at once blows up the
        vision-encoder attention memory. We cap concurrency at `sample_micro_batch`.
        """
        prompt_len = inputs.input_ids.shape[1]
        micro = self.gen.get("sample_micro_batch") or n
        results = []
        remaining = n
        while remaining > 0:
            k = min(micro, remaining)
            out = self.policy.generate(
                **inputs,
                max_new_tokens=self.gen["max_new_tokens"],
                do_sample=True,
                temperature=self.gen["temperature"],
                top_p=self.gen["top_p"],
                num_return_sequences=k,
            )
            for i in range(out.shape[0]):
                seq = out[i : i + 1].detach().cpu()
                text = self.processor.decode(
                    out[i][prompt_len:], skip_special_tokens=True
                ).strip()
                results.append((seq, text))
            remaining -= k
        return results, prompt_len

    @torch.no_grad()
    def _greedy_prompt(self, inputs):
        out = self.policy.generate(
            **inputs, max_new_tokens=self.gen["max_new_tokens"], do_sample=False
        )
        prompt_len = inputs.input_ids.shape[1]
        return self.processor.decode(
            out[0][prompt_len:], skip_special_tokens=True
        ).strip()

    @torch.no_grad()
    def run_episode(self, x0: Image.Image, base_path: str) -> List[Group]:
        """Roll one image through rec_num scales; return Groups at trained scales."""
        upscale = self.roll["upscale"]
        rec_num = self.roll["rec_num"]
        trained_scales = set(self.roll["scales"])

        x0_caption = self._anchor_caption(x0, base_path)
        prev_sr = x0          # x_0 acts as the scale-0 "SR output"
        prev_crop = None      # x_{i-2}
        prev_prompts: List[str] = []
        groups: List[Group] = []

        for rec in range(rec_num):
            scale_idx = rec + 1
            crop = _center_zoom_crop(prev_sr, upscale)            # x_{i-1}
            scale_factor = upscale ** scale_idx
            messages = build_messages(x0, prev_crop, crop, scale_factor)
            inputs = process_state(self.processor, messages, self.device)

            if scale_idx in trained_scales:
                samples, prompt_len = self._sample_group(
                    inputs, self.gen["group_size"]
                )
                group = Group(
                    pixel_values=inputs["pixel_values"].detach().cpu(),
                    image_grid_thw=inputs["image_grid_thw"].detach().cpu(),
                    prompt_len=prompt_len,
                    scale=scale_idx,
                    base_path=base_path,
                )
                ctx = RewardContext(
                    crop_pil=crop, x0_caption=x0_caption,
                    prev_prompts=list(prev_prompts),
                )
                for seq, text in samples:
                    total, raw = self.rewards.compute(text or "", ctx)
                    group.completions.append(
                        Completion(seq, prompt_len, total, raw, text)
                    )
                groups.append(group)
                # advance with the best prompt of the group
                best = max(group.completions, key=lambda c: c.reward)
                chosen_prompt = best.text or ""
            else:
                chosen_prompt = self._greedy_prompt(inputs)

            prev_prompts.append(chosen_prompt)
            if self.sr is not None:
                prev_sr, _ = self.sr.render(crop, chosen_prompt)
            else:
                # text-only mode (no SR backbone): chain the bicubic crops
                prev_sr = crop
            prev_crop = crop

        return groups
