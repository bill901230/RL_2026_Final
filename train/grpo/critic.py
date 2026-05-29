"""Critic preference reward R_crit.

Original CoZ uses a larger VLM to score how well a generated prompt aligns with
the image crop. Two backends:
  - "vlm"       : Qwen2.5-VL-7B-Instruct rates alignment 0-10 (parsed -> [0,1]).
  - "clipscore" : cosine(CLIP image, CLIP text), cheap fallback (no extra model).
"""
import re

import torch

CRITIC_PROMPT = (
    "You are grading how well a set of descriptive words matches an image. "
    "Image is provided. Candidate words: \"{prompt}\". "
    "Rate the alignment from 0 (unrelated/hallucinated) to 10 (perfectly accurate). "
    "Answer with ONLY a single integer."
)


class Critic:
    def __init__(self, cfg, clip_embedder=None, device="cuda:2"):
        rc = cfg["rewards"]["r_crit"]
        self.backend = rc.get("backend", "vlm")
        self.device = device
        self.clip = clip_embedder

        if self.backend == "vlm":
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
            )
            from qwen_vl_utils import process_vision_info

            self._process_vision_info = process_vision_info
            critic_id = rc.get("critic_id", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                critic_id, torch_dtype=torch.float16, device_map=device
            ).eval()
            self.processor = AutoProcessor.from_pretrained(critic_id)
            for p in self.model.parameters():
                p.requires_grad_(False)
        elif self.backend == "clipscore":
            assert self.clip is not None, "clipscore backend needs a ClipEmbedder"
        else:
            raise ValueError(f"Unknown R_crit backend: {self.backend}")

    @torch.no_grad()
    def score(self, crop_pil, prompt):
        """Return alignment reward in [0,1]."""
        if self.backend == "clipscore":
            img = self.clip.embed_image(crop_pil)
            txt = self.clip.embed_text(prompt)
            return float(self.clip.cosine(img, txt).item())

        # vlm backend
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop_pil},
                    {"type": "text", "text": CRITIC_PROMPT.format(prompt=prompt)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        gen = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        trimmed = gen[0][inputs.input_ids.shape[1]:]
        out = self.processor.decode(trimmed, skip_special_tokens=True)
        m = re.search(r"\d+(\.\d+)?", out)
        val = float(m.group()) if m else 0.0
        return max(0.0, min(1.0, val / 10.0))
