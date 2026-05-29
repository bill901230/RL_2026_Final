"""Frozen SR backbone wrapper used as the environment for the R_fb reward.

Reproduces the SR construction + invocation from inference_coz.py
(lines 18-20, 183-192, 359-380): SD3Euler + OSEDiff_SD3_TEST, fed a [-1,1]
tensor, returning a [-1,1] image which we map to PIL / [0,1].

The whole backbone is frozen — we only need its pixels to score prompts.
"""
from types import SimpleNamespace

import torch
from torchvision import transforms

from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler

_to_tensor = transforms.ToTensor()
_to_pil = transforms.ToPILImage()


class FrozenSRBackbone:
    def __init__(self, cfg, device="cuda:1"):
        self.device = device
        model = SD3Euler()
        model.text_enc_1.to(device)
        model.text_enc_2.to(device)
        model.text_enc_3.to(device)
        model.transformer.to(device)
        model.vae.to(device)
        for m in [model.text_enc_1, model.text_enc_2, model.text_enc_3,
                  model.transformer, model.vae]:
            m.requires_grad_(False)

        # OSEDiff_SD3_TEST reads lora_path / vae_path / lora_rank off args.
        sr_args = SimpleNamespace(
            lora_path=cfg["model"]["sr_lora_path"],
            vae_path=cfg["model"]["sr_vae_path"],
            lora_rank=cfg["model"].get("sr_lora_rank", 4),
        )
        self.model_test = OSEDiff_SD3_TEST(sr_args, model)

    @torch.no_grad()
    def render(self, crop_pil, prompt):
        """Run one SR step. crop_pil: PIL RGB. Returns (pil, tensor[0,1] (3,H,W))."""
        lq = _to_tensor(crop_pil).unsqueeze(0).to(self.device)  # [0,1]
        lq = lq * 2 - 1                                          # [-1,1]
        out = self.model_test(lq, prompt=prompt)
        out = torch.clamp(out[0].cpu().float(), -1.0, 1.0)
        out01 = out * 0.5 + 0.5                                  # [0,1]
        return _to_pil(out01), out01
