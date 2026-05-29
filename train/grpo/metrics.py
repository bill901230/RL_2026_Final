"""No-reference image-quality metrics via pyiqa.

Mirrors the metric set already used in visualize.py:84-124 (NIQE, MUSIQ,
MANIQA, CLIPIQA). Reused both for the R_fb reward and for periodic evaluation.
Metrics operate on a torch tensor in [0, 1], shape (B, 3, H, W).
"""
import torch

import pyiqa

# True  -> higher is better
# False -> lower is better (NIQE)
HIGHER_BETTER = {
    "niqe": False,
    "musiq": True,
    "maniqa": True,
    "clipiqa": True,
}


class IQAMetrics:
    def __init__(self, metric_names, device="cuda"):
        self.device = device
        self.models = {}
        for name in metric_names:
            self.models[name] = pyiqa.create_metric(name.lower(), device=device)

    @torch.no_grad()
    def score(self, name, img):
        """img: (B,3,H,W) float tensor in [0,1] on any device. Returns (B,) tensor."""
        img = img.to(self.device).float().clamp(0, 1)
        out = self.models[name](img)
        return out.reshape(-1).detach().cpu()

    @torch.no_grad()
    def score_all(self, img):
        return {name: self.score(name, img) for name in self.models}
