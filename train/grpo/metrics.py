"""No-reference image-quality metrics via pyiqa.

Mirrors the metric set already used in visualize.py:84-124 (NIQE, MUSIQ,
MANIQA, CLIPIQA). Reused both for the R_fb reward and for periodic evaluation.
Metrics operate on a torch tensor in [0, 1], shape (B, 3, H, W).
"""
from collections.abc import Iterable
from typing import Protocol, cast

import torch

import pyiqa  # pyright: ignore[reportMissingTypeStubs]

# True  -> higher is better
# False -> lower is better (NIQE)
HIGHER_BETTER: dict[str, bool] = {
    "niqe": False,
    "musiq": True,
    "maniqa": True,
    "clipiqa": True,
}


class _Metric(Protocol):
    def __call__(self, img: torch.Tensor) -> torch.Tensor: ...


class _CreateMetric(Protocol):
    def __call__(self, metric_name: str, *, device: str) -> _Metric: ...


class IQAMetrics:
    def __init__(self, metric_names: Iterable[str], device: str = "cuda") -> None:
        self.device: str = device
        self.models: dict[str, _Metric] = {}
        create_metric = cast(_CreateMetric, getattr(pyiqa, "create_metric"))
        for name in metric_names:
            key = name.lower()
            self.models[key] = create_metric(key, device=device)

    @staticmethod
    def _validate_img(img: object) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError("img must be a torch.Tensor")
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError("img must be NCHW with shape (B, 3, H, W)")
        if not torch.is_floating_point(img):
            raise TypeError("img must be a floating-point tensor")
        if not bool(torch.isfinite(img).all().item()):
            raise ValueError("img must contain only finite values")
        if float(img.amin().item()) < 0.0 or float(img.amax().item()) > 1.0:
            raise ValueError("img values must be in [0, 1]")
        return img

    @staticmethod
    def _higher_better(name: str, score: torch.Tensor) -> torch.Tensor:
        return score if HIGHER_BETTER.get(name, True) else -score

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def score(self, name: str, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) float tensor in [0, 1]. Returns raw pyiqa scores."""
        key = name.lower()
        img = self._validate_img(img)
        img = img.to(self.device).float()
        out = self.models[key](img)
        return out.reshape(-1).detach().cpu()

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def score_all(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: self._higher_better(name, self.score(name, img)) for name in self.models}
