"""Multi-objective reward orchestrator for GRPO.

Computes the five reward components and combines them into a single scalar:

    R = sum_k weight_k * znorm(R_k)

Components (all config-toggleable via grpo_default.yaml `rewards`):
  R_anc  : semantic anchor   -> cosine(prompt, x_0 caption)            [fix drift]
  R_rep  : repetition penalty -> -sim(prompt, previous-scale prompts)  [fix stagnation]
  R_fb   : SR feedback       -> pyiqa quality(SR output) + consistency
  R_crit : critic preference -> critic VLM / CLIPScore alignment
  R_phr  : phrase exclusion  -> -#conversational fillers

Heavy sub-models (CLIP, pyiqa, SR backbone, critic) are injected by the trainer
so device placement and loading happen once.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import math

from PIL import Image

from .text_sim import ngram_overlap


@dataclass
class RewardContext:
    crop_pil: Image.Image                 # current zoom crop x_{i-1} (LQ / SR input)
    x0_caption: str                       # cached anchor caption for R_anc
    prev_prompts: List[str] = field(default_factory=list)  # earlier-scale prompts


class _RunningNorm:
    """Welford running mean/std for online reward normalisation."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def normalize(self, x):
        if self.n < 2:
            return x
        std = math.sqrt(self.m2 / (self.n - 1))
        return (x - self.mean) / (std + 1e-6)


class RewardOrchestrator:
    def __init__(self, cfg, clip=None, metrics=None, sr_backbone=None, critic=None):
        self.cfg = cfg["rewards"]
        self.normalize = self.cfg.get("normalize", True)
        self.clip = clip
        self.metrics = metrics
        self.sr = sr_backbone
        self.critic = critic
        self.enabled = {k: self.cfg[k]["enabled"] for k in
                        ("r_anc", "r_rep", "r_fb", "r_crit", "r_phr")}
        self.weights = {k: self.cfg[k]["weight"] for k in self.enabled}
        self.norms = {k: _RunningNorm() for k in self.enabled}

    # ---- individual components (all return "higher is better" raw scalars) ----
    def _r_anc(self, prompt, ctx):
        p = self.clip.embed_text(prompt)
        a = self.clip.embed_text(ctx.x0_caption)
        return float(self.clip.cosine(p, a).item())

    def _r_rep(self, prompt, ctx):
        if not ctx.prev_prompts:
            return 0.0
        n = self.cfg["r_rep"].get("ngram", 2)
        lex = ngram_overlap(prompt, ctx.prev_prompts, n=n)
        p = self.clip.embed_text(prompt)
        prev = self.clip.embed_text(ctx.prev_prompts)
        sem = float(self.clip.cosine(p, prev).max().item())
        return -(0.5 * lex + 0.5 * sem)        # higher (less repetition) is better

    def _r_fb(self, prompt, ctx):
        cfb = self.cfg["r_fb"]
        sr_pil, sr01 = self.sr.render(ctx.crop_pil, prompt)
        q = self.metrics.score(cfb["quality_metric"], sr01.unsqueeze(0))
        q = float(q.item())
        from .metrics import HIGHER_BETTER
        if not HIGHER_BETTER.get(cfb["quality_metric"], True):
            q = -q                              # invert NIQE-style metrics
        cons = 0.0
        cw = cfb.get("consistency_weight", 0.0)
        if cw and self.clip is not None:
            sim = self.clip.cosine(
                self.clip.embed_image(sr_pil), self.clip.embed_image(ctx.crop_pil)
            )
            cons = float(sim.item())
        return q + cw * cons

    def _r_crit(self, prompt, ctx):
        return self.critic.score(ctx.crop_pil, prompt)

    def _r_phr(self, prompt, ctx):
        low = prompt.lower()
        fillers = self.cfg["r_phr"].get("fillers", [])
        hits = sum(1 for f in fillers if f in low)
        return -float(hits)

    _FNS = {
        "r_anc": "_r_anc", "r_rep": "_r_rep", "r_fb": "_r_fb",
        "r_crit": "_r_crit", "r_phr": "_r_phr",
    }

    def compute(self, prompt, ctx: RewardContext):
        """Return (total_reward, raw_components dict)."""
        raw = {}
        total = 0.0
        for key, fn_name in self._FNS.items():
            if not self.enabled[key]:
                continue
            val = getattr(self, fn_name)(prompt, ctx)
            raw[key] = val
            self.norms[key].update(val)
            use = self.norms[key].normalize(val) if self.normalize else val
            total += self.weights[key] * use
        return total, raw
