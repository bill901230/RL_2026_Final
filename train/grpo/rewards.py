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
import math
from dataclasses import dataclass, field
from typing import List

from PIL import Image

from .text_sim import ngram_overlap


_EPS = 1.0e-6


@dataclass
class RewardContext:
    crop_pil: Image.Image                 # current zoom crop x_{i-1} (LQ / SR input)
    x0_caption: str                       # cached anchor caption for R_anc
    prev_prompts: List[str] = field(default_factory=list)  # earlier-scale prompts


class _RunningNorm:
    """Welford running mean/std for legacy single-process normalization."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def state_dict(self):
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state):
        self.n = int(state.get("n", 0))
        self.mean = float(state.get("mean", 0.0))
        self.m2 = float(state.get("m2", 0.0))

    def update(self, x):
        x = float(x)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def normalize(self, x):
        x = float(x)
        if self.n < 2:
            return x
        std = math.sqrt(self.m2 / (self.n - 1))
        return (x - self.mean) / (std + _EPS)


class RewardOrchestrator:
    def __init__(self, cfg, clip=None, metrics=None, sr_backbone=None, critic=None):
        self.cfg = cfg["rewards"]
        self.normalize = self.cfg.get("normalize", True)
        self.normalization_mode = self._normalization_mode()
        self.clip = clip
        self.metrics = metrics
        self.sr = sr_backbone
        self.critic = critic
        self.enabled = {k: self.cfg[k]["enabled"] for k in
                        ("r_anc", "r_rep", "r_fb", "r_crit", "r_phr")}
        self.weights = {k: self.cfg[k]["weight"] for k in self.enabled}
        self.r_anc_mode = str(self.cfg["r_anc"].get("mode", "cosine")).lower()
        self.norms = {k: _RunningNorm() for k in self.enabled}

    def _normalization_mode(self):
        if not self.normalize:
            return "none"
        mode = self.cfg.get("normalization_mode", self.cfg.get("norm_mode", "per_group"))
        mode = str(mode).lower().replace("-", "_")
        aliases = {
            "batch": "per_group",
            "group": "per_group",
            "global": "running",
            "global_running": "running",
            "welford": "running",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"per_group", "running", "none"}:
            raise ValueError(f"unsupported reward normalization_mode={mode!r}")
        return mode

    def reset_norms(self):
        for norm in self.norms.values():
            norm.reset()

    def state_dict(self):
        return {
            "normalization_mode": self.normalization_mode,
            "norms": {key: norm.state_dict() for key, norm in self.norms.items()},
        }

    def load_state_dict(self, state):
        for key, norm_state in state.get("norms", {}).items():
            if key in self.norms:
                self.norms[key].load_state_dict(norm_state)

    # ---- individual components (all return "higher is better" raw scalars) ----
    def _r_anc(self, prompt, ctx):
        clip = self.clip
        if clip is None:
            raise RuntimeError("r_anc requires a clip embedder")
        p = clip.embed_text(prompt)
        a = clip.embed_text(ctx.x0_caption)
        return float(clip.cosine(p, a).item())

    def _r_rep(self, prompt, ctx):
        if not ctx.prev_prompts:
            return 0.0
        clip = self.clip
        if clip is None:
            raise RuntimeError("r_rep requires a clip embedder")
        n = self.cfg["r_rep"].get("ngram", 2)
        lex = ngram_overlap(prompt, ctx.prev_prompts, n=n)
        p = clip.embed_text(prompt)
        prev = clip.embed_text(ctx.prev_prompts)
        sem = float(clip.cosine(p, prev).max().item())
        return -(0.5 * lex + 0.5 * sem)        # higher (less repetition) is better

    def _r_fb(self, prompt, ctx):
        if self.sr is None or self.metrics is None:
            raise RuntimeError("r_fb requires sr_backbone and metrics")
        cfb = self.cfg["r_fb"]
        sr_pil, sr01 = self.sr.render(ctx.crop_pil, prompt)
        q = self.metrics.score(cfb["quality_metric"], sr01.unsqueeze(0))
        q = float(q.item())
        from .metrics import HIGHER_BETTER
        if not HIGHER_BETTER.get(cfb["quality_metric"], True):
            q = -q                              # invert NIQE-style metrics
        cons = 0.0
        cw = cfb.get("consistency_weight", 0.0)
        clip = self.clip
        if cw and clip is not None:
            sim = clip.cosine(
                clip.embed_image(sr_pil), clip.embed_image(ctx.crop_pil)
            )
            cons = float(sim.item())
        return q + cw * cons

    def _r_crit(self, prompt, ctx):
        if self.critic is None:
            raise RuntimeError("r_crit requires a critic")
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

    def _raw_components(self, prompt, ctx):
        raw = {}
        for key, fn_name in self._FNS.items():
            if self.enabled[key]:
                raw[key] = float(getattr(self, fn_name)(prompt, ctx))
        return raw

    def _running_components(self, raw):
        use = {}
        for key, val in raw.items():
            self.norms[key].update(val)
            use[key] = self.norms[key].normalize(val)
        return use

    def _per_group_components(self, raws):
        uses = [{key: 0.0 for key in raw} for raw in raws]
        if not raws:
            return uses
        for key in raws[0]:
            vals = [raw[key] for raw in raws]
            if key == "r_anc" and self.r_anc_mode == "margin":
                if max(vals) - min(vals) <= _EPS:
                    continue

                center = (len(vals) - 1) / 2.0
                if center <= _EPS:
                    continue

                ranks = [0.0 for _ in vals]
                ordered = sorted(enumerate(vals), key=lambda item: item[1])
                start = 0
                while start < len(ordered):
                    end = start + 1
                    while end < len(ordered) and ordered[end][1] == ordered[start][1]:
                        end += 1
                    rank = (start + end - 1) / 2.0
                    for idx, _ in ordered[start:end]:
                        ranks[idx] = rank
                    start = end

                for use, rank in zip(uses, ranks):
                    use[key] = (rank - center) / center
                continue

            mean = sum(vals) / len(vals)
            var = sum((val - mean) ** 2 for val in vals) / len(vals)
            std = math.sqrt(var)
            if std <= _EPS:
                continue
            for use, raw in zip(uses, raws):
                use[key] = (raw[key] - mean) / (std + _EPS)
        return uses

    def _normalized_group(self, raws):
        if self.normalization_mode == "none":
            return [dict(raw) for raw in raws]
        if self.normalization_mode == "running":
            return [self._running_components(raw) for raw in raws]
        return self._per_group_components(raws)

    def _weighted_total(self, use):
        return sum(self.weights[key] * val for key, val in use.items())

    def compute(self, prompt, ctx: RewardContext):
        """Return (total_reward, raw_components dict)."""
        if isinstance(prompt, (list, tuple)):
            return self.compute_group(prompt, ctx)
        raw = self._raw_components(prompt, ctx)
        if self.normalization_mode == "running":
            use = self._running_components(raw)
        else:
            use = dict(raw)
        return self._weighted_total(use), raw

    def compute_group(self, prompts, ctx: RewardContext):
        """Return per-prompt rewards with default per-group component z-scoring.

        The default `per_group` mode is order-invariant and intended for GRPO/veRL
        batches. `running` preserves the old global Welford path for single-process
        torch experiments; it is order-, rank-, and restart-dependent. Trainer-side
        GRPO advantages are standardized again, so enabling reward normalization
        intentionally normalizes components before the trainer normalizes totals.
        """
        raws = [self._raw_components(prompt, ctx) for prompt in prompts]
        uses = self._normalized_group(raws)
        return [(self._weighted_total(use), raw) for use, raw in zip(uses, raws)]
