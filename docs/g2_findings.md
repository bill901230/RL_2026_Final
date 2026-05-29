# G2 Correlation-Gate Findings (text-only prototypes)

## Setup
- Continue-trained author `ckpt/VLM_LoRA/checkpoint-10000` with text-only GRPO rewards.
- Trainer: validated teammate torch GRPO (single-GPU). Prototype scale: `max_steps=25`, `group_size=4`, `images_per_step=1`, 50-image train subset (DIV2K 0001-0050).
- Eval: fixed 30-image held-out subset (DIV2K 0801-0830), recursive_multiscale 4-step (256x), greedy `max_new_tokens=32`. All axes higher=better (NIQE stored inverted). n=30 per arm.

## Results (mean over 30 images)

| arm | niqe | musiq | maniqa | clipiqa | consistency (drift) | unique_token_ratio (convergence) |
|---|---|---|---|---|---|---|
| A3 (author, baseline) | -7.6018 | 50.2718 | 0.3958 | 0.6070 | 0.7808 | 0.6234 |
| A4 (+R_rep) | -7.5929 | 50.5245 | 0.3976 | 0.6060 | 0.7840 | 0.6000 |
| A5 (+R_anc) | -7.4076 | 51.1002 | 0.3937 | 0.5885 | 0.7843 | 0.6158 |
| A6 (text-only combined) | -7.4362 | 50.2986 | 0.3970 | 0.5920 | 0.7820 | 0.6107 |

## Verdict: WEAK / INCONCLUSIVE (not a clean pass)

- Consistent positive *direction* across all 3 arms on niqe, musiq, and consistency (drift axis). The anti-drift reward (R_anc, A5) shows the cleanest trend: musiq +0.83, consistency +0.0035 vs A3.
- BUT clipiqa and unique_token_ratio (convergence axis) regressed on every arm; uniqtok dropped even on the R_rep arm (opposite of intended).
- Effect sizes are within noise for 25 steps / n=30. We deliberately do NOT claim success on these deltas (measurement-integrity guard R4: no single-metric / noise-level claims).

## What this establishes
- POSITIVE: the full pipeline is functional end-to-end — validated trainer (G0) trains (G1), adapter plugs into `inference_coz.py`, and `evaluate.py` produces all 6 axes. The measurement loop is trustworthy.
- LIMITATION: 25 torch steps are far too few to move SR no-reference IQA meaningfully. The teammate torch trainer runs ~200 s/step (single-GPU, no batched rollout), so meaningful step counts are infeasible on it.

## Implication for the path forward
- A defensible numerical improvement requires many more training steps, which requires the FAST trainer: veRL + FSDP2 + vLLM (the user's explicit ask). That is the critical path.
- veRL is currently STACK_BLOCKED (see docs/verl_status / W3.T1 evidence): dependency conflicts in `.venv-train` + a Ray 2.47.1 GPU-actor hang. Next focused task: clean `.venv-train` rebuild with pinned compatible versions + resolve the Ray GPU-actor issue, then re-run the veRL cycle (G3) and scale up.
- Phasing note: if, after sufficient steps, text-only proxies still do not move IQA, escalate to R_fb (SR-in-loop reward) which optimizes IQA directly.

## Evidence
- Per-arm CSVs: `results/{A3,A4,A5,A6}.csv` (gitignored).
- Training logs: `.sisyphus/evidence/proto-{A4,A5,A6}-train.log`; eval logs: `.sisyphus/evidence/proto-{A4,A5,A6}-eval.log`.
