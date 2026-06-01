# Higher-R_rep reward-tuning ablation

Date: 2026-06-01

## Goal

Run one seed (`seed=123`) with the same constrained 2-GPU veRL GRPO setup as `task_w4v2_seed789_2gpu.sh`, but raise the repetition penalty to directly target the weak anti-convergence axis (`unique_token_ratio`).

## Reward weights

Old W4-v2 weights:

- `R_anc`: `0.2`
- `R_rep`: `1.0`
- `R_fb`: `1.0`
- `R_phr`: `0.1`
- `R_fb` consistency subweight: `0.5`

Tune-rrep weights:

- `R_anc`: `0.2 -> 0.2` (kept small; anchor saturates)
- `R_rep`: `1.0 -> 3.0` (3x; directly targets prompt convergence)
- `R_fb`: `1.0 -> 1.25` (modest +25%)
- `R_phr`: `0.1 -> 0.1`
- `R_fb` consistency subweight: `0.5 -> 0.5`

Implementation:

- `verl_custom_reward.py` defaults now match the ablation (`COZ_R_REP_WEIGHT` default `3.0`, `COZ_R_FB_WEIGHT` default `1.25`).
- Launcher `.sisyphus/evidence/task_tune_rrep_seed123_2gpu.sh` sets the same values explicitly in the environment.

## Validation log

- Reward finite/unit check: `source .venv-train/bin/activate && pytest tests/test_verl_reward.py -q` -> `3 passed in 4.33s`.
- Train seed123: reached `global_step_100` on the 2-GPU config. Operational note: the first tmux run stopped during the `global_step_75` save after model shards; resumed from the complete `global_step_50` checkpoint and completed the remaining steps. The final `global_step_100` actor model shards were saved; `fsdp_config.json` and `huggingface/` metadata were restored from the complete step-50 actor folder for the FSDP merger, then intermediate step-25/50/75 checkpoints were deleted to keep disk below quota.
- Reward signal from persisted train log (`900` `[coz_reward]` records through step 74): `score` std `3.0926` (nonzero), `raw_r_rep` std `0.0426`; with `R_rep=3.0`, weighted R_rep std is `0.1278` vs W4-v2 `~0.0442` at weight `1.0`, so the repetition term contributed ~3x more. `raw_r_fb` std `14.8212`; with `R_fb=1.25`, weighted R_fb std `18.5265`.
- Merge: `python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/tune_rrep/global_step_100/actor --target_dir ckpt/VLM_FT/coz_tune_rrep` -> PASS (`task-tune-rrep-merge.log`).
- Eval: `.venv` inference on `data/eval_subset` -> `results/tune_rrep_sr`, then `python evaluate.py --arm tune_rrep --seed 123 --images results/tune_rrep_sr/per-sample --out results/tune_rrep.csv` -> `30` rows (`task-tune-rrep-eval.log`).

## Honest comparison

All axes are higher-is-better; NIQE is the inverted evaluator value.

| axis | A3 mean | tune_rrep mean | tune - A3 | w4v2 3-seed mean | w4v2 - A3 | tune - w4v2 |
|---|---:|---:|---:|---:|---:|---:|
| NIQE (inverted) | -7.6018 | -7.7405 | -0.1388 | -7.5674 | +0.0344 | -0.1731 |
| MUSIQ | 50.2718 | 50.8590 | +0.5872 | 51.1622 | +0.8904 | -0.3032 |
| MANIQA | 0.3958 | 0.3922 | -0.0036 | 0.3987 | +0.0029 | -0.0065 |
| CLIPIQA | 0.6070 | 0.5970 | -0.0100 | 0.6133 | +0.0063 | -0.0164 |
| consistency | 0.7808 | 0.7841 | +0.0033 | 0.7820 | +0.0012 | +0.0022 |
| unique-token ratio | 0.6234 | 0.6442 | +0.0208 | 0.6274 | +0.0040 | +0.0168 |

Verdict: higher `R_rep` did move the anti-convergence axis in the intended direction for this seed (`unique_token_ratio` `0.6442`, +`0.0208` vs A3 and +`0.0168` vs w4v2 mean), and consistency improved more than w4v2. However this is **not a robust win yet** under the prior 3-seed noise yardstick: the unique-token gain is still below the w4v2 seed std (`0.0263`). It also traded off IQA axes: inverted NIQE, MANIQA, and CLIPIQA regressed vs A3, and MUSIQ is positive vs A3 but below the prior w4v2 mean. Honest read: mixed/negative as a general replacement; useful evidence that `R_rep` can lift diversity, but 3 seeds or a less aggressive weight are needed before claiming robust anti-convergence improvement.
