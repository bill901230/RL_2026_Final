# W4v2 CoZ GRPO headline result

Date: 2026-05-30

## Data / reward fixes

- `data/parquet/coz_states_train.parquet` rebuilt with `extra_info.crop_path` for R_fb.
- Verified `prev_prompt` is populated for all scale 2/3/4 rows:
  - all-train parquet: 200 rows, counts `{1:50, 2:50, 3:50, 4:50}`, nonempty prev prompts `{2:50, 3:50, 4:50}`.
  - W4v2 train parquet: `data/parquet/coz_states_train_s2_4.parquet`, 150 rows, counts `{2:50, 3:50, 4:50}`, 150/150 nonempty prev prompts and 150/150 crop paths.
- `verl_custom_reward.py` now enables headline `R_rep + R_fb + small R_anc`:
  - `R_anc` weight 0.2, `R_rep` weight 1.0, `R_fb` weight 1.0, `R_phr` weight 0.1.
  - R_fb lazily loads `FrozenSRBackbone` + `IQAMetrics(musiq)` and CLIP consistency on dedicated reward GPU.
  - Fixed `train/grpo/sr_env.py` to construct `SD3Euler(device=device)` so the SR reward can run on `cuda:5` instead of mismatching `cuda:0` inputs.
- Toy R_fb check on physical `cuda:5`: finite scores with nonzero variance (`[-1.79997, +1.79997]`, raw R_fb `[59.0700, 61.5720]`).

## veRL run

- Launcher: `.sisyphus/evidence/task_w4v2_run_coz_grpo_direct.sh`
- Log: `.sisyphus/evidence/task-w4v2-train.log`
- Config highlights: full-FT/no LoRA, merged author base, FSDP2 actor/ref, vLLM rollout, `rollout.n=6`, train on scales 2-4, reward GPU `cuda:5`, 4 trainer GPUs, `trainer.total_training_steps=100`, `trainer.total_epochs=10`, `trainer.save_freq=25`.
- Completed 100/100 GRPO steps.
- Saved final checkpoint: `checkpoints/w4v2/coz_grpo/global_step_100/actor` with `model_world_size_4_rank_{0..3}.pt`.
- Reward variance / advantage: PASS.
  - 2400 `[coz_reward]` records.
  - `score`: std 1.2978, min -4.9149, max 4.5547.
  - `raw_r_fb`: std 14.2343, min 19.7918, max 74.9965.
  - `raw_r_rep`: std 0.0442, min -0.4776, max -0.2553.
  - Step 1 advantages: min -1.536, max 1.667.
  - Step 100 advantages: min -1.891, max 1.510.

## Merge + eval

- Merged HF model: `ckpt/VLM_FT/coz_w4v2` via `python -m verl.model_merger merge --backend fsdp ...`.
- Inference log: `.sisyphus/evidence/task-w4v2-infer.log`
- Eval log: `.sisyphus/evidence/task-w4v2-eval.log`
- Output CSV: `results/w4v2.csv` (30 rows).

## Means vs A3

Higher is better for all reported axes here; NIQE is the inverted evaluator value.

| axis | A3 mean | w4v2 mean | delta |
|---|---:|---:|---:|
| niqe | -7.6018 | -7.7823 | -0.1805 |
| musiq | 50.2720 | 51.8138 | +1.5418 |
| maniqa | 0.3958 | 0.4043 | +0.0085 |
| clipiqa | 0.6070 | 0.6148 | +0.0078 |
| consistency | 0.7810 | 0.7826 | +0.0016 |
| uniqtok | 0.6234 | 0.6571 | +0.0337 |

Paired deltas vs `results/A3.csv` have SEMs: NIQE 0.3100, MUSIQ 0.7944, MANIQA 0.0066, CLIPIQA 0.0150, consistency 0.0023, uniqtok 0.0269.

## Verdict

W4v2 is the first real veRL numerical CoZ GRPO result with nonzero reward/advantage signal and a saved/evaluable full-FT checkpoint. It improves 5/6 axes vs A3, but it does **not** strictly beat A3 on all axes because inverted NIQE regresses by -0.1805 (small relative to paired SEM/noise). Honest verdict: promising multi-axis improvement, not a clean all-axis win.
