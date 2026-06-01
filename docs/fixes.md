# W5 fixes and numerical deltas

## Fixes implemented

- `ratio ≡ 1` made PPO/GRPO clipping a no-op -> cached rollout-time `old_logp` and trained on `new_logp - old_logp` ratios -> G0: `43 passed, 1 xfailed` in `14.96s`.
- Best-of-group trajectory advance biased the recursive data stream -> defaulted rollout advance to sampled completions while keeping `best` selectable -> G0: `43 passed, 1 xfailed` in `14.96s`.
- Process-global reward z-norm made rewards order/rank/restart dependent -> switched default reward normalization to per-group component z-norm -> G0: `43 passed, 1 xfailed` in `14.96s`.
- CLIP cosine lived in `[-1,1]` while reward terms expected `[0,1]` -> mapped cosine by `(cos+1)/2` and clamped -> G0: `43 passed, 1 xfailed` in `14.96s`.
- NIQE direction was inconsistent with higher-is-better aggregate reporting -> kept raw `score()` but negated NIQE in `score_all()`/eval CSVs -> G0: `43 passed, 1 xfailed` in `14.96s`.
- `R_phr` blacklist missed prompt fillers like “the first image shows” -> added 5 first/second/third-image blacklist variants -> G0: `43 passed, 1 xfailed` in `14.96s`.
- veRL+FSDP2 was blocked by Ray/worker and LoRA weight-sync failures -> rebuilt the train env, fixed Ray launch/temp-dir issues, used direct-controller veRL, then pivoted from broken Qwen2.5-VL LoRA+vLLM sync to full-FT on the merged author checkpoint -> G3 smoke completed `1/1` GRPO step; W4-v2 completed `100/100` full-FT GRPO steps and saved `global_step_100`.
- Saturated `R_anc` gave near-zero GRPO advantage (`raw_r_anc≈0.99-1.0`, `raw_r_rep=0` at scale 1) -> trained scales `2-4` with populated `prev_prompt`, activated `R_rep`, and added SR-in-loop `R_fb` -> W4-v2 logged `2400` reward records, `score` std `1.2978`, `raw_r_fb` std `14.2343`, and nonzero advantages at step 1 (`-1.536..+1.667`) and step 100 (`-1.891..+1.510`).

## Numerical results

W4-v2 vs A3 author baseline on `eval_subset` (`n=30`, `3` seeds, `100` GRPO steps). All reported axes are higher-is-better; NIQE is the inverted evaluator value. Result: `6/6` axes improved by three-seed mean, with no single-metric claim.

| axis | A3 mean | w4v2 mean±std | delta mean±std | seeds improving | reading |
|---|---:|---:|---:|---:|---|
| NIQE (inverted) | -7.60 | -7.57±0.19 | +0.03±0.19 | 2/3 | slight mean recovery; still high variance |
| MUSIQ | 50.27 | 51.16±0.63 | +0.89±0.63 | 3/3 | improved |
| MANIQA | 0.3958 | 0.3987±0.0054 | +0.0029±0.0054 | 2/3 | modest improvement |
| CLIPIQA | 0.6070 | 0.6133±0.0060 | +0.0063±0.0060 | 2/3 | improved |
| consistency | 0.7808 | 0.7820±0.0006 | +0.0012±0.0006 | 3/3 | anti-drift improved |
| unique-token ratio | 0.6234 | 0.6274±0.0263 | +0.0040±0.0263 | 1/3 | mean anti-convergence improved but seed-fragile |

Exact CSV recomputation is in `results/aggregate_deltas.csv`; the headline table above keeps rounded 3-seed mean±std deltas used for the W4-v2 verdict. Caveat: seed789 completed on the constrained 2-GPU allocation with `train_batch_size=2` / `ppo_mini_batch_size=2`, while earlier seeds used the original larger micro-run settings, so treat the aggregate as confirmation rather than a final significance claim.

### n=100 full-valid confirmation (eval-only F3)

A clean eval-only rerun on the full DIV2K valid split (`data/div2k/valid`, `0801-0900`, `n=100`) compared the author A3 adapter (`ckpt/VLM_LoRA/checkpoint-10000`) against the seed-123 balanced full-FT model (`ckpt/VLM_FT/coz_w4v2`) with the same fixed protocol (`recursive_multiscale`, same SR LoRA/VAE, greedy VLM prompting). This is larger-sample confirmation of the seed-123 point only, not a new multi-seed claim.

| axis | A3 full mean±img std | w4v2 full mean±img std | delta mean±paired std | vs prior `n=30` reading |
|---|---:|---:|---:|---|
| NIQE (inverted) | -8.456±3.863 | -8.629±4.057 | -0.173±1.676 | seed-123 NIQE regression holds; flips negative vs the 3-seed `+0.03` mean |
| MUSIQ | 50.107±13.626 | 50.644±13.226 | +0.537±4.621 | positive direction holds, but smaller than seed-123 `+1.54` / 3-seed `+0.89` on `n=30` |
| MANIQA | 0.4085±0.0674 | 0.4107±0.0642 | +0.0022±0.0311 | positive direction holds, small effect |
| CLIPIQA | 0.6121±0.1467 | 0.6199±0.1398 | +0.0079±0.0658 | positive direction holds |
| consistency | 0.7922±0.0465 | 0.7910±0.0455 | -0.00125±0.0158 | flips negative; the `n=30` anti-drift gain does **not** hold on full-valid seed-123 |
| unique-token ratio | 0.6182±0.1007 | 0.6894±0.1142 | +0.0712±0.1281 | positive direction holds and is larger than `n=30` seed-123 / 3-seed means |

Exact full-valid recomputation is in `results/full_valid_deltas.csv`. Honest verdict: the larger `n=100` rerun supports a positive MUSIQ direction for seed-123, but the effect is smaller and should not be over-read as significance from one seed; it does **not** confirm the prior consistency/anti-drift gain because that axis flips sign. Four of six axes remain positive on full-valid (`MUSIQ`, `MANIQA`, `CLIPIQA`, unique-token ratio); inverted `NIQE` and consistency regress.

## 0064 probe

- Input `samples/0064.png`, A3=`ckpt/VLM_LoRA/checkpoint-10000`, w4v2=`ckpt/VLM_FT/coz_w4v2` via `--vlm_model_path`, `recursive_multiscale`, `--save_prompts`.
- Subject-token check (`eye|fur|animal|dog`) at deep scales: **PASS** — w4v2 scale 3 has `Fur`/`Animal`; scale 4 has `Fur`/`Animal`.
- Neural drift check (`neuron|synapse|dendrite|axon`): **PASS for no neural terms, tie vs A3** — w4v2 `0`, A3 rerun `0`; historical fail-case drift emitted those terms.
- Cross-scale diversity check: **PASS** — A3 `34/72 = 0.4722`; w4v2 `38/55 = 0.6909`; delta `+0.2187`. Caveat: w4v2 scale 2 is terse (`dog`).

## Reward-tuning ablation

- One 2-GPU seed-123 ablation raised `R_rep` `1.0 -> 3.0` and `R_fb` `1.0 -> 1.25` (`R_anc=0.2`, `R_phr=0.1` unchanged) to target prompt convergence directly.
- `tune_rrep` vs A3 (`n=30`): unique-token ratio `0.6234 -> 0.6442` (`+0.0208`), consistency `0.7808 -> 0.7841` (`+0.0033`), MUSIQ `50.27 -> 50.86` (`+0.59`), but inverted NIQE `-7.60 -> -7.74` (`-0.14`), MANIQA `-0.0036`, and CLIPIQA `-0.0100` regressed.
- Honest verdict: diversity moved in the intended direction and beat the w4v2 3-seed mean by `+0.0168`, but it is **not robust yet** because the gain is below the prior w4v2 seed std (`0.0263`) and it trades off several IQA axes. Do not replace W4-v2 with this weight without a 3-seed follow-up or a milder `R_rep` setting.

## Caveats & next steps

- Statistical defensibility is still limited: `3` seeds at `n=30` plus one clean seed-123 full-valid rerun at `n=100`, `100` GRPO steps, and one constrained 2-GPU seed789 run with smaller batch settings. The full-valid rerun reduces the sample-count caveat for the seed-123 point, but final significance still needs multi-seed full-valid evaluation.
- Stabilize NIQE/consistency and verify diversity across seeds: inverted NIQE is only `+0.03±0.19` on the 3-seed subset and regresses `-0.173` on full-valid seed-123, while consistency flips to `-0.00125` on full-valid. Unique-token ratio is strong on the full-valid seed-123 rerun but was seed-fragile at `n=30` (`+0.004±0.026`), so tune against seed variance rather than optimizing a single metric.
- Remaining arms: A7 state-expansion and A8 higher-`R_fb` weight are still unrun; both should report all `6` axes, not a single metric.
