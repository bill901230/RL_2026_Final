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

W4-v2 vs A3 author baseline on `eval_subset` (`n=30`, `1` seed, `100` GRPO steps). All reported axes are higher-is-better; NIQE is the inverted evaluator value. Result: `5/6` axes improved, with no single-metric claim.

| axis | A3 mean | w4v2 mean | delta | reading |
|---|---:|---:|---:|---|
| NIQE (inverted) | -7.60 | -7.78 | -0.18 | regression; MUSIQ↑/NIQE↓ no-ref-IQA tradeoff |
| MUSIQ | 50.27 | 51.81 | +1.54 | improved |
| MANIQA | 0.3958 | 0.4043 | +0.0085 | improved |
| CLIPIQA | 0.6070 | 0.6148 | +0.0078 | improved |
| consistency | 0.7810 | 0.7826 | +0.0016 | anti-drift improved |
| unique-token ratio | 0.6234 | 0.6571 | +0.0337 | anti-convergence improved |

Exact CSV recomputation is in `results/aggregate_deltas.csv`; the headline table above keeps the rounded deltas used for the W4-v2 verdict.

## 0064 probe

- Input `samples/0064.png`, A3=`ckpt/VLM_LoRA/checkpoint-10000`, w4v2=`ckpt/VLM_FT/coz_w4v2` via `--vlm_model_path`, `recursive_multiscale`, `--save_prompts`.
- Subject-token check (`eye|fur|animal|dog`) at deep scales: **PASS** — w4v2 scale 3 has `Fur`/`Animal`; scale 4 has `Fur`/`Animal`.
- Neural drift check (`neuron|synapse|dendrite|axon`): **PASS for no neural terms, tie vs A3** — w4v2 `0`, A3 rerun `0`; historical fail-case drift emitted those terms.
- Cross-scale diversity check: **PASS** — A3 `34/72 = 0.4722`; w4v2 `38/55 = 0.6909`; delta `+0.2187`. Caveat: w4v2 scale 2 is terse (`dog`).

## Caveats & next steps

- Statistical defensibility is limited: only `1` seed, `n=30`, and `100` GRPO steps; run `>=3` seeds and full DIV2K-valid before final significance claims.
- Recover NIQE: current inverted NIQE delta is `-0.18`, so tune reward weights against the MUSIQ↑/NIQE↓ tradeoff.
- Remaining arms: A7 state-expansion and A8 higher-`R_fb` weight are still unrun; both should report all `6` axes, not a single metric.
