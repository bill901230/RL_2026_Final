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
- A7 exposed a near-empty/off-language reward hack -> added `COZ_VALIDITY_GUARD` before reward z-norm (`<8` model tokens / `<5` unique content tokens / `>30%` CJK => raw total `-2.0` and `R_rep=-1.0`) and reran A7 against a matched W4-v2 stabilized control (`kl_loss_coef=0.02`, `lr=3e-7`, `entropy_coeff=0.005`, `grad_clip=0.5`).

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

### Definitive 3-seed × n=100 full-valid result (eval-only F3)

A clean eval-only rerun on the full DIV2K valid split (`data/div2k/valid`, `0801-0900`, `n=100`) now compares A3 against all three W4-v2 full-FT seeds (`seed123`, `seed456`, `seed789`) with the same fixed protocol (`recursive_multiscale`, same SR LoRA/VAE, greedy VLM prompting). The table reports the mean across the three seed-means and the seed std across those three means; all axes are higher-is-better, with NIQE already inverted by the evaluator.

Robust-win criterion used below: `delta > 0` for `3/3` seeds **and** `delta > seed-std`. By that strict rule, **MUSIQ is the only robust W4-v2 win** on full-valid. Unique-token ratio is positive in `3/3` seeds but remains within seed noise because seed123 is much stronger than seed456/seed789; consistency does not hold.

| axis | A3 full mean | w4v2 3-seed mean±seed std | delta vs A3 | seeds improving | reading |
|---|---:|---:|---:|---:|---|
| NIQE (inverted) | -8.456 | -8.482±0.128 | -0.026 | 2/3 | mean regresses; seed123 negative dominates two small positive seeds |
| MUSIQ | 50.107 | 50.502±0.163 | +0.395 | 3/3 | **ROBUST win** (`+0.395 > 0.163`) |
| MANIQA | 0.4085 | 0.4093±0.0030 | +0.0008 | 2/3 | positive mean but within noise / not seed-unanimous |
| CLIPIQA | 0.6121 | 0.6151±0.0043 | +0.0030 | 2/3 | positive mean but within noise / not seed-unanimous |
| consistency | 0.7922 | 0.7910±0.0001 | -0.00119 | 0/3 | regresses in all seeds; prior subset anti-drift claim does **not** hold |
| unique-token ratio | 0.6182 | 0.6495±0.0346 | +0.0313 | 3/3 | consistent positive direction, but **not robust** by strict rule (`+0.0313 < 0.0346`) |

Exact recomputation is in `results/aggregate_full_3seed.csv`. This resolves the two evaluation caveats (`n=30 → n=100`, `1 seed → 3 seeds`) for W4-v2 on DIV2K-valid: the defensible headline is a robust MUSIQ gain plus a seed-unanimous but still noisy prompt-diversity gain. Do **not** claim a single-metric or across-the-board win: NIQE and consistency regress by mean, and MANIQA/CLIPIQA are small, seed-mixed positives.

## 0064 probe

- Input `samples/0064.png`, A3=`ckpt/VLM_LoRA/checkpoint-10000`, w4v2=`ckpt/VLM_FT/coz_w4v2` via `--vlm_model_path`, `recursive_multiscale`, `--save_prompts`.
- Subject-token check (`eye|fur|animal|dog`) at deep scales: **PASS** — w4v2 scale 3 has `Fur`/`Animal`; scale 4 has `Fur`/`Animal`.
- Neural drift check (`neuron|synapse|dendrite|axon`): **PASS for no neural terms, tie vs A3** — w4v2 `0`, A3 rerun `0`; historical fail-case drift emitted those terms.
- Cross-scale diversity check: **PASS** — A3 `34/72 = 0.4722`; w4v2 `38/55 = 0.6909`; delta `+0.2187`. Caveat: w4v2 scale 2 is terse (`dog`).

## Reward-tuning ablation

- One 2-GPU seed-123 ablation raised `R_rep` `1.0 -> 3.0` and `R_fb` `1.0 -> 1.25` (`R_anc=0.2`, `R_phr=0.1` unchanged) to target prompt convergence directly.
- `tune_rrep` vs A3 (`n=30`): unique-token ratio `0.6234 -> 0.6442` (`+0.0208`), consistency `0.7808 -> 0.7841` (`+0.0033`), MUSIQ `50.27 -> 50.86` (`+0.59`), but inverted NIQE `-7.60 -> -7.74` (`-0.14`), MANIQA `-0.0036`, and CLIPIQA `-0.0100` regressed.
- Honest verdict: diversity moved in the intended direction and beat the w4v2 3-seed mean by `+0.0168`, but it is **not robust yet** because the gain is below the prior w4v2 seed std (`0.0263`) and it trades off several IQA axes. Do not replace W4-v2 with this weight without a 3-seed follow-up or a milder `R_rep` setting.

### Operating points @ n=100

The higher-`R_rep` / higher-`R_fb` checkpoint (`ckpt/VLM_FT/coz_tune_rrep`) was re-evaluated on the full DIV2K-valid split (`0801-0900`, `n=100`) with the same fixed greedy `recursive_multiscale` protocol and seed label `123`. This makes the operating points directly comparable at scale: A3 author baseline, balanced W4-v2 seed123 (`results/w4v2_full.csv`), and max-anti-convergence tuned seed123 (`results/tune_rrep_full.csv`). All axes are higher-is-better; NIQE is the inverted evaluator value.

| axis | A3 full mean | balanced W4-v2 seed123 | tuned `tune_rrep` seed123 | tuned delta vs A3 | tuned minus balanced | reading |
|---|---:|---:|---:|---:|---:|---|
| NIQE (inverted) | -8.456 | -8.629 | -8.414 | +0.041 | +0.214 | tuned recovers NIQE vs both A3 and balanced |
| MUSIQ | 50.107 | 50.644 | 49.813 | -0.294 | -0.831 | tuned gives up the balanced robust MUSIQ win |
| MANIQA | 0.4085 | 0.4107 | 0.4053 | -0.0032 | -0.0054 | tuned regresses |
| CLIPIQA | 0.6121 | 0.6199 | 0.6140 | +0.0019 | -0.0060 | tuned is slightly above A3 but below balanced |
| consistency | 0.7922 | 0.7910 | 0.7952 | +0.0030 | +0.0042 | tuned improves anti-drift where balanced regressed |
| unique-token ratio | 0.6182 | 0.6894 | 0.6342 | +0.0160 | -0.0552 | tuned improves over A3 but **does not** beat balanced at `n=100` |

Exact operating-point recomputation is in `results/operating_points_n100.csv`. Compared with the balanced 3-seed reference (`results/aggregate_full_3seed.csv`), tuned unique-token ratio (`0.6342`, `+0.0160` vs A3) is also below the balanced 3-seed mean (`0.6495`, `+0.0313`) and below the W4-v2 seed std threshold (`0.0346`), so the higher-`R_rep` arm is **not** a larger or more-robust anti-convergence win at `n=100`. The scaled tradeoff is clearer than the `n=30` probe: tuned buys NIQE and consistency recovery, but pays in MUSIQ, MANIQA, CLIPIQA-vs-balanced, and prompt diversity-vs-balanced. Keep W4-v2 as the balanced operating point; treat `tune_rrep` as a single-seed stress point, not a replacement.

## State-expansion ablation (A7) — stabilized matched rerun

A7 adds text-only AR-2 expanded state: the VLM system prompt additionally receives a generated caption of `x_{i-2}` (the two-steps-back zoom state) alongside the original-image caption; the image input remains a single current crop. The first A7 seed-123 run with the old W4-v2 stability settings collapsed (`response_length/mean 45.1 -> 3.3`, `actor/entropy 2.16 -> 0.34`, `actor/kl_loss 0.003 -> 8.4`, identical `新规发育` outputs). That old run is kept only as diagnostic history: it measured a broken model, not the expanded-state hypothesis.

The valid comparison below reran **both** cells under one stabilized config: validity guard on (`<8` model tokens / `<5` unique content tokens / `>30%` CJK => raw total `-2.0` and `R_rep=-1.0` before group z-norm), `kl_loss_coef=0.02`, `lr=3e-7`, `entropy_coeff=0.005`, `grad_clip=0.5`, `rollout.n=6`, `train_batch_size=2`, seed `123`, balanced rewards (`R_anc=0.2`, `R_rep=1.0`, `R_fb=1.0`, `R_phr=0.1`), `100` GRPO steps, and full-FT from the same merged author checkpoint. A 20-step A7 smoke passed before the full runs (`response_length/mean=32.5`, `actor/entropy=3.11`, `actor/kl_loss=0.169`, no >30% CJK completions; short outputs were forced to `R_rep=-1.0` / `score=-2.0`). Final health also stayed non-collapsed: A7-stable `response_length/mean=24.3`, `entropy=2.88`, `kl_loss=0.269`; W4-v2-stable `response_length/mean=48.1`, `entropy=1.89`, `kl_loss=0.094`.

Matched `n=100` DIV2K-valid eval (all axes higher-is-better; NIQE is inverted; `1` seed only):

| axis | A3 full | W4-v2-stable | A7-stable | A7 - W4-v2-stable | reading |
|---|---:|---:|---:|---:|---|
| NIQE (inverted) | -8.456 | -8.551 | -8.300 | +0.252 | A7 recovers NIQE vs matched W4-v2 and beats A3 |
| MUSIQ | 50.107 | 50.620 | 51.224 | +0.604 | A7 improves perceptual MUSIQ most in this seed |
| MANIQA | 0.4085 | 0.4090 | 0.4064 | -0.0026 | A7 regresses vs both controls |
| CLIPIQA | 0.6121 | 0.6194 | 0.6034 | -0.0160 | A7 hurts CLIPIQA |
| consistency | 0.7922 | 0.7926 | 0.7922 | -0.0004 | essentially tied / slightly below matched W4-v2 |
| unique-token ratio | 0.6182 | 0.6160 | 0.6333 | +0.0174 | A7 improves prompt diversity in this seed |

Exact recomputation is in `results/A7_stable_matched_comparison.csv`. Honest verdict: once collapse is prevented, the text-caption `x_{i-2}` state is **mixed, not a clean win**. It helps NIQE, MUSIQ, and prompt diversity versus the matched stabilized W4-v2 control, but hurts MANIQA and CLIPIQA and does not improve consistency. Treat this as a one-seed hypothesis signal, not a headline replacement for the existing 3-seed W4-v2 result. The old collapsed A7 and old W4-v2 cells remain diagnostic history only; the causal state-expansion comparison is A7-stable vs W4-v2-stable.

Future-work recommendation: test composite-image state injection instead of longer text state — one VLM image containing the current crop large plus a small labeled `x_{i-2}` thumbnail, with a prompt kept close to standard W4-v2. That keeps visual trajectory context while avoiding the long free-form text prompt that destabilized the original A7 run.

## Baseline ladder (A0/A1/A2/A3 vs ours) @ n=100

To state our GRPO improvement against the *original* Chain-of-Zoom (not only the author checkpoint), three reference baselines were produced under the **identical locked protocol** used for A3 and W4-v2: `recursive_multiscale`, `crop_strategy=center`, `rec_num=4`, `upscale=4`, `process_size=512`, `align_method=nofix`, SR LoRA `model_20001.pkl` + VAE `vae_encoder_20001.pt`, SD3-medium, greedy `max_new_tokens=32`, on the same `n=100` DIV2K-valid split (`0801-0900`). Only the prompt/VLM source differs per rung:

- **A0 — NN interpolation** (`--rec_type nearest`): no SR, no VLM. Each scale crops the matching region from the `512x512` source and NEAREST-upscales back to `512x512`; the deepest scale is the same `256x` (`4^4`) center zoom the SR arms reach (`results/A0_full.csv`).
- **A1 — Direct-SR, null prompt** (`--rec_type recursive_multiscale --prompt_type null`, empty text): the frozen SR pipeline with an empty prompt, identical recursion/crop/decode to A2/A3 — only the VLM prompt is removed (`results/A1_full.csv`).
- **A2 — original CoZ** (`--rec_type recursive_multiscale --prompt_type vlm_base`, stock Qwen2.5-VL-3B, **no VLM LoRA**): the proposal's "original Chain-of-Zoom" baseline. Differs from A3 only in the VLM weights (`results/A2_full.csv`).
- **A3 — author checkpoint** (`ckpt/VLM_LoRA/checkpoint-10000`) and **ours** = balanced W4-v2 full-FT GRPO (3-seed mean from `results/aggregate_full_3seed.csv`).

All axes are higher-is-better; NIQE is the inverted evaluator value. Means over `n=100` (exact recomputation in `results/baseline_ladder_n100.csv`):

| axis | A0 NN | A1 SR null | A2 original CoZ | A3 author | ours (W4-v2, 3-seed) | ours − A2 | ours − A3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NIQE (inverted) | -31.924 | -9.384 | -8.872 | -8.456 | -8.482 | +0.390 | -0.026 |
| MUSIQ | 26.469 | 48.427 | 50.072 | 50.107 | 50.502 | +0.430 | +0.395 |
| MANIQA | 0.4453 | 0.3980 | 0.4054 | 0.4085 | 0.4093 | +0.0038 | +0.0008 |
| CLIPIQA | 0.5170 | 0.6005 | 0.6104 | 0.6121 | 0.6151 | +0.0047 | +0.0030 |
| consistency | 0.7491 | 0.7889 | 0.7902 | 0.7922 | 0.7910 | +0.0008 | -0.0012 |
| unique-token ratio | 0.0000 | 0.0000 | 0.7038 | 0.6182 | 0.6495 | -0.0543 | +0.0313 |

Honest, multi-axis reading (no single-metric claim):

- **MUSIQ and CLIPIQA form a clean monotonic ladder** `A0 < A1 < A2 < A3 < ours`. Our GRPO is top of the ladder on both, beating original CoZ (A2) by `+0.430` MUSIQ / `+0.0047` CLIPIQA and the author (A3) by `+0.395` / `+0.0030`. MUSIQ is also the strictly-robust 3-seed win (see the definitive table above).
- **SR matters most, then the prompt:** A0 (no SR) is far below everything (MUSIQ `26.5`, NIQE `-31.9`); adding the frozen SR with no prompt (A1) jumps MUSIQ to `48.4`; adding any VLM prompt (A2) reaches `50.1`; fine-tuning (A3 -> ours) adds the final `+0.4`.
- **Against original CoZ (A2) our GRPO wins 5/6 axes** (NIQE `+0.390`, MUSIQ `+0.430`, MANIQA `+0.0038`, CLIPIQA `+0.0047`, consistency `+0.0008`) and loses only unique-token ratio.
- **Against the author (A3) our GRPO wins 4/6 axes** (MUSIQ, MANIQA, CLIPIQA, unique-token ratio) and slightly regresses on NIQE (`-0.026`) and consistency (`-0.0012`), exactly as the 3-seed table reports; on NIQE/consistency ours sits **between A2 and A3** (above original CoZ, just below the author).
- **Caveat — MANIQA is not monotonic:** A0 (degenerate NN) scores the *highest* MANIQA (`0.4453`), above every SR arm. A `2x2 -> 512` NEAREST image is near-flat, which MANIQA rewards; MUSIQ/CLIPIQA/NIQE all correctly rank A0 last. Treat A0's MANIQA as a no-reference-metric artifact, not real quality. (NIQE itself is ill-conditioned on `2/100` A0 images; those two are excluded from the A0 NIQE mean, which is computed over `98/100`.)
- **Caveat — unique-token ratio favors the un-tuned base VLM:** A2 (stock Qwen) has the highest token diversity (`0.7038`), above both A3 (`0.6182`) and ours (`0.6495`). Base Qwen emits long, free-form captions (high raw token variety) — the same verbosity that drives the semantic-drift failure case — so this axis should not be read as "original CoZ has better prompts". A0/A1 are `0.0` by construction (no prompt / empty prompt). Our anti-convergence gain (`+0.0313`) is therefore claimed **only vs the author checkpoint (A3)**, which is the fine-tuned regime; we do **not** claim to beat raw base-Qwen verbosity on this metric.

Net: on the locked protocol our GRPO finetune is at the top of the ladder for the two cleanest perceptual axes (MUSIQ, CLIPIQA), beats original CoZ (A2) on 5/6 axes, and beats the author (A3) on 4/6 — with the two honest exceptions called out above (MANIQA's A0 artifact and base-Qwen's raw token diversity).

## Caveats & next steps

- The outstanding eval caveats are now resolved for W4-v2 (`3` seeds on all `100` DIV2K-valid images), and `tune_rrep` / A7-stable now have directly comparable `n=100` evals, but those arms are still only `1` seed; interpret the seed-std robustness rule as a practical filter, not a formal significance test.
- For the balanced W4-v2 operating point, stabilize NIQE/consistency and reduce diversity variance: consistency regresses in `3/3` full-valid seeds, inverted NIQE regresses by mean, and the unique-token gain is seed-unanimous but below seed std.
- A7 text-caption state expansion is now a stabilized, matched one-seed result: mixed (NIQE/MUSIQ/diversity up; MANIQA/CLIPIQA down; consistency tied/slightly down). The next state-context test should use composite-image injection rather than more free-form text.
- Remaining arm: A8 higher-`R_fb` weight is still unrun and should report all `6` axes, not a single metric.
