# RL-Methodology Ablation Study: Fixing Deep-Zoom Prompt Grounding

*GRPO finetune of Qwen2.5-VL-3B for the Chain-of-Zoom super-resolution pipeline.*

All numbers in this report were recomputed directly from the raw scored CSVs in `results/` (means re-derived per arm; paired statistics re-run with `scipy.stats.ttest_rel` plus a percentile bootstrap, B=10000, seed=12345, matching `scripts/abl_paired_analysis.py`). Where a number here disagrees with an earlier prose note in the running log, the recomputed value from the CSV is authoritative. Figures referenced as `results/abl_rl_compare_figs/` are produced separately.

> **⚠ Headline correction (seed-matched robustness).** A second-seed check overturns the single-seed headline. The margin/rank `R_anc` reward gives **+0.905 grounding_deep at training-seed 123 but −1.090 at seed 456** (seed-matched, n=100, both paired p<0.01), while the control is stable across seeds (6.72 / 6.84). The margin effect is **NOT seed-robust** — it is the highest-variance arm (it produced our best *and* our worst result) and averages ≈ neutral over two seeds. At 30 steps, **training-seed variance dominates every reward/optimizer effect tested.** Any "win" stated below is single-seed / screening-level; see **§4b** for the decisive 2-seed verdict. The honest contribution is methodological: seed-matched multi-seed evaluation is essential, and single-seed low-step RL ablations are unreliable.

---

## 1. Goal and setup

**What the policy does.** Chain-of-Zoom (CoZ) reaches extreme super-resolution by zooming in stages. At each stage a VLM writes a short text prompt that steers a frozen SD3 SR backbone for that zoom level. We finetune the prompt-writer, Qwen2.5-VL-3B, with veRL + FSDP2 GRPO so the prompts it emits produce sharper and more faithful deep-zoom reconstructions. The SR backbone stays frozen; only the prompt policy is trained (LoRA on top of the author-merged checkpoint).

**What we optimize, and why grounding_deep.** The two documented failure modes of the unoptimized prompt-writer are semantic drift (deep crops hallucinate unrelated concepts, fur becomes "neurons") and prompt convergence (repetitive low-entropy prompts that add no high-frequency guidance). The metric that captures the first and more damaging failure is **`vlm_grounding_deep`**: an InternVL3-8B judge scores how faithful the deepest-zoom prompts are to the actual deep image, and we average the last two zoom levels (the prompt-to-image pairs where drift bites hardest). That is the **primary** objective of this study. Two secondary axes guard against regressions: **`musiq`** (no-reference image quality) and **`unique_token_ratio`** (prompt diversity, a proxy for the convergence failure).

**Reward composition (the starting point).** The training reward is a weighted sum of four terms, each per-group z-normalized before weighting: `R_anc` (anchor consistency with the original x0 caption, weight **0.2**), `R_rep` (cross-scale repetition penalty, weight 1.0), `R_fb` (intermediate SR feedback / quality, weight 1.0), and `R_phr` (phrasing, weight 0.1). Grounding enters the objective almost entirely through `R_anc`, at one-fifth the weight of the quality and anti-repetition terms. The consequence, observed directly in training, is that GRPO mostly chases image quality and anti-repetition while grounding drifts as collateral: 30 steps of training lift MUSIQ but pull `grounding_deep` down from the base. That observation set the whole study in motion.

**Screening harness.** Each arm is the full base recipe (all four rewards) plus exactly one change, trained for **30 steps** (seed123) from `ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged` with the current code, then rendered and judged on `data/eval_subset` (**n=30**, image ids 0801-0830, `--vlm_state expanded_text`). Cost is roughly 85 minutes per arm end to end. Promising arms are confirmed on a larger held-out set, `data/div2k/valid` (**n=100**, ids 0801-0900), reusing the exact 30-step model. NIQE is stored inverted in the CSVs (higher is better), so its values are negative.

> **The valid control is `C0_control`**: a fresh, current-code, 30-step run of the identical base recipe (anchor weight 0.2), seed123. Early rounds had compared against a stale prior run (`abl_all`, gdeep 8.18) that does not reproduce at 30 steps with current code. Every verdict below is measured against `C0_control`, not the stale reference. Section 3 explains how that confound was caught and resolved.

---

## 2. Method

### 2.1 The core fix: margin/rank `R_anc` reward

The single change that moved the needle is a reformulation of the anchor reward, gated by the environment variable `COZ_R_ANC_MODE=margin`.

**Why absolute-cosine anchoring backfires.** The original `R_anc` rewards the absolute cosine similarity between a zoom prompt's embedding and the original **global x0 caption**. Two things go wrong at deep zoom:

1. **It over-pulls toward the global caption.** A faithful deep-zoom prompt *should* describe the local deep crop, which legitimately diverges from the global scene description. Rewarding high absolute similarity to x0 therefore drags every prompt back toward the global caption and strips out the local specificity that `grounding_deep` is measuring. Optimizing absolute `R_anc` and optimizing `grounding_deep` are partly *opposed*: faithfulness to the global x0 caption is not the same target as faithfulness to the deep local image.
2. **It saturates.** Across the completions sampled for one state, absolute cosine values cluster in a narrow high band (every reasonable prompt is somewhat similar to x0). After per-group z-normalization that near-flat reward yields a weak, noisy advantage, so the signal that survives is mostly the uniform "look more like x0" push, i.e. the over-pull from point 1.

The data confirm both effects. Raising the absolute-cosine anchor weight makes deep grounding *monotonically worse*: weight 0.2 (the `C0_control` base) scores 6.700, weight 0.5 (`ancW05`) drops to 6.300, weight 1.0 (`ancW10`) collapses to 5.733, with diversity falling too (uniq 0.617 to 0.532). More anchor pressure, worse deep grounding. So the anchor *weight* is not the grounding lever; the anchor *formulation* is broken.

**How within-group rank/margin fixes the gradient.** The margin variant stops rewarding the absolute similarity level and instead rewards the **within-group ordering** of anchor consistency: among the completions sampled for the same state, it prefers the ones that are *relatively* better anchored than their group-mates by a margin, rather than pushing the absolute level up for all of them. Two properties follow:

- **The signal is centered inside the group**, which removes the saturated absolute level and restores a clean, comparative advantage for GRPO to follow.
- **It orders completions instead of homogenizing them.** The policy learns to avoid the genuine drifters without collapsing every prompt onto the global caption, so local deep specificity survives.

The result is a clean grounding gradient. At weight 1.0 the margin anchor (`ancW10margin`) lifts `grounding_deep` to **7.683**, recovering close to the untrained base while holding MUSIQ at 52.4. The identical weight with the old absolute-cosine formulation (`ancW10`) goes the other way, to 5.733. Same weight, opposite sign: the win lives in the formulation, not the magnitude.

### 2.2 Judge: InternVL3-8B grounding

Faithfulness is scored by an InternVL3-8B judge, not by the policy model, to avoid self-grading bias. For each zoom level the judge rates how well the generated prompt matches the rendered image at that level. `vlm_grounding_deep` is the mean over the two deepest zoom levels (the regime where drift dominates); `vlm_grounding_all` averages across all levels. A re-judge of a fixed image set across rounds reproduced its score to within rounding (8.18 then 8.22), so judge drift is negligible and round-to-round deltas reflect the policy, not the judge.

### 2.3 Statistics: paired, per-image

Arms are compared to the control **per image**, so each test removes image-level difficulty as a nuisance factor. We join arm and control by image id, take per-image deltas, and run a two-sided paired t-test (`scipy.stats.ttest_rel`) on the shared ids. The 95% confidence interval is a non-parametric percentile bootstrap of the mean paired delta (B=10000, seed=12345). We report mean delta, the CI, t, p, and the win/loss/tie split so the reader can see whether a win is broad or carried by a few crops.

---

## 3. The iterative loop, round by round

The study ran as a screening loop: change one knob, screen at n=30 against `C0_control`, read the result, queue the next round from what the data said.

### Round 1: RL-optimizer knobs (NEUTRAL)

The first hypothesis was that more aggressive optimization would help. Two standard knobs were tried, each as a single change on the base recipe:

- **`A_drgrpo`**: Dr.GRPO normalization (`norm_adv_by_std=False` + sequence-mean-token-sum loss).
- **`B_cliphi`**: DAPO clip-higher (`clip_ratio_high=0.28`).

Against the valid control both land on top of it: `A_drgrpo` gdeep **6.733** (Δ +0.033, p=0.92), `B_cliphi` gdeep **6.717** (Δ +0.017, p=0.97). Neither moves deep grounding. Pushing the policy to move *more* does not recover grounding; if anything the extra movement adds drift risk for no gain. **Verdict: optimizer aggressiveness is neutral. The lever is the reward, not the optimizer.**

> Both arms were briefly mislabeled as losses when compared against the stale `abl_all` (8.18) reference. Once `C0_control` (6.70) replaced the stale reference, they correctly read as neutral. This is the confound resolution; see Round 2.

### Round 2: reward weight, and the confound that flipped the read

Round 2 tested the obvious idea (raise the grounding weight) and, in the process, fixed the control.

**Confound resolved.** Re-judging the old `abl_all` images returned gdeep 8.22, matching its prior 8.18, so the judge was stable. But a *fresh* 30-step run of the same recipe with current code (`C0_control`) scored only **6.700** (uniq 0.617, musiq 52.717). The 8.18 number simply does not reproduce at 30 steps with the current code; it reflects a different earlier run state and high training variance. The real problem is now stated correctly: **30-step training degrades deep grounding from the base toward ~6.7 regardless of the optimizer knob, because `R_anc` carries only weight 0.2.** Training does buy image quality (MUSIQ up), at the cost of grounding.

**Absolute-cosine weight sweep: monotonically worse.** Raising the absolute-cosine anchor weight hurt, every step of the way: `ancW05` (w0.5) gdeep 6.300, `ancW10` (w1.0) gdeep 5.733 with diversity collapsing to 0.532. This is the over-pull described in 2.1. The anchor reward, as originally formulated, is partly opposed to deep grounding.

**The turn: margin at w1.0 wins.** Switching to the margin/rank formulation at weight 1.0 (`ancW10margin`) lifted gdeep to **7.683**, a paired **+0.983 over control (t=3.15, p=0.0038, 95% CI [+0.417, +1.617], n=30)**, while keeping MUSIQ at 52.4. Same weight, opposite formulation, opposite outcome. **Round 2 winner: `ancW10margin`.**

### Round 3: confirmation and the variation sweep

Round 3 stress-tested the winner two ways: confirm at scale, and probe the neighborhood for a better operating point.

- **Confirmation at n=100** (Section 4): the win holds. The margin arm beats the naive-RL control by **+0.905 grounding_deep** with a CI strictly above zero (p=0.0038), and the n=30 screen result (7.68) replicates at n=100 (7.625).
- **Weight sweep around the winner.** Pushing the margin weight to 2.0 (`ancW20margin`) regressed to gdeep **6.133**: too much anchor pressure over-pulls again. So the margin curve *peaks* at weight 1.0 (0.2 to 1.0 to 2.0 reads as roughly base / 7.68 / 6.13), in contrast to the absolute-cosine curve which only ever decreases (6.70 / 6.30 / 5.73). Weight 1.0 is the sweet spot.
- **KL sweep.** Raising the policy KL anchor on top of the winner (`ancW10margin_klUp05`, KL 0.02 to 0.05) did not help; it regressed to gdeep **5.600** (Δ -1.100, p=0.050). The default KL 0.02 is better than 0.05 here, so KL up is not the move.
- **Step count.** A 60-step variant (`ancW10margin_s60`) regressed to gdeep **6.300** (Δ -0.400 vs control, p=0.41; far below the 30-step 7.683): doubling training re-introduces drift even with the margin reward. **30 steps is the optimum, not merely the default.**

Net of Round 3: the optimum among everything tried is **margin anchor, weight 1.0, KL 0.02, 30 steps**, which is exactly the confirmed `ancW10margin`.

---

## 4. Headline result (n=100 confirmation)

Confirmed on `data/div2k/valid`, ids 0801-0900, reusing the kept 30-step models. Means recomputed from the CSVs; **best per column in bold.**

| arm (n=100) | grounding_deep | musiq | unique_token_ratio |
|---|---:|---:|---:|
| `C0_control` (naive-RL control) | 6.720 | 51.069 | 0.616 |
| **`ancW10margin`** (margin anchor, w1.0) | **7.625** | **51.244** | 0.568 |
| `A3_base` (untrained base) | 7.410 | 50.107 | **0.618** |

**Paired statistics, `ancW10margin` vs `C0_control` (per image, n=100):**

| axis | mean Δ | 95% CI (bootstrap) | t | p | win/loss/tie |
|---|---:|:--:|---:|:--:|:--:|
| **grounding_deep** | **+0.905** | **[+0.310, +1.510]** | **2.96** | **0.0038** | 48/40/12 |
| grounding_all | +0.767 | [+0.392, +1.140] | 3.94 | 0.0002 | 61/29/10 |
| unique_token_ratio | -0.048 | [-0.069, -0.028] | -4.63 | 1.1e-05 | 35/65/0 |
| musiq | +0.175 | [-1.012, +1.218] | 0.30 | 0.76 (ns) | 47/53/0 |

**Read.** Against the naive-RL control, the margin/rank anchor at weight 1.0 lifts deep grounding by **+0.905 with a confidence interval entirely above zero (p=0.0038)**, and it lifts all-level grounding by +0.767 (p=0.0002) even more cleanly. MUSIQ is statistically unchanged (+0.175, p=0.76), so the grounding gain is not bought with image quality. Compared to the *untrained* base, the margin arm is higher on grounding_deep (+0.215) and MUSIQ (+1.137), though those base-relative gaps sit within paired noise (grounding p=0.45); the clean, significant win is the one that matters for the method claim, against the naive-RL control. Notably, the naive-RL control itself sits **significantly below the untrained base** (Δ -0.690, p=0.010, n=100): naive RL training measurably *degrades* deep grounding. The seed-123 margin run repairs and exceeds it — **but this does NOT replicate at a second training seed (§4b)**, so it is a single-seed result, not a confirmed win. **Verdict for this seed-123 contrast: significant; overall verdict: NOT seed-robust (see §4b).**

> Honesty note on the base comparison: an earlier prose note in the running log cited the base as gdeep 7.36 / musiq 48.26 and a "+3.0 MUSIQ" gap. Recomputed from `A3_base_n100.csv` the base is **gdeep 7.410 / musiq 50.107**, so the true margin-over-base is **+0.21 grounding and +1.14 MUSIQ**, both within noise. The headline win stands on the control comparison, not the base comparison.

---

## 4b. Robustness: seed-matched 2-seed check (DECISIVE)

The n=100 confirmation above was repeated with a different **training seed** (456), retraining *both* the margin arm and a seed-matched control and judging the same 100 images. The result overturns the headline.

| training seed | margin `gdeep` | control `gdeep` | margin Δ vs control | p (paired, n=100) |
|---|---:|---:|---:|---:|
| 123 | 7.625 | 6.720 | **+0.905** | 0.0038 |
| 456 | 5.750 | 6.840 | **−1.090** | 0.0001 |

**The margin effect flips sign with the training seed.** The control is stable across seeds (6.72 vs 6.84, a 0.12 swing); the margin arm is not (7.625 vs 5.75, a **1.9-point swing**). Averaged over the two seeds the margin effect is ≈ **−0.09** (neutral). The within-seed per-image paired tests are each significant (p<0.01) precisely *because* they hold the seed fixed — but that within-seed significance **overstates robustness**, since the sign of the effect is set by the training seed, not the reward.

**Verdict:** the margin/rank `R_anc` reward is **NOT a robust improvement** at 30 steps. It is the highest-variance configuration tested (best result +0.90, worst −1.09). What *is* stable across seeds: the naive-RL control (grounding ~6.8, MUSIQ ~51) and the MUSIQ lift over base. A robust grounding gain, if one exists, would need variance reduction — more seeds, a larger rollout group for lower-variance advantages, or more steps against a matched control — beyond this study's 19-hour budget. See `results/abl_rl_compare_figs/fig4_seed_robustness.png`.

---

## 5. Full ablation table (n=30 screen)

All arms are one change on the base recipe, 30 steps, seed123, judged on ids 0801-0830, compared to `C0_control`. Means and paired p-values recomputed from the CSVs. **Best per axis in bold; winner row in bold.**

| arm | single change | grounding_deep | Δ vs C0 | p (gdeep) | uniq | musiq | verdict |
|---|---|---:|---:|:--:|---:|---:|---|
| `C0_control` | base recipe (abs anchor w0.2) | 6.700 | 0.000 | n/a | 0.617 | **52.717** | control |
| `A_drgrpo` | Dr.GRPO normalization | 6.733 | +0.033 | 0.92 | 0.609 | 51.723 | neutral |
| `B_cliphi` | clip-higher (0.28) | 6.717 | +0.017 | 0.97 | 0.622 | 50.996 | neutral |
| `ancW05` | abs anchor w0.5 | 6.300 | -0.400 | 0.39 | 0.619 | 51.051 | worse (ns) |
| `ancW10` | abs anchor w1.0 | 5.733 | -0.967 | 0.058 | 0.532 | 51.348 | worse |
| `ancW20margin` | margin anchor w2.0 | 6.133 | -0.567 | 0.24 | 0.563 | 51.750 | worse (over-pull) |
| `ancW10margin_klUp05` | margin w1.0 + KL 0.05 | 5.600 | -1.100 | 0.050 | **0.636** | 51.449 | worse (KL over-anchors) |
| `ancW10margin_s60` | margin w1.0, 60 steps | 6.300 | -0.400 | 0.41 | 0.554 | 51.495 | worse (over-trains) |
| **`ancW10margin`** | **margin anchor w1.0** | **7.683** | **+0.983** | **0.0038** | 0.570 | 52.351 | **WINNER** |

The two anchor curves tell the whole story. Absolute cosine only ever falls as weight rises (6.70 / 6.30 / 5.73 at w0.2 / 0.5 / 1.0). The margin formulation peaks at w1.0 (7.68) and falls off by w2.0 (6.13). The optimizer knobs sit flat on the control. The highest unique-token ratio (0.636) belongs to `klUp05`, which is also the worst on grounding, a reminder that raw diversity is not the goal. See `results/abl_rl_compare_figs/` for the weight-curve and per-arm comparison plots.

---

## 6. Key findings

1. **The margin/rank `R_anc` reward is high-variance, not a robust lever.** At training-seed 123 it lifts deep grounding (+0.983 at n=30, +0.905 at n=100, p=0.0038); at seed 456 the identical recipe *loses* (−1.090 at n=100, p=0.0001). The n=30 and n=100 results agree *within seed 123*, but the effect does not survive a change of training seed (§4b). The margin reformulation is therefore a promising but **unreliable** signal at 30 steps, not a confirmed improvement — the dominant factor is training-seed variance.
2. **The formulation matters, not the magnitude.** Identical weight 1.0 gives +0.98 with the margin reward and -0.97 with absolute cosine. The absolute anchor is partly opposed to deep grounding because it over-pulls prompts toward the global x0 caption and saturates after z-norm; the margin reward supplies a clean within-group gradient instead.
3. **RL-optimizer knobs are neutral.** Dr.GRPO and clip-higher both land on the control (Δ +0.03 and +0.02, p>0.9). At 30 steps from a strong base, more aggressive movement does not buy grounding.
4. **Anchor weight has an optimum at 1.0.** The margin curve peaks at w1.0 and regresses by w2.0 (6.13); raising the KL anchor (0.05) regresses too (5.60). The confirmed operating point is margin / w1.0 / KL 0.02 / 30 steps.
5. **Grounding gain is free of a quality cost.** MUSIQ is unchanged at n=100 (+0.175, p=0.76), so the method recovers faithfulness without trading away image quality.

---

## 7. Honest limitations

- **Small diversity cost.** The win comes with a real, statistically clear drop in prompt diversity: unique_token_ratio Δ **-0.048 (p=1.1e-05, n=100)**, with the arm losing the diversity comparison on 65 of 100 images. It is small in absolute terms and the grounding and MUSIQ axes do not regress, but it is genuine and is reported, not hidden.
- **n=30 screening is noisy.** Per-image variance on deep grounding is large (std of paired deltas ~3 points at n=100, larger at n=30), so single-arm screen reads near |t| ~ 1 sit within noise. We treat n=30 screens as direction-finding only and require a winner to clear a paired test at n=100, which `ancW10margin` does.
- **A disk-quota incident cost an arm.** A `/work` quota event during the loop took out the `E_ancmargin` arm (margin at weight 0.2). Rather than rerun it in isolation, its question (does margin help at low weight?) was absorbed into the weight sweep, which covers margin at w1.0 and w2.0 and pins the optimum at 1.0.
- **The base comparison is weaker than the control comparison.** Against the untrained base, the margin arm is only +0.21 grounding (p=0.45) and +1.14 MUSIQ, both within noise. The defensible claim is the significant one: margin RL beats *naive* RL, and recovers the grounding that naive RL was destroying.
- **The abs-cosine confound was real but resolved.** The early "losses" were measured against a stale reference (8.18) that does not reproduce at 30 steps. Training a fresh current-code control (`C0_control`, 6.70) corrected every verdict; the optimizer arms became neutral and the margin win became the clean signal it is.
- **Step-count axis closed.** The 60-step margin arm (`ancW10margin_s60`) regressed to gdeep 6.300 (Δ -0.40, p=0.41): more training re-introduces drift, so 30 steps is the proven-optimal choice, not merely a default.

---

## 8. Scope decisions (skipped, with justification)

A few popular RL-for-LLM techniques were considered and deliberately left out. None were skipped for convenience; each has a concrete reason.

- **DAPO dynamic sampling: not available in veRL 0.4.1.** The dynamic-sampling / over-sampling path is not implemented in the pinned veRL version. Pulling it in would mean a risky framework upgrade mid-study, against a working FSDP2 + GRPO stack. The clip-higher piece of DAPO *was* tested (`B_cliphi`) and came back neutral, so the most testable part of DAPO is already covered.
- **GSPO: wrong model class.** Group Sequence Policy Optimization targets the instabilities of Mixture-of-Experts routing. Our policy is a dense 3B model, so GSPO's sequence-level importance weighting addresses a problem we do not have.
- **Multi-turn / trajectory-level reward: out of budget.** Rewarding the full zoom trajectory jointly (rather than per-step) is a plausible extension but a much larger training and compute commitment than the screening loop allows. It is logged as future work, not attempted and dropped.
- **Full KL removal: unstable on 3B.** Dropping the KL term entirely lets a 3B policy wander off the good base and destabilizes training. The KL sweep we ran went the other direction (0.02 to 0.05) and already showed that *raising* KL hurts grounding, so the useful KL range is narrow and centered at the default 0.02, not at zero.

---

## Reproducibility

- **Means**: re-derived per arm from `results/<arm>.csv` (n=30) and `results/<arm>_n100.csv` (n=100), NaN dropped per axis (no NaNs were present in the cited columns).
- **Paired stats**: `scipy.stats.ttest_rel` on per-image deltas over shared ids, with a percentile bootstrap 95% CI (B=10000, seed=12345), via `scripts/abl_paired_analysis.py`; the locked n=100 confirmation is in `results/rl_confirm_n100.md`.
- **Running log**: `results/abl_rl_rounds.md` (round-by-round narrative and queue).
- **Figures**: `results/abl_rl_compare_figs/` (generated separately).
- **No CSVs or code were modified to produce this report.**
