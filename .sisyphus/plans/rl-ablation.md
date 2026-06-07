# RL-Methodology Ablation Study — Executable Task Graph
**Repo:** /work/seanma0627/RL_2026_Final · **Branch:** grpo/coz-vlm · **Hardware:** 2×H200 (both idle)
**Deadline:** tomorrow 21:00 (~19h from 02:00) · **Goal:** prove independent RL-optimization gains on the
veRL+FSDP2 GRPO finetune of Qwen2.5-VL-3B (CoZ per-zoom prompt generator), combine winners, confirm at n=100.

> Mantra: **ablate each change independently → screen → pick winners → combine → confirm n=100 with paired stats.**
> Bias toward RL-optimization changes, one reward-side substitute (margin-R_anc). False-gain guard throughout.

---

## 0. Locked decisions (do not re-litigate)

**CONTROL.** ALL reward recipe (R_anc+R_rep+R_fb+R_phr) + current RL config.
- *Screening control* = **reuse `ckpt/VLM_FT/coz_abl_all` (30-step ALL)**; do NOT retrain. Re-render + re-judge it
  fresh, in the same batch/session as the arms, so the paired baseline shares judge conditions. (Decision: screen at
  **30 steps** to make reuse valid and save one control retrain; G0 may bump to 50 only if timing is cheap.)
- *Confirmation control* = **fresh retrain at 120 steps × 2 seeds** (30-step ckpt is invalid for a 120-step paired claim).

**>>> CORRECTION (atlas, vs plan-agent draft): REWARD ENV MUST MATCH THE REUSED CONTROL <<<**
`coz_abl_all` was trained with the launcher exports **R_ANC=0.2, R_REP=1.0, R_FB=1.0, ENABLE_RFB=1, R_PHR=0.1**
(the `task_abl_arm.sh` defaults override the `verl_custom_reward._cfg` defaults of 3.0/1.25). Therefore **every
screening arm and the confirmation control/ALL-RL MUST use R_REP=1.0 and R_FB=1.0**, NOT 3.0/1.25 — otherwise an
arm differs from control by both the RL knob AND the reward weights (confounded, broken ablation). W0.5 must VERIFY
the exact env by reading `.sisyphus/evidence/task-abl-all-train.log` (line ~6 echo) before launching any arm.

**Common reward env for EVERY training arm (screening + confirmation):**
```
COZ_R_ANC_WEIGHT=0.2 COZ_R_REP_WEIGHT=1.0 COZ_R_FB_WEIGHT=1.0 COZ_ENABLE_RFB=1 COZ_R_PHR_WEIGHT=0.1
# script constants already set: COZ_VALIDITY_GUARD=1 COZ_RFB_METRIC=musiq COZ_RFB_CONSISTENCY_WEIGHT=0.5
# ^ these values are provisional pending W0.5 verification against task-abl-all-train.log; match whatever it shows.
```

**ARMS (each = CONTROL + exactly ONE change):**
| Arm | Change | Mechanism |
|---|---|---|
| `A_drgrpo` | `norm_adv_by_std_in_grpo=False` + `loss_agg_mode=seq-mean-token-sum-norm` (Dr.GRPO norm; **KEEP KL=0.02**) | CLI override |
| `B_cliphi` | `clip_ratio_low=0.2`, `clip_ratio_high=0.28` (DAPO clip-higher) | CLI override |
| `C_kl` | `kl_loss_coef=0.01` (0.5×; KL **not** removed) | CLI override |
| `D_group` | `rollout.n=8` (from 6) | CLI override |
| `E_anc_margin` | within-group margin/rank R_anc (reward-side substitute for dynamic sampling) | **new code** `COZ_R_ANC_MODE=margin` |
| `F_temp` *(opt)* | `rollout.temperature=1.1` | CLI override |
| `G_fbnorm` *(opt)* | gated R_fb internal rescale so consistency isn't numerically dead | **new code** `COZ_RFB_NORM=1` |

**SKIP (with justification, stated in writeup):**
- True DAPO dynamic sampling (`filter_groups`/`gen_batch_size`) — **not in veRL 0.4.1**; upgrade is high-risk vs the custom controller. Margin-R_anc is the in-budget substitute.
- GSPO — designed for MoE; ours is a dense 3B.
- Multi-turn/trajectory reward — too risky for the budget.
- Full KL removal — unstable on a 3B; C_kl only halves it.
- Per-image baseline subtraction / reward whitening for R_anc — cannot create signal in flat groups; use rank/margin instead.

---

## 1. Wave / dependency graph

```
WAVE 0  (parallel; code on CPU ∥ G0 on GPU)
  W0.1 G0 pipeline smoke (GPU)  ──────────────┐  emits: s/step, s/img render, s/img judge → finalizes screen sizes
  W0.2 margin-R_anc  (TDD, CPU) ──┐           │
  W0.3 R_fb-norm     (TDD, CPU) ──┤ commits   │
  W0.4 stats upgrade (TDD, CPU) ──┤ 1,2,3     │
  W0.5 arm-override passthrough + recipe sanity (CPU) ─┘ (blocks all training)
                         │                    │
WAVE 1  SCREENING (gated by W0.1 sizing + W0.5; E needs W0.2; G needs W0.3)
  Train arms @30 steps, 2-GPU SEQUENTIAL: A → B → E → C → D  (→ F,G if budget)
  Pipeline: merge(arm) → render(arm, free GPU) → judge(arm, API/CPU ∥ next train)
  Control = reuse coz_abl_all ckpt → render+judge fresh at screening n
                         │
WAVE 2  ANALYSIS / WINNER PICK (CPU) — needs all W1 judged + W0.4
  paired ttest_rel + bootstrap CI vs control; false-gain guard (reward↔grounding corr); pick winners
                         │
WAVE 3  CONFIRMATION (GPU sequential) — needs W2 winners
  Build ALL-RL combined arm. Train {control, ALL-RL} × {seed123, seed456} @120 → merge → render+judge n=100
  → paired ttest_rel + bootstrap CI (n=100)
                         │
WAVE 4  DOCS / FIGURES / PUSH (CPU) — needs W3
  writeup md + figures; commits 4,5; push branch
```

**Parallelism summary**
- W0.2/W0.3/W0.4 fully parallel with each other and with W0.1 (CPU vs GPU). W0.5 is a tiny edit, parallel too.
- W1 trainings are **strictly sequential** (1-GPU concurrency failed before). The *only* overlap: render needs a free GPU (run between trains or after last train); **judge is HTTP → run on CPU concurrently with the next arm's training**.
- W2/W4 are CPU and can overlap with tail-end GPU work.

---

## 2. Tasks (objective · commands/files · category+skills · done-when · estimate)

### WAVE 0

**W0.1 — G0 pipeline smoke + sizing** `category=deep, load_skills=[]`
- **Objective:** measure real per-step train time, per-image render time, per-image judge latency on a tiny end-to-end run; lock screening/confirmation sizes.
- **Do:** in tmux, train a 3-step throwaway with the ALL recipe (verified env):
  `COZ_ARM=g0_smoke COZ_TOTAL_STEPS=3 COZ_SAVE_FREQ=3 <verified ALL reward env> bash .sisyphus/evidence/task_abl_arm.sh`
  then `python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/g0_smoke/global_step_3/actor --target_dir ckpt/VLM_FT/g0_smoke`, render **2 images** (cmd template §3), judge those 2 (cmd §3).
- **Done when:** log shows real s/step (expect ~60–65s w/ R_fb), measured s/img render + s/img judge; a one-line **sizing decision** recorded (screen n, screen steps=30 default, confirm n, confirm steps/seeds) that fits §4 budget. Full loop ran without error (validates merge+render+judge wiring & env tokens).
- **Est:** 30–45 min. **Blocks:** all of Wave 1/3 sizing.

**W0.2 — margin/rank R_anc (TDD)** `category=ultrabrain, load_skills=[]`
- **Objective:** replace saturated absolute-cosine R_anc with a within-group rank/margin signal; flat group (cosine range < eps) contributes 0.
- **Files/functions:**
  - `train/grpo/rewards.py`: `RewardOrchestrator.__init__` read `self.r_anc_mode = self.cfg["r_anc"].get("mode","cosine")`; in `_per_group_components` (rewards.py:188-201) branch `key=="r_anc" and mode=="margin"`: collect group raw cosines; if `max-min <= _EPS` → all 0 (skip); else rank → centered spread `(rank-(n-1)/2)/((n-1)/2)` ∈ [-1,1], zero-mean, monotone in cosine. Other components keep z-norm. Keep `_r_anc` (117-123) returning raw cosine.
  - `verl_custom_reward.py`: `_cfg()` r_anc block add `"mode": os.environ.get("COZ_R_ANC_MODE","cosine")`.
- **TDD (write tests first):**
  - `tests/test_rewards.py::test_r_anc_margin_ranks_and_skips_flat`: distinct cosines → r_anc uses monotonic + sum≈0; equal cosines → all 0; **regression**: default mode path unchanged (existing tests stay green).
  - `tests/test_verl_reward.py::test_anc_margin_env`: `COZ_R_ANC_MODE=margin` + FakeClip with distinct anchors → highest-anchor completion ranks top (not flattened).
- **Done when:** `pytest tests/test_rewards.py tests/test_verl_reward.py -q` green; text-only smoke with `COZ_R_ANC_MODE=margin` shows within-group r_anc std > 0 when anchors differ. → **commit 1**.
- **Est:** 60–90 min. **Blocks:** E_anc_margin, ALL-RL combo.

**W0.3 — R_fb internal rescale (TDD, gated, optional)** `category=deep, load_skills=[]`
- **Objective:** stop MUSIQ (std~14) from drowning consistency (∈[0,1]); make the term influence R_fb again. **Gated, default OFF** so control/all arms are byte-identical unless `COZ_RFB_NORM=1`.
- **Files/functions:** `train/grpo/rewards.py::_r_fb` (138-156) — when `COZ_RFB_NORM`, rescale quality (e.g. MUSIQ/100) before `q + cw*cons` so cons is not numerically dead. **Documented deferral:** cons referencing degraded crop x_{i-1} (not anchor x_0) is left as a noted limitation (needs x0-image plumbing; out of budget).
- **TDD:** `tests/test_rewards.py::test_rfb_norm_makes_consistency_matter`: with flag ON, varying `cons` measurably moves R_fb; with flag OFF, output byte-identical to current.
- **Done when:** tests green; default-off regression proven. → **commit 2**.
- **Est:** 45–60 min. **Priority:** LOW (reward-eng; only screens as G_fbnorm if budget). **Blocks:** G_fbnorm only.

**W0.4 — paired-stats upgrade (TDD)** `category=unspecified-low, load_skills=[]`
- **Objective:** replace hand-rolled `t_read` with `scipy.stats.ttest_rel` + bootstrap 95% CI; generalize hardcoded `ARM_FILES`/`BASE` so any arm-vs-control set works at n=100.
- **Files/functions:** `scripts/abl_paired_analysis.py`: `paired_stats()` add `t,p = ttest_rel(arm,base)` and `bootstrap_ci(delta, B=10000)` (lower/upper); add argparse `--base LABEL` + `--arm label=file ...` (keep current defaults as fallback).
- **TDD:** `tests/test_paired_analysis.py` (new): on a tiny synthetic paired set, assert returned p ≈ `scipy.stats.ttest_rel` and bootstrap CI brackets mean delta; sign of `t` matches mean delta. Smoke-run on existing `results/abl_*.csv` (no crash).
- **Done when:** `pytest tests/test_paired_analysis.py -q` green; rerun on existing CSVs reproduces ladder + adds p & CI columns. → **commit 3**.
- **Est:** 45–60 min. **Blocks:** W2, W3 stats.

**W0.5 — arm-override passthrough + recipe sanity** `category=quick, load_skills=[]`
- **Objective:** enable one-line RL-knob ablation without editing the python block per arm; VERIFY the control reward env.
- **File:** `.sisyphus/evidence/task_abl_arm.sh` — append `${COZ_EXTRA_OVERRIDES:-}` as the final args of the `verl_direct_controller.py` invocation (Hydra: last assignment wins, so it overrides the hardcoded `kl_loss_coef`/`rollout.n`/etc.).
- **VERIFY:** read `.sisyphus/evidence/task-abl-all-train.log` echoed env (line ~6) and confirm the exact COZ_R_*/COZ_ENABLE_RFB values `coz_abl_all` used; lock the "Common reward env" block in §0 to those exact values.
- **Done when:** `bash -n task_abl_arm.sh` parses; a 1-step dry arm with `COZ_EXTRA_OVERRIDES="actor_rollout_ref.rollout.n=8"` shows the override in the resolved config echo; empty override = identical to current behavior; control reward env confirmed. (Folded into commit 4 with run scripts.)
- **Est:** 15–25 min. **Blocks:** A/B/C/D/F training.

### WAVE 1 — Screening (after W0.1 sizing, W0.5; E after W0.2; G after W0.3)

**W1.x — per-arm screening run** `category=deep, load_skills=[]` (one logical task, looped per arm)
- **Objective:** train each arm @30 steps, 1 seed (123), merge, render + judge n≈40 (ids 0801-0840) under live health monitoring.
- **Train (tmux, sequential A→B→E→C→D), verified ALL reward env + per-arm delta:**
  ```
  COZ_ARM=<arm> COZ_TOTAL_STEPS=30 COZ_SAVE_FREQ=30 \
  COZ_R_ANC_WEIGHT=0.2 COZ_R_REP_WEIGHT=1.0 COZ_R_FB_WEIGHT=1.0 COZ_ENABLE_RFB=1 COZ_R_PHR_WEIGHT=0.1 \
  [COZ_EXTRA_OVERRIDES="..."] [COZ_R_ANC_MODE=margin] \
  bash .sisyphus/evidence/task_abl_arm.sh
  ```
  Per-arm delta: A=`algorithm.norm_adv_by_std_in_grpo=False actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm` · B=`actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28` · C=`actor_rollout_ref.actor.kl_loss_coef=0.01` · D=`actor_rollout_ref.rollout.n=8` · E=`COZ_R_ANC_MODE=margin` (no CLI override).
- **Merge → render → judge:** §3 templates. Control = reuse `ckpt/VLM_FT/coz_abl_all`; render+judge fresh at the same 40 ids.
- **Live monitoring (poll every 25–30 min):** entropy/token, KL mean+p95, reward mean/std, **within-group reward std**, **active-group %**, response length, validity-fail rate, clip-frac, grad norm; `nvidia-smi` VRAM/util (prior runs under-utilized).
- **Done when:** each arm has `results/<arm>.csv` (12 cols) at n=40 + a 1-line health verdict (stable / unstable→note+possibly revert). Control re-judged at matched 40 ids.
- **Est:** ~40 min train + render + judge per arm. 5 arms ≈ 3.5–4.5h wall (judge overlaps next train).

### WAVE 2 — Analysis & winner pick `category=ultrabrain, load_skills=[]`
- **Objective:** paired delta vs control per arm; pick winners; run false-gain guard.
- **Do:** `python scripts/abl_paired_analysis.py --base control --arm A_drgrpo=results/A_drgrpo.csv ...` → grounding_deep/all + uniqtok + MUSIQ/NIQE/CLIPIQA, with ttest_rel p & bootstrap CI. Compute **train-reward (from `[coz_reward]` json) ↔ grounding_deep correlation**.
- **Screen win bar (softer):** clear positive paired Δ on grounding_deep + healthy training curves + no MUSIQ/CLIPIQA collapse + reward gain that actually moves grounding.
- **Done when:** ranked arm table written (`results/abl_rl_screen.md`); explicit **winner set** chosen for the combined ALL-RL arm; arms that gained train-reward but not grounding flagged as NON-wins.
- **Est:** 45–60 min.

### WAVE 3 — Confirmation (n=100) `category=deep, load_skills=[]`
- **Objective:** combine winners into ALL-RL; confirm vs fresh control at n=100, 2 seeds, 120 steps.
- **Build ALL-RL:** verified ALL reward env + winners' overrides merged into one `COZ_EXTRA_OVERRIDES` (+ `COZ_R_ANC_MODE=margin` if E won).
- **Train (sequential):** `control@120{seed123,seed456}`, `ALL-RL@120{seed123,seed456}` via `COZ_TOTAL_STEPS=120 COZ_SAVE_FREQ=120 COZ_EXTRA_OVERRIDES="... ++data.seed=<s> ++actor_rollout_ref.rollout.seed=<s>"`. Both control and ALL-RL use the SAME verified ALL reward env (differ only by winning RL knobs). Merge each → render+judge n=100 (ids 0801-0900) — schedule renders on both GPUs while no training runs.
- **Confirm win bar:** paired grounding_deep **Δ ≥ +0.35** vs control; bootstrap 95% CI **lower bound > 0**; **both seeds positive**; validity clean; MUSIQ/CLIPIQA not materially regressed.
- **Done when:** `results/<control120_sX>.csv`, `results/<allrl120_sX>.csv` (n=100); `scripts/abl_paired_analysis.py` produces ttest_rel p + bootstrap CI; verdict PASS/FAIL vs the bar.
- **Est:** 4 trains × ~2.1–3.5h = 8.5–14h sequential + render/judge 1–3h → **the budget driver** (see §4 fallbacks).

### WAVE 4 — Docs / figures / push
- **W4.1 writeup + figures** `category=writing, load_skills=[]`: `results/abl_rl_writeup.md` (method, per-arm screen table, combined confirmation table w/ p+CI, false-gain guard, SKIP justifications, limitations incl. R_fb cons deferral). Figures: screen Δ bar chart, confirmation paired scatter, reward↔grounding scatter (matplotlib → `results/figs/`). **Done when:** md renders, every number traces to a CSV.
- **W4.2 commits + push** `category=quick, load_skills=["git-master"]`: atomic commits (§5); push `grpo/coz-vlm`. **Done when:** `git status` clean of intended files, branch pushed, no ckpt/SR-tree blobs committed.
- **Est:** 60–90 min.

---

## 3. Command templates (canonical)

**Merge:** `python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/<arm>/global_step_<N>/actor --target_dir ckpt/VLM_FT/<arm>`

**Render (full-FT):**
```
python inference_coz.py -i <div2k_valid_input> -o results/<arm>_sr \
  --rec_type recursive_multiscale --prompt_type vlm \
  --vlm_model_path ckpt/VLM_FT/<arm> --vlm_state expanded_text --rec_num 4 --save_prompts \
  --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --ram_ft_path ckpt/DAPE/DAPE.pth --ram_path ckpt/RAM/ram_swin_large_14m.pth \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers
```
(G0 confirms exact `-i` form for ids 0801-0840 / 0801-0900; reuse the input spec the existing abl_*_sr runs used.)

**Judge/eval (eval venv):**
```
source activate.sh
export VLM_JUDGE_BASE_URL=https://intern-vl3-8b.seanmamasde.me/v1 \
       VLM_JUDGE_API_KEY=sk-internvl3-zpC5lCRvqeRnjxFanKbfttAQsLrz1BD0SVLvbb8i7aA \
       VLM_JUDGE_MODEL=InternVL3-8B VLM_JUDGE_TRANSPORT=curl
python evaluate.py --arm <arm> --images results/<arm>_sr/per-sample --vlm-judge --out results/<arm>.csv
# cross-check flaky cases on gemma-4: VLM_JUDGE_BASE_URL=https://gemma4-26b-a4b.seanmamasde.me/v1 KEY=sk-gemma4-...
```

---

## 4. Time budget (~19h, 02:00→21:00) + prioritization

| Block | Work | Wall |
|---|---|---|
| 02:00–03:00 | W0.1 G0 smoke ∥ W0.2/0.3/0.4/0.5 code (parallel) | ~1.0h |
| 03:00–07:30 | W1 screening: A,B,E,C,D (train sequential; judge overlaps) | ~4.5h |
| 07:30–08:30 | W2 analysis + winner pick | ~1.0h |
| 08:30–19:00 | W3 confirmation: 4×120-step trains + n=100 render/judge + stats | ~10.5h |
| 19:00–20:30 | W4 writeup/figures/commits/push | ~1.5h |
| 20:30–21:00 | **safety buffer** | 0.5h |

**MUST-SHIP (guarantee, in priority order):**
1. W0.2 margin-R_anc + W0.4 stats land & green (commits 1,3).
2. Screen **A_drgrpo, B_cliphi, E_anc_margin** at n=40 vs control.
3. **ONE** combined ALL-RL arm confirmed at **n=100** vs fresh control, with paired ttest_rel + bootstrap CI.

**Degrade-gracefully ladder (if behind by W2/W3):**
- Drop C_kl, D_group, F_temp, G_fbnorm from screening first.
- Confirmation: 2 seeds → **1 seed** (report Δ+CI, flag "single-seed"); then 120 → **100 steps**; then n=100 → **n=60–80** (same ids both arms).
- Last resort: confirm only the single strongest screened arm (not a combo).

---

## 5. Atomic commit strategy (force-add small artifacts; never ckpt/SR trees)

`results/`, `checkpoints/`, `.sisyphus/` are gitignored → use `git add -f` for the **small** files only (CSV/MD/PNG); never add `*_sr/` trees, `ckpt/VLM_FT/*`, or `checkpoints/*`.

1. `feat(reward): within-group margin/rank R_anc gated by COZ_R_ANC_MODE` — rewards.py, verl_custom_reward.py, test_rewards.py, test_verl_reward.py. *(gate: pytest green)*
2. `fix(reward): rescale R_fb quality vs consistency gated by COZ_RFB_NORM` — rewards.py, test_rewards.py. *(gate: pytest green, default-off identical)*
3. `feat(analysis): ttest_rel + bootstrap CI and arm-mapping in paired analysis` — scripts/abl_paired_analysis.py, test_paired_analysis.py. *(gate: pytest green)*
4. `chore(ablation): RL-knob override passthrough + screening run scripts/results` — task_abl_arm.sh, run scripts, `-f` screen CSVs + `results/abl_rl_screen.md`. *(gate: W2 done)*
5. `docs(ablation): n=100 confirmation, paired stats, writeup + figures` — `-f` confirm CSVs, `results/abl_rl_writeup.md`, `results/figs/*.png`. *(gate: W3 done)* → push `grpo/coz-vlm`.

Commit after each gate passes; never amend a failed commit.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Render/judge time blows up** (unknown until G0) | G0 gates sizing; cut screen n (40→24) then confirm n (100→60) before cutting steps/seeds. |
| **An RL change destabilizes** (KL/entropy/grad-norm blow-up, validity-fail spike) | Live monitor; on instability **revert that arm**, record "unstable", exclude from combo. Dr.GRPO+seq-mean-token-sum-norm is the top instability suspect → watch grad-norm/clip-frac first. |
| **Judge endpoint flaky** | `VLM_JUDGE_TRANSPORT=curl` + retry; cross-check disputed cases on gemma-4; judge control+arms in one session to control drift. |
| **False gain** (train-reward up, grounding flat) | Mandatory reward↔grounding correlation in W2; such arms are NOT winners. |
| **`COZ_EXTRA_OVERRIDES` key not in config / needs `+`** | G0/W0.5 validate the resolved-config echo before committing arms; add `+` prefix if Hydra rejects. |
| **Reward-env confound** (arms ≠ control recipe) | W0.5 verifies control's exact COZ_R_* env; all arms use those identical values. |
| **GPU under-utilization** (history) | poll `nvidia-smi`; if util low, it's wall-clock cost only — bake into budget, don't chase. |
| **1-GPU concurrency** (failed before) | All trains 2-GPU sequential; only HTTP judge runs concurrently. |
| **Confirmation overruns deadline** | Degrade ladder §4; ALL-RL combo + 1 seed is the floor for a defensible n=100 claim. |
| **Merge step missing global_step dir** | Confirm `checkpoints/<arm>/global_step_<N>/actor` exists post-train before merge (save_freq == total_steps). |

---

## 7. TDD checklist (code tasks)
- Write the failing test first; implement; green; then run the **full** reward suite (`pytest tests/test_rewards.py tests/test_verl_reward.py tests/test_paired_analysis.py -q`) to prove no regression.
- Every new behavior is **gated** (`COZ_R_ANC_MODE`, `COZ_RFB_NORM`) so default = current control byte-identical → ablations stay independent and the control is never silently altered mid-study.

---

## 8. Iterative improvement loop (run rounds until T-2h)

This study is an **OUTER LOOP**, not a single pass. Maintain one **running-best** config (reward env + `COZ_EXTRA_OVERRIDES` + `COZ_R_ANC_MODE`), initialized to the current ALL recipe + stock RL config (= `coz_abl_all`). Track it at the top of `results/abl_rl_rounds.md`.

**Per round:**
1. **SCREEN** a small set (≤5) of INDEPENDENT candidate changes, each = running-best + ONE change, at 30 steps / n≈40 / seed123 (cheap).
2. **ANALYZE**: paired `ttest_rel` + bootstrap CI vs the round's control (= current running-best). False-gain guard (reward↔grounding corr).
3. **FOLD** winners into running-best (each winner must beat the running-best, so gains compound and stay independent). If two winners conflict/overlap, screen their pairwise combo before folding both.
4. **CONFIRM-LITE**: re-render+judge the new running-best at n=100, seed123 only (cheap check the fold held at scale).
5. **RETROSPECTIVE** (append to `results/abl_rl_rounds.md`): what won/lost (+stats), the new bottleneck (from training-health curves + grounding failure cases), and the candidate list for the next round.

**FINAL (at T≈2h):** freeze running-best → full confirm (n=100 × 2 seeds, 120 steps) vs the ORIGINAL control → writeup → figures → push. Never leave training running past the buffer.

**Per-round budget:** keep each round ≤ ~3–4h (screen + lite-confirm) so 3–4 rounds fit in 19h. Round 1 = {A_drgrpo, B_cliphi, E_anc_margin} (must-ship trio) + C_kl/D_group if cheap.

**Candidate backlog (draw from, data-driven by each retrospective):**
- *Exploration/entropy:* `entropy_coeff` 0.005→0.01/0.02; `rollout.temperature` 1.0→1.1/1.2; `clip_ratio_high` 0.28→0.32.
- *Capacity/estimate:* `rollout.n` 8→12; `lr` 3e-7→5e-7; steps 30→60→120 (under-training check).
- *Reward:* margin-R_anc variants (rank vs margin, eps threshold); R_fb-norm (G) + x0-anchored consistency; R_rep ngram/weight; phrase reward.
- *Algorithm:* KL schedule/anneal; `loss_agg_mode` variants; pairwise combine of top-2 winners.
- *State:* revisit A7 expanded-state / composite-image injection if grounding plateaus.

**Stop conditions:** abort a candidate on instability (KL/grad-norm/entropy blow-up, validity spike) → record + exclude. Abort the LOOP hard at **T-2h** and ship the running-best.
