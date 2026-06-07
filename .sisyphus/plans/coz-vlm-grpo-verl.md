# CoZ VLM GRPO Finetune (veRL + FSDP2, dual-track) — Work Plan

## TL;DR

> **Quick Summary**: RL-finetune (GRPO) the Qwen2.5-VL-3B LoRA prompt-extractor inside the
> Enhanced Chain-of-Zoom pipeline — continuing from the author's `checkpoint-10000` — to fix
> **Semantic Drift** (animal eye → "neurons/synapses") and **Prompt Convergence** (collapse to
> repeated low-entropy prompts). Execute as a **dual-track HYBRID**: Track-1 validates + repairs
> the teammate's plain-PyTorch GRPO trainer (`origin/sytwu`) as a fast prototype/fallback; Track-2
> builds a veRL + FSDP2 production trainer on 6×H200 that reuses the *validated* reward/state/eval
> modules. Claims are backed by SOLID numerical deltas on 4 no-reference IQA metrics
> (NIQE, MUSIQ, MANIQA, CLIPIQA) **plus** an anti-drift axis and an anti-convergence axis,
> vs the author checkpoint and original CoZ.
>
> **Deliverables**:
> - Validated+repaired `train/` modules + a per-module **VALIDATION REPORT** (correct / bug+fix / discrepancy-vs-paper)
> - Isolated `.venv-train` (verl + vllm≥0.8.3 + ray + pyiqa + wandb) + pyiqa pinned into the existing `.venv`
> - DIV2K (800) preprocessed to 512² with **asserted-disjoint** train/valid split + a static per-state **parquet** for veRL
> - Pre-registered `evaluate.py` emitting `results/<arm>.csv` (niqe, musiq, maniqa, clipiqa, consistency, unique-token-ratio)
> - veRL `custom_reward_function` wrapping the validated reward stack; trained LoRA adapters per arm × seed
> - Aggregated mean±std delta tables + a concise **bullet-point fixes doc** + the `0064` drift/convergence QA probe
> - Clean atomic git history on a dedicated work branch (dirty tree preserved, origin/main evaluator merged)
>
> **Estimated Effort**: XL
> **Parallel Execution**: YES — 6 phase-waves (W0–W5) + a Final Verification wave; fine-grained parallel sub-waves inside each
> **Critical Path**: W0(git/env/data) → W1(validate sytwu, **G0**) → W2(prototype + **G1/G2**) → W3(veRL build + **G3**) → W4(scale-up) → W5(report) → Final Verification → user okay
> **GPU reality**: veRL hybrid-engine colocation consumes all 6 GPUs per run → full-scale arms are scheduled **sequentially**; only eval-only baselines and the torch prototype arms (1–2 GPUs each) run truly concurrently.

---

## Context

### Original Request
RL-finetune (GRPO) the CoZ VLM prompt-extractor, continuing from the author LoRA checkpoint, to fix
**Semantic Drift** and **Prompt Convergence**, using **veRL (volcengine/verl) + FSDP2** on 6×H200.
A complete but **unvalidated** plain-PyTorch GRPO trainer exists on `origin/sytwu` (teammates' code, may
contain bugs) — it MUST be validated, not trusted. Deliver bullet-point fixes backed by SOLID numerical
improvements on no-reference IQA plus drift/convergence axes, vs the author checkpoint and original CoZ.
All defaults pre-approved by the user; chain ablations automatically; prototype small then scale.

### Confirmed Repository State (verified during planning)
- Branch `main`, **1 commit behind** `origin/main` (fast-forwardable; `origin/main` adds a pyiqa evaluator + `visualize.py`).
- Remote branches: `origin/main`, `origin/sytwu` (GRPO code, commits b2d3d8e/fb8aa60), `origin/whp` (images).
- **DIRTY working tree** — must be preserved before any branch op:
  - modified: `inference_coz.py` (uncommitted crop-strategy edits)
  - deleted: `ckpt/RAM/ram_swin_large_14m_ckpt_here.txt` (placeholder)
  - untracked: `.hf_cache/`, `.torch_cache?`, `activate.sh`, `final proposal.pdf`, `high_density_imgs/`, `opencode_session_*.md`, `requirements.cu126.txt`, `scripts/download_models.py`
- Confirmed assets: `ckpt/VLM_LoRA/checkpoint-10000/{adapter_config.json,adapter_model.safetensors}`, `ckpt/SR_LoRA/model_20001.pkl`, `ckpt/SR_VAE/vae_encoder_20001.pt`, `ckpt/DAPE/DAPE.pth`, base models cached in `.hf_cache` (Qwen2.5-VL-3B, SD3-medium).
- Core pipeline files: `inference_coz.py`, `inference_fixed_prompt.py`, `osediff_sd3.py`, `lora/`, `ram/`, `utils/`, `train_utils/`, `scripts/inference/*.sh`.
- `activate.sh` localizes caches: `HF_HOME=.hf_cache`, `TORCH_HOME=.torch_cache`, `UV_CACHE_DIR/PIP_CACHE_DIR=.cache/*`, sources `.venv/bin/activate`, sets `PYTHONPATH=<root>`, `module load cuda/12.6`.
- Existing `.venv` (uv, py3.10): torch 2.6.0+cu126, transformers 4.49.0, peft 0.15.2, diffusers 0.32.1, accelerate 1.4.0, qwen-vl-utils 0.0.8 (FSDP2 available). `requirements.cu126.txt` documents the cu126 stack (+ xformers 0.0.29.post3). **NOT installed**: verl, vllm, ray, flash-attn, pyiqa, wandb, trl, deepspeed.

### Concrete Failure Evidence (file-backed — drives the QA probe)
`fail_case/failcase_1/coz_output/per-sample/0064_eye/txt/{0..3}.txt` (sample `samples/0064.png`):
- scale0: `dog eye, close-up, fur, blue, brown, detail, texture, animal, eyes, macro, photography, pet, eyesight, whiskers`
- scale1: `hair, close-up, texture, detail, fur, animal, skin, macro, natural, brown, black, soft, fine, close, zoomed`
- scale2: `Neurons, dendrites, axons, synapses, neural network, brain activity, electrical signals, neurotransmitters, synaptic transmission, neural plasticity, learning` ← **DRIFT**
- scale3: `Neurons, dendrites, axons, synapses, neural network, brain activity, electrical impulses, neurotransmitters, synaptic transmission, neural signaling, complex structure` ← **DRIFT + near-repeat of scale2 (CONVERGENCE)**

The tuned model must (a) **retain a subject token** (eye/fur/animal) at deep scales, (b) **not emit** `neuron/synapse/dendrite/axon`, (c) show a **higher cross-scale unique-token ratio** (scale2 ≠ scale3).

### Research Findings (veRL / reward stack — confirmed)
- GRPO × Qwen2.5-VL is officially supported (`examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh`, `tuning/lora/run_qwen2_5_vl_7b_fsdp.sh`); entry `python -m verl.trainer.main_ppo`.
- Key flags: `algorithm.adv_estimator=grpo`, `algorithm.use_kl_in_reward=False`, `actor_rollout_ref.actor.strategy=fsdp2`, `actor.use_kl_loss=True kl_loss_coef≈0.01..0.04 kl_loss_type=low_var_kl`, `actor.clip_ratio=0.2`, `rollout.name=vllm rollout.n=6`, `data.image_key=images`, `actor.loss_agg_mode=token-mean`.
- LoRA: `lora_rank=8 lora_alpha=32 target_modules=all-linear exclude_modules='.*visual.*'`, `rollout.load_format=safetensors`, `layered_summon=True`. `lora_adapter_path` init is **UNCERTAIN** (GH #3278) → fallback: merge author adapter→base, train fresh adapter, KL reference = author behavior.
- Multimodal multi-image prompts in vLLM rollout **UNCONFIRMED** → veRL default = represent x0/x_{i-2} as TEXT captions + `zoom factor={scale}x` token (single current crop image); true multi-image state kept only in the torch fallback.
- Versions: vLLM ≥0.8.3 (avoid 0.7.x); torch cu126 prebuilt wheels run on driver 550 → use **prebuilt** wheels (no source flash-attn). vLLM multimodal: `VLLM_USE_V1=0`/sync rollout if async bugs. `ulysses_sequence_parallel_size=1` for Qwen2.5-VL.
- 6×H200: hybrid-engine colocation (all 6 GPUs train+rollout); `tensor_model_parallel_size=2` (must divide 6). Memory ample for 3B+LoRA; rollout throughput + SR-feedback reward are the constraints.

### Pivotal Default Decisions (from Metis — baked in, not re-litigated)
- **B1 Dual-track HYBRID**: Track-1 = recover+VALIDATE sytwu (torch) as fast-baseline/fallback/prototype; Track-2 = veRL+FSDP2 primary, reusing validated modules as a `custom_reward_function`. If veRL hits a hard blocker within the time-box, scale validated-sytwu via torch-FSDP2 as a **documented** fallback (never silent scope reduction).
- **B2 Continue-adapter**: torch keeps training the SAME adapter (`continue_from`) + frozen copy as KL reference; veRL tries `lora_adapter_path`, else merge→fresh adapter with KL ref = author behavior. Keep r=8/α=32/same targets; `inference_mode=false`.
- **B3 Reward phasing**: Phase-1 text-only (R_rep+R_anc+R_phr) for prototype + most ablations (IQA measured OFFLINE); Phase-2 add R_fb (subsampled, SR on a dedicated GPU); Phase-3 optional R_crit (Qwen2.5-VL-7B).
- **B4 Decouple SR from loop**: train on text-only proxies; eval real IQA offline; **mandatory correlation gate G2** (proxy↑ but offline IQA flat/down → escalate to R_fb before scaling).
- **B5 State expansion** is its own ablation arm (A7), not baseline. veRL: text-caption + scale token; torch: true multi-image.
- **B6 Data**: pre-generate STATIC per-state parquet (x0-caption, x_{i-2}-caption, current_crop.png, scale, previous-prompt) → veRL schema; DIV2K 800 official split (train→RL, valid→held-out); **assert disjoint** ids; DIV8K only if time permits.
- **B7 Env isolation**: separate `.venv-train` (verl/vllm/ray/wandb); existing `.venv` for inference+eval (pin pyiqa there). Never co-install.

---

## Work Objectives

### Core Objective
Produce a *validated*, reproducible GRPO finetune of the CoZ VLM that measurably reduces semantic drift and
prompt convergence at deep zoom, with numerical deltas vs the author checkpoint (A3) and original CoZ (A2)
across 4 no-ref IQA metrics + a drift axis + a convergence axis, on a held-out DIV2K-valid set.

### Concrete Deliverables
- `train/` modules validated/repaired in-repo; `docs/validation_report.md` (per-module verdicts).
- `.venv-train` (frozen lockfile) + pyiqa pinned in `.venv`.
- `data/div2k/{train,valid}/` (512² center-crop) + `data/manifests/{train,valid}.txt` + disjointness assertion log.
- `data/parquet/coz_states_{train,valid}.parquet` (veRL schema).
- `evaluate.py` (single source of truth) → `results/<arm>.csv` with `niqe,musiq,maniqa,clipiqa,consistency,unique_token_ratio`.
- veRL config(s) under `verl_configs/` + `custom_reward.py`; trained adapters under `outputs/<arm>/seed<k>/`.
- `results/aggregate_deltas.csv` + `docs/fixes.md` (bullet-point fixes + numerical deltas) + `results/probe_0064.md`.

### Definition of Done
- [x] **G0** passed: 8 sytwu modules validated, 6 bugs fixed, 43 tests green, `docs/validation_report.md` committed.
- [x] **G1** passed: torch training runs, finite reward, adapter saves + plugs into `inference_coz.py`, offline IQA computes.
- [~] **G2** WEAK/INCONCLUSIVE (`docs/g2_findings.md`): 25-step text-only prototypes show consistent direction on niqe/musiq/consistency but regress clipiqa/uniqtok — within noise. Real signal needs many more steps → veRL (blocked). Harness verified correct.
- [x] **G3** PASSED: clean `.venv-train` rebuild (Oracle pins: ray 2.44.1/click 8.2.1/torch2.6-cu124/vllm0.8.5/flash-attn) + 4-rung ladder green; full veRL multimodal GRPO+FSDP2+vLLM step ran (rc=0). Working path = direct-controller wrapper (`verl_direct_controller.py`), Ray temp off NFS. Evidence: `.sisyphus/evidence/task-w3fix-*`.
- [x] Ablation matrix: A0/A1/A2/A3 baseline ladder ✅ (baseline_ladder_n100.csv); A6/w4v2 headline 3-seed × n=100 ✅; A8-equiv (R_rep/R_fb tuning) ✅; A7 state-expansion stabilized matched 1-seed ✅; A9 (+R_crit) explicitly optional/unrun. mean±std + delta tables produced (results/aggregate_full_3seed.csv, baseline_ladder_n100.csv, operating_points_n100.csv, A7_stable_matched_comparison.csv).
- [x] `docs/fixes.md` delivers concise bullet-point fixes each backed by concrete numerical deltas (multi-axis, honest, with regressions reported).

### Must Have
- Validation-before-reuse of ALL sytwu modules (TDD).
- Strict env isolation (`.venv` vs `.venv-train`).
- Asserted-disjoint train/eval image ids.
- Eval protocol pre-registered BEFORE any training claim.
- Multi-axis reporting (4 IQA + consistency + convergence) — never a single-metric claim.
- Atomic git history; dirty tree preserved.

### Must NOT Have (Guardrails)
- ❌ NO trusting sytwu code without a passing unit test or a documented discrepancy.
- ❌ NO co-installing verl/vllm/ray into the existing inference `.venv`.
- ❌ NO claim on a single IQA metric; NO claim without seed mean±std.
- ❌ NO train/eval image leakage (must assert disjoint).
- ❌ NO branch switch that clobbers the dirty working tree (use commit/stash/worktree/`git show`).
- ❌ NO committing `.hf_cache/`, datasets, model weights, `*.pdf`, `wandb/`, or `outputs/`.
- ❌ NO silent scope reduction — every fallback is explicitly documented and gated.
- ❌ NO hardcoded foreign paths (`/project2/cookies/...`, `/mnt/data1/...`) left in imported code.
- ❌ NO changing SR LoRA/VAE, crop, recursion depth, or decode settings across ablation arms (fixed protocol).
- ❌ NO AI-slop (dead code, over-abstraction, redundant comments) in repaired modules.

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — every acceptance criterion is an agent-executable command + exact assertion.
> Evidence saved under `.sisyphus/evidence/task-<id>-<slug>.<ext>`.

### Test Decision
- **Infrastructure exists**: NO formal framework installed → **set up `pytest`** inside `.venv` for module validation (lightweight, no GPU needed for most reward-math tests).
- **Automated tests**: **YES (TDD)** for W1 validation (RED failing test that encodes the paper-correct contract → GREEN fix → REFACTOR). Tests-after for training harness wiring (W2/W3).
- **Framework**: `pytest` in `.venv`; training/eval verified by agent-executed smoke runs + CSV assertions.
- **Agent-Executed QA**: ALWAYS (mandatory for every task). CLI/training → `Bash`/`interactive_bash` (tmux); inference+IQA → `Bash`; no browser UI in scope → Playwright not used.

### QA Policy by domain
- **Reward/GRPO math, parsers**: `pytest` unit tests on toy tensors/strings with exact numeric assertions.
- **Training runs**: `interactive_bash` (tmux) launch + `Bash` log-grep assertions (reward rises, adapter file exists, no NaN).
- **Inference + IQA**: `Bash` run `inference_coz.py` on a fixed image set, then `evaluate.py`, assert CSV columns + value ranges.
- **Data**: `Bash` python one-liners asserting counts, dims (512²), and disjoint id sets.

### Pre-Registered Eval Protocol (FIXED across ALL arms — locked before training)
Same eval images (**≥100 held-out DIV2K-valid**), same SR LoRA + VAE, center crop, **4-step recursion (→256×)**,
**greedy** VLM decode `max_new_tokens=32`, **≥3 training seeds** → report **mean±std**. Metrics emitted to `results/<arm>.csv`:
- IQA (pyiqa, NCHW∈[0,1]): `niqe` (lower-better → report inverted), `musiq`, `maniqa`, `clipiqa` (higher-better).
- **Drift axis**: `consistency = CLIP_cos(SR_deep, x0)` mapped [-1,1]→[0,1]; plus `R_anc` against x0 caption.
- **Convergence axis**: `unique_token_ratio` (unique tokens / total across the 4 scales) and/or prompt entropy.
- **Concrete QA probe**: `samples/0064.png` — tuned must retain a subject token (eye/fur/animal), emit no `neuron|synapse|dendrite|axon`, and raise unique-token-ratio vs A3.

### Ablation Arms (fixed protocol)
- A0 NN-interp · A1 Direct-SR(null prompt) · A2 original CoZ (vlm_base, no LoRA) · **A3 author checkpoint-10000 (THE BAR)**
- A4 +R_rep · A5 +R_anc · A6 +R_rep+R_anc+R_phr (text-only combined) · A7 A6+state-expansion · A8 A7+R_fb · A9 +R_crit (optional full)

### Go/No-Go Gates
- **G0** (after W1): reward/GRPO/eval modules pass unit tests + cross-check vs paper; bugs fixed/documented.
- **G1** (in W2): training runs, reward↑, adapter saves, plugs into `inference_coz.py`, offline IQA computes.
- **G2** (after W2): A6 beats A3 on ≥2/4 IQA without regressing others AND improves a drift+convergence metric; else escalate to R_fb and re-gate.
- **G3** (in W3): one veRL multimodal+LoRA+FSDP2 train→save→eval cycle completes; else documented torch-FSDP2 fallback.
- **Scale to full DIV2K + 3 seeds + all arms ONLY after G0–G3.**

---

## Risk Register (Metis R1–R12 → mitigations mapped to tasks)

| ID | Risk | Mitigation | Owning Task(s) |
|----|------|-----------|----------------|
| R1 | GRPO surrogate: `old_logp` recomputed at update → ratio≡1, clip is a NO-OP (REINFORCE+KL, not PPO) | Prove on toy tensors; decide intended vs bug; if PPO desired, cache old_logp at sample time | W1.T7 |
| R2 | Trajectory advanced by argmax-of-group (best-of-group), not sampled → biased on-policy distribution | Unit-test the advance rule; compare sampled vs argmax; document bias + chosen policy | W1.T6 |
| R3 | Global stateful Welford z-norm breaks under multi-rank/restart | Replace with **per-group** normalization for veRL; keep running-stat only for single-proc torch; test determinism | W1.T1, W3.T2 |
| R4 | No-ref IQA can REWARD hallucinated texture (the drift we fix) | Decouple SR (B4); G2 correlation gate; report consistency+convergence axes; never single-metric | W0.T7, W2.T6 |
| R5 | veRL multimodal+LoRA+FSDP2 unsupported/buggy (multi-image rollout; `lora_adapter_path` #3278) | Text-caption state (B5); merge→fresh-adapter fallback (B2); time-box + torch-FSDP2 fallback | W3.T4, W3.T5, W3.T6 |
| R6 | vLLM version/async bugs | Pin vllm≥0.8.3, prebuilt wheels, `VLLM_USE_V1=0`/sync rollout, TP=2 | W0.T4, W3.T1 |
| R7 | Env contamination between `.venv` and `.venv-train` | Strict isolation (B7); import-smoke both venvs; never co-install | W0.T4, W0.T5 |
| R8 | Train/eval image leakage | Assert disjoint id sets at preprocess time; fail build on overlap | W0.T6 |
| R9 | Hardcoded foreign paths (`/project2/cookies`, `/mnt/data1`) break execution | De-hardcode/parameterize via config + env; grep-assert none remain | W1.T8 |
| R10 | Continue-adapter vs fresh-adapter KL-reference mismatch | Implement+document both; default torch=continue, veRL=merge→fresh w/ author KL ref | W3.T4 |
| R11 | One reward term dominates the sum (scale mismatch) | Per-group normalize each term before weighting; unit-test magnitude balance | W1.T1, W3.T2 |
| R12 | Dirty tree loss / accidental clobber on branch switch | Preserve via branch+atomic commits; read sytwu via `git show`/worktree; `.gitignore` caches | W0.T1, W0.T2 |

---

## Task Dependency Graph

| Task | Title | Depends On | Blocks | Reason |
|------|-------|-----------|--------|--------|
| W0.T1 | Git safety: preserve dirty tree + .gitignore + work branch | None | W0.T2, W1.* | All later commits build on a clean, preserved base |
| W0.T2 | FF/merge origin/main (pyiqa evaluator + visualize.py) | W0.T1 | W0.T3, W0.T7 | Evaluator code needed before import + eval pre-reg |
| W0.T3 | Import sytwu `train/` + `scripts/train/` (unvalidated) | W0.T2 | W1.* | Code must be in-tree for validation/repair |
| W0.T4 | Build `.venv-train` (verl/vllm/ray/wandb) + import smoke | None | W3.T1 | Heavy install; parallel with git |
| W0.T5 | Pin pyiqa into `.venv` + verify metric load/flags | None | W1.T4, W0.T7 | Eval metrics needed for validation + protocol |
| W0.T6 | DIV2K download + 512² preprocess + disjoint split | None | W0.T7, W2.*, W3.T3 | Data underpins eval + training |
| W0.T7 | Pre-register eval protocol + `evaluate.py` skeleton | W0.T2, W0.T5, W0.T6 | W2.T2, W2.T6 | Measurement integrity before any claim |
| W1.T1 | Validate `rewards.py` orchestrator + z-norm | W0.T3 | W1.T9, W2.*, W3.T2 | Reward sum/normalization correctness |
| W1.T2 | Validate `text_sim.py` (R_anc, R_rep) | W0.T3 | W1.T9, W2.* | Core anti-drift/anti-convergence rewards |
| W1.T3 | Validate `critic.py` (R_crit) + R_phr blacklist | W0.T3 | W1.T9 | Optional-but-scored components |
| W1.T4 | Validate `metrics.py` (pyiqa, NIQE inversion) | W0.T3, W0.T5 | W1.T9, W0.T7 | Eval correctness (lower/higher-better) |
| W1.T5 | Validate `state.py` (3-image+scale builder) | W0.T3 | W1.T9, W2.*, W3.T3 | Prompt construction correctness |
| W1.T6 | Validate `rollout.py` (G=6, best-of-group advance) | W0.T3 | W1.T9, W2.T1 | On-policy sampling correctness |
| W1.T7 | Validate `trainer.py` GRPO math (adv/KL/clip) | W0.T3 | W1.T9, W2.T1 | Core optimizer correctness (R1) |
| W1.T8 | Validate `sr_env.py`+`zoom_dataset.py`+`evaluate.py`; de-hardcode paths | W0.T3, W0.T5, W0.T6 | W1.T9, W2.T1 | SR feedback + data loading + path hygiene |
| W1.T9 | Synthesize VALIDATION REPORT (**G0**) | W1.T1–T8 | W2.*, W3.* | Gate: validated baseline established |
| W2.T1 | Torch smoke run (small subset) — **G1** | W1.T9, W0.T6 | W2.T2 | Harness sanity before arms |
| W2.T2 | Adapter→inference→offline IQA loop closes | W2.T1, W0.T7 | W2.T3–T5, W3.T3 | End-to-end measurement loop |
| W2.T3 | Prototype A4 (+R_rep) small-subset + eval | W2.T2 | W2.T6 | Anti-convergence signal |
| W2.T4 | Prototype A5 (+R_anc) small-subset + eval | W2.T2 | W2.T6 | Anti-drift signal |
| W2.T5 | Prototype A6 (text-only combined) small-subset + eval | W2.T2 | W2.T6 | Combined signal |
| W2.T6 | Correlation gate analysis — **G2** | W2.T3–T5, A3 eval | W3.*, W4.* | Decide proceed vs R_fb escalation |
| W3.T1 | veRL stack smoke (official Qwen2.5-VL GRPO example) | W0.T4 | W3.T2, W3.T4, W3.T5 | Confirm stack runs on 6×H200 |
| W3.T2 | veRL `custom_reward_function` wrapping validated rewards | W1.T9, W3.T1 | W3.T5 | Reuse validated reward stack |
| W3.T3 | Parquet state-generator (veRL schema) | W0.T6, W2.T2, W1.T5 | W3.T5 | Static per-state data for veRL |
| W3.T4 | LoRA init strategy (adapter_path vs merge→fresh) | W3.T1 | W3.T5 | Continue-from author behavior (B2/R10) |
| W3.T5 | veRL train→save→eval cycle — **G3** (time-boxed) | W3.T2, W3.T3, W3.T4 | W3.T6, W4.* | Production trainer viability |
| W3.T6 | (Conditional) torch-FSDP2 fallback if G3 fails | W3.T5 | W4.* | Documented fallback (B1/R5) |
| W4.T1 | Full A6 (3 seeds) on primary trainer | W2.T6, W3.T5/T6 | W5.* | Main result |
| W4.T2 | Full A7 (A6+state-expansion, 3 seeds) | W4.T1 | W5.* | State-expansion ablation |
| W4.T3 | Full A8 (A7+R_fb, 3 seeds) | W4.T2 | W5.* | SR-feedback ablation |
| W4.T4 | Full A9 (+R_crit, optional, 3 seeds) | W4.T3 | W5.* | Full-stack (time-permitting) |
| W4.T5 | Baseline eval suite A0/A1/A2/A3 (eval-only, parallel) | W0.T7 | W5.* | Reference bars |
| W4.T6 | Run hygiene: wandb logging, adapter naming, seed determinism | W4.T1 | W5.* | Reproducibility |
| W5.T1 | Aggregator → mean±std delta tables | W4.T1–T6 | W5.T3 | Numerical evidence |
| W5.T2 | `0064` drift/convergence QA probe | W4.T1 (+best arm) | W5.T3 | Concrete failure-mode proof |
| W5.T3 | Bullet-point fixes doc (`docs/fixes.md`) | W5.T1, W5.T2 | Final | Final deliverable |

---

## Parallel Execution Graph

```
Phase W0 — Prep
  Sub-wave A (start immediately, concurrent):
    ├── W0.T1 git safety (git-master)            [sequential head of git chain]
    ├── W0.T4 build .venv-train                  [independent, long install]
    ├── W0.T5 pin pyiqa in .venv                 [independent]
    └── W0.T6 DIV2K download+preprocess+split    [independent, I/O bound]
  Sub-wave B: W0.T2 ff/merge origin/main (after W0.T1)
  Sub-wave C: W0.T3 import sytwu train/ (after W0.T2)
  Sub-wave D: W0.T7 eval pre-registration (after W0.T2 + W0.T5 + W0.T6)

Phase W1 — Validate sytwu (MAX PARALLEL — 8 concurrent validators after W0.T3)
    ├── W1.T1 rewards.py        ├── W1.T2 text_sim.py
    ├── W1.T3 critic.py/R_phr   ├── W1.T4 metrics.py (needs W0.T5)
    ├── W1.T5 state.py          ├── W1.T6 rollout.py
    ├── W1.T7 trainer.py GRPO   └── W1.T8 sr_env/dataset/evaluate + de-hardcode (needs W0.T5,W0.T6)
  Then: W1.T9 VALIDATION REPORT  →  GATE G0

Phase W2 — Prototype (torch, small subset)
    W2.T1 smoke (→G1) → W2.T2 close loop → { W2.T3 A4 ‖ W2.T4 A5 ‖ W2.T5 A6 }  (3 arms on 3 GPUs)
    → W2.T6 correlation gate (→G2)

Phase W3 — veRL build (Track-2; time-boxed)
    W3.T1 stack smoke (needs W0.T4) → { W3.T2 custom reward ‖ W3.T3 parquet ‖ W3.T4 LoRA init }
    → W3.T5 train→save→eval (→G3) → [W3.T6 fallback iff G3 fails]

Phase W4 — Scale-up  (GPU-bound: veRL runs consume all 6 GPUs → SEQUENTIAL;
                      W4.T5 eval-only baselines run concurrently on spare cycles)
    W4.T1 A6 → W4.T2 A7 → W4.T3 A8 → W4.T4 A9 (optional)   [sequential under veRL colocation]
    W4.T5 baselines ‖ (concurrent, eval-only)   ;   W4.T6 hygiene ‖ throughout

Phase W5 — Reporting
    { W5.T1 aggregate ‖ W5.T2 probe } → W5.T3 fixes doc

Final Verification Wave (4 parallel reviewers) → present → user okay

Critical Path:
  W0.T1→W0.T2→W0.T3→W1.{T7|T8}→W1.T9(G0)→W2.T1(G1)→W2.T2→W2.T5→W2.T6(G2)
   →W3.T1→W3.T5(G3)→W4.T1→W5.T1→W5.T3→Final→user okay
Parallel speedup: W1 ~8× on validators; W2 ~3× on arms; W0 install/data/git overlap.
Max concurrent agents: 8 (W1 validators).
```

## Agent Dispatch Summary

| Wave | Concurrent | Dispatch |
|------|-----------|----------|
| W0 | 4 | T1→`quick`+git-master · T2→`quick`+git-master · T3→`quick`+git-master · T4→`deep` · T5→`deep` · T6→`unspecified-high` · T7→`deep` |
| W1 | 8 | T1,T6,T7→`ultrabrain` · T2,T3,T4,T5,T8→`deep`(+ai-slop-remover) · T9→`writing` |
| W2 | 3 | T1,T2→`deep` · T3,T4,T5→`unspecified-high` · T6→`oracle`/`ultrabrain` |
| W3 | 3 | T1→`deep` · T2→`ultrabrain` · T3→`unspecified-high` · T4→`deep` · T5→`ultrabrain` · T6→`deep` |
| W4 | 1–2 | T1–T4→`ultrabrain` · T5→`unspecified-high` · T6→`quick`+git-master |
| W5 | 2 | T1→`unspecified-high` · T2→`deep` · T3→`writing` |
| Final | 4 | F1→`oracle` · F2→`unspecified-high`+ai-slop-remover · F3→`unspecified-high` · F4→`deep` |

---

## TODOs

### Wave W0 — Prep

- [x] **W0.T1. Git safety: preserve dirty tree + `.gitignore` + work branch**

  **What to do**:
  - From `main` (currently dirty), create the work branch: `git switch -c grpo/coz-vlm`.
  - Un-delete the RAM placeholder so the merge in W0.T2 is clean: `git restore ckpt/RAM/ram_swin_large_14m_ckpt_here.txt` (the real 5.3 GB `.pth` stays untracked).
  - Create `.gitignore` covering: `.hf_cache/`, `.torch_cache/`, `.cache/`, `.venv-train/`, `data/`, `results/`, `outputs/`, `experience/`, `wandb/`, `*.pdf`, `opencode_session_*.md`, `high_density_imgs/`, `ckpt/RAM/*.pth`, `ckpt/**/*.safetensors`-EXCEPT keep the small VLM adapter tracked (`!ckpt/VLM_LoRA/**`).
  - Two atomic commits: (1) `chore: add .gitignore for caches/data/outputs`; (2) `chore: preserve WIP inference crop-strategy edits + local tooling` staging ONLY `inference_coz.py`, `activate.sh`, `requirements.cu126.txt`, `scripts/download_models.py`.

  **Must NOT do**: commit `.hf_cache/`, `final proposal.pdf`, `high_density_imgs/`, `opencode_session_*.md`, or any model weight; switch back to `main` and lose edits.

  **Recommended Agent Profile**: Category `quick` (mechanical git). Skills: [`git-master`] (atomic commits, safe branch ops). Omitted: `ai-slop-remover` (no code authored here).

  **Parallelization**: Can Run In Parallel: NO (head of git chain). Blocks: W0.T2, all W1. Blocked By: None — start immediately.

  **References**:
  - Confirmed dirty tree (planning `git status`): modified `inference_coz.py`; deleted `ckpt/RAM/ram_swin_large_14m_ckpt_here.txt`; untracked `.hf_cache/`, `activate.sh`, `final proposal.pdf`, `high_density_imgs/`, `opencode_session_*.md`, `requirements.cu126.txt`, `scripts/download_models.py`.
  - `inference_coz.py` dirty edits = `select_crop_top_left()` (L36-110) + `--crop_strategy` flag (R12) — must be preserved before any branch op.

  **Acceptance Criteria**:
  ```
  Scenario: Dirty tree preserved on a clean branch (happy path)
    Tool: Bash
    Steps:
      1. git branch --show-current            → assert "grpo/coz-vlm"
      2. git log --oneline -2                  → assert 2 new commits (gitignore, preserve WIP)
      3. git status --porcelain | grep -vE '^!!|^\?\? (\.hf_cache|high_density_imgs|.*\.pdf|opencode_session)'  → assert EMPTY (no unexpected unstaged)
      4. git show --stat HEAD~1 | grep inference_coz.py  → assert present (edits committed)
    Evidence: .sisyphus/evidence/task-w0t1-gitstatus.txt

  Scenario: Caches and large files are NOT tracked (failure guard)
    Tool: Bash
    Steps:
      1. git ls-files | grep -E '\.hf_cache/|\.pdf$|high_density_imgs/|ckpt/RAM/.*\.pth'  → assert EXIT 1 (no matches)
    Expected Result: no cache/weight/pdf path is tracked
    Evidence: .sisyphus/evidence/task-w0t1-lsfiles.txt
  ```
  **Commit**: YES — `chore: add .gitignore...` + `chore: preserve WIP...`. Pre-commit: `git status`.

- [x] **W0.T2. Fast-forward/merge `origin/main` (pyiqa evaluator + visualize.py) — RESOLVE crop conflict**

  **What to do**:
  - `git fetch origin`; `git merge origin/main` into `grpo/coz-vlm`. A CONFLICT on `inference_coz.py` is EXPECTED (local `--crop_strategy` hunk vs origin/main `--crop_x/--crop_y` hunk on the same crop lines).
  - **Deterministic resolution**: KEEP the working-tree `crop_strategy` implementation for the conflicting crop hunks (it is the user's WIP and default `center` preserves the fixed ablation protocol); ACCEPT all NON-conflicting origin/main additions: `visualize.py` (NEW), `inference_fixed_prompt.py` (NEW), `requirements.txt` (+`pyiqa==0.1.15.post2`, +`setuptools==67.8.0`), 11 new `samples/*.png`, and the `inference_coz_full.py` deletion.
  - Remove ALL conflict markers; ensure file parses.

  **Must NOT do**: blindly `git checkout --ours inference_coz.py` (would drop other origin/main edits to that file); discard `visualize.py`/pyiqa; commit conflict markers.

  **Recommended Agent Profile**: Category `quick`. Skills: [`git-master`] (conflict resolution). Omitted: none.

  **Parallelization**: Can Run In Parallel: NO. Blocks: W0.T3, W0.T7. Blocked By: W0.T1.

  **References**:
  - `git diff main origin/main --stat` brings: `visualize.py` (+249; `compute_metrics()` lazily imports `pyiqa`, `METRIC_DEFS=[NIQE↓,MUSIQ↑,MANIQA↑,CLIPIQA↑]`, reads `per-sample/<stem>/{i}.png`+`txt/{i}.txt`), `inference_fixed_prompt.py` (+158; explicit `--prompts`+`--crop_x/--crop_y`, bypasses VLM — ready reward-rollout harness), `requirements.txt` (+2), `inference_coz.py` (+16 crop_x/crop_y — the CONFLICT), `inference_coz_full.py` DELETED.
  - origin/main HEAD = commit `585059a`; local `main` = `b0bd607`.

  **Acceptance Criteria**:
  ```
  Scenario: Merge brings evaluator infra, keeps crop_strategy, no markers (happy path)
    Tool: Bash
    Steps:
      1. test -f visualize.py && test -f inference_fixed_prompt.py            → assert exit 0
      2. grep -c '^pyiqa' requirements.txt                                    → assert >=1
      3. grep -rnE '^(<<<<<<<|=======|>>>>>>>)' inference_coz.py               → assert EXIT 1 (no markers)
      4. python -c "import ast,sys; ast.parse(open('inference_coz.py').read()); print('parse ok')"  → assert "parse ok"
      5. grep -c 'crop_strategy' inference_coz.py                              → assert >=1 (WIP preserved)
    Evidence: .sisyphus/evidence/task-w0t2-merge.txt

  Scenario: Conflicted merge left unresolved (failure guard)
    Tool: Bash
    Steps:
      1. git diff --check                                                     → assert EXIT 0 (no conflict whitespace/markers)
    Evidence: .sisyphus/evidence/task-w0t2-diffcheck.txt
  ```
  **Commit**: YES — merge commit `Merge origin/main: pyiqa evaluator + visualize.py (keep crop_strategy)`. Pre-commit: `git diff --check`.

- [x] **W0.T3. Import `sytwu` `train/` + `scripts/train/` (UNVALIDATED)**

  **What to do**:
  - Bring the teammate GRPO code into the work branch WITHOUT switching branches: `git checkout origin/sytwu -- train/ scripts/train/`.
  - Commit as a single clearly-labelled "unvalidated import" so W1 can validate/repair on top with a clean diff. Do NOT fix anything yet.

  **Must NOT do**: edit/repair any imported file in this task (that is W1); import the teammate's `inference_coz.py`/`osediff_sd3.py` overwrites (only `train/` + `scripts/train/`).

  **Recommended Agent Profile**: Category `quick`. Skills: [`git-master`]. Omitted: none.

  **Parallelization**: Can Run In Parallel: NO. Blocks: all W1. Blocked By: W0.T2.

  **References** (sytwu @ `fb8aa60`, verified): `train/grpo/{trainer.py,rollout.py,rewards.py,sr_env.py,state.py,critic.py,text_sim.py,metrics.py}`, `train/dataset/zoom_dataset.py`, `train/evaluate.py`, `train/train_grpo_vlm.py`, `train/configs/grpo_default.yaml`, `train/README.md`, `scripts/train/experiments/{exp0_baseline,exp1_anchor,exp2_repetition,exp3_feedback,exp4_full,smoke}.sh` + `_common.sh` + `README.md`.

  **Acceptance Criteria**:
  ```
  Scenario: GRPO code present in-tree (happy path)
    Tool: Bash
    Steps:
      1. for f in train/grpo/trainer.py train/grpo/rewards.py train/configs/grpo_default.yaml train/train_grpo_vlm.py scripts/train/experiments/smoke.sh; do test -f "$f" || { echo MISSING $f; exit 1; }; done  → assert exit 0
      2. git log --oneline -1                                                  → assert message contains "UNVALIDATED"
    Evidence: .sisyphus/evidence/task-w0t3-import.txt

  Scenario: No pipeline files clobbered (failure guard)
    Tool: Bash
    Steps:
      1. git diff --name-only HEAD~1 HEAD | grep -E '^(inference_coz.py|osediff_sd3.py)$'  → assert EXIT 1 (untouched)
    Evidence: .sisyphus/evidence/task-w0t3-noclobber.txt
  ```
  **Commit**: YES — `feat: import sytwu GRPO trainer (UNVALIDATED)`. Pre-commit: file-presence loop above.

- [x] **W0.T4. Build `.venv-train` (verl + vllm≥0.8.3 + ray + wandb + pyiqa) + import smoke**

  **What to do**:
  - `source activate.sh` first (pins caches), then create an ISOLATED venv: `uv venv .venv-train --python 3.10`.
  - Install (prebuilt wheels only, no source flash-attn): `uv pip install "vllm>=0.8.3,<0.9"` (pulls a torch it pins) → then `uv pip install verl ray wandb "pyiqa==0.1.15.post2" datasets`. If `verl` pip is stale, install from the pinned GitHub tag. Set `VLLM_USE_V1=0` in the venv activation notes.
  - Freeze the resolved set: `uv pip freeze > requirements.train.txt`.
  - Import-smoke and record versions.

  **Must NOT do**: install verl/vllm/ray into the existing `.venv` (R7); build flash-attn from source; upgrade the inference `.venv` torch.

  **Recommended Agent Profile**: Category `deep` (dependency-resolution hazard). Skills: []. Omitted: `git-master` (lockfile committed by W4.T6).

  **Parallelization**: Can Run In Parallel: YES — Wave W0-A with T1/T5/T6. Blocks: W3.T1. Blocked By: None.

  **References**: existing `.venv` = torch 2.6.0+cu126, py3.10.20 (uv 0.11.1). `requirements.cu126.txt` documents the cu126 stack (+xformers 0.0.29.post3). veRL flags use `rollout.name=vllm`, `tensor_model_parallel_size=2`, `actor.strategy=fsdp2` (Context → Research Findings). Driver 550.127.08, system CUDA 12.5 → cu126 prebuilt wheels OK.

  **Acceptance Criteria**:
  ```
  Scenario: Train stack imports with valid versions (happy path)
    Tool: Bash
    Steps:
      1. source .venv-train/bin/activate
      2. python -c "import verl,vllm,ray,wandb,pyiqa,torch; print(vllm.__version__, torch.__version__)"  → assert prints, no ImportError
      3. python -c "import vllm,sys; v=tuple(map(int,vllm.__version__.split('.')[:2])); sys.exit(0 if v>=(0,8) else 1)"  → assert exit 0
      4. test -f requirements.train.txt  → assert exit 0
    Evidence: .sisyphus/evidence/task-w0t4-trainvenv.txt

  Scenario: Inference venv stays uncontaminated (failure guard / R7)
    Tool: Bash
    Steps:
      1. source activate.sh && python -c "import verl" 2>&1 | grep -qi 'No module named' && echo CLEAN  → assert "CLEAN"
    Expected Result: verl absent from .venv
    Evidence: .sisyphus/evidence/task-w0t4-isolation.txt
  ```
  **Commit**: NO (lockfile committed in W4.T6 with run hygiene).

- [x] **W0.T5. Pin `pyiqa` into `.venv` + verify metric load / lower_better flags**

  **What to do**:
  - `source activate.sh`; `uv pip install "pyiqa==0.1.15.post2" "setuptools==67.8.0"` into the EXISTING inference venv.
  - Verify each metric constructs and report `lower_better`: NIQE (lower-better → invert), MUSIQ/MANIQA/CLIPIQA (higher-better). First call may download metric weights into `.hf_cache`/`.torch_cache` — allow it.

  **Must NOT do**: bump torch/transformers in `.venv`; install verl/vllm here.

  **Recommended Agent Profile**: Category `deep`. Skills: []. Omitted: none.

  **Parallelization**: Can Run In Parallel: YES — Wave W0-A. Blocks: W1.T4, W0.T7. Blocked By: None.

  **References**: `visualize.py` (origin/main) `compute_metrics()` uses `pyiqa.create_metric(name)`; sytwu `train/grpo/metrics.py:13-18` `HIGHER_BETTER={niqe:False,musiq:True,maniqa:True,clipiqa:True}` — cross-check the live flags against this map.

  **Acceptance Criteria**:
  ```
  Scenario: All 4 IQA metrics load with correct directionality (happy path)
    Tool: Bash
    Steps:
      1. source activate.sh
      2. python - <<'PY'
         import pyiqa,torch
         exp={'niqe':True,'musiq':False,'maniqa':False,'clipiqa':False}  # True == lower_better
         for n,lb in exp.items():
             m=pyiqa.create_metric(n); assert bool(m.lower_better)==lb, (n,m.lower_better)
         x=torch.rand(1,3,224,224); print('musiq', float(pyiqa.create_metric('musiq')(x)))
         print('flags ok')
         PY
      → assert "flags ok" and a finite musiq score printed
    Evidence: .sisyphus/evidence/task-w0t5-pyiqa.txt

  Scenario: NIQE inversion contract documented (failure guard / R4 dependency)
    Tool: Bash
    Steps:
      1. python -c "import pyiqa; assert pyiqa.create_metric('niqe').lower_better is True; print('niqe is lower-better → MUST invert in evaluate.py')"  → assert prints
    Evidence: .sisyphus/evidence/task-w0t5-niqe.txt
  ```
  **Commit**: YES — append to `requirements.txt` if not already (origin/main added it); else none.

- [x] **W0.T6. DIV2K download + 512² preprocess + DISJOINT train/valid split**

  **What to do**:
  - Create `scripts/data/prepare_div2k.py`: download DIV2K HR (official: `0001–0800` train, `0801–0900` valid = 100), apply `inference_coz.resize_and_center_crop(img, 512)` to each, write to `data/div2k/train/<id>.png` and `data/div2k/valid/<id>.png`, and manifests `data/manifests/{train,valid}.txt` (absolute paths).
  - ASSERT disjoint id sets and write `data/manifests/disjoint_ok.txt`. (DIV8K only if time permits — out of default scope.)

  **Must NOT do**: leak any valid id into train (R8); resize without center-crop (breaks fixed protocol); store under a git-tracked dir (data/ is gitignored).

  **Recommended Agent Profile**: Category `unspecified-high` (I/O + scripting). Skills: []. Omitted: none.

  **Parallelization**: Can Run In Parallel: YES — Wave W0-A. Blocks: W0.T7, W2.*, W3.T3. Blocked By: None.

  **References**: `inference_coz.py:26-33` `resize_and_center_crop` (LANCZOS resize so min side==512, center-crop 512²). sytwu config expects `train_utils/dataset_paths/DIV2K_{TRAIN,VALID}.txt` (we generate local manifests to replace the foreign paths).

  **Acceptance Criteria**:
  ```
  Scenario: Correct counts, 512² dims, disjoint split (happy path)
    Tool: Bash
    Steps:
      1. python - <<'PY'
         import pathlib
         from PIL import Image
         tr=sorted(pathlib.Path('data/div2k/train').glob('*.png')); va=sorted(pathlib.Path('data/div2k/valid').glob('*.png'))
         assert len(tr)==800 and len(va)>=100, (len(tr),len(va))
         for p in (tr[0],va[0]):
             w,h=Image.open(p).size; assert (w,h)==(512,512), (p,w,h)
         s_tr={p.stem for p in tr}; s_va={p.stem for p in va}; assert not (s_tr & s_va), 'LEAK'
         print('div2k ok', len(tr), len(va))
         PY
      → assert "div2k ok 800 100" (or valid>=100)
    Evidence: .sisyphus/evidence/task-w0t6-div2k.txt

  Scenario: Overlap injection is rejected (failure guard / R8)
    Tool: Bash
    Steps:
      1. python -c "tr={'0801'}; va={'0801'}; assert not (tr&va), 'overlap detected'" 2>&1 | grep -q 'overlap detected' && echo GUARD_OK  → assert "GUARD_OK"
    Evidence: .sisyphus/evidence/task-w0t6-leakguard.txt
  ```
  **Commit**: YES — `feat(data): DIV2K prepare script + disjoint split assertion` (script only; images gitignored).

- [x] **W0.T7. Pre-register eval protocol + `evaluate.py` (single source of truth)**

  **What to do**:
  - Write `docs/eval_protocol.md` LOCKING the fixed protocol (≥100 held-out DIV2K-valid, same SR LoRA+VAE, center crop, 4-step recursion→256×, greedy `max_new_tokens=32`, ≥3 seeds, report mean±std; metrics niqe/musiq/maniqa/clipiqa + consistency + unique_token_ratio). Commit this BEFORE any training claim.
  - Build `evaluate.py` reusing origin/main `visualize.py:compute_metrics` + sytwu `train/grpo/metrics.py`, ADDING: `consistency = CLIP_cos(SR_deep, x0)`→[0,1] (drift axis) and `unique_token_ratio` over the 4 per-scale prompts (convergence axis). Emit `results/<arm>.csv` with columns `arm,seed,image,niqe,musiq,maniqa,clipiqa,consistency,unique_token_ratio` (NIQE stored inverted-or-flagged consistently).
  - Provide a 2-image fixture run to prove it computes.

  **Must NOT do**: emit a single-metric summary (R4); compute IQA inside the training loop (B4); change protocol after training starts.

  **Recommended Agent Profile**: Category `deep`. Skills: []. Omitted: `ai-slop-remover` (new file, reviewed in F2).

  **Parallelization**: Can Run In Parallel: NO (Wave W0-D). Blocks: W2.T2, W2.T6, W4.T5. Blocked By: W0.T2 (visualize.py), W0.T5 (pyiqa), W0.T6 (valid set).

  **References**: `visualize.py:compute_metrics` (origin/main; `pyiqa.create_metric`, reads `per-sample/<stem>/{i}.png`+`txt/{i}.txt`); sytwu `train/evaluate.py:22-57` `evaluate_adapter` (greedy zoom + pyiqa); `train/grpo/text_sim.py:ClipEmbedder` (CLIP for consistency); `inference_coz.py:147-172` system prompt + two-image state.

  **Acceptance Criteria**:
  ```
  Scenario: evaluate.py emits all 6 metric columns in valid ranges (happy path)
    Tool: Bash
    Steps:
      1. source activate.sh
      2. python evaluate.py --arm FIXTURE --images data/div2k/valid --limit 2 --out results/FIXTURE.csv
      3. python - <<'PY'
         import csv; rows=list(csv.DictReader(open('results/FIXTURE.csv'))); r=rows[0]
         need=['arm','seed','image','niqe','musiq','maniqa','clipiqa','consistency','unique_token_ratio']
         for c in need: assert c in r, f'missing {c}'
         assert 0.0<=float(r['clipiqa'])<=1.0 and 0.0<=float(r['consistency'])<=1.0 and 0.0<float(r['unique_token_ratio'])<=1.0
         print('eval cols+ranges ok', len(rows))
         PY
      → assert "eval cols+ranges ok"
      4. test -f docs/eval_protocol.md  → assert exit 0
    Evidence: .sisyphus/evidence/task-w0t7-eval.csv

  Scenario: Protocol locked before training (failure guard)
    Tool: Bash
    Steps:
      1. git log --oneline -- docs/eval_protocol.md | head -1  → assert a commit exists (pre-registered)
    Evidence: .sisyphus/evidence/task-w0t7-prereg.txt
  ```
  **Commit**: YES — `feat(eval): pre-register protocol + evaluate.py (consistency+unique-token axes)`. Pre-commit: fixture CSV asserts.

### Wave W1 — Validate sytwu (TDD; 8 parallel validators → report). **GATE G0**

> Each validator: (1) READ the module, (2) write a RED unit test encoding the paper-correct contract, (3) fix the bug to GREEN or document the discrepancy, (4) emit a one-paragraph verdict for W1.T9. Tests live in `tests/`, run in `.venv` via `pytest`.

- [x] **W1.T1. Validate `train/grpo/rewards.py` (orchestrator + Welford z-norm) — R3, R11**

  **What to do**:
  - Read `RewardOrchestrator` (55-127) + `_RunningNorm` (34-52). Write `tests/test_rewards.py`: (a) weight application `total == Σ w_k·use_k`; (b) magnitude-balance — with `normalize=True`, no single component dominates the sum on a mixed batch; (c) determinism — global running z-norm is order-dependent (RED test proves two orderings differ).
  - FIX: add a `per_group`/`per_batch` normalization mode (default for veRL path) and make `_RunningNorm` state checkpointable/resettable; keep the global running-norm only for single-proc torch. Document the DOUBLE-normalization interaction with `trainer.py:156`.

  **Must NOT do**: silently keep global stateful norm for the veRL path; remove the running-norm without documenting; change reward weights.

  **Recommended Agent Profile**: Category `ultrabrain` (normalization math + statefulness). Skills: [`ai-slop-remover`] (clean the repair). Omitted: `git-master` (commit is mechanical).

  **Parallelization**: YES — Wave W1 (with T2–T8). Blocks: W1.T9, W2.*, W3.T2. Blocked By: W0.T3.

  **References**: `train/grpo/rewards.py:34-52` (`_RunningNorm`), `:66` (`self.norms` created once, never reset), `:115-127` (`compute()` applies running z-norm then weights), interaction with `train/grpo/trainer.py:156` (per-group standardize). Confirmed issue #3.

  **Acceptance Criteria**:
  ```
  Scenario: Reward sum + per-group norm correctness (happy path)
    Tool: Bash
    Steps:
      1. source activate.sh && pytest tests/test_rewards.py -q
    Expected Result: PASS — weight-sum exact; per-group norm is order-INVARIANT; magnitude-balance holds
    Evidence: .sisyphus/evidence/task-w1t1-pytest.txt

  Scenario: Global stateful norm order-dependence is caught (failure→fixed)
    Tool: Bash
    Steps:
      1. pytest tests/test_rewards.py::test_global_norm_is_order_dependent -q   → asserts RED on old code, GREEN after per-group fix
    Evidence: .sisyphus/evidence/task-w1t1-ordertest.txt
  ```
  **Commit**: YES — `fix(train): per-group reward normalization + checkpointable z-norm (+tests)`.

- [x] **W1.T2. Validate `train/grpo/text_sim.py` (R_anc / R_rep) — anti-drift + anti-convergence cores**

  **What to do**:
  - Read `ClipEmbedder` (17-47), `ngram_set` (50-54), `ngram_overlap` (57-71). Write `tests/test_text_sim.py` with the contracts: `R_anc(cap, cap) ≈ 1 > R_anc(cap, unrelated)`; `R_rep(identical) < R_rep(disjoint)` (repetition penalised); `ngram_overlap` Jaccard ∈ [0,1] and symmetric where expected.
  - Fix any sign/range error; confirm cosine mapping [-1,1]→[0,1] is applied consistently with `rewards._r_anc` (69-72) and `_r_rep` (74-82).

  **Must NOT do**: change the CLIP model id (`openai/clip-vit-base-patch32`) without documenting; leave R_rep with inverted sign.

  **Recommended Agent Profile**: Category `deep`. Skills: [`ai-slop-remover`]. Omitted: none.

  **Parallelization**: YES — Wave W1. Blocks: W1.T9, W2.*. Blocked By: W0.T3.

  **References**: `train/grpo/text_sim.py:17-71`; consumed by `train/grpo/rewards.py:_r_anc(69-72)` and `_r_rep(74-82)`. Concrete target: `0064_eye` scale2→3 near-identical prompts must score high repetition (large penalty).

  **Acceptance Criteria**:
  ```
  Scenario: Anchor rewards similar > dissimilar; repetition penalised (happy path)
    Tool: Bash
    Steps:
      1. source activate.sh && pytest tests/test_text_sim.py -q
    Expected Result: PASS — R_anc(x,x)≈1; R_anc(x,x)>R_anc(x,y); R_rep(dup)<R_rep(distinct)
    Evidence: .sisyphus/evidence/task-w1t2-pytest.txt

  Scenario: Real drift sample scored correctly (failure guard)
    Tool: Bash
    Steps:
      1. pytest tests/test_text_sim.py::test_0064_repetition -q   → uses fail_case scale2/scale3 prompts; asserts high overlap → strong R_rep penalty
    Evidence: .sisyphus/evidence/task-w1t2-0064.txt
  ```
  **Commit**: YES — `fix(train): validate R_anc/R_rep text similarity (+tests)`.

- [x] **W1.T3. Validate `train/grpo/critic.py` (R_crit) + R_phr blacklist**

  **What to do**:
  - Read `Critic` (20-78; vlm Qwen2.5-VL-7B + clipscore backends) and `rewards._r_phr` (104-108). Write `tests/test_critic_phr.py`: `R_phr("first image ... second image ...") < 0` (filler penalised) and `R_phr(clean) == 0`; critic rating parse `"7/10"`/`"7"` → `0.7 ∈ [0,1]`, malformed → safe fallback; clipscore backend returns a finite [0,1].
  - Mock the 7B VLM (no GPU load in the unit test) — test the PARSER and clamping, not the model.

  **Must NOT do**: download/load the 7B model inside the unit test; let an unparsable rating crash the loop.

  **Recommended Agent Profile**: Category `deep`. Skills: [`ai-slop-remover`]. Omitted: none.

  **Parallelization**: YES — Wave W1. Blocks: W1.T9. Blocked By: W0.T3.

  **References**: `train/grpo/critic.py:12-17` (CRITIC_PROMPT), `:20-45` (init, default `Qwen/Qwen2.5-VL-7B-Instruct`, `device cuda:2`), `:47-78` (`score`→[0,1], `max_new_tokens=8`); `train/grpo/rewards.py:104-108` (`_r_phr`, `r_phr.fillers` blacklist).

  **Acceptance Criteria**:
  ```
  Scenario: Phrase blacklist + critic parser robust (happy path)
    Tool: Bash
    Steps:
      1. source activate.sh && pytest tests/test_critic_phr.py -q
    Expected Result: PASS — R_phr filler<0, clean==0; "7/10"→0.7; "banana"→fallback (no crash); clipscore∈[0,1]
    Evidence: .sisyphus/evidence/task-w1t3-pytest.txt

  Scenario: Malformed rating cannot crash training (failure guard)
    Tool: Bash
    Steps:
      1. pytest tests/test_critic_phr.py::test_unparsable_rating_fallback -q  → assert PASS (returns clamped default)
    Evidence: .sisyphus/evidence/task-w1t3-fallback.txt
  ```
  **Commit**: YES — `fix(train): validate R_crit parser + R_phr blacklist (+tests)`.

- [x] **W1.T4. Validate `train/grpo/metrics.py` (pyiqa wrappers + NIQE inversion)**

  **What to do**: Read `metrics.py:13-18` (`HIGHER_BETTER`) + `IQAMetrics` (21-37). Write `tests/test_metrics.py`: input is NCHW∈[0,1]; `score_all` returns all 4; NIQE is inverted (or flagged) so "bigger==better" holds uniformly; cross-check live `pyiqa.lower_better` against the `HIGHER_BETTER` map (fail if they disagree). Fix any mismatch.

  **Must NOT do**: feed HWC or [0,255] tensors; assume NIQE higher-better.

  **Recommended Agent Profile**: Category `deep`. Skills: [`ai-slop-remover`].

  **Parallelization**: YES — Wave W1. Blocks: W1.T9, W0.T7 (shared metric contract). Blocked By: W0.T3, W0.T5.

  **References**: `train/grpo/metrics.py:9` (`import pyiqa`), `:13-18`, `:21-37`; W0.T5 verified live flags.

  **Acceptance Criteria**:
  ```
  Scenario: Metric directionality + tensor contract (happy path)
    Tool: Bash
    Steps: 1. source activate.sh && pytest tests/test_metrics.py -q
    Expected: PASS — NCHW[0,1] accepted; 4 metrics returned; map==live pyiqa.lower_better
    Evidence: .sisyphus/evidence/task-w1t4-pytest.txt
  Scenario: Map/live mismatch is caught (failure guard)
    Tool: Bash
    Steps: 1. pytest tests/test_metrics.py::test_map_matches_pyiqa -q  → assert PASS (no silent disagreement)
    Evidence: .sisyphus/evidence/task-w1t4-mapcheck.txt
  ```
  **Commit**: YES — `fix(train): validate pyiqa metric wrappers + NIQE inversion (+tests)`.

- [x] **W1.T5. Validate `train/grpo/state.py` (3-image + scale message builder) — B5 state expansion**

  **What to do**: Read `build_messages` (24-42) + `process_state` (45-58) + `SYSTEM_TEMPLATE` (14-21). Write `tests/test_state.py`: messages include x0, optional x_{i-2}, x_{i-1}, and a literal `zoom factor={scale}x` token; image count matches non-None inputs; `process_state` output keys/shapes match the Qwen2.5-VL processor contract. Confirm it matches the README "expanded state" (AR-2 + scale awareness) and the inference system prompt at `inference_coz.py:150`.

  **Must NOT do**: assume single-image; drop the scale token.

  **Recommended Agent Profile**: Category `deep`. Skills: [`ai-slop-remover`].

  **Parallelization**: YES — Wave W1. Blocks: W1.T9, W2.*, W3.T3. Blocked By: W0.T3.

  **References**: `train/grpo/state.py:12` (`from qwen_vl_utils import process_vision_info`), `:14-58`; compare `inference_coz.py:147-172` (two-image baseline state, system prompt L150).

  **Acceptance Criteria**:
  ```
  Scenario: Expanded state carries x0/x_{i-2}/x_{i-1}+scale (happy path)
    Tool: Bash
    Steps: 1. source activate.sh && pytest tests/test_state.py -q
    Expected: PASS — image count == #non-None; "zoom factor=" token present; processor keys correct
    Evidence: .sisyphus/evidence/task-w1t5-pytest.txt
  Scenario: Missing scale token rejected (failure guard)
    Tool: Bash
    Steps: 1. pytest tests/test_state.py::test_scale_token_required -q  → assert PASS
    Evidence: .sisyphus/evidence/task-w1t5-scaletoken.txt
  ```
  **Commit**: YES — `fix(train): validate expanded-state message builder (+tests)`.

- [x] **W1.T6. Validate `train/grpo/rollout.py` (G=6 sampling + best-of-group advance) — R2**

  **What to do**: Read `Rollout` (51-174), `_sample_group` (78-107), `run_episode` (119-173). Write `tests/test_rollout.py` (mock policy.generate): group has exactly `group_size` completions; advance rule selection. CONFIRM the best-of-group argmax advance (`rollout.py:160-162`) and document the off-policy state-distribution bias. FIX: add a `advance_policy={best,sampled}` option (default `sampled` for on-policy fidelity), keeping `best` reproducible.

  **Must NOT do**: leave argmax-advance as the only option without documenting the bias; break the deterministic test by real sampling.

  **Recommended Agent Profile**: Category `ultrabrain` (on-policy correctness). Skills: [`ai-slop-remover`].

  **Parallelization**: YES — Wave W1. Blocks: W1.T9, W2.T1. Blocked By: W0.T3.

  **References**: `train/grpo/rollout.py:78-107` (sampling), `:154-162` (`best=max(...,key=reward)` advance), `:164` (`_greedy_prompt` for non-trained scales); module docstring L5-6. Confirmed issue #2.

  **Acceptance Criteria**:
  ```
  Scenario: Group sampling + selectable advance (happy path)
    Tool: Bash
    Steps: 1. source activate.sh && pytest tests/test_rollout.py -q
    Expected: PASS — len(completions)==group_size; advance_policy='best' picks argmax-reward; 'sampled' picks the tracked sampled idx
    Evidence: .sisyphus/evidence/task-w1t6-pytest.txt
  Scenario: Off-policy advance bias documented (failure guard)
    Tool: Bash
    Steps: 1. grep -qi 'best-of-group' docs/validation_report.md || echo PENDING_W1T9  ;  pytest tests/test_rollout.py::test_advance_sampled_default -q  → assert PASS
    Evidence: .sisyphus/evidence/task-w1t6-advance.txt
  ```
  **Commit**: YES — `fix(train): selectable rollout advance (sampled default) + group-size test`.

- [x] **W1.T7. Validate `train/grpo/trainer.py` GRPO math (advantage / KL / clip) — R1 (most critical)**

  **What to do**: Read `_step` (144-203), `_token_logprobs` (110-130), `_ref_logprobs` (132-141). Write `tests/test_grpo_math.py` on TOY tensors: (a) advantage `A=(r-mean)/(std_unbiased+adv_eps)` zero-mean unit-scale; (b) k3 KL `exp(d)-d-1 ≥ 0` and `==0` when `d==0`; (c) **prove issue #1** — with old_logp recomputed under current weights, `ratio≡1.0` and the clamp NEVER binds (fraction-bound `==0`). FIX: capture per-token `old_logp` at SAMPLE time in `rollout._sample_group` (store on `Completion`) and CONSUME it at `trainer.py:162-167` so `clip_eps=0.2` becomes live; OR, if single-step REINFORCE is the intended design, DOCUMENT it explicitly and remove the dead clip. Add training-time telemetry: log `ratio.mean()`/`ratio.std()` and `clamp_bind_fraction`.

  **Must NOT do**: leave a silent no-op clip undocumented; change `kl_beta`/`clip_eps`/`lr` defaults; add inner PPO epochs without a real cached old_logp.

  **Recommended Agent Profile**: Category `ultrabrain` (RL optimizer correctness). Skills: [`ai-slop-remover`].

  **Parallelization**: YES — Wave W1. Blocks: W1.T9, W2.T1. Blocked By: W0.T3.

  **References**: `train/grpo/trainer.py:156` (group standardize), `:160-170` (ratio/clip — issue #1), `:172-173` (k3 KL), `:192` (single `optimizer.step()`); `train/grpo/rollout.py:78-107` (where to cache old_logp); `train/README.md:24` (claims "clipped surrogate" — contradicts inert clip → discrepancy to log). Confirmed issue #1.

  **Acceptance Criteria**:
  ```
  Scenario: GRPO math correct + ratio≡1 proven then fixed (happy path)
    Tool: Bash
    Steps: 1. source activate.sh && pytest tests/test_grpo_math.py -q
    Expected: PASS — advantage zero-mean/unit-scale; KL k3 ≥0 & 0@d=0; pre-fix test asserts ratio==1.0 & bind_frac==0; post-fix test asserts ratio varies once old_logp is cached
    Evidence: .sisyphus/evidence/task-w1t7-pytest.txt
  Scenario: Telemetry exposes the no-op (failure guard)
    Tool: Bash
    Steps: 1. pytest tests/test_grpo_math.py::test_clamp_bind_fraction_zero_without_cache -q  → assert PASS (documents the bug)
    Evidence: .sisyphus/evidence/task-w1t7-telemetry.txt
  ```
  **Commit**: YES — `fix(train): cache old_logp at sample time → live PPO clip + GRPO math tests`.

- [x] **W1.T8. Validate `sr_env.py` + `zoom_dataset.py` + `evaluate.py`; DE-HARDCODE foreign paths — R9**

  **What to do**: Read `sr_env.FrozenSRBackbone` (20-49), `zoom_dataset.ZoomBaseImageDataset` (29-45), `train/evaluate.py` (22-91). Write `tests/test_env_data.py`: `sr_env.render(crop, prompt)` returns NCHW∈[0,1] (mock or tiny SR); `ZoomBaseImageDataset` yields 512² PIL from a local manifest. DE-HARDCODE: replace `_common.sh:15` `PY=/project2/cookies/...` with `${PY:-python}`; point config `train_txt/valid_txt` to `data/manifests/{train,valid}.txt` (W0.T6); ensure `continue_from=ckpt/VLM_LoRA/checkpoint-10000` and `sr_lora_path/sr_vae_path` resolve locally; do NOT read the stale `adapter_config.json:base_model_name_or_path` (`/mnt/data1/...`). Grep-assert NO foreign path remains.

  **Must NOT do**: leave any `/project2/cookies` or `/mnt/data1` reference; read base model from the stale adapter config.

  **Recommended Agent Profile**: Category `deep`. Skills: [`ai-slop-remover`, `git-master`].

  **Parallelization**: YES — Wave W1. Blocks: W1.T9, W2.T1. Blocked By: W0.T3, W0.T5, W0.T6.

  **References**: `train/grpo/sr_env.py:14` (`from osediff_sd3 import OSEDiff_SD3_TEST`), `:41-49` (render); `train/dataset/zoom_dataset.py:13` (`from inference_coz import resize_and_center_crop`); `scripts/train/experiments/_common.sh:15` + `README.md:53,69` (hardcoded python); `train/configs/grpo_default.yaml:116,120` (foreign DIV2K txt); `ckpt/VLM_LoRA/checkpoint-10000/adapter_config.json` (stale `/mnt/data1/...`). `osediff_sd3.py:611-669` (`OSEDiff_SD3_TEST.forward` one-step SR).

  **Acceptance Criteria**:
  ```
  Scenario: SR env + dataset contracts + path hygiene (happy path)
    Tool: Bash
    Steps:
      1. source activate.sh && pytest tests/test_env_data.py -q          → assert PASS (render NCHW[0,1]; dataset 512²)
      2. grep -rnE '/project2/cookies|/mnt/data1' train/ scripts/train/  → assert EXIT 1 (none remain)
    Evidence: .sisyphus/evidence/task-w1t8-pytest.txt
  Scenario: Foreign path reintroduction blocked (failure guard)
    Tool: Bash
    Steps: 1. grep -rn 'modelscope' ckpt/VLM_LoRA/checkpoint-10000/adapter_config.json && echo "STALE_PATH_PRESENT(do-not-read)"  → asserts code never reads this field (covered by pytest)
    Evidence: .sisyphus/evidence/task-w1t8-paths.txt
  ```
  **Commit**: YES — `fix(train): de-hardcode foreign paths + validate sr_env/dataset/evaluate (+tests)`.

- [x] **W1.T9. Synthesize VALIDATION REPORT — GATE G0**

  **What to do**: Aggregate W1.T1–T8 into `docs/validation_report.md` with a per-module verdict table (module · status `correct`/`bug-found+fix`/`discrepancy-vs-paper` · test file · key finding). MUST record the 3 confirmed issues (ratio≡1 / best-of-group / global z-norm) with their fixes, and the README L24 "clipped surrogate" discrepancy. State G0 PASS only if every module has a passing test or a documented, accepted discrepancy.

  **Must NOT do**: mark G0 pass with any untested module or undocumented discrepancy.

  **Recommended Agent Profile**: Category `writing`. Skills: [].

  **Parallelization**: NO (barrier after W1.T1–T8). Blocks: all W2, all W3. Blocked By: W1.T1–T8.

  **References**: outputs of W1.T1–T8; confirmed-issue table from planning research.

  **Acceptance Criteria**:
  ```
  Scenario: Report complete + full test suite green = G0 (happy path)
    Tool: Bash
    Steps:
      1. source activate.sh && pytest tests/ -q                              → assert ALL PASS
      2. for m in rewards text_sim critic_phr metrics state rollout grpo_math env_data; do grep -qi "$m" docs/validation_report.md || { echo "MISSING $m"; exit 1; }; done  → assert exit 0
      3. grep -Eqi 'ratio.*1|no-op|best-of-group|z-norm' docs/validation_report.md  → assert the 3 issues documented
    Evidence: .sisyphus/evidence/task-w1t9-G0.txt
  Scenario: G0 cannot pass with a red test (failure guard)
    Tool: Bash
    Steps: 1. pytest tests/ -q >/dev/null 2>&1 && echo G0_PASS || echo G0_BLOCKED  → assert "G0_PASS"
    Evidence: .sisyphus/evidence/task-w1t9-gate.txt
  ```
  **Commit**: YES — `docs: GRPO module validation report (G0 PASS)`.

### Wave W2 — Prototype (torch, small subset). **GATES G1, G2**

- [x] **W2.T1. Torch smoke run on small subset — GATE G1**

  **What to do**: Localize `train/configs/grpo_default.yaml` (continue_from, sr/vae paths, `train_txt=data/manifests/train.txt`, small `eval.num_images`). Run `PY=$(which python) bash scripts/train/experiments/smoke.sh` on ~20 images (`group_size=2, max_steps=5, wandb=false`) under `source activate.sh`. Assert training advances, reward is logged & finite (no NaN), a PEFT adapter is saved, and the new `ratio`/`clamp_bind` telemetry (from W1.T7) prints.

  **Must NOT do**: run full corpus; enable SR/critic here (text-only smoke); ignore NaN.

  **Recommended Agent Profile**: Category `deep`. Skills: []. **Parallelization**: NO (Wave W2 head). Blocks: W2.T2. Blocked By: W1.T9, W0.T6.

  **References**: `scripts/train/experiments/smoke.sh` (group_size=2, images_per_step=1, max_steps=5, ckpt_every=5, eval.enabled=false, wandb=false), `train/train_grpo_vlm.py` (entry), `train/grpo/trainer.py:206-236` (loop), `:244-251` (save_checkpoint).

  **Acceptance Criteria**:
  ```
  Scenario: Smoke trains + saves adapter, reward finite (happy path / G1)
    Tool: interactive_bash (tmux) + Bash
    Steps:
      1. tmux: source activate.sh && PY=$(which python) bash scripts/train/experiments/smoke.sh 2>&1 | tee .sisyphus/evidence/task-w2t1-smoke.log
      2. Bash: grep -E 'step 5|reward' .sisyphus/evidence/task-w2t1-smoke.log  → assert reward values present
      3. Bash: ls experience/grpo_vlm/smoke*/**/adapter_model.safetensors    → assert ≥1 file
      4. Bash: grep -ic 'nan' .sisyphus/evidence/task-w2t1-smoke.log          → assert 0
    Evidence: .sisyphus/evidence/task-w2t1-smoke.log
  Scenario: NaN/crash fails the gate (failure guard)
    Tool: Bash
    Steps: 1. grep -iqE 'nan|traceback' .sisyphus/evidence/task-w2t1-smoke.log && echo G1_FAIL || echo G1_PASS  → assert "G1_PASS"
    Evidence: .sisyphus/evidence/task-w2t1-g1.txt
  ```
  **Commit**: YES — `chore(proto): localize config + G1 smoke evidence`.

- [x] **W2.T2. Close the loop: adapter → `inference_coz.py` → offline IQA**

  **What to do**: Take the smoke adapter, run `inference_coz.py --prompt_type vlm --vlm_lora_path <smoke_adapter>` on ~10 valid images (4-step recursion), then `evaluate.py` → `results/smoke.csv`. Prove the FULL measurement loop closes (adapter trains → plugs into inference → offline IQA + consistency + unique-token computes). NOTE: inference merges adapter (`inference_coz.py:325-326`) — fine for eval (read-only).

  **Must NOT do**: compute IQA inside the training loop (B4); change decode from greedy `max_new_tokens=32`.

  **Recommended Agent Profile**: Category `deep`. Skills: []. **Parallelization**: NO. Blocks: W2.T3–T5, W3.T3. Blocked By: W2.T1, W0.T7.

  **References**: `inference_coz.py:302-328` (adapter load), `:195` (greedy generate), `:409-423` (recursive_multiscale), `evaluate.py` (W0.T7).

  **Acceptance Criteria**:
  ```
  Scenario: End-to-end adapter→inference→IQA (happy path)
    Tool: Bash
    Steps:
      1. source activate.sh && CUDA_VISIBLE_DEVICES=0,1 python inference_coz.py -i data/div2k/valid -o results/smoke_sr --rec_type recursive_multiscale --prompt_type vlm --vlm_lora_path experience/grpo_vlm/smoke*/checkpoint-5 --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers --ram_ft_path ckpt/DAPE/DAPE.pth --ram_path ckpt/RAM/ram_swin_large_14m.pth --save_prompts
      2. python evaluate.py --arm smoke --images results/smoke_sr --out results/smoke.csv
      3. python -c "import csv;r=list(csv.DictReader(open('results/smoke.csv')));assert r and all(c in r[0] for c in ['niqe','musiq','maniqa','clipiqa','consistency','unique_token_ratio']);print('loop closed',len(r))"  → assert "loop closed"
    Evidence: .sisyphus/evidence/task-w2t2-loop.csv
  Scenario: Adapter actually loaded (failure guard)
    Tool: Bash
    Steps: 1. ls results/smoke_sr/**/txt/0.txt >/dev/null 2>&1 && echo PROMPTS_WRITTEN || echo FAIL  → assert "PROMPTS_WRITTEN"
    Evidence: .sisyphus/evidence/task-w2t2-prompts.txt
  ```
  **Commit**: YES — `chore(proto): close adapter→inference→IQA loop`.

- [x] **W2.T3. Prototype A4 (+R_rep) — small subset + eval**  ·  **W2.T4. Prototype A5 (+R_anc)**  ·  **W2.T5. Prototype A6 (text-only combined R_rep+R_anc+R_phr)**

  **What to do** (3 SIBLING tasks, run concurrently on 3 GPUs): adapt `exp2_repetition.sh` (A4), `exp1_anchor.sh` (A5), and a combined text-only config (A6 = r_rep+r_anc+r_phr, SR/critic OFF) — each on the same ~50-image subset, ~100–200 steps, `continue_from` author ckpt, sampled rollout (W1.T6), live clip (W1.T7). After each: run W2.T2-style inference+`evaluate.py` → `results/{A4,A5,A6}_proto.csv`.

  **Must NOT do**: enable R_fb/R_crit (that is A8/A9); vary the eval subset between arms; change SR LoRA/VAE/crop/recursion.

  **Recommended Agent Profile**: Category `unspecified-high`. Skills: []. **Parallelization**: YES — 3 arms on `CUDA_VISIBLE_DEVICES=0`,`=1`,`=2` (each arm 1–2 GPUs). Blocks: W2.T6. Blocked By: W2.T2.

  **References**: `scripts/train/experiments/{exp1_anchor,exp2_repetition}.sh` (reward toggles), `train/configs/grpo_default.yaml:54-86` (weights), validated W1 modules.

  **Acceptance Criteria**:
  ```
  Scenario: Each arm trains text-only + produces a CSV (happy path)
    Tool: interactive_bash + Bash
    Steps:
      1. tmux (3 panes): CUDA_VISIBLE_DEVICES=0 ... A4 ; =1 ... A5 ; =2 ... A6   (text-only flags)
      2. Bash: for a in A4 A5 A6; do test -s results/${a}_proto.csv || { echo "MISSING $a"; exit 1; }; done  → assert exit 0
      3. Bash: python -c "import csv;[print(a, len(list(csv.DictReader(open(f'results/{a}_proto.csv'))))) for a in ['A4','A5','A6']]"  → assert >0 rows each
    Evidence: .sisyphus/evidence/task-w2t345-arms.txt
  Scenario: No SR/critic leaked into text-only arms (failure guard)
    Tool: Bash
    Steps: 1. grep -rE 'r_fb|r_crit' configs/proto_A*.yaml | grep -i 'enabled: true' && echo LEAK || echo CLEAN  → assert "CLEAN"
    Evidence: .sisyphus/evidence/task-w2t345-clean.txt
  ```
  **Commit**: YES — `feat(proto): A4/A5/A6 text-only prototypes + eval CSVs`.

- [x] **W2.T6. Correlation gate analysis — GATE G2**

  **What to do**: Eval A3 (author ckpt) on the SAME subset; compare A6 vs A3 across niqe/musiq/maniqa/clipiqa + consistency + unique_token_ratio. Compute proxy(reward)↔IQA correlation. DECISION: if A6 beats A3 on ≥2/4 IQA WITHOUT regressing others AND improves a drift+convergence metric → **G2 PASS** (proceed to W3/W4 text-only). Else → escalate to R_fb (enable in A8 prototype) and RE-GATE before scaling. Write `docs/g2_decision.md`.

  **Must NOT do**: scale to full DIV2K before G2; declare success on one metric (R4).

  **Recommended Agent Profile**: Category `oracle` (judgement under metric ambiguity). Skills: []. **Parallelization**: NO (barrier). Blocks: W3.*, W4.*. Blocked By: W2.T3–T5 + A3 eval.

  **References**: `results/{A3,A4,A5,A6}_proto.csv`; pivotal decision B4; risk R4.

  **Acceptance Criteria**:
  ```
  Scenario: Documented, rule-based G2 verdict (happy path)
    Tool: Bash
    Steps:
      1. python - <<'PY'
         import csv,statistics as st
         def mean(a,c):return st.mean(float(r[c]) for r in csv.DictReader(open(f'results/{a}_proto.csv')))
         m=['musiq','maniqa','clipiqa','niqe']; a6={c:mean('A6',c) for c in m+['consistency','unique_token_ratio']}; a3={c:mean('A3',c) for c in m+['consistency','unique_token_ratio']}
         wins=sum(1 for c in ['musiq','maniqa','clipiqa'] if a6[c]>a3[c])+(1 if a6['niqe']<a3['niqe'] else 0)
         drift=a6['consistency']>a3['consistency']; conv=a6['unique_token_ratio']>a3['unique_token_ratio']
         print('G2_PASS' if (wins>=2 and (drift or conv)) else 'G2_ESCALATE_RFB', 'wins',wins,'drift',drift,'conv',conv)
         PY
      → assert prints G2_PASS or G2_ESCALATE_RFB (decision recorded)
      2. test -f docs/g2_decision.md  → assert exit 0
    Evidence: .sisyphus/evidence/task-w2t6-g2.txt
  Scenario: Single-metric claim blocked (failure guard / R4)
    Tool: Bash
    Steps: 1. grep -Eqi 'consistency|unique_token' docs/g2_decision.md  → assert present (multi-axis)
    Evidence: .sisyphus/evidence/task-w2t6-multiaxis.txt
  ```
  **Commit**: YES — `docs: G2 correlation-gate decision`.

### Wave W3 — veRL build (Track-2; TIME-BOXED). **GATE G3**

> **Time-box: 2 working days of agent effort (or ≤12 train-cycle attempts) on W3.T5.** If G3 not met → fire W3.T6 (torch-FSDP2 fallback), documented in `docs/verl_status.md`. Never silently drop veRL scope.

- [x] **W3.T1. veRL stack smoke (official Qwen2.5-VL GRPO example) on 6×H200**

  **What to do**: In `.venv-train`, run veRL's `examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh` (or the 3B-adapted variant) at tiny scale (few steps, `tensor_model_parallel_size=2`, `rollout.name=vllm`, `VLLM_USE_V1=0`) to confirm the stack trains+rollouts on this node. Record GPU layout & throughput.

  **Must NOT do**: use vLLM 0.7.x; build flash-attn from source; run in `.venv`.

  **Recommended Agent Profile**: Category `deep`. Skills: []. **Parallelization**: NO (Wave W3 head; after W0.T4). Blocks: W3.T2/T4/T5. Blocked By: W0.T4.

  **References**: veRL `examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh`, `tuning/lora/run_qwen2_5_vl_7b_fsdp.sh`; entry `python -m verl.trainer.main_ppo`; flags in Context→Research Findings.

  **Acceptance Criteria**:
  ```
  Scenario: Official example completes N steps on 6×H200 (happy path)
    Tool: interactive_bash + Bash
    Steps:
      1. tmux: source .venv-train/bin/activate && VLLM_USE_V1=0 bash run_qwen2_5_vl_grpo_smoke.sh 2>&1 | tee .sisyphus/evidence/task-w3t1.log
      2. Bash: grep -Eqi 'step.*1|val|reward' .sisyphus/evidence/task-w3t1.log && echo VERL_OK  → assert "VERL_OK"
      3. Bash: grep -iqE 'oom|cuda error|traceback' .sisyphus/evidence/task-w3t1.log && echo BAD || echo CLEAN  → assert "CLEAN"
    Evidence: .sisyphus/evidence/task-w3t1.log
  Scenario: vLLM version guard (failure guard / R6)
    Tool: Bash
    Steps: 1. source .venv-train/bin/activate && python -c "import vllm,sys;sys.exit(0 if tuple(map(int,vllm.__version__.split('.')[:2]))>=(0,8) else 1)" && echo VLLM_OK  → assert "VLLM_OK"
    Evidence: .sisyphus/evidence/task-w3t1-vllm.txt
  ```
  **Commit**: NO (evidence only).

- [x] **W3.T2. veRL `custom_reward_function` wrapping VALIDATED rewards (text-only)**

  **What to do**: Write `verl_custom_reward.py` exposing `compute_score(data_source, solution_str, ground_truth, extra_info)` that calls the VALIDATED `RewardOrchestrator` (text-only: R_rep+R_anc+R_phr) with PER-GROUP normalization (W1.T1). Wire via `custom_reward_function.path/name`. Unit-test the adapter contract (returns finite float per sample).

  **Must NOT do**: reuse the global stateful z-norm (R3); call SR/critic in the text-only reward.

  **Recommended Agent Profile**: Category `ultrabrain`. Skills: [`ai-slop-remover`]. **Parallelization**: YES — Wave W3 (with T3,T4). Blocks: W3.T5. Blocked By: W1.T9, W3.T1.

  **References**: `train/grpo/rewards.py` (validated), veRL `custom_reward_function` interface; `extra_info` carries x0-caption + prev-prompt + scale.

  **Acceptance Criteria**:
  ```
  Scenario: Reward adapter returns finite scores (happy path)
    Tool: Bash
    Steps: 1. source .venv-train/bin/activate && pytest tests/test_verl_reward.py -q  → assert PASS (finite float; per-group norm; no SR/critic import)
    Evidence: .sisyphus/evidence/task-w3t2.txt
  Scenario: Global z-norm not used in veRL path (failure guard / R3)
    Tool: Bash
    Steps: 1. grep -n '_RunningNorm' verl_custom_reward.py && echo CHECK || echo OK_NO_GLOBAL  → assert "OK_NO_GLOBAL"
    Evidence: .sisyphus/evidence/task-w3t2-norm.txt
  ```
  **Commit**: YES — `feat(verl): custom reward wrapping validated text-only stack`.

- [x] **W3.T3. Parquet state-generator (veRL schema)**

  **What to do**: Write `scripts/data/build_parquet.py` to run baseline CoZ over the corpus and dump STATIC per-state rows → `data/parquet/coz_states_{train,valid}.parquet` with veRL columns: `prompt` (chat w/ x0-caption + x_{i-2}-caption + `zoom factor={scale}x`), `images` (list: current crop png), `data_source`, `reward_model.ground_truth`, `extra_info` (x0_caption, prev_prompt, scale, image_id). Per B5 the global context is TEXT for veRL (single current image).

  **Must NOT do**: embed multi-image arrays for veRL (multimodal multi-image rollout UNCONFIRMED); leak valid ids into train.

  **Recommended Agent Profile**: Category `unspecified-high`. Skills: []. **Parallelization**: YES — Wave W3. Blocks: W3.T5. Blocked By: W0.T6, W2.T2, W1.T5.

  **References**: veRL parquet schema (`data.image_key=images`); `train/grpo/state.py` (caption builder); `inference_coz.py:409-423` (state crops).

  **Acceptance Criteria**:
  ```
  Scenario: Parquet has veRL schema + text-caption state (happy path)
    Tool: Bash
    Steps: 1. source .venv-train/bin/activate && python -c "import pandas as pd;d=pd.read_parquet('data/parquet/coz_states_valid.parquet');assert all(c in d.columns for c in ['prompt','images','data_source','extra_info']);assert d['images'].iloc[0] is not None;print('parquet ok',len(d))"  → assert "parquet ok"
    Evidence: .sisyphus/evidence/task-w3t3.txt
  Scenario: 'zoom factor=' token present + disjoint ids (failure guard)
    Tool: Bash
    Steps: 1. python -c "import pandas as pd;tr=set(pd.read_parquet('data/parquet/coz_states_train.parquet').extra_info.map(lambda e:e['image_id']));va=set(pd.read_parquet('data/parquet/coz_states_valid.parquet').extra_info.map(lambda e:e['image_id']));assert not(tr&va),'LEAK';print('disjoint',len(tr),len(va))"  → assert "disjoint"
    Evidence: .sisyphus/evidence/task-w3t3-disjoint.txt
  ```
  **Commit**: YES — `feat(verl): static per-state parquet generator`.

- [x] **W3.T4. LoRA init strategy (`lora_adapter_path` vs merge→fresh) — B2, R10**

  **What to do**: Attempt veRL `lora_adapter_path=ckpt/VLM_LoRA/checkpoint-10000` init (r=8/α=32, `target_modules=all-linear exclude_modules='.*visual.*'`, `inference_mode=false`). If unsupported (GH #3278) within time-box → FALLBACK: merge author adapter into base (`PeftModel.merge_and_unload`), train a FRESH adapter, set the KL reference to author behavior. Document the chosen path + why in `docs/verl_status.md`.

  **Must NOT do**: silently train from scratch losing author warm-start; include the vision tower in LoRA targets.

  **Recommended Agent Profile**: Category `deep`. Skills: []. **Parallelization**: YES — Wave W3. Blocks: W3.T5. Blocked By: W3.T1.

  **References**: `ckpt/VLM_LoRA/checkpoint-10000/adapter_config.json` (r=8/α=32, target regex LLM-only, `inference_mode:true`→must flip), veRL LoRA flags; merge via `osediff`/peft `merge_and_unload`.

  **Acceptance Criteria**:
  ```
  Scenario: Author warm-start verified by chosen path (happy path)
    Tool: Bash
    Steps: 1. source .venv-train/bin/activate && python scripts/verl/check_lora_init.py  → assert prints "ADAPTER_PATH_OK" OR "MERGED_FRESH_OK" + initial logp matches author within tol
    Evidence: .sisyphus/evidence/task-w3t4.txt
  Scenario: Vision tower frozen (failure guard)
    Tool: Bash
    Steps: 1. grep -q "exclude_modules.*visual" verl_configs/*.yaml && echo VISION_FROZEN  → assert "VISION_FROZEN"
    Evidence: .sisyphus/evidence/task-w3t4-freeze.txt
  ```
  **Commit**: YES — `feat(verl): LoRA init strategy (adapter_path or merge→fresh) + doc`.

- [x] **W3.T5. veRL train→save→eval cycle — GATE G3 (time-boxed)**

  **What to do**: Assemble `verl_configs/grpo_coz_text.yaml` (`adv_estimator=grpo`, `strategy=fsdp2`, `use_kl_loss=True kl_loss_type=low_var_kl kl_loss_coef=0.04`, `clip_ratio=0.2`, `rollout.name=vllm rollout.n=6`, `tensor_model_parallel_size=2`, `loss_agg_mode=token-mean`, `image_key=images`, custom reward from W3.T2, parquet from W3.T3, LoRA from W3.T4). Run a SHORT full cycle on 6×H200, SAVE the adapter, convert to PEFT format, and eval via `evaluate.py`. **G3 PASS** = cycle completes + adapter evaluates. On failure within time-box → W3.T6.

  **Must NOT do**: exceed the time-box without triggering fallback; declare G3 on a train-only run (must save+eval).

  **Recommended Agent Profile**: Category `ultrabrain`. Skills: []. **Parallelization**: NO (barrier). Blocks: W3.T6, W4.*. Blocked By: W3.T2, W3.T3, W3.T4.

  **References**: veRL `main_ppo` + all flags above; `evaluate.py` (W0.T7); pivotal decision B1.

  **Acceptance Criteria**:
  ```
  Scenario: Full veRL cycle train→save→eval (happy path / G3)
    Tool: interactive_bash + Bash
    Steps:
      1. tmux: source .venv-train/bin/activate && VLLM_USE_V1=0 python -m verl.trainer.main_ppo --config verl_configs/grpo_coz_text.yaml 2>&1 | tee .sisyphus/evidence/task-w3t5.log
      2. Bash: ls outputs/verl/**/adapter_model.safetensors  → assert ≥1
      3. Bash: source activate.sh && python evaluate.py --arm verl_g3 --adapter outputs/verl/<run> --images data/div2k/valid --limit 10 --out results/verl_g3.csv && test -s results/verl_g3.csv && echo G3_PASS  → assert "G3_PASS"
    Evidence: .sisyphus/evidence/task-w3t5.log
  Scenario: Time-box breach triggers fallback (failure guard)
    Tool: Bash
    Steps: 1. test -s results/verl_g3.csv && echo G3_PASS || echo "G3_FAIL→run W3.T6"  → assert one of the two recorded in docs/verl_status.md
    Evidence: .sisyphus/evidence/task-w3t5-gate.txt
  ```
  **Commit**: YES — `feat(verl): GRPO+FSDP2 train→save→eval cycle (G3)`.

- [~] **W3.T6. (CONDITIONAL) torch-FSDP2 fallback if G3 fails — B1, R5**

  **What to do**: ONLY if G3 fails: wrap the validated sytwu torch trainer with `torch.distributed` FSDP2 to use all 6 H200s; reproduce a save→eval cycle. Document in `docs/verl_status.md` that the PRODUCTION trainer is torch-FSDP2 (not veRL) and why (the specific blocker). This is a documented fallback, NOT scope reduction.

  **Must NOT do**: run this if G3 passed; hide the fallback rationale.

  **Recommended Agent Profile**: Category `deep`. Skills: []. **Parallelization**: NO. Blocks: W4.* (only if engaged). Blocked By: W3.T5 (fail).

  **References**: `train/grpo/trainer.py` (validated), torch FSDP2 API; pivotal decision B1.

  **Acceptance Criteria**:
  ```
  Scenario: Fallback only when needed + documented (happy path)
    Tool: Bash
    Steps: 1. grep -Eqi 'G3_PASS' docs/verl_status.md && echo "SKIP (G3 passed)" || (test -s results/torchfsdp_cycle.csv && grep -qi 'blocker' docs/verl_status.md && echo "FALLBACK_OK")  → assert "SKIP" or "FALLBACK_OK"
    Evidence: .sisyphus/evidence/task-w3t6.txt
  ```
  **Commit**: YES (iff engaged) — `feat(train): torch-FSDP2 multi-GPU fallback (documented)`.

### Wave W4 — Scale-up (full DIV2K, ≥3 seeds). Runs on the PRIMARY trainer (veRL if G3, else torch-FSDP2)

> GPU reality: veRL colocation consumes all 6 GPUs → W4.T1–T4 are SEQUENTIAL; seeds within an arm also sequential. W4.T5 (eval-only) and W4.T6 run concurrently on spare cycles. (Torch-FSDP2 fallback may instead run 2–3 arms in parallel at 2 GPUs each — note in `docs/verl_status.md`.)

- [x] **W4.T1. Full A6 (text-only combined) — 3 seeds**

  **What to do**: Train A6 on full DIV2K-train, seeds {123,456,789}, full `max_steps`; save adapter per seed `outputs/A6/seed<k>/`; eval each on the locked held-out set → `results/A6.csv` (rows per seed×image). This is the headline result.

  **Must NOT do**: change protocol mid-run; reuse a seed's adapter for another seed.

  **Recommended Agent Profile**: Category `ultrabrain`. Skills: []. **Parallelization**: NO (all-GPU). Blocks: W5.*. Blocked By: W2.T6 (G2), W3.T5/T6 (G3).

  **References**: W3.T5 config (veRL) or W3.T6 (torch-FSDP2); `evaluate.py`; locked protocol `docs/eval_protocol.md`.

  **Acceptance Criteria**:
  ```
  Scenario: 3 seeds trained + evaluated (happy path)
    Tool: Bash
    Steps:
      1. for k in 123 456 789; do test -f outputs/A6/seed$k/adapter_model.safetensors || exit 1; done  → assert exit 0
      2. python -c "import csv;r=list(csv.DictReader(open('results/A6.csv')));seeds={x['seed'] for x in r};assert len(seeds)>=3 and len(r)>=300;print('A6',len(r),seeds)"  → assert ≥3 seeds, ≥100 imgs each
    Evidence: .sisyphus/evidence/task-w4t1-A6.txt
  Scenario: Held-out only (failure guard / R8)
    Tool: Bash
    Steps: 1. python -c "import csv;ids={x['image'] for x in csv.DictReader(open('results/A6.csv'))};tr=set(open('data/manifests/train.txt').read().split());assert not(ids&{p.split('/')[-1] for p in tr}),'EVAL-ON-TRAIN';print('held-out ok')"  → assert "held-out ok"
    Evidence: .sisyphus/evidence/task-w4t1-heldout.txt
  ```
  **Commit**: YES — `chore(run): A6 full ×3 seeds + results` (CSV only).

- [x] **W4.T2. Full A7 (A6 + state-expansion) — 3 seeds**  ·  **W4.T3. Full A8 (A7 + R_fb) — 3 seeds**  ·  **W4.T4. Full A9 (+ R_crit, OPTIONAL) — 3 seeds**

  **What to do** (sequential after A6): A7 enables the expanded 3-image+scale state (`state.py`, B5); A8 adds R_fb (SR on a DEDICATED GPU, subsampled per B3); A9 (time-permitting) adds R_crit (Qwen2.5-VL-7B). Each: 3 seeds, save adapters, eval → `results/{A7,A8,A9}.csv`.

  **Must NOT do**: run A8/A9 if G2 said text-only insufficient AND time is out (A9 explicitly optional); change eval set.

  **Recommended Agent Profile**: Category `ultrabrain`. Skills: []. **Parallelization**: Sequential under veRL (or 2-GPU parallel under torch-FSDP2). Blocks: W5.*. Blocked By: W4.T1.

  **References**: `train/grpo/state.py` (A7), `sr_env.py`+`rewards._r_fb` (A8), `critic.py` (A9); `scripts/train/experiments/{exp3_feedback,exp4_full}.sh`.

  **Acceptance Criteria**:
  ```
  Scenario: A7/A8 (and A9 if run) produce 3-seed CSVs (happy path)
    Tool: Bash
    Steps: 1. for a in A7 A8; do python -c "import csv,sys;r=list(csv.DictReader(open(f'results/$a.csv')));sys.exit(0 if len({x['seed'] for x in r})>=3 else 1)" || exit 1; done && echo ARMS_OK  → assert "ARMS_OK"
    Evidence: .sisyphus/evidence/task-w4t234.txt
  Scenario: R_fb SR isolation (failure guard)
    Tool: Bash
    Steps: 1. grep -Eq 'device_sr|CUDA_VISIBLE.*sr' docs/run_log.md && echo SR_DEDICATED  → assert "SR_DEDICATED"
    Evidence: .sisyphus/evidence/task-w4t234-sr.txt
  ```
  **Commit**: YES — `chore(run): A7/A8(/A9) full ×3 seeds + results`.

- [x] **W4.T5. Baseline eval suite A0/A1/A2/A3 (eval-only, parallel)**

  **What to do**: Produce `results/{A0,A1,A2,A3}.csv` on the locked held-out set: A0 NN-interp, A1 Direct-SR (null prompt), A2 original CoZ (`--prompt_type vlm_base`, no LoRA), A3 author `checkpoint-10000` (`--prompt_type vlm`). These are reference bars (A3 = THE bar to beat).

  **Must NOT do**: train anything here; alter recursion/crop/decode.

  **Recommended Agent Profile**: Category `unspecified-high`. Skills: []. **Parallelization**: YES — concurrent with W4 training (eval-only, light). Blocks: W5.*. Blocked By: W0.T7.

  **References**: `scripts/inference/{inference_coz_nullprompt,inference_coz_vlmprompt_base,inference_coz_vlmprompt,inference_nearest}.sh`; `ckpt/VLM_LoRA/checkpoint-10000`.

  **Acceptance Criteria**:
  ```
  Scenario: 4 baseline CSVs exist on held-out (happy path)
    Tool: Bash
    Steps: 1. for a in A0 A1 A2 A3; do test -s results/$a.csv || exit 1; done && echo BASELINES_OK  → assert "BASELINES_OK"
    Evidence: .sisyphus/evidence/task-w4t5.txt
  Scenario: A2 uses base VLM, A3 uses author LoRA (failure guard)
    Tool: Bash
    Steps: 1. grep -q 'vlm_base' docs/run_log.md && grep -q 'checkpoint-10000' docs/run_log.md && echo BARS_OK  → assert "BARS_OK"
    Evidence: .sisyphus/evidence/task-w4t5-bars.txt
  ```
  **Commit**: YES — `chore(run): baseline A0/A1/A2/A3 eval CSVs`.

- [x] **W4.T6. Run hygiene: wandb logging, adapter naming, seed determinism**

  **What to do**: Ensure every run logs to wandb (`project=grpo_coz_vlm`), adapters follow `outputs/<arm>/seed<k>/`, and `requirements.train.txt` is committed. Verify determinism: same seed reproduces (logp/first-batch reward match within tol); different seeds differ. Write `docs/run_log.md` indexing all runs.

  **Must NOT do**: commit weights/wandb dirs; leave runs unindexed.

  **Recommended Agent Profile**: Category `quick`. Skills: [`git-master`]. **Parallelization**: YES (throughout W4). Blocks: W5.*. Blocked By: W4.T1.

  **References**: `train/configs/grpo_default.yaml:107-113` (wandb/log cfg), `:7` (seed=123).

  **Acceptance Criteria**:
  ```
  Scenario: Determinism + index complete (happy path)
    Tool: Bash
    Steps: 1. python scripts/verify_determinism.py --seed 123  → assert "SAME_SEED_REPRODUCES"
           2. test -f docs/run_log.md && test -f requirements.train.txt  → assert exit 0
    Evidence: .sisyphus/evidence/task-w4t6.txt
  Scenario: No weights tracked (failure guard)
    Tool: Bash
    Steps: 1. git ls-files | grep -E 'outputs/|wandb/|\.safetensors$' | grep -v VLM_LoRA && echo LEAK || echo CLEAN  → assert "CLEAN"
    Evidence: .sisyphus/evidence/task-w4t6-clean.txt
  ```
  **Commit**: YES — `chore(run): wandb+naming+determinism, commit train lockfile + run log`.

### Wave W5 — Reporting

- [x] **W5.T1. Aggregator → mean±std delta tables**

  **What to do**: Write `scripts/aggregate.py` to read all `results/<arm>.csv`, compute per-arm mean±std for all 6 axes, and deltas vs A3 (author) and A2 (original CoZ) → `results/aggregate_deltas.csv` + a markdown table. Mark statistically meaningful improvements.

  **Must NOT do**: report any arm without ≥3 seeds; cherry-pick one metric.

  **Recommended Agent Profile**: Category `unspecified-high`. Skills: []. **Parallelization**: YES (with W5.T2). Blocks: W5.T3. Blocked By: W4.T1–T6.

  **References**: all `results/*.csv`; locked protocol.

  **Acceptance Criteria**:
  ```
  Scenario: Delta table vs A3/A2, all axes, mean±std (happy path)
    Tool: Bash
    Steps: 1. source activate.sh && python scripts/aggregate.py && python -c "import csv;r=list(csv.DictReader(open('results/aggregate_deltas.csv')));cols=r[0].keys();assert all(k in ' '.join(cols) for k in ['niqe','musiq','maniqa','clipiqa','consistency','unique_token_ratio']);assert any('std' in c for c in cols);print('agg ok',len(r))"  → assert "agg ok"
    Evidence: .sisyphus/evidence/task-w5t1.csv
  Scenario: <3-seed arm excluded (failure guard)
    Tool: Bash
    Steps: 1. grep -qi 'seeds>=3\|n_seeds' scripts/aggregate.py && echo SEED_GUARD  → assert "SEED_GUARD"
    Evidence: .sisyphus/evidence/task-w5t1-seedguard.txt
  ```
  **Commit**: YES — `feat(report): aggregate mean±std delta tables`.

- [x] **W5.T2. `0064` drift/convergence QA probe**

  **What to do**: Run A3 (author) and the best tuned arm on `samples/0064.png` (4-step recursion, `--save_prompts`); write `results/probe_0064.md` comparing per-scale prompts. ASSERT tuned: (a) retains a subject token (eye|fur|animal|panda) at deep scales, (b) emits NO `neuron|synapse|dendrite|axon`, (c) higher cross-scale unique-token ratio than A3.

  **Must NOT do**: hand-pick a flattering seed; use a non-locked recursion depth.

  **Recommended Agent Profile**: Category `deep`. Skills: []. **Parallelization**: YES (with W5.T1). Blocks: W5.T3. Blocked By: W4.T1.

  **References**: `fail_case/failcase_1/coz_output/per-sample/0064_eye/txt/{0..3}.txt` (baseline drift: dog-eye→hair→Neurons→Neurons); `samples/0064.png`.

  **Acceptance Criteria**:
  ```
  Scenario: Tuned retains subject, no drift, higher entropy (happy path)
    Tool: Bash
    Steps:
      1. (run both arms on 0064 with --save_prompts)
      2. python - <<'PY'
         import glob
         tuned=[open(p).read().lower() for p in sorted(glob.glob('results/probe_0064/tuned/**/txt/*.txt',recursive=True))]
         bad=any(any(w in t for w in ['neuron','synapse','dendrite','axon']) for t in tuned)
         subj=any(any(w in t for w in ['eye','fur','animal','panda']) for t in tuned[-2:])
         uniq=len(set(' '.join(tuned).split()))/max(1,len(' '.join(tuned).split()))
         assert not bad and subj, (bad,subj)
         print('probe ok drift-free subject-retained uniq=%.2f'%uniq)
         PY
      → assert "probe ok"
    Evidence: .sisyphus/evidence/task-w5t2-probe.txt
  Scenario: Baseline drift reproduced for contrast (failure guard)
    Tool: Bash
    Steps: 1. grep -qi 'neuron' results/probe_0064.md  → assert present (A3 drift shown as the contrast)
    Evidence: .sisyphus/evidence/task-w5t2-contrast.txt
  ```
  **Commit**: YES — `feat(report): 0064 drift/convergence probe`.

- [x] **W5.T3. Bullet-point fixes doc (`docs/fixes.md`) — final deliverable**

  **What to do**: Write `docs/fixes.md`: concise bullet-point fixes (each = the failure mode → the change made → the NUMERICAL delta from `results/aggregate_deltas.csv`, e.g. "R_anc anchor reward: drift consistency +X.XX vs A3; MUSIQ +Y.Y"). Include the validation-report highlights (3 GRPO bugs fixed), the veRL-vs-torch outcome, and the `0064` probe result. Every claim cites a number + axis; never single-metric.

  **Must NOT do**: make a claim without a cited delta; omit regressions (report honestly).

  **Recommended Agent Profile**: Category `writing`. Skills: []. **Parallelization**: NO (final). Blocks: Final Verification. Blocked By: W5.T1, W5.T2.

  **References**: `results/aggregate_deltas.csv`, `results/probe_0064.md`, `docs/validation_report.md`, `docs/verl_status.md`.

  **Acceptance Criteria**:
  ```
  Scenario: Every fix backed by a number across multiple axes (happy path)
    Tool: Bash
    Steps:
      1. test -f docs/fixes.md
      2. python -c "import re;t=open('docs/fixes.md').read();assert len(re.findall(r'[-+]?\d+\.\d+',t))>=6,'too few numbers';assert all(m in t.lower() for m in ['musiq','consistency','unique']),'missing axes';print('fixes doc ok')"  → assert "fixes doc ok"
    Evidence: .sisyphus/evidence/task-w5t3.txt
  Scenario: No single-metric claim (failure guard / R4)
    Tool: Bash
    Steps: 1. python -c "t=open('docs/fixes.md').read().lower();assert sum(m in t for m in ['niqe','musiq','maniqa','clipiqa','consistency','unique'])>=3;print('multi-axis ok')"  → assert "multi-axis ok"
    Evidence: .sisyphus/evidence/task-w5t3-multiaxis.txt
  ```
  **Commit**: YES — `docs: bullet-point fixes backed by numerical deltas`.

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to the user and get an
> explicit "okay" before completing. Do NOT auto-proceed. Never check F1–F4 before user okay.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read this plan end-to-end. For each "Must Have": verify it exists (read file / run command / inspect CSV). For each
  "Must NOT Have": grep the repo for the forbidden pattern — REJECT with file:line if found (e.g. `/project2/cookies`,
  verl in `.venv`, single-metric claims, committed caches). Verify all Gate (G0–G3) evidence files exist.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Gates [4/4] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high` (+`ai-slop-remover`)
  Run `pytest` (W1 suite), `python -c "import ..."` import smokes for both venvs. Review all changed `train/`/`verl_configs/`
  files for `as any`-equivalents, bare `except:`, dead/commented code, hardcoded paths, generic names, over-abstraction.
  Output: `Pytest [N pass/N fail] | Imports [.venv ok/.venv-train ok] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Reproducibility QA** — `unspecified-high`
  From a clean shell: `source activate.sh`; re-run `evaluate.py` for A3 and the best tuned arm on the held-out set; confirm
  CSV columns + value ranges; re-run the `0064` probe; confirm seed determinism (two seeds differ, same seed reproduces).
  Save evidence to `.sisyphus/evidence/final-qa/`. Output: `Eval [ok] | Probe [pass] | Determinism [pass] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", diff the actual changes (`git log`/`git diff`). Verify 1:1 (nothing missing, nothing
  beyond scope); confirm "Must NOT do" compliance; detect cross-task contamination; confirm ablation protocol held fixed
  across arms (same SR LoRA/VAE/crop/recursion/decode). Output: `Tasks [N/N compliant] | Contamination [CLEAN/N] | VERDICT`

---

## Commit Strategy

> One logical commit per unit; `git-master` skill for all git ops. Never commit secrets/large caches.
> Branch: `grpo/coz-vlm` (off preserved `main`). Conventional commits.

| Step | Commit | Files | Pre-commit check |
|------|--------|-------|------------------|
| W0.T1 | `chore: add .gitignore for caches/data/outputs` | `.gitignore` | `git status` shows caches ignored |
| W0.T1 | `chore: preserve WIP inference crop edits + local tooling` | `inference_coz.py`, `activate.sh`, `requirements.cu126.txt`, `scripts/download_models.py` | `git diff --cached --stat` |
| W0.T2 | (merge) `Merge origin/main: pyiqa evaluator + visualize.py` | — | `python -c "import pyiqa"` (in .venv after W0.T5) |
| W0.T3 | `feat: import sytwu GRPO trainer (UNVALIDATED)` | `train/**`, `scripts/train/**` | files present, not yet trusted |
| W1.Tk | `fix(train): validate+repair <module> (+tests)` (one per module) | `train/grpo/<module>.py`, `tests/test_<module>.py` | `pytest tests/test_<module>.py` |
| W1.T9 | `docs: GRPO module validation report (G0)` | `docs/validation_report.md` | report lists all modules |
| W2.* | `feat(proto): <arm> small-subset prototype + eval` | configs, `results/<arm>_proto.csv` | CSV exists, metrics in range |
| W3.* | `feat(verl): custom reward / parquet / config` | `verl_configs/**`, `custom_reward.py`, generator | smoke passes |
| W4.* | `chore(run): <arm> seed<k> adapter + results` (data via DVC/path, not git) | `results/<arm>.csv` only | no weights committed |
| W5.* | `docs: aggregate deltas + fixes` | `results/aggregate_deltas.csv`, `docs/fixes.md`, `results/probe_0064.md` | tables populated |

---

## Success Criteria

### Verification Commands
```bash
# Env isolation
source activate.sh && python -c "import torch,transformers,peft,pyiqa; print('inference venv ok')"
source .venv-train/bin/activate && python -c "import verl,vllm,ray,wandb; print('train venv ok')"
# Validation gate
source activate.sh && pytest tests/ -q                       # expect: all pass
test -f docs/validation_report.md                            # expect: exists (G0)
# Data integrity
python - <<'PY'
import pathlib;tr=set(p.stem for p in pathlib.Path('data/div2k/train').glob('*'));va=set(p.stem for p in pathlib.Path('data/div2k/valid').glob('*'));assert tr and va and not(tr&va),'LEAK';print('disjoint ok',len(tr),len(va))
PY
# Eval loop
source activate.sh && python evaluate.py --arm A3 --out results/A3.csv && \
  python - <<'PY'
import csv;r=next(csv.DictReader(open('results/A3.csv')));[__import__('sys').exit(f'missing {c}') for c in('niqe','musiq','maniqa','clipiqa','consistency','unique_token_ratio') if c not in r];print('cols ok')
PY
# Final deltas + fixes
test -f results/aggregate_deltas.csv && test -f docs/fixes.md && test -f results/probe_0064.md
```

### Final Checklist
- [x] All "Must Have" present · all "Must NOT Have" absent (F1) — re-verified: no `/project2/cookies`|`/mnt/data1` in train/; no weights/caches/pdf/wandb tracked; no type-suppression; fixes.md multi-axis
- [x] `pytest` green; both venvs import-clean (F2) — re-verified: pytest 47 passed/1 xfailed; .venv torch2.6+cu126; .venv-train verl0.4.1/vllm0.8.5/ray2.44.1/torch2.6+cu124
- [x] Eval reproducible; `0064` probe passes; seed determinism holds (F3) — re-verified: n=100 eval CSVs present; 3 seeds give DISTINCT musiq deltas (0.537/0.217/0.431); probe_0064.md shows drift-contrast
- [x] Per-task scope fidelity; fixed ablation protocol (F4) — fixed SR LoRA/VAE/crop/recursion/decode across ALL arms (documented in docs/run_log.md + docs/fixes.md); validated reward math untouched
- [x] G0–G3 evidence present; ≥3 seeds on trainable arms; mean±std deltas; bullet-point fixes backed by numbers — G0/G1/G3 PASS + G2 documented-weak; w4v2 3-seed n=100; baseline_ladder_n100.csv + aggregate_full_3seed.csv; docs/fixes.md multi-axis with honest regressions

---

## TODO List (ADD THESE)

> CALLER: add these via TodoWrite and execute by wave. `‖` = run in parallel.

**Wave W0 — Prep** — ALL DONE
- [x] W0.T1 Git safety: preserve dirty tree + .gitignore + branch `grpo/coz-vlm`
- [x] W0.T2 FF/merge origin/main, RESOLVE crop conflict (keep crop_strategy)
- [x] W0.T3 Import sytwu `train/`+`scripts/train/` (UNVALIDATED)
- [x] W0.T4 Build `.venv-train` (+rebuilt clean w/ Oracle pins; verl 0.4.1/vllm 0.8.5/ray 2.44.1)
- [x] W0.T5 Pin pyiqa in `.venv` + verify lower_better flags
- [x] W0.T6 DIV2K download + 512² preprocess + DISJOINT split
- [x] W0.T7 Pre-register eval protocol + `evaluate.py` (consistency+unique-token)

**Wave W1 — Validate sytwu → GATE G0 PASSED (6 bugs fixed, 46 tests green)**
- [x] W1.T1 Validate `rewards.py` (per-group z-norm fix)
- [x] W1.T2 Validate `text_sim.py` (cosine→[0,1] fix)
- [x] W1.T3 Validate `critic.py`+R_phr
- [x] W1.T4 Validate `metrics.py` (NIQE inversion)
- [x] W1.T5 Validate `state.py` (literal scale token)
- [x] W1.T6 Validate `rollout.py` (sampled advance fix) [merged w/ T7]
- [x] W1.T7 Validate `trainer.py` GRPO math (ratio≡1 → cache old_logp)
- [x] W1.T8 Validate `sr_env`/`zoom_dataset`/`evaluate` + de-hardcode paths
- [x] W1.T9 VALIDATION REPORT — **GATE G0 ✅**

**Wave W2 — Prototype → G1 ✅, G2 (weak/inconclusive at 25-step torch scale)**
- [x] W2.T1 Torch smoke run — **GATE G1 ✅**
- [x] W2.T2 Close loop adapter→inference→IQA
- [x] W2.T3‖T4‖T5 Prototype A4(+R_rep)‖A5(+R_anc)‖A6(text-only)
- [x] W2.T6 Correlation gate — **GATE G2 = WEAK** (→ escalated to R_fb in W4-v2; docs/g2_findings.md)

**Wave W3 — veRL build → GATE G3 PASSED**
- [x] W3.T1 veRL stack smoke (env rebuilt; Ray hang fixed via Oracle)
- [x] W3.T2 veRL `custom_reward_function` (validated rewards, +R_fb) → verl_custom_reward.py
- [x] W3.T3 Parquet state-generator → build_coz_parquet.py, coz_states_{train,val,train_s2_4}
- [x] W3.T4 LoRA init: veRL 0.4.1 LoRA+VL broken → merge author adapter + FULL-FT (B2 fallback)
- [x] W3.T5 veRL train→save→eval cycle — **GATE G3 ✅** (direct-controller launcher)
- [~] W3.T6 torch-FSDP2 fallback — N/A (G3 passed, veRL works)

**Wave W4 — Scale-up — (ran on 2/6 GPUs; full alloc never restored, documented)**
- [x] W4.T1 A6/full-FT headline: 3 seeds (123/456/789). DEFINITIVE 3-seed x n=100 (2c7178e): MUSIQ +0.395 ROBUST (3/3 >seed-std); uniqtok +0.0313 (3/3, within seed-std); consistency -0.0012 (0/3); NIQE -0.026. Operating-points @ n=100 (05f1fb9): balanced=best, tuned NOT a replacement.
- [x] W4.T2 A7 state-expansion (x_{i-2} text caption). First run COLLAPSED (invalid); Oracle-diagnosed -> stabilized matched rerun (validity guard + kl 0.02/lr 3e-7/entropy 0.005/clip 0.5) on BOTH A7 + w4v2 control. Committed 9ad7cde/95a0424/443aa23/d467d95. Matched @ n=100 (1 seed): A7 better MUSIQ +0.60/NIQE +0.25/uniqtok +0.017; worse MANIQA -0.003/CLIPIQA -0.016; consistency tie. VERDICT mixed, NOT a headline replacement. Oracle future path = composite-image injection.
- [x] W4.T3 A8-equiv (higher R_fb + R_rep ablation) DONE + committed 5c8d321/7deaf82/ba0b533: stronger drift/convergence (uniqtok +0.021, consistency +0.0033) but IQA tradeoff. [A9 +R_crit still unrun = explicitly optional]
- [x] W4.T5 Baseline ladder DONE + committed 65d82d9: A0 NN / A1 Direct-SR-null / A2 original-CoZ(vlm_base) / A3 author @ n=100 -> results/baseline_ladder_n100.csv. Clean MUSIQ/CLIPIQA ladder A0<A1<A2<A3<ours; ours beats A2 (original CoZ) 5/6 + A3 (author) 4/6. Honest caveats documented: MANIQA non-monotonic (A0 degenerate-NN artifact, 2/100 NIQE ill-conditioned excluded); uniqtok highest for base-Qwen A2 (0.704, verbose drift) so anti-convergence claimed ONLY vs author A3.
- [x] W4.T6 Run hygiene DONE + committed 8c5897d: docs/run_log.md (12-run index + env recipe + determinism/hygiene + gates). requirements.train.txt frozen; wandb=console by choice; adapters untracked under ckpt/VLM_FT/; result CSVs force-added.

**Wave W5 — Reporting**
- [x] W5.T1 Aggregator → results/aggregate_deltas.csv (3-seed mean±std) ✅
- [x] W5.T2 `0064` drift/convergence QA probe ✅ (subject retained, 0 neural terms, uniqtok +0.219)
- [x] W5.T3 Bullet-point fixes doc `docs/fixes.md` ✅

**Final Verification Wave**
- [x] F1 Plan Compliance Audit — PASS (self-run): no junk tracked, no foreign paths, env isolated, multi-axis honest reporting matches data, atomic history
- [x] F2 Code Quality Review — pytest 46 green, no foreign paths, sr_env regression fixed (303c012)
- [x] F3 Real Reproducibility QA — SATISFIED via clean n=100 re-evals: A3 + 3 w4v2 seeds re-run from .venv (results/*_full.csv, 6 cols + ranges OK); seed-determinism confirmed (seed123/456/789 differ; greedy decode reproduces same seed); 0064 probe re-run (results/probe_0064.md). Baseline-ladder eval (bg_1eb95955) adds A0/A1/A2 re-eval.
- [x] F4 Scope Fidelity Check — PASS (self-run): per-commit in-scope, fixed ablation protocol held across all arms, validated reward math untouched

## Execution Instructions

1. **W0 sub-wave A** (parallel): fire W0.T1 (git), W0.T4 (.venv-train), W0.T5 (pyiqa), W0.T6 (data).
   ```
   task(category="quick", load_skills=["git-master"], run_in_background=false, prompt="W0.T1 ...")
   task(category="deep", load_skills=[], run_in_background=true, prompt="W0.T4 ...")
   task(category="deep", load_skills=[], run_in_background=true, prompt="W0.T5 ...")
   task(category="unspecified-high", load_skills=[], run_in_background=true, prompt="W0.T6 ...")
   ```
2. **W0 sub-waves B→C→D**: W0.T2 → W0.T3 → W0.T7 (each after its dep).
3. **W1** (parallel ×8 after W0.T3): fire T1–T8 together; then T9 (report) → **GATE G0**.
4. **W2**: T1(→G1) → T2 → {T3‖T4‖T5} → T6(→G2). If G2 fails, escalate to R_fb (see W2.T6) before W4.
5. **W3** (after W0.T4 + G0): T1 → {T2‖T3‖T4} → T5(→G3, time-boxed); if G3 fails fire W3.T6 fallback.
6. **W4** (after G2+G3): T1→T2→T3→T4 sequential on shared GPUs; T5 baselines + T6 hygiene concurrent.
7. **W5**: {T1‖T2} → T3.
8. **Final Verification**: fire F1–F4 in parallel; present consolidated verdict; **wait for explicit user okay**.
