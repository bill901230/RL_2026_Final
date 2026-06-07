# CoZ VLM GRPO — Progress Notepad

## ===== CURRENT STATUS (Jun 2) — READ FIRST =====
- BASELINE LADDER (A0/A1/A2) @ n=100 DONE + COMMITTED 65d82d9. Eval-only, locked A3 protocol (recursive_multiscale, center crop, rec_num=4, SR LoRA model_20001+VAE, SD3-medium, greedy, data/div2k/valid 0801-0900). A0=NN (rec_type=nearest) results/A0_full.csv; A1=direct-SR null prompt (recursive_multiscale + prompt_type=null) results/A1_full.csv; A2=ORIGINAL CoZ (base Qwen2.5-VL, prompt_type=vlm_base, NO LoRA) results/A2_full.csv. Ladder results/baseline_ladder_n100.csv + docs/fixes.md "Baseline ladder" subsection. HONEST: ours(w4v2 3-seed) > A2 on 5/6 axes (loses only uniqtok: base-Qwen verbose=0.704 > ours 0.649), > A3 on 4/6 (loses niqe -0.026, consistency -0.0012). MUSIQ/CLIPIQA clean monotonic A0<A1<A2<A3<ours. Caveats: MANIQA non-monotonic (A0 degenerate-NN highest=metric artifact); NIQE ill-conditioned on 2/100 A0 imgs (excluded from A0 niqe mean, 98/100). Launchers .sisyphus/evidence/task_baseline_{A1A0_gpu0,A2_gpu1}.sh + build_baseline_ladder.py + eval_a0_robust.py (niqe-guard). _sr dirs kept (disk 350G free). NOTE: tmux lanes need `umask 022` (inherited umask strips dir +x -> /tmp temp writes fail) + TMPDIR=/tmp/$USER/...
- REWARD-DESIGN HALF: DONE + committed (up to 05f1fb9). Definitive 3-seed x n=100: MUSIQ +0.395 ROBUST (3/3, >seed-std); uniqtok +0.0313 (3/3 but within seed-std); consistency -0.0012 (0/3); NIQE -0.026. Operating-points @ n=100: balanced w4v2 = best; tuned(higher R_rep) recovers NIQE/consistency but loses MUSIQ + NO bigger anti-convergence -> NOT a replacement. docs/fixes.md is accurate (cross-checked vs results/aggregate_full_3seed.csv).
- A7 STATE-EXPANSION (x_{i-2} text caption): RAN but **TRAINING COLLAPSED = INVALID**. response_length 45->3.3, entropy 2.16->0.34, kl_loss 0.003->8.4, grad_norm 130-172; model emits identical degenerate `新规发育` at every scale (results/A7_stateexp_sr/.../txt/*.txt). Cause: reward-hack to near-empty output (R_rep~0 when nothing repeats) + kl_loss_coef=0.001 too weak (same coef survived on w4v2's shorter prompt). TRAINING DATA captions are CLEAN (data/parquet/coz_prev2_captions.json); impl code (build_coz_parquet.py --state expanded, inference_coz.py --vlm_state expanded_text) is SOUND. docs/fixes.md A7 section CORRECTED to "INVALID/collapsed, inconclusive" (NOT a clean negative). NOT yet committed (working tree: M docs/fixes.md, inference_coz.py, build_coz_parquet.py).
- ORACLE VERDICT (DONE, ses_17b15539fffe00VPtH0gXYB0qY): root cause = reward-validity failure (short off-lang phrase escapes R_rep) + weak KL. FIX: (1) validity guard BEFORE z-norm: invalid if <8 resp tokens / <5 unique content tokens / >30% CJK -> raw -2.0 + worst R_rep (degenerate = worst-in-group); (2) kl_loss_coef=0.02; (3) lr=3e-7; (4) entropy_coeff=0.005; (5) grad_clip=0.5. DESIGN: rerun BOTH A7 + w4v2 under SAME stabilized config (matched cells); old runs = diagnostic history. Smoke 20 steps + gate (resp_len>8, entropy>1.0, kl<3, no CJK) before full. GO/NO-GO: free-form text-caption = NO-GO as main path; composite-image (current crop large + x_{i-2} thumbnail in ONE image) = better future path (separate test).
- STABILIZED RERUN DONE + VERIFIED + COMMITTED (9ad7cde state code, 95a0424 validity guard+test, 443aa23 docs, d467d95 result CSVs force-added). Collapse FIXED: validity guard (invalid <8 tok / <5 uniq / >30% CJK -> raw -2.0 + R_rep=-1.0 before z-norm; COZ_VALIDITY_GUARD) + kl_loss_coef=0.02 + lr=3e-7 + entropy_coeff=0.005 + grad_clip=0.5. Smoke PASSED; both full runs HEALTHY (A7-stable resp_len 24.3/entropy 2.88/kl 0.27; w4v2-stable 48.1/1.89/0.09). MATCHED @ n=100 (A7-stable vs w4v2-stable, both stabilized, 1 seed): A7 BETTER musiq +0.60, niqe +0.25, uniqtok +0.017; A7 WORSE maniqa -0.003, clipiqa -0.016; consistency tie. VERDICT = MIXED 1-seed signal, NOT a headline replacement for the 3-seed w4v2. Verified: commits clean (no blobs), docs honest+accurate, CSVs tracked, pytest 47 passed/1 xfailed, lsp clean. ckpt/VLM_FT/coz_A7_stable + coz_w4v2_stable on disk (untracked, intentional).
- PLAN RECONCILED (Jun 2): plan checkboxes ticked to reflect reality — W0/W1/W2 all done, W3.T1-5 done (T6 N/A), W4.T1/T2 done, W5 all done, Final F1-F4 done. Open: W4.T5 (baselines, running), W4.T6 (hygiene, partial), A9 (optional).
- W4.T5 BASELINE LADDER DONE + committed 65d82d9 + verified: A0/A1/A2/A3 @ n=100 -> results/baseline_ladder_n100.csv. Clean MUSIQ/CLIPIQA ladder A0<A1<A2<A3<ours; ours beats A2(original CoZ) 5/6, A3(author) 4/6. Honest caveats in docs/fixes.md: MANIQA non-monotonic (A0 degenerate-NN artifact); uniqtok highest for base-Qwen A2 (0.704 = verbose drift) -> anti-convergence claimed ONLY vs author A3. docs/fixes.md ladder subsection accurate (matches CSV).
- W4.T6 DONE + committed 8c5897d: docs/run_log.md. 
- FINAL VERIFICATION RE-EXECUTED (fresh evidence, this session): F1 guards clean (no foreign paths / no tracked weights-caches-pdf / no type-suppression / fixes.md multi-axis); F2 pytest 47 passed+1 xfailed, BOTH venvs import (.venv torch2.6+cu126; .venv-train verl0.4.1/vllm0.8.5/ray2.44.1/torch2.6+cu124); F3 n=100 CSVs present + seeds distinct (musiq 0.537/0.217/0.431) + 0064 drift-contrast; F4 fixed protocol across arms. All 5 Final-Checklist boxes -> [x].
- PLAN STATUS: ALL checkbox tasks [x]. Only 2 [~] remain = honestly-resolved OUTCOMES not pending work (G2=documented-weak; W3.T6=N/A since G3 passed). Deliverable commits all in (HEAD 8c5897d); working tree clean except intentional untracked ckpt/ dirs. .sisyphus/ (plan+notepad) gitignored -> no commit needed.
- OPTIONAL (not plan checkboxes; available on request): A9 +R_crit; Oracle composite-image state injection; 3-seed A7-stable robustness.
- OPTIONAL FUTURE (not started): (1) Oracle's composite-image state injection (current crop large + x_{i-2} thumbnail in ONE image, prompt ~ w4v2) — recommended state-expansion path; (2) 3-seed A7-stable+w4v2-stable for robust state-expansion claim; (3) A9 +R_crit; (4) W4.T6 docs/run_log.md index. Launchers: .sisyphus/evidence/task_{A7_stateexp,w4v2}_seed123_2gpu_stable.sh.
- A7 launcher: .sisyphus/evidence/task_A7_stateexp_seed123_2gpu.sh (kl_loss_coef=0.001, lr=default ~1e-6, max_prompt_length=2048, rollout.n=6, train_batch=2). A7 merged model (collapsed): ckpt/VLM_FT/coz_A7_stateexp. A7 train log: .sisyphus/evidence/task-A7-stateexp-train.log.
- NEXT after Oracle: (1) stabilized A7 rerun + matched w4v2 control per Oracle; (2) eval@n100; (3) finalize+commit A7 honestly; (4) optional A8 higher-R_fb. NOTE: Glob tool was returning false "No files found" — use Bash ls to verify filesystem.

## !!! BLOCKER (as of Jun 1, HISTORICAL — mostly resolved) + RESUME POINTER !!!
- ENV CHANGED: node now exposes ONLY 2 H200s (idx 0,1), both idle (confirmed via clean-env `nvidia-smi -L` — REAL, not a shell artifact; cluster reclaimed 4). The user's "use all 6" can't be met. BUT 2 GPUs is ENOUGH for the runs (seed123/456 used ~2-3) -> seed789 + 3-seed aggregate are NOT blocked, resuming on 2 GPUs (bg_8119ddf4, tmux+poll). Only A7/A8 multi-arm parallelism + true 6-GPU throughput await full alloc.
- DONE + COMMITTED (intact): G0 validation (6 bugs, 43 tests, docs/validation_report.md), working veRL+FSDP2 full-FT GRPO pipeline (commit d4fe10d), reward-variance fix + R_fb (2252347), inference full-FT override (d8776f0), W5 deliverable docs/fixes.md + results/aggregate_deltas.csv + results/probe_0064.md (62c778d). Result: veRL GRPO beats author A3 on 5/6 axes (musiq+1.54, uniqtok+0.034, consistency+0.0016; niqe-0.18); 0064 probe fixes drift (subject retained, 0 neuron terms, uniqtok 0.472->0.691).
- SEEDS: seed123 (results/w4v2.csv) + seed456 (100/100 done; ckpt checkpoints/w4v2_seed456 — may still need merge+eval) DONE. seed789 frozen at step17 (process died; GPUs idle).
- F2 CODE-QUALITY (NON-GPU) DONE + committed 303c012: fixed regressed sr_env render-contract TEST (W4-v2 device fix made the test STUB stale; prod sr_env.py was correct, stub updated to accept device kwarg, [0,1]/float asserts kept). No real foreign paths in pipeline (only a benign commented /home line in ram/models/utils.py). pytest = 46 passed, 1 xfailed, 0 failed. Tracked pipeline is clean + green.
- TO RESUME (fresh /start-work after 6-GPU alloc restored): (1) finish seed789 (config = seed456 launcher, +n_gpus_per_node=6/TP dividing 6 for util) in tmux+poll; (2) merge+eval seed456/789 -> 3-seed mean±std vs A3 -> update docs/fixes.md; (3) GPU-util note (gpu-util.log monitor); (4) A7 state-exp + A8 higher-R_fb arms; (5) NIQE-recovery reward-weight tuning; (6) Final Verification F1/F3/F4. EXACT veRL invocation/env in '## W4-v2 RESULT' + '## *** GATE G3 PASSED ***' sections below.


## DONE (committed on branch grpo/coz-vlm)
- W0: git safety, merge origin/main (kept crop_strategy), import sytwu train/, .venv-train, pyiqa in .venv, DIV2K 800+100 (disjoint), evaluate.py (6 axes).
- W1 + **G0 PASS**: validated/repaired ALL 8 teammate modules; 6 bugs fixed (ratio≡1 clip no-op, best-of-group advance, global z-norm, cosine[0,1], NIQE inversion, R_phr blacklist). 43 tests green. docs/validation_report.md.
- **G1 PASS**: torch trainer smoke ran, adapter saved, telemetry live.
- A3 author baseline (eval_subset 30 imgs, 0801-0830): niqe -7.6018 musiq 50.272 maniqa 0.3958 clipiqa 0.607 consistency 0.781 uniqtok 0.6234.
- evaluate.py concat-strip fix (a906306).
- W2 prototypes A4/A5/A6 (text-only, 25 steps, train50=0001-0050) trained+evaled. **G2 = WEAK/INCONCLUSIVE** (within noise; consistency↑ direction but clipiqa+uniqtok regressed). docs/g2_findings.md (3abe14b).

## KEY LEARNINGS
- Torch trainer ~200s/step (single-GPU, 16 completions/step x ~4 fwd passes over 3-img seqs, no batching) -> scale-up INFEASIBLE on torch. veRL is mandatory for real results.
- Run long jobs in tmux + poll from main orchestrator (sub-agents hit 30-min inactivity timeout; setsid agents respawn-on-death = chaos). python -u REQUIRED (stdout block-buffered to files hides progress).
- Killing CUDA procs on networked /work -> D-state -> pkill/ps HANG the persistent shell. Guard with `timeout`; GPUs do free eventually.
- evaluate.py: inference_coz writes concat strip <stem>.png INTO per-sample/<stem>/ -> fixed to ignore it + *_input.png.

## BLOCKER: veRL STACK_BLOCKED (W3.T1, 2h smoke)
.venv-train (verl 0.4.1, vllm 0.8.5.post1, ray 2.47.1, torch 2.6.0+cu124) issues:
- dep conflicts: datasets 2.14.4 vs pyarrow 24; fsspec 2026 local-proto; hydra mismatch (fixed 1.3.2/omegaconf 2.3); missing qwen-vl-utils; missing flash-attn (patched ->sdpa).
- **Ray 2.47.1 GPU-actor creation HANGS** (minimal 2x num_gpus=1 actors never return; gRPC abort queue.num_items()==0; `ray stop` errors). This is BELOW veRL.
- Agent left hacky shims + patched installed verl -> .venv-train is messy; REBUILD CLEAN.
- FIX PLAN: Oracle-guided clean .venv-train rebuild w/ pinned compatible versions + resolve Ray GPU-actor hang, then re-run veRL cycle (G3), then scale-up (W4), then report (W5).

## ORACLE REMEDIATION (bg_dec4df20) — IN FLIGHT as bg_0a25b5c3
Root causes of veRL block: (1) Ray 2.47.1 + `uv run` worker hang -> pin ray[default]==2.44.1, launch via venv python NOT `uv run`; (2) click>=8.3 breaks ray CLI -> click==8.2.1; (3) Ray temp on NFS /work hangs -> RAY_TMPDIR/TMPDIR to local /tmp; (4) cgroup ~1 CPU -> num_cpus=0 smoke actors.
PINNED SET: torch 2.6.0(cu124)/tv0.21/ta2.6, vllm 0.8.5.post1, ray[default] 2.44.1, click 8.2.1, grpcio 1.62.1, tensordict 0.6.2, transformers[hf_xet] 4.51.3, accelerate 1.6.0, datasets 3.6.0, pyarrow 19.0.1, fsspec 2025.3.0, numpy 1.26.4, qwen-vl-utils 0.0.11, hydra-core 1.3.2, omegaconf 2.3.0, flash-attn 2.7.4.post1 (prebuilt wheel), verl 0.4.1 (--no-deps).
LADDER (G3): Rung1 ray GPU-actor smoke (MUST pass first) -> Rung2 FSDP2 1-GPU -> Rung3 vLLM 2-GPU -> Rung4 minimal verl GRPO step.
RESUME: if continuing fresh, `git show`/read .sisyphus/evidence/task-w3fix-status.md + task-w3fix-rung{1..4}; requirements.train.txt = frozen pins.

## *** GATE G3 PASSED (bg_0a25b5c3) ***
- Clean .venv-train rebuilt (requirements.train.txt). Resolver quirk: uv can't co-solve vllm0.8.5.post1 + ray[default]2.44.1 -> solve with ray[default]==2.43.0 THEN `uv pip install --reinstall --no-deps ray==2.44.1 click==8.2.1`. Also needed setuptools==75.8.0.
- Ladder: Rung1 ray GPU-actor PASS (orig 30s timeout was torch COLD-START not hang; warm torch in actor __init__ + 240s timeout). Rung2 FSDP2 1-GPU PASS. Rung3 vLLM 2-GPU Qwen2.5-VL PASS (MUST clean sys.path so leftover .sisyphus/evidence/flash_attn shim doesn't shadow installed flash-attn). Rung4 = G3 PASS: full verl GRPO step rc=0 (Training 100% 1/1, global_step 1.0).
- WORKING INVOCATION: stock `python -m verl.trainer.main_ppo` HANGS at Ray TaskRunner actor. Use the DIRECT-CONTROLLER wrapper `.sisyphus/evidence/verl_direct_controller.py`. Smoke flags that worked: avoid ROCR/HIP env, inherit CUDA_VISIBLE_DEVICES=1,3, tensor_model_parallel_size=1, actor.use_kl_loss=False (ref off for smoke), rollout.n=1, 2 images/prompt allowed. For REAL run: turn KL/ref back ON, rollout.n>=4, scale TP + GPUs.
- ENV for any verl run: source .venv-train; CUDA_VISIBLE_DEVICES; RAY_TMPDIR=/tmp/$USER/ray-manual TMPDIR=/tmp/$USER/tmp-manual (OFF /work); RAY_DEDUP_LOGS=0 RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0. NEVER `uv run`.

## W3 WIRING (bg_93b24db4) — built; fixing 1 LoRA bug
- DONE: verl_custom_reward.py (wraps validated train/grpo rewards, text-only, per-group norm; reward LOADS in verl), tests/test_verl_reward.py, scripts/data/build_coz_parquet.py, data/parquet/coz_states_{train,val}.parquet (built), data/parquet/coz_x0_captions.json, LoRA-init decision: veRL 0.4.1 worker ignores exclude_modules + can't load pretrained adapter -> MERGE author adapter into base = ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged, train FRESH r8 LoRA on top (KL ref = author behavior).
- BUG (fixing): tiny real-CoZ GRPO failed `KeyError: blocks.0.mlp.gate_proj.base_layer.weight` in vLLM rollout weight-sync. Cause: target_modules=all-linear wrapped the VISION tower (visual.blocks.*); veRL ignores exclude_modules. FIX = set actor_rollout_ref.model.target_modules to an LLM-ONLY regex (matches model.language_model...q/k/v/o/gate/up/down_proj, EXCLUDES visual/blocks). Config-only, no verl patch. [resumed bg_93b24db4]
- CRITICAL for W4: once the LLM-only target_modules regex confirms a real-CoZ step rc=0 with custom reward firing, that exact regex + the direct-controller invocation + merged-base + CoZ parquet + verl_custom_reward.py = the scale-up recipe.

## *** W3 WIRING COMPLETE — veRL CoZ pipeline WORKS end-to-end (committed d4fe10d) ***
Tiny real-CoZ full-FT GRPO step rc=0: 16 [coz_reward] records, Training 100% 1/1, FSDP2 checkpoint saved (checkpoints/w3wire/coz_real_tiny/global_step_1/actor/model_world_size_2_rank_*.pt). WORKING CMD: `bash .sisyphus/evidence/task_w3wire_run_coz_grpo_direct.sh` (after `ray stop --force`). Config: NO LoRA, model.path=ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged, fsdp2 actor+ref, vllm rollout, rollout.max_model_len=4096, rollout.n=4, custom_reward=verl_custom_reward.py:compute_score, reward_manager=batch.

## W4 ATTEMPT 1 (bg_b8e9f5f4) — ran, reward fired, but NO ckpt saved + agent timed out
- veRL full-FT GRPO ran; [coz_reward] records confirm custom reward fires in the real loop. BUT process died before save_freq -> no W4 checkpoint, no merge, no eval. tmux gone.
- GPU CONTENTION: another user (edison0223, Sana job) now uses ~13GB on ALL 6 GPUs (~130GB free each still). Schedule around this.
- *** REWARD-DESIGN FINDING (key for continuation) ***: raw_r_anc ~0.99-1.0 SATURATED, raw_r_rep=0.0, raw_r_phr=0.0 at scale 1. Anchor reward near-max -> little gradient; r_rep inactive at scale1 (prev_prompt empty). Explains weak G2. FIX before scale-up: (a) rebalance reward weights / use anchor DELTA not absolute, (b) ensure r_rep active across scales 2-4 (needs prev_prompt populated in parquet — verify build_coz_parquet fills prev_prompt for scale>=2), (c) per phased plan add R_fb (SR-in-loop, directly optimizes IQA) — likely NEEDED since text-only proxies are weak/saturated. 
- W4 RE-RUN must: set trainer.save_freq to save a final ckpt at the chosen step count; run actively in tmux+poll (agent died idle); schedule around edison0223.

## *** 3-SEED CONFIRMATION DONE + committed 8e8ae21 (honest) ***
seeds 123/456/789 (seed789 on 2 GPUs, train_batch_size=2 caveat). aggregate_deltas.csv + docs/fixes.md updated (3-seed mean±std). HONEST verdict: 6/6 axes positive-in-MEAN, but only 2 are ROBUST (delta>std, 3/3 seeds): MUSIQ +0.89±0.63 (3/3) and consistency/anti-drift +0.0012±0.0006 (3/3). Others WITHIN NOISE: niqe +0.03±0.19 (2/3), maniqa +0.003±0.005 (2/3), clipiqa +0.006±0.006 (2/3), uniqtok +0.004±0.026 (1/3 — the 1-seed +0.034 did NOT replicate). 0064 probe (qualitative, 1 img) strong: subject retained, 0 neural terms, uniqtok 0.472->0.691. Deliverable does NOT over-claim. W4.T1 (3-seed headline) + W5.T1 (aggregate) DONE.
REMAINING (each ~2h on 2 GPUs): A7 state-expansion arm, A8 higher-R_fb arm, NIQE/diversity reward-weight tuning (to make more axes robust), F1/F3/F4 verification.

## REWARD-TUNING ABLATION DONE + committed 5c8d321/7deaf82 (honest, valuable)
Changed verl_custom_reward.py: COZ_R_REP_WEIGHT 1.0->3.0, COZ_R_FB_WEIGHT 1.0->1.25. seed123, 2 GPUs. results/tune_rrep.csv vs A3: uniqtok +0.021 (0.623->0.644, STRONGER anti-convergence than w4v2's +0.004), consistency +0.0033 (stronger anti-drift); BUT IQA TRADEOFF: niqe -0.139, maniqa -0.0036, clipiqa -0.010, musiq +0.59 (vs w4v2's +0.89). 
=> CLEAN FINDING: higher text-reward weight more strongly fixes PROMPT-level failure modes (drift+convergence) but costs PIXEL-level IQA. Tradeoff documented in docs/fixes.md "Reward-tuning ablation". 1 seed (not robust claim). w4v2-balanced remains the best all-round (robust musiq+consistency); tune_rrep is the "max-anti-convergence" point.
TWO documented operating points now: w4v2 (balanced, 3-seed robust musiq+consistency) + tune_rrep (max drift/convergence fix, IQA tradeoff).
REMAINING (incremental, ~2h each on 2 GPUs): A7 state-expansion, more seeds for tune_rrep, full DIV2K-valid eval for significance, F3 reproducibility (GPU).

## F1 COMPLIANCE AUDIT = PASS (self-run; oracle agent timed out)
Verified: no caches/weights/data/checkpoints tracked (only legit VLM_LoRA adapter); no foreign paths in pipeline; .venv has NO verl (isolation OK); docs/fixes.md multi-axis + honest (mean±std + within-noise/seed caveats, no single-metric over-claim, matches aggregate_deltas.csv); all 5 deliverable docs present; atomic conventional git history (d4fe10d..7deaf82). Deliverable is COMPLIANT + HONEST.

## F4 SCOPE-FIDELITY AUDIT = PASS (self-run)
Per-commit footprint clean (8 commits, each in-scope, zero cross-contamination). FIXED ABLATION PROTOCOL held across ALL arms: A3/w4v2(x3 seeds)/tune_rrep all used same SR LoRA model_20001.pkl + VAE vae_encoder_20001.pt + 30-row eval_subset + recursive_multiscale + greedy. Reward tuning changed ONLY weight constants (4 lines), validated train/grpo/rewards.py math UNTOUCHED.
=> NON-GPU FINAL VERIFICATION COMPLETE: F1 (compliance/integrity) PASS, F2 (code quality, 46 tests green) PASS, F4 (scope fidelity) PASS.

## *** n=100 FULL-VALID RESULT DONE + committed ba0b533 (honest refinement) ***
w4v2(seed123) vs A3 on FULL 100-img valid: uniqtok +0.0712 (0.618->0.689, STRONG anti-CONVERGENCE — the standout, much bigger than n=30's +0.004), musiq +0.54 (holds, smaller than n=30), maniqa +0.0022, clipiqa +0.0079; consistency -0.0012 (FLIPPED negative — n=30 anti-drift signal was subset-dependent, did NOT hold at n=100); niqe -0.173 (regresses, consistent w/ seed123). 4/6 axes positive at n=100. results/{A3_full,w4v2_full}.csv (100 rows), results/full_valid_deltas.csv, docs/fixes.md updated honestly (F3 reproducibility done too).
HONEST HEADLINE (refined): the veRL GRPO finetune ROBUSTLY fixes PROMPT CONVERGENCE (uniqtok +0.071 @ n=100) + modestly improves MUSIQ/MANIQA/CLIPIQA; anti-drift consistency NOT robust (subset-dependent); NIQE regresses. 1 seed @ n=100 (seed123). The 3-seed n=30 robust wins were musiq+consistency; n=100 shows convergence is the durable win, consistency is not. Need multi-seed @ n=100 for final claim.

## *** DEFINITIVE 3-SEED x n=100 RESULT DONE + committed 2c7178e ***
Strict robustness rule (3/3 seeds improve AND delta>seed-std). w4v2 3-seed mean±std vs A3 @ n=100:
- MUSIQ +0.395 (3/3, delta>std) = ONLY STRICTLY-ROBUST WIN.
- unique-token (anti-convergence) +0.0313 (3/3 seeds, seed-unanimous) but delta < seed-std(0.0346) -> directionally consistent, not strictly robust.
- MANIQA +0.0008 (2/3), CLIPIQA +0.0030 (2/3) = small seed-mixed.
- NIQE -0.026 (2/3, slight regression), consistency -0.0012 (0/3, REGRESSES — anti-drift did NOT hold at scale).
HONEST FINAL HEADLINE: veRL GRPO finetune of author ckpt -> robustly improves perceptual MUSIQ (+0.40, 3/3, beyond variance) + seed-unanimously raises prompt diversity/anti-convergence (+0.031, 3/3, within variance); slight NIQE/consistency regression. results/aggregate_full_3seed.csv + docs/fixes.md committed. No over-claim.
=> 3-seed x n=100 DONE. OPERATING-POINTS @ n=100 DONE + committed 05f1fb9: tune_rrep (higher R_rep) recovers NIQE +0.21 + consistency vs balanced BUT loses MUSIQ -0.83 (below A3!) + does NOT give bigger anti-convergence (uniqtok +0.016 < balanced +0.031, n=30 stronger-result was small-sample artifact). VERDICT: balanced w4v2 = best operating point; tuned = NIQE/consistency-vs-MUSIQ stress point, NOT a replacement. Honest negative result. results/operating_points_n100.csv.
=> REWARD-DESIGN half of proposal FULLY characterized. LAST untested proposal idea = EXPANDED STATE (condition on x_{i-2}) = A7.

## A7 STATE-EXPANSION ARM (the proposal's 2nd core idea) — final arm
veRL path (B5): add x_{i-2} as TEXT caption to the prompt (alongside x0-caption + scale token; avoid multi-image vLLM rollout risk). Parquet regen (add prev2_caption) -> train balanced reward (same as w4v2) on expanded-state parquet -> eval vs w4v2/A3 @ n=100. Feasible on 2 GPUs ~3h. Tests whether AR-2 (x_{i-2}) conditioning helps.

## DEFINITIVE 3-SEED x n=100 [DONE above] (bg_d675b8a0)
All 3 merged seed models exist (coz_w4v2/seed456/seed789); disk OK (18%, 3PB free); GPUs idle. Eval seed456+seed789 on FULL 100-img valid (parallel GPU 0,1, eval-only ~40min) -> aggregate w/ seed123_full (results/w4v2_full.csv) vs A3_full -> results/aggregate_full_3seed.csv (per-axis seed-mean+/-seed-std, delta, #seeds-improving) -> docs/fixes.md DEFINITIVE "3-seed x n=100" table (honest: which axes robust 3/3 + delta>std). Expect uniqtok(anti-convergence) strong; consistency was subset-dependent. THIS is the strongest defensible statement.

## F3 + FULL-VALID EVAL (bg_8c3ec149) — n=30 -> n=100 [DONE above]
EVAL-ONLY (feasible on 2 GPUs, ~40-60min). A3 (GPU0) + coz_w4v2 seed123 (GPU1) on FULL data/div2k/valid (100 imgs) -> results/{A3_full,w4v2_full}.csv -> n=100 deltas all 6 axes -> update docs/fixes.md "n=100 full-valid confirmation" + commit. Addresses the n=30 caveat + doubles as F3 reproducibility. Honest: 1 seed (123) on n=100 = larger-sample confirmation of the seed123 point, NOT a new multi-seed claim.

## W5 DELIVERABLE DONE (commits 2252347/d8776f0/62c778d)
docs/fixes.md (concise bullet fixes + numerical deltas + honest caveats), results/aggregate_deltas.csv, results/probe_0064.md. 0064 probe: tuned retains Fur/Animal at deep scales, 0 neuron/synapse terms, uniqtok 0.472->0.691 (+0.219). Working tree clean.

## W4 MULTI-SEED CONFIRM LAUNCHED (bg_5672afb5, ~6h)
Seeds 456+789 of the winning W4-v2 config (full-FT merged base, R_rep+R_fb, s2-4 parquet, ~100 steps) -> merge -> eval on eval_subset -> 3-seed mean+/-std vs A3 -> update docs/fixes.md + commit. Schedules around edison0223 GPU job (tmux + tight poll). Addresses #1 caveat (1 seed -> 3 seeds). seed123 done = results/w4v2.csv.
REMAINING after this: NIQE recovery (reward-weight tuning), A7 state-expansion arm, A8 higher-R_fb arm, Final Verification F1-F4. Each veRL run ~3h.

## *** W4-v2 RESULT: WIN on 5/6 axes (bg_93f50c64, 3h7m) ***
Reward-variance fix WORKED: built coz_states_train_s2_4.parquet (scales 2-4, prev_prompt + crop_path populated), wired R_fb (FrozenSRBackbone SD3 1-step + pyiqa MUSIQ + CLIP consistency) on cuda:5. 100/100 full-FT GRPO steps (NO LoRA), reward std 1.30, R_fb std 14.23, advantage +-1.5..1.9 (NONZERO -> real learning). Ckpt -> merged HF ckpt/VLM_FT/coz_w4v2.
EVAL vs A3 (n=30, 1 seed): musiq 50.27->51.81 (+1.54), uniqtok 0.6234->0.6571 (+0.034 ANTI-CONVERGENCE fix), maniqa +0.0085, clipiqa +0.0078, consistency 0.781->0.7826 (+0.0016 ANTI-DRIFT), niqe -7.60->-7.78 (-0.18 REGRESSION, honest; MUSIQ^/NIQE_v texture tradeoff). 5/6 improved incl BOTH target failure modes.
W5 (bg_6b90ccd5): 0064 probe + docs/fixes.md (deliverable) + aggregate + commit.
REMAINING for full defensibility: >=3 seeds + full DIV2K-valid eval; reward-weight tuning to recover NIQE; A7(state-exp)/A8(more R_fb) arms. Each veRL run ~3h (GPU shared w/ edison0223).

## W4-v2 LAUNCHED (bg_93f50c64) — fix reward variance + add R_fb
ROOT INSIGHT: GRPO advantage=(r-mean)/std; saturated r_anc(~1.0)=zero group variance=zero advantage=NO learning. Prior W4 had no usable gradient. FIX: (1) train scales>=2 with prev_prompt populated (r_rep varies) -> verify/fix build_coz_parquet.py prev_prompt; (2) ADD R_fb (sr_env SD3 one-step + pyiqa) to verl_custom_reward.py -> varies per prompt + directly targets eval IQA; (3) modest r_anc weight. Re-run full-FT W4 with SAVE_FREQ set (prior saved nothing!), tmux+tight-poll, schedule around edison0223 GPU job. Merge shards->ckpt/VLM_FT/coz_w4v2-> inference_coz (--vlm_model_path override for full-FT) -> evaluate.py -> results/w4v2.csv vs A3.

## W4 RECIPE (orig plan):
- Scale proven cmd to 6 GPU, ~100-150 steps, rollout.n=6, text-only combined reward (A6-equiv), save final ckpt -> merge veRL FSDP shards to HF (verl.model_merger) -> ckpt/VLM_FT/coz_a6_verl -> inference_coz (point VLM base at the FT model; full-FT not LoRA) on 30-img eval_subset -> evaluate.py -> results/A6verl.csv -> deltas vs A3 (niqe -7.60/musiq 50.27/maniqa 0.396/clipiqa 0.607/consistency 0.781/uniqtok 0.623).
- EVAL WRINKLE: full-FT model replaces the VLM base in inference_coz (not --vlm_lora_path). Agent adds a --vlm_model_path override (keep crop_strategy intact).
- REMAINING after headline: A7 (state-expansion), A8 (+R_fb), >=3 seeds, aggregate mean+/-std, 0064 probe, docs/fixes.md. All HOURS of GPU compute -> continuation.

## *** DECISIVE PIVOT: LoRA bug -> FULL-PARAM FT on merged base ***
- veRL 0.4.1 LoRA+vLLM+Qwen2.5-VL is BROKEN: FSDP->vLLM weight-sync rewrites VISION-tower keys to `.base_layer.*` (KeyError blocks.0.mlp.gate_proj.base_layer.weight) EVEN with LLM-only target_modules (252 LLM modules, 0 visual wrapped). Not fixable by config; would need patching verl (forbidden).
- RESOLUTION = FULL-PARAMETER GRPO (NO LoRA) on the MERGED author base `ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged`. PROVEN: G3 geo3k smoke passed with lora_rank=0 (full-FT) -> verl full-FT + FSDP2 + vLLM + Qwen2.5-VL WORKS on this node. Full-FT on merged base == "continue finetuning the author checkpoint" (user's actual intent) + FSDP2 (user's ask). 3B full-FT shards fine on 6xH200. KL ref = merged-author behavior.
- WORKING ASSETS (all built): verl_custom_reward.py (compute_score, text-only R_anc+R_rep+R_phr, reward_manager=batch for per-group norm, logs [coz_reward]); data/parquet/coz_states_{train(200rows/50img),val(120rows/30img)}.parquet (disjoint); merged base; direct-controller launcher .sisyphus/evidence/verl_direct_controller.py.
- W4 RECIPE (once tiny full-FT confirm passes): direct-controller + model.path=merged base + NO LoRA + strategy=fsdp2 + rollout.name=vllm + custom_reward_function=verl_custom_reward.py:compute_score + reward_manager=batch + CoZ parquet + KL on. Scale: 6 GPU, more steps, arms A6(text)/A7(state)/A8(+R_fb), >=3 seeds. Eval each adapter via inference_coz + evaluate.py on eval_subset vs A3.
- earlier max_num_seqs<1 error -> set rollout.max_model_len ~4096 + sane batch sizes.

## AFTER G3
- W4 scale-up: port validated reward/state modules as verl custom_reward; run A6/A7/A8 (FSDP2, 6 GPU, many steps, 3 seeds) on DIV2K; eval on held-out.
- W5: scripts/aggregate.py (mean±std deltas vs A3/A2) + 0064 probe + docs/fixes.md.
- FALLBACK (if G3 fails): documented torch-FSDP2 (W3.T6) — but torch ~200s/step is slow; veRL strongly preferred.
