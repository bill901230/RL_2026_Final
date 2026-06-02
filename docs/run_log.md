# GRPO run-log index

Consolidated index of every GRPO training/eval run for the CoZ VLM project (W4.T6 run hygiene). Every number and hash below is traced to a committed source CSV, a launcher under `.sisyphus/evidence/`, or a commit on `grpo/coz-vlm`. For the full multi-axis verdicts and caveats see `docs/fixes.md`; this file is the reference index only.

## Environment

Two isolated venvs (both gitignored, never committed):

- `.venv` (inference / eval): `torch 2.6+cu126`, `pyiqa` for the six IQA axes (NIQE/MUSIQ/MANIQA/CLIPIQA + CLIP consistency + unique-token ratio). No veRL here (isolation check passed).
- `.venv-train` (RL): `verl==0.4.1`, `vllm==0.8.5.post1`, `ray[default]==2.44.1`, `torch==2.6.0` (cu124, `cupy-cuda12x`), `tensordict==0.6.2`, `transformers==4.51.3`, `flash-attn 2.7.4.post1+cu12torch2.6`. Frozen lockfile: `requirements.train.txt` (204 pins, present and tracked).

Working veRL launch recipe (stock `python -m verl.trainer.main_ppo` hangs at the Ray TaskRunner actor):

- Launch via the direct-controller wrapper `.sisyphus/evidence/verl_direct_controller.py`, never `uv run`.
- `RAY_TMPDIR` / `TMPDIR` -> `/tmp/$USER/...` (OFF the NFS `/work` mount; on-NFS Ray temp hangs).
- `umask 022` in every tmux lane (inherited umask strips dir `+x`, breaking `/tmp` temp writes).
- `CUDA_VISIBLE_DEVICES=0,1` for the stabilized 2-GPU runs.
- `RAY_DEDUP_LOGS=0 RAY_USAGE_STATS_ENABLED=0 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0 WANDB_MODE=disabled`.
- Full-parameter GRPO (NO LoRA) on the merged author base `ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged`; `strategy=fsdp2` (actor+ref); `rollout.name=vllm`; `reward_manager=batch`; `custom_reward_function=verl_custom_reward.py:compute_score`. (Qwen2.5-VL LoRA+vLLM weight-sync is broken in veRL 0.4.1, hence full-FT.)

GPU reality (documented constraint, not a silent scope cut): the node was reclaimed from 6 to **2 of 6 H200s** (idx 0,1) partway through. seed123/seed456 trained under the original multi-GPU settings (`n_gpus_per_node=4`, `train_batch_size=4`, `R_fb` on `cuda:5`); seed789 and every later run (tune_rrep, A7, both stabilized cells) ran on 2 GPUs with `train_batch_size=2` / `ppo_mini_batch_size=2` and `R_fb` sharing `cuda:1`. This batch-size mismatch is the standing aggregate caveat.

## Reward-config legend

`R_fb` is `musiq + 0.5*consistency` in all RFB-enabled runs. `lr`/`entropy`/`grad_clip` left blank = veRL 0.4.1 defaults (`lr 1e-6`, `entropy_coeff 0`, `grad_clip 1.0`). Validity guard (`COZ_VALIDITY_GUARD`): a completion is invalid if `<8` model tokens OR `<5` unique content tokens OR `>30%` CJK, forcing raw total `-2.0` and `R_rep=-1.0` before group z-norm.

| cfg | R_anc/R_rep/R_fb/R_phr | guard | kl_loss_coef | lr | entropy | grad_clip |
|---|---|---|---:|---|---:|---:|
| `balanced` | 0.2 / 1.0 / 1.0 / 0.1 | off | 0.001 | default | default | default |
| `hi-Rrep` | 0.2 / 3.0 / 1.25 / 0.1 | off | 0.001 | default | default | default |
| `stable` | 0.2 / 1.0 / 1.0 / 0.1 | ON | 0.02 | 3e-7 | 0.005 | 0.5 |
| `n/a` | eval-only, no training | - | - | - | - | - |

## Run index

All evals: DIV2K-valid `0801-0900` (`n=100`), locked protocol (`recursive_multiscale`, `crop_strategy=center`, `rec_num=4`, `upscale=4`, `process_size=512`, SR LoRA `model_20001.pkl` + VAE `vae_encoder_20001.pt`, SD3-medium, greedy `max_new_tokens=32`). Merged models live under `ckpt/VLM_FT/` (untracked); CSV paths are relative to `results/`. "tracked" marks a force-added CSV on the branch; "untracked" = on disk only (diagnostic).

| arm | seed(s) | cfg | train parquet | steps | merged model | eval CSV | commit | final health | verdict (vs A3 unless noted) |
|---|---|---|---|---:|---|---|---|---|---|
| A3 author | n/a | `n/a` | none | - | `ckpt/VLM_LoRA/checkpoint-10000` (via `--vlm_lora_path`) | `aggregate_full_3seed.csv` + `baseline_ladder_n100.csv` (tracked); raw `A3_full.csv` (untracked) | `75a2e79`, `ba0b533` | eval-only | reference: musiq 50.107, niqe -8.456, uniqtok 0.6182, consistency 0.7922 |
| w4v2 | 123 | `balanced` | `coz_states_train_s2_4.parquet` | 100 | `coz_w4v2` | `aggregate_full_3seed.csv` (tracked); raw `w4v2_full.csv` (untracked) | `d4fe10d`, `2252347`, `2c7178e` | healthy: score std 1.30, R_fb std 14.23, advantage ±1.5..1.9 (nonzero) | strongest seed: musiq +0.537, uniqtok +0.0712; niqe -0.173 |
| w4v2 | 456 | `balanced` | `coz_states_train_s2_4.parquet` | 100 | `coz_w4v2_seed456` | `aggregate_full_3seed.csv` (tracked); raw `w4v2_seed456_full.csv` (untracked) | `8e8ae21`, `2c7178e` | healthy (100/100) | musiq +0.217, uniqtok +0.0107 |
| w4v2 | 789 | `balanced` | `coz_states_train_s2_4.parquet` | 100 | `coz_w4v2_seed789` | `aggregate_full_3seed.csv` (tracked); raw `w4v2_seed789_full.csv` (untracked) | `8e8ae21`, `2c7178e` | healthy (100/100); 2-GPU `bs=2` caveat | musiq +0.431, uniqtok +0.0119 |
| **w4v2 (3-seed headline)** | 123/456/789 | `balanced` | `coz_states_train_s2_4.parquet` | 100 ea. | three above | `aggregate_full_3seed.csv` (tracked) | `2c7178e` | all 3 non-collapsed | **MUSIQ +0.395 robust (3/3 and >seed-std 0.163); uniqtok +0.0313 (3/3, within std); niqe -0.026, consistency -0.0012 regress. The defensible result.** |
| tune_rrep | 123 | `hi-Rrep` | `coz_states_train_s2_4.parquet` | 100 | `coz_tune_rrep` | `operating_points_n100.csv` + `tune_rrep_full.csv` (tracked) | `5c8d321`, `7deaf82`, `05f1fb9` | healthy (100/100) | operating point: recovers niqe +0.041 / consistency +0.0030, but musiq -0.294 and uniqtok +0.016 < balanced. Not a replacement (1 seed) |
| A7 state-exp | 123 | `balanced` | `coz_states_train_s2_4_expanded.parquet` | 100 | `coz_A7_stateexp` | `A7_stateexp_comparison.csv` (untracked, diagnostic) | `9ad7cde` (code; run not committed) | **COLLAPSED**: resp_len 45.1->3.3, entropy 2.16->0.34, kl 0.003->8.4, grad_norm 130-172 | **INVALID**: reward-hacked to near-empty `新规发育`; measures a broken model, not the hypothesis. Diagnostic history only |
| w4v2-stable | 123 | `stable` | `coz_states_train_s2_4.parquet` | 100 | `coz_w4v2_stable` | `A7_stable_matched_comparison.csv` + `w4v2_stable_full.csv` (tracked) | `95a0424`, `443aa23`, `d467d95` | healthy: resp_len 48.1, entropy 1.89, kl 0.094 | matched control for A7-stable: musiq +0.513 vs A3 |
| A7-stable | 123 | `stable` | `coz_states_train_s2_4_expanded.parquet` | 100 | `coz_A7_stable` | `A7_stable_matched_comparison.csv` + `A7_stable_full.csv` (tracked) | `9ad7cde`, `95a0424`, `443aa23`, `d467d95` | healthy: resp_len 24.3, entropy 2.88, kl 0.269 (20-step smoke passed first) | vs matched w4v2-stable: MIXED. musiq +0.604, niqe +0.252, uniqtok +0.017 up; maniqa -0.003, clipiqa -0.016 down; consistency tied. 1-seed hypothesis signal, not a headline replacement |
| A0 NN | n/a | `n/a` | none | - | none (`--rec_type nearest`) | `baseline_ladder_n100.csv` + `A0_full.csv` (tracked) | `65d82d9` | eval-only | floor: musiq 26.5, niqe -31.9, uniqtok 0. MANIQA 0.445 is highest = no-ref artifact on degenerate NN; niqe over 98/100 (2 ill-conditioned excluded) |
| A1 SR-null | n/a | `n/a` | none | - | none (`prompt_type=null`) | `baseline_ladder_n100.csv` + `A1_full.csv` (tracked) | `65d82d9` | eval-only | SR-only, no prompt: musiq 48.4, uniqtok 0 |
| A2 original CoZ | n/a | `n/a` | none | - | none (stock Qwen2.5-VL, `prompt_type=vlm_base`, no VLM LoRA) | `baseline_ladder_n100.csv` + `A2_full.csv` (tracked) | `65d82d9` | eval-only | original CoZ: musiq 50.07; uniqtok 0.704 (highest = verbose base-Qwen drift). Ours beats A2 on 5/6 axes, loses only uniqtok |

Ladder summary (`baseline_ladder_n100.csv`): MUSIQ and CLIPIQA are clean monotonic `A0 < A1 < A2 < A3 < ours`. Ours (w4v2 3-seed) beats A2 on 5/6 axes and A3 on 4/6; the anti-convergence (uniqtok) claim is made only vs A3, since base-Qwen A2 is the most verbose.

## Determinism and hygiene

- **Seeds give distinct results (confirmed).** `{123, 456, 789}` produce different per-seed means in `aggregate_full_3seed.csv` (e.g. uniqtok delta `+0.0712 / +0.0107 / +0.0119`; musiq `+0.537 / +0.217 / +0.431`). This is real seed variance, not a frozen-output bug, which is why only MUSIQ clears the `delta > seed-std` robustness bar.
- **Eval decode is deterministic.** Greedy VLM prompting (`max_new_tokens=32`, no sampling) under the locked protocol, so eval CSVs are reproducible given a fixed model.
- **wandb disabled by choice.** `WANDB_MODE=disabled` + `trainer.logger=["console"]`; all telemetry lives in `.sisyphus/evidence/*.log`, none phoned out.
- **No weights/caches tracked.** Merged HF models under `ckpt/VLM_FT/` (and the merged author base `ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged`) are untracked, never `git add`-ed. Note: `.gitignore` only pattern-ignores weight globs under `ckpt/{RAM,DAPE,SR_LoRA,SR_VAE}`; `ckpt/VLM_FT/` is kept out by simply not staging it, confirmed by `git status` showing it as the only untracked tree.
- **Result CSVs force-added.** `results/` is gitignored at repo root, so the 13 canonical CSVs + `probe_0064.md` were added with `-f`. Per-image raw seed CSVs (`w4v2_full`, `*_seed456/789_full`, `A3_full`) and the collapsed-A7 evals (`A7_stateexp_*`) are intentionally left untracked as diagnostic-only.
- `/data/`, `.venv*`, `.hf_cache/`, `.torch_cache/`, `.sisyphus/` all gitignored; no datasets, parquet, or HF caches in history.

## Gates

| gate | status | evidence |
|---|---|---|
| G0 (reward-module correctness) | **PASS** | 8 teammate modules validated, 6 bugs fixed, `43 passed, 1 xfailed`. `docs/validation_report.md` (commit `aa2b263`) |
| G1 (training smoke) | **PASS** | torch trainer smoke ran, adapter saved, telemetry live (progress notepad) |
| G2 (text-only prototypes) | **WEAK / INCONCLUSIVE** | A4/A5/A6 text-only 25-step protos within noise; `docs/g2_findings.md` (commit `3abe14b`); raw `results/A{4,5,6}.csv` (untracked). Later explained: saturated `R_anc` gave near-zero advantage |
| G3 (veRL GRPO end-to-end) | **PASS** | direct-controller veRL GRPO step `rc=0`, then W4-v2 `100/100` full-FT steps + ckpt saved; commit `d4fe10d`; logs `.sisyphus/evidence/task-w3fix-rung4.log`, `task-w4v2-train.log`; frozen `requirements.train.txt` |
