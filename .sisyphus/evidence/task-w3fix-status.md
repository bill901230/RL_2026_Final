# W3 fix status

Final verdict: **G3 PASS**.

## Step 0: node health + backup

PASS. Evidence: `.sisyphus/evidence/task-w3fix-node.txt`.

- GPUs 1 and 3 were selected for the ladder.
- Broken `.venv-train` was moved to `.venv-train.broken.1780061659`.

## Step 1: clean rebuild

PASS. Evidence: `.sisyphus/evidence/task-w3fix-build.txt`, `requirements.train.txt`.

- Built a fresh `.venv-train` without touching `.venv`.
- Verified versions: `torch 2.6.0+cu124`, CUDA `12.4`, `ray 2.44.1`, `vllm 0.8.5.post1`, `click 8.2.1`.
- Used the Ray resolver workaround documented in the build log; no flash-attn source build was attempted.

## Rung 1: Ray GPU-actor smoke

PASS. Evidence: `.sisyphus/evidence/task-w3fix-rung1a.txt`, `.sisyphus/evidence/task-w3fix-rung1b.txt`.

- A1 trivial Ray actor passed.
- A2 warm torch actor passed on distinct visible devices for GPUs 1 and 3 with CUDA available.

## Rung 2: FSDP2 sanity

PASS. Evidence: `.sisyphus/evidence/task-w3fix-rung2.txt`.

- `.sisyphus/evidence/fsdp2_sanity.py` printed `FSDP2_SANITY_OK`.

## Rung 3: 2-GPU vLLM rollout

PASS. Evidence: `.sisyphus/evidence/task-w3fix-rung3.txt`.

- `.sisyphus/evidence/vllm_rollout.py` completed a Qwen2.5-VL rollout across GPUs 1 and 3.

## Rung 4 / GATE G3: minimal veRL GRPO step

PASS. Evidence: `.sisyphus/evidence/task-w3fix-rung4.log`.

- Final successful attempt ended `rc=0` at `2026-05-29T22:58:15+08:00`.
- It reached `Training Progress: 100%|██████████| 1/1`, printed step metrics for `training/global_step:1.000`, and ended with `'Final validation metrics: None'`.
- Runtime fixes used in the evidence wrapper only:
  - direct controller wrapper to avoid the veRL register-center wait path;
  - empty `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` in Ray worker runtime env;
  - inherited `CUDA_VISIBLE_DEVICES=1,3` via `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`;
  - `rollout.tensor_model_parallel_size=1`, `actor.use_kl_loss=False`, `rollout.n=1`, and `+rollout.limit_images=2` for this 2-GPU multimodal smoke.
