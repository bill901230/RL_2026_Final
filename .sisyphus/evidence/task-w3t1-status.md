# W3.T1 veRL stack smoke status

Verdict: **STACK_BLOCKED** — no GRPO rollout/train step completed.

Update after continuation: a clean minimal Ray 2-GPU actor repro also blocked on
this node before any actor probe returned. This confirms the current blocker is
below veRL/FSDP2/vLLM initialization: Ray can start a local instance, but 1-GPU
actors do not become usable on the filtered `CUDA_VISIBLE_DEVICES=1,3` view.

## Environment/version check

`source .venv-train/bin/activate`:

```text
verl 0.4.1, vllm 0.8.5.post1, ray 2.47.1, torch 2.6.0+cu124
```

Assigned GPU layout only:

```text
GPU 1: NVIDIA H200, 143771 MiB total, final 1 MiB used, 0% util
GPU 3: NVIDIA H200, 143771 MiB total, final 1 MiB used, 0% util
```

## Dataset

Created `data/geo3k/train.parquet` (8 rows) and `data/geo3k/test.parquet` (2 rows) from `data/eval_subset/*.png` with veRL columns: `prompt`, `images`, `data_source`, `reward_model`, `extra_info`.

Validated `RLHFDataset` + Qwen2.5-VL processor on a 2-image row: `multi_modal_data {'image': 2}` and Qwen image tensors were produced. Multi-image **dataset preprocessing works**, but multi-image **vLLM rollout was not reached**.

## Acceptance grep

```text
grep -Eqi 'step|reward|val' .sisyphus/evidence/task-w3t1.log && echo VERL_RUNS -> VERL_RUNS
grep -iqE 'oom|cuda error|traceback|Error' .sisyphus/evidence/task-w3t1.log && echo HAS_ERRORS || echo CLEAN -> HAS_ERRORS
```

## Final blocker

The latest run reached Ray startup, dataset construction, config validation, and dataloader sizing, then blocked during Ray GPU WorkerDict actor creation. Current log shows Ray/gRPC worker abort and then actor workers stuck without GPU allocation:

```text
Total training steps: 1
ASSERTION FAILED: queue.num_items() == 0
*** SIGABRT ... grpc::ServerCompletionQueue::~ServerCompletionQueue()
```

The named WorkerDict actors remained present, but they did not reach FSDP2 model init or vLLM init in the final run; GPU memory stayed ~1-4 MiB on GPUs 1 and 3. Throughput/sec-per-step: **N/A (0 train steps completed)**.

## Minimal Ray GPU actor repro

Added `.sisyphus/evidence/task_w3t1_ray_gpu_actor_repro.py` and ran it from
`.venv-train` with:

```text
CUDA_VISIBLE_DEVICES=1,3 VLLM_USE_V1=0 PYTHONPATH=$PWD/.sisyphus/evidence:$PYTHONPATH
python .sisyphus/evidence/task_w3t1_ray_gpu_actor_repro.py
```

Evidence log: `.sisyphus/evidence/task-w3t1-ray-gpu-actor-repro.log`.

Observed:

```text
driver host=hgpn40 pid=1480046
driver CUDA_VISIBLE_DEVICES=1,3
Started a local Ray instance.
```

No `ACTOR_0`, `ACTOR_1`, or `RAY_GPU_ACTOR_REPRO_OK` line was emitted before
the 100s foreground timeout. A subsequent `ray stop --force` through the Ray CLI
also failed during CLI import with `ValueError: <object object at ...> is not a
valid Sentinel`, so the Ray install/CLI stack itself is suspect in addition to
GPU actor scheduling.

Next retry should first repair or downgrade the Ray dependency stack in
`.venv-train` and verify this minimal actor repro completes. Only after that
passes should the veRL smoke be retried.

## Integration issues surfaced

1. `.venv-train` had `hydra-core==0.11.3`/`omegaconf==1.4.1`; `python -m verl.trainer.main_ppo` failed immediately with `TypeError: main() got an unexpected keyword argument 'config_name'`. Upgraded to `hydra-core==1.3.2`, `omegaconf==2.3.0` inside the isolated venv.
2. `datasets==2.14.4` was incompatible with `pyarrow==24.0.0` and `fsspec==2026.4.0`; added local `sitecustomize.py` shims for `PyExtensionType` and `LocalFileSystem.protocol`.
3. `qwen_vl_utils` was missing; added a local image-only shim for file-URI images.
4. `flash_attn` was missing. veRL imports `flash_attn.bert_padding` unconditionally and `fsdp_workers.py` hardcoded `attn_implementation="flash_attention_2"`; added a local `flash_attn.bert_padding` shim and patched veRL FSDP worker config load to `attn_implementation="sdpa"` to avoid building flash-attn.
5. `main_ppo`'s Ray controller actor stayed pending, so a direct-controller runner was used for diagnosis. Ray job-level `runtime_env={env_vars: ...}` also caused GPU actor startup hangs on this node; worker-specific envs were used instead.
6. veRL's register-center wait path timed out; a skip-register RayWorkerGroup path was tried, but final blocking moved to Ray WorkerDict actor creation / gRPC worker abort before rollout.

## Flags/changes from requested command

- `data.max_response_length=128` (smaller smoke).
- Added microbatch flags required by veRL 0.4.1 validation: actor/ref/rollout `*_micro_batch_size_per_gpu=1`.
- `trainer.total_training_steps=1`, `trainer.val_before_train=False`, `trainer.logger=["console"]`, `ray_init.num_cpus=4`.
- `+actor_rollout_ref.rollout.limit_images=2` for multi-image rows.
- `actor_rollout_ref.actor.use_torch_compile=False`, `actor_rollout_ref.ref.use_torch_compile=False`.
- Custom trivial reward: `.sisyphus/evidence/task_w3t1_reward.py`.

LoRA: not enabled (`lora_rank=0`). FSDP2/vLLM: not successfully initialized end-to-end.
