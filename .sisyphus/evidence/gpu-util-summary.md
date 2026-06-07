# GPU utilization summary — W4-v2 seed789 full-6 attempt

Date: 2026-06-01
Host/job checked: `hgpn44`, Slurm job `222799` (`2xH200`).

## Allocation status

- Current active allocation exposes only **2 GPUs**, not 6:
  - `nvidia-smi -L` lists GPU 0 and GPU 1 only.
  - `torch.cuda.device_count()` in `.venv-train` returns `2`.
  - Slurm `AllocTRES=...gres/gpu=2`, `TresPerNode=gres/gpu:2`.
- Existing requested full allocation `222830` (`6xH200`) is **PENDING** with `Reason=Priority` and estimated `StartTime=2026-06-03T13:01:15`.
- SSH to the prior 6-GPU host `hgpn40` is denied by `pam_slurm_adopt` because there is no active job on that node.

Because the user's main constraint is **USE ALL 6 GPUs**, seed789 was **not** relaunched in the current 2-GPU allocation; doing so would repeat the slow under-utilized condition the user explicitly rejected.

## Monitor

Started a tmux-launched monitor loop writing `.sisyphus/evidence/gpu-util.log` every 60s, then stopped it after confirming the 6-GPU training blocker to avoid an idle logger on the wrong 2-GPU allocation:

```bash
while true; do
  date >> .sisyphus/evidence/gpu-util.log
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> .sisyphus/evidence/gpu-util.log
  sleep 60
done
```

Observed during seed456 inference/eval cleanup on the current 2-GPU allocation:

```text
Mon Jun  1 18:13:18 CST 2026
0, 0 %, 34600 MiB
1, 0 %, 1 MiB
Mon Jun  1 18:14:21 CST 2026
0, 54 %, 34600 MiB
1, 0 %, 1 MiB
Mon Jun  1 18:15:25 CST 2026
0, 12 %, 34600 MiB
1, 0 %, 1 MiB
Mon Jun  1 18:16:31 CST 2026
0, 0 %, 1 MiB
1, 0 %, 1 MiB
```

No full-6 seed789 training utilization exists yet because the 6-GPU allocation is not active.

## Prepared full-6 launcher

Prepared `.sisyphus/evidence/task_w4v2_seed789_full6.sh` for the next 6-GPU allocation. It uses:

- `.venv-train` only; no `uv run`.
- `trainer.n_gpus_per_node=6`.
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`.
- FSDP2 actor/ref, no LoRA, merged author base.
- `R_rep + R_fb` on `data/parquet/coz_states_train_s2_4.parquet`.
- Shared reward GPU `cuda:5` within the 6-GPU pool.
- `data.train_batch_size=6` and `actor_rollout_ref.actor.ppo_mini_batch_size=6` because veRL asserts equal `DataProto` chunks over 6 workers; the prior 6-GPU attempt failed with `DataProto 4 and chunk 6`.

Expected launch command once inside a 6-GPU allocation:

```bash
cd /work/seanma0627/RL_2026_Final
tmux new-session -d -s w4v2-seed789-full6 "bash .sisyphus/evidence/task_w4v2_seed789_full6.sh"
```

Then poll every ~25-30 min with `tail` + `nvidia-smi`, appending the current GPU snapshot to `.sisyphus/evidence/gpu-util.log`.

## Seed456 cleanup completed

Seed456 final checkpoint was merged and evaluated on the current allocation:

- Merged model: `ckpt/VLM_FT/coz_w4v2_seed456`
- Eval CSV: `results/w4v2_seed456.csv`
- Log: `.sisyphus/evidence/task-w4v2-seed456-merge-eval.log`

Seed456 mean deltas vs A3:

| axis | delta |
|---|---:|
| niqe | +0.1176 |
| musiq | +0.2883 |
| maniqa | +0.0024 |
| clipiqa | +0.0114 |
| consistency | +0.0006 |
| uniqtok | -0.0054 |
