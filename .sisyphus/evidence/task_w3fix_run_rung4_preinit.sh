#!/usr/bin/env bash
set -euo pipefail

cd /work/seanma0627/RL_2026_Final

LOG=.sisyphus/evidence/task-w3fix-rung4.log
exec > >(tee -a "$LOG") 2>&1

on_exit() {
  rc=$?
  echo "=== rung4 preinit end $(date -Is) rc=${rc} ==="
  exit "$rc"
}
trap on_exit EXIT

echo "=== rung4 preinit start $(date -Is) ==="

source .venv-train/bin/activate

export CUDA_VISIBLE_DEVICES=1,3
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export RAY_TMPDIR=/tmp/$USER/ray-manual
export TMPDIR=/tmp/$USER/tmp-manual
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export HF_HOME="$PWD/.hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PWD/.torch_cache"
export PYTHONPATH="$PWD"

timeout 30 ray stop --force || true

if [[ ! -f data/geo3k/train.parquet || ! -f data/geo3k/test.parquet ]]; then
  python .sisyphus/evidence/task_w3t1_make_dataset.py
fi

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

python .sisyphus/evidence/verl_main_ppo_preinit_ray.py \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PWD/.sisyphus/evidence/task_w3t1_reward.py" \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  data.train_files=data/geo3k/train.parquet \
  data.val_files=data/geo3k/test.parquet \
  data.image_key=images \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.trust_remote_code=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_model_len=1152 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.use_torch_compile=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  'trainer.logger=["console"]' \
  trainer.project_name=w3fix \
  trainer.experiment_name=verl_grpo_smoke \
  ray_init.num_cpus=8
