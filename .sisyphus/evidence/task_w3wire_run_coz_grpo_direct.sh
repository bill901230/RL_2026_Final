#!/usr/bin/env bash
set -euo pipefail

cd /work/seanma0627/RL_2026_Final

LOG=.sisyphus/evidence/task-w3wire-coz-grpo.log
exec > >(tee "$LOG") 2>&1

on_exit() {
  rc=$?
  echo "=== w3wire real-coz grpo end $(date -Is) rc=${rc} ==="
  exit "$rc"
}
trap on_exit EXIT

echo "=== w3wire real-coz grpo start $(date -Is) ==="

source .venv-train/bin/activate

export CUDA_VISIBLE_DEVICES=1,3
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export RAY_TMPDIR=/tmp/$USER/ray-manual
export TMPDIR=/tmp/$USER/tmp-manual
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
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

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

python .sisyphus/evidence/verl_direct_controller.py \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PWD/verl_custom_reward.py" \
  custom_reward_function.name=compute_score \
  reward_model.reward_manager=batch \
  actor_rollout_ref.model.path="$PWD/ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  data.train_files=data/parquet/coz_states_train.parquet \
  data.val_files=data/parquet/coz_states_val.parquet \
  data.image_key=images \
  data.train_batch_size=4 \
  data.max_prompt_length=1536 \
  data.max_response_length=64 \
  data.shuffle=False \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.trust_remote_code=True \
  trainer.balance_batch=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=1 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=200704 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.use_torch_compile=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.default_local_dir=checkpoints/w3wire/coz_real_tiny \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=1 \
  'trainer.logger=["console"]' \
  trainer.project_name=w3wire \
  trainer.experiment_name=coz_real_tiny \
  ray_init.num_cpus=8
