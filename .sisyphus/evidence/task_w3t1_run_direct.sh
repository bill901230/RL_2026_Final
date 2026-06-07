#!/usr/bin/env bash
set -euo pipefail

cd /work/seanma0627/RL_2026_Final
source .venv-train/bin/activate

export CUDA_VISIBLE_DEVICES=1,3
export VLLM_USE_V1=0
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
unset W3T1_PRECREATE_REGISTER
export W3T1_SKIP_REGISTER=1
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/.sisyphus/evidence:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=true
export W3T1_TRACE_DIST=1
unset W3T1_FORCE_SDPA

python .sisyphus/evidence/task_w3t1_direct_ppo.py \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PWD/.sisyphus/evidence/task_w3t1_reward.py" \
  data.train_files=data/geo3k/train.parquet data.val_files=data/geo3k/test.parquet \
  data.image_key=images data.train_batch_size=8 data.max_prompt_length=1024 data.max_response_length=128 \
  data.filter_overlong_prompts=False data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_model_len=1152 actor_rollout_ref.rollout.max_num_seqs=32 \
  +actor_rollout_ref.rollout.limit_images=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 actor_rollout_ref.ref.use_torch_compile=False \
  trainer.nnodes=1 trainer.n_gpus_per_node=2 trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.val_before_train=False \
  'trainer.logger=["console"]' trainer.project_name=w3t1 trainer.experiment_name=verl_smoke_direct \
  ray_init.num_cpus=4
