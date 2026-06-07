#!/usr/bin/env bash
set -euo pipefail
umask 022

cd /work/seanma0627/RL_2026_Final

ARM="${COZ_ARM:-${COZ_RUN_TAG:-abl}}"
TOTAL_STEPS="${COZ_TOTAL_STEPS:-30}"
SAVE_FREQ="${COZ_SAVE_FREQ:-30}"
RUN_TAG="${COZ_RUN_TAG:-${ARM}}"
NGPU="${COZ_NGPU:-2}"
CHECKPOINT_DIR="${COZ_CHECKPOINT_DIR:-checkpoints/${ARM}}"
LOG="${COZ_LOG:-.sisyphus/evidence/task-${ARM}-train.log}"
PROJECT_NAME="${COZ_PROJECT_NAME:-coz_ablation}"
EXPERIMENT_NAME="${COZ_EXPERIMENT_NAME:-coz_grpo_${ARM}_seed123_${NGPU}gpu_${RUN_TAG}}"

mkdir -p "$(dirname "$LOG")"
if [[ -f "$LOG" ]]; then
  mv "$LOG" "${LOG}.prev.$(date +%Y%m%dT%H%M%S)"
fi
exec > >(tee "$LOG") 2>&1

on_exit() {
  rc=$?
  echo "=== ablation ${ARM} ${RUN_TAG} seed123 ${NGPU}gpu real-coz grpo end $(date -Is) rc=${rc} ==="
  exit "$rc"
}
trap on_exit EXIT

echo "=== ablation ${ARM} ${RUN_TAG} seed123 ${NGPU}gpu real-coz grpo start $(date -Is) ==="
hostname
echo "TOTAL_STEPS=${TOTAL_STEPS} SAVE_FREQ=${SAVE_FREQ} CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "PROJECT_NAME=${PROJECT_NAME} EXPERIMENT_NAME=${EXPERIMENT_NAME}"

source .venv-train/bin/activate

export CI=true GIT_TERMINAL_PROMPT=0 GIT_EDITOR=: GIT_PAGER=cat PAGER=cat
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/${USER}/ray-${ARM}}"
export TMPDIR="${TMPDIR:-/tmp/${USER}/tmp-${ARM}}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
chmod 700 "$RAY_TMPDIR" "$TMPDIR"

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

# Caller-driven ablation knobs. Defaults are base + phrase reward only.
export COZ_R_ANC_WEIGHT="${COZ_R_ANC_WEIGHT:-0}"
export COZ_R_REP_WEIGHT="${COZ_R_REP_WEIGHT:-0}"
export COZ_R_FB_WEIGHT="${COZ_R_FB_WEIGHT:-1.0}"
export COZ_ENABLE_RFB="${COZ_ENABLE_RFB:-0}"

# Constants across all ablation arms.
export COZ_VALIDITY_GUARD=1
export COZ_R_PHR_WEIGHT=0.1
export COZ_RFB_METRIC=musiq
export COZ_RFB_CONSISTENCY_WEIGHT=0.5

# Device knobs for 1-GPU concurrent runs. With CUDA_VISIBLE_DEVICES set to a
# single physical GPU, cuda:0 maps to that arm's isolated visible device.
export COZ_REWARD_DEVICE="${COZ_REWARD_DEVICE:-cuda:0}"
export COZ_REWARD_SR_DEVICE="${COZ_REWARD_SR_DEVICE:-${COZ_REWARD_DEVICE}}"
export COZ_REWARD_IQA_DEVICE="${COZ_REWARD_IQA_DEVICE:-${COZ_REWARD_DEVICE}}"
export COZ_REWARD_CLIP_DEVICE="${COZ_REWARD_CLIP_DEVICE:-${COZ_REWARD_DEVICE}}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} COZ_NGPU=${NGPU} RAY_TMPDIR=${RAY_TMPDIR} TMPDIR=${TMPDIR}"
echo "REWARD_CFG COZ_R_ANC_WEIGHT=${COZ_R_ANC_WEIGHT} COZ_R_REP_WEIGHT=${COZ_R_REP_WEIGHT} COZ_R_FB_WEIGHT=${COZ_R_FB_WEIGHT} COZ_ENABLE_RFB=${COZ_ENABLE_RFB} COZ_R_PHR_WEIGHT=${COZ_R_PHR_WEIGHT} COZ_VALIDITY_GUARD=${COZ_VALIDITY_GUARD} COZ_RFB_METRIC=${COZ_RFB_METRIC} COZ_RFB_CONSISTENCY_WEIGHT=${COZ_RFB_CONSISTENCY_WEIGHT}"
echo "REWARD_DEVICES COZ_REWARD_DEVICE=${COZ_REWARD_DEVICE} COZ_REWARD_SR_DEVICE=${COZ_REWARD_SR_DEVICE} COZ_REWARD_IQA_DEVICE=${COZ_REWARD_IQA_DEVICE} COZ_REWARD_CLIP_DEVICE=${COZ_REWARD_CLIP_DEVICE}"

# Do not ray stop here: multiple 1-GPU arms may run concurrently.

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader || true

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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.02 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.optim.lr=3e-7 \
  actor_rollout_ref.actor.entropy_coeff=0.005 \
  actor_rollout_ref.actor.grad_clip=0.5 \
  data.train_files=data/parquet/coz_states_train_s2_4.parquet \
  data.val_files=data/parquet/coz_states_val_s2_4.parquet \
  data.image_key=images \
  data.train_batch_size=2 \
  data.max_prompt_length=2048 \
  data.max_response_length=64 \
  data.shuffle=False \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.trust_remote_code=True \
  ++data.seed=123 \
  trainer.balance_batch=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.n=6 \
  ++actor_rollout_ref.rollout.seed=123 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_seqs=24 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=1 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=200704 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.use_torch_compile=False \
  trainer.n_gpus_per_node="${NGPU}" \
  trainer.nnodes=1 \
  trainer.total_epochs=10 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=1 \
  'trainer.logger=["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  ray_init.num_cpus=8 \
  ${COZ_EXTRA_OVERRIDES:-}
