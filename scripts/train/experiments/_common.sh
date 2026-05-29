# Shared setup sourced by every exp*.sh launcher. Not meant to be run directly.
# Provides: repo-root cwd, the coz-env python (PY), CONFIG, and run_grpo().
#
# HARDWARE PROFILE — pick with the HW env var (default 4090):
#   HW=4090    4x RTX 4090 (24GB each): one model per GPU, memory-safe knobs.
#   HW=a6000   1x RTX A6000 (48GB):     all models on one card, conservative knobs.
#   HW=pro6000 1x RTX PRO 6000 (96GB):  all models on one card, full-speed knobs.
# The profile sets device placement + memory knobs; the exp script only chooses
# which rewards are on. Same exp*.sh therefore runs on any of the three cards:
#   HW=a6000 bash scripts/train/experiments/exp4_full.sh
# Anything is still overridable, e.g.  GPUS=2,3  or  appending key=val to the call.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."            # -> repo root
PY="${PY:-/project2/cookies/miniconda3/envs/coz/bin/python}"   # always the coz env
CONFIG="${CONFIG:-train/configs/grpo_default.yaml}"

run_grpo () {
  local hw="${HW:-4090}"
  local -a HW_ARGS
  case "$hw" in
    4090)      # 24GB/card, one model per GPU. SR experiments (NEED_SR) load 3 models
               # -> 3 cards; the others load only policy+critic -> 2 cards.
      if [[ -n "${NEED_SR:-}" ]]; then
        : "${GPUS:=0,1,2}"
        HW_ARGS=( device_policy=cuda:0 device_sr=cuda:1 device_critic=cuda:2 )
      else
        : "${GPUS:=0,1}"
        HW_ARGS=( device_policy=cuda:0 device_critic=cuda:1 )
      fi
      HW_ARGS+=(
        optim.gradient_checkpointing=true
        generation.group_size=6 generation.sample_micro_batch=2
        rollout.images_per_step=2
      ) ;;
    a6000)     # 48GB, everything on one card; keep checkpointing + small micro-batch
      : "${GPUS:=0}"
      HW_ARGS=(
        device_policy=cuda:0 device_sr=cuda:0 device_critic=cuda:0
        optim.gradient_checkpointing=true
        generation.group_size=6 generation.sample_micro_batch=3
        rollout.images_per_step=2
      ) ;;
    pro6000)   # 96GB, everything on one card; checkpointing off, big batches
      : "${GPUS:=0}"
      HW_ARGS=(
        device_policy=cuda:0 device_sr=cuda:0 device_critic=cuda:0
        optim.gradient_checkpointing=false
        generation.group_size=8 generation.sample_micro_batch=8
        rollout.images_per_step=4
      ) ;;
    *) echo "[exp] unknown HW='$hw' (use 4090 | a6000 | pro6000)" >&2; return 1 ;;
  esac

  echo "[exp] ${NAME} | HW=${hw} | GPUs=${GPUS} | out=experience/grpo_vlm/${NAME}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "$PY" -m train.train_grpo_vlm \
    --config "$CONFIG" --set \
      "${HW_ARGS[@]}" \
      "$@" \
      output_dir="experience/grpo_vlm/${NAME}" \
      logging.run_name="${NAME}"
}
