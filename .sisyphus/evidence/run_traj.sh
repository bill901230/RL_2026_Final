#!/usr/bin/env bash
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
ARM="${ARM:?set ARM}"
SEEDX="${SEEDX:-123}"
ANC_MODE="${ANC_MODE:-cosine}"
ANC_W="${ANC_W:-0.2}"
INPUT="${INPUT:-data/eval_subset}"
export TMPDIR="/tmp/seanma0627/tmp-${ARM}" RAY_TMPDIR="/tmp/seanma0627/ray-${ARM}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
exec > >(tee ".sisyphus/evidence/traj-${ARM}.log") 2>&1
echo "### TRAJ ${ARM} seed=${SEEDX} anc=${ANC_MODE} w=${ANC_W} start $(date -Is)"

SEEDEX=""
[ "$SEEDX" != "123" ] && SEEDEX="++data.seed=${SEEDX} ++actor_rollout_ref.rollout.seed=${SEEDX}"
COZ_ARM="$ARM" COZ_TOTAL_STEPS=30 COZ_SAVE_FREQ=5 \
COZ_R_ANC_WEIGHT="$ANC_W" COZ_R_REP_WEIGHT=1.0 COZ_R_FB_WEIGHT=1.0 COZ_ENABLE_RFB=1 COZ_R_ANC_MODE="$ANC_MODE" \
COZ_EXTRA_OVERRIDES="trainer.max_actor_ckpt_to_keep=10 actor_rollout_ref.actor.checkpoint.save_contents=[model] ${SEEDEX}" \
TMPDIR="$TMPDIR" RAY_TMPDIR="$RAY_TMPDIR" \
bash .sisyphus/evidence/task_abl_arm.sh
[ $? -ne 0 ] && { echo "### ${ARM} TRAIN FAILED"; exit 11; }

for N in 5 10 15 20 25 30; do
  CK="checkpoints/${ARM}/global_step_${N}/actor"
  [ -d "$CK" ] || { echo "### no ckpt step ${N}"; continue; }
  TAG="${ARM}_st${N}"
  ( source .venv-train/bin/activate && python -m verl.model_merger merge --backend fsdp --local_dir "$CK" --target_dir "ckpt/VLM_FT/${TAG}" ) || { echo "### merge ${N} FAILED"; rm -rf "$CK"; continue; }
  rm -rf "$CK"
  ( source activate.sh
    python inference_coz.py -i "$INPUT" -o "results/${TAG}_sr" \
      --rec_type recursive_multiscale --prompt_type vlm \
      --vlm_model_path "ckpt/VLM_FT/${TAG}" --vlm_state expanded_text --rec_num 4 --save_prompts \
      --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
      --ram_ft_path ckpt/DAPE/DAPE.pth --ram_path ckpt/RAM/ram_swin_large_14m.pth \
      --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers
    export VLM_JUDGE_BASE_URL=https://intern-vl3-8b.seanmamasde.me/v1 \
           VLM_JUDGE_API_KEY=sk-internvl3-zpC5lCRvqeRnjxFanKbfttAQsLrz1BD0SVLvbb8i7aA \
           VLM_JUDGE_MODEL=InternVL3-8B VLM_JUDGE_TRANSPORT=curl
    python evaluate.py --arm "${TAG}" --images "results/${TAG}_sr/per-sample" --vlm-judge --out "results/${TAG}.csv" )
  rm -rf "ckpt/VLM_FT/${TAG}"
  echo "### step ${N} eval done $(date -Is)"
done
rm -rf "checkpoints/${ARM}"
echo "### TRAJ ${ARM} DONE $(date -Is)"
