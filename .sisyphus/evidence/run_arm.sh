#!/usr/bin/env bash
# Per-arm RL-ablation pipeline: train -> merge -> render -> judge.
# Env: ARM (req), STEPS=30, EXTRA="<hydra overrides>", ANC_MODE=cosine|margin,
#      INPUT=data/eval_subset, RFB=1, REP_W=1.0, FB_W=1.0, ANC_W=0.2
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final

ARM="${ARM:?set ARM}"
STEPS="${STEPS:-30}"
EXTRA="${EXTRA:-}"
ANC_MODE="${ANC_MODE:-cosine}"
INPUT="${INPUT:-data/eval_subset}"
RFB="${RFB:-1}"
ANC_W="${ANC_W:-0.2}"
REP_W="${REP_W:-1.0}"
FB_W="${FB_W:-1.0}"

export TMPDIR="/tmp/seanma0627/tmp-${ARM}" RAY_TMPDIR="/tmp/seanma0627/ray-${ARM}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
RLOG=".sisyphus/evidence/run-${ARM}.log"
exec > >(tee "$RLOG") 2>&1
echo "### RUN ${ARM} steps=${STEPS} rfb=${RFB} anc=${ANC_MODE} extra='${EXTRA}' input=${INPUT} start $(date -Is)"

# 1) TRAIN (task script sources .venv-train internally)
COZ_ARM="$ARM" COZ_TOTAL_STEPS="$STEPS" COZ_SAVE_FREQ="$STEPS" \
COZ_R_ANC_WEIGHT="$ANC_W" COZ_R_REP_WEIGHT="$REP_W" COZ_R_FB_WEIGHT="$FB_W" COZ_ENABLE_RFB="$RFB" \
COZ_R_ANC_MODE="$ANC_MODE" COZ_EXTRA_OVERRIDES="$EXTRA" \
TMPDIR="$TMPDIR" RAY_TMPDIR="$RAY_TMPDIR" \
bash .sisyphus/evidence/task_abl_arm.sh
if [ $? -ne 0 ]; then echo "### ${ARM} TRAIN FAILED"; exit 11; fi

CKPT="checkpoints/${ARM}/global_step_${STEPS}/actor"
if [ ! -d "$CKPT" ]; then echo "### ${ARM} NO CKPT at $CKPT"; exit 12; fi

# 2) MERGE (subshell isolates .venv-train)
( source .venv-train/bin/activate && python -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT" --target_dir "ckpt/VLM_FT/${ARM}" )
if [ $? -ne 0 ]; then echo "### ${ARM} MERGE FAILED"; exit 13; fi
rm -rf "checkpoints/${ARM}"   # free disk; merged model retained in ckpt/VLM_FT

# 3) RENDER + 4) JUDGE (subshell isolates eval venv)
( set -uo pipefail
  source activate.sh
  python inference_coz.py -i "$INPUT" -o "results/${ARM}_sr" \
    --rec_type recursive_multiscale --prompt_type vlm \
    --vlm_model_path "ckpt/VLM_FT/${ARM}" --vlm_state expanded_text --rec_num 4 --save_prompts \
    --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
    --ram_ft_path ckpt/DAPE/DAPE.pth --ram_path ckpt/RAM/ram_swin_large_14m.pth \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers
  if [ $? -ne 0 ]; then echo "### ${ARM} RENDER FAILED"; exit 14; fi
  export VLM_JUDGE_BASE_URL=https://intern-vl3-8b.seanmamasde.me/v1 \
         VLM_JUDGE_API_KEY=sk-internvl3-zpC5lCRvqeRnjxFanKbfttAQsLrz1BD0SVLvbb8i7aA \
         VLM_JUDGE_MODEL=InternVL3-8B VLM_JUDGE_TRANSPORT=curl
  python evaluate.py --arm "${ARM}" --images "results/${ARM}_sr/per-sample" --vlm-judge --out "results/${ARM}.csv"
  if [ $? -ne 0 ]; then echo "### ${ARM} JUDGE FAILED"; exit 15; fi )
src=$?
if [ $src -ne 0 ]; then echo "### ${ARM} EVAL FAILED rc=$src"; exit $src; fi
echo "### RUN ${ARM} DONE $(date -Is)"
