#!/usr/bin/env bash
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
ARM="${ARM:?set ARM}"
INPUT="${INPUT:-data/div2k/valid}"
TAG="${TAG:-${ARM}_n100}"
MODEL="${MODEL:-ckpt/VLM_FT/${ARM}}"
export TMPDIR="/tmp/seanma0627/tmp-${TAG}"; mkdir -p "$TMPDIR"
exec > >(tee ".sisyphus/evidence/eval-${TAG}.log") 2>&1
echo "### EVAL ${TAG} model=${MODEL} input=${INPUT} start $(date -Is)"
( source activate.sh
  python inference_coz.py -i "$INPUT" -o "results/${TAG}_sr" \
    --rec_type recursive_multiscale --prompt_type vlm \
    --vlm_model_path "$MODEL" --vlm_state expanded_text --rec_num 4 --save_prompts \
    --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
    --ram_ft_path ckpt/DAPE/DAPE.pth --ram_path ckpt/RAM/ram_swin_large_14m.pth \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers
  if [ $? -ne 0 ]; then echo "### ${TAG} RENDER FAILED"; exit 14; fi
  export VLM_JUDGE_BASE_URL=https://intern-vl3-8b.seanmamasde.me/v1 \
         VLM_JUDGE_API_KEY=sk-internvl3-zpC5lCRvqeRnjxFanKbfttAQsLrz1BD0SVLvbb8i7aA \
         VLM_JUDGE_MODEL=InternVL3-8B VLM_JUDGE_TRANSPORT=curl
  python evaluate.py --arm "${TAG}" --images "results/${TAG}_sr/per-sample" --vlm-judge --out "results/${TAG}.csv" )
echo "### EVAL ${TAG} DONE rc=$? $(date -Is)"
