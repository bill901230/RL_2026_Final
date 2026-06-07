#!/usr/bin/env bash
# Baseline ladder GPU1 lane: A2 = ORIGINAL CoZ (base Qwen2.5-VL, NO VLM LoRA).
# LOCKED PROTOCOL == A3: recursive_multiscale, crop_strategy=center, rec_num=4,
# upscale=4, process_size=512, align_method=nofix, SR LoRA model_20001.pkl +
# VAE vae_encoder_20001.pt, SD3 medium, greedy max_new_tokens=32 (hardcoded).
# A2 differs from A3 ONLY in the VLM weights: A2 uses the stock base Qwen
# (--prompt_type vlm_base, no --vlm_lora_path / --vlm_model_path), A3 uses the
# author GRPO LoRA. Same n=100 valid set data/div2k/valid (0801-0900).
set -eo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
source ./activate.sh
export CUDA_VISIBLE_DEVICES=1
export TMPDIR=/tmp/$USER/tmp-baseline-gpu1
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
EV=.sisyphus/evidence
VALID=data/div2k/valid
SD3='stabilityai/stable-diffusion-3-medium-diffusers'

echo "===== A2: original CoZ, base Qwen2.5-VL (vlm_base, recursive_multiscale) ====="
python -u inference_coz.py \
  -i "$VALID" \
  -o results/A2_full_sr \
  --rec_type recursive_multiscale \
  --prompt_type vlm_base \
  --lora_path ckpt/SR_LoRA/model_20001.pkl \
  --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --pretrained_model_name_or_path "$SD3" \
  --ram_ft_path ckpt/DAPE/DAPE.pth \
  --ram_path ckpt/RAM/ram_swin_large_14m.pth \
  --process_size 512 --upscale 4 --rec_num 4 \
  --crop_strategy center --align_method nofix --mixed_precision fp16 \
  --save_prompts 2>&1 | tee "$EV/task-baseline-A2-infer.log"
echo "A2 inference done"

python -u evaluate.py --arm A2_full --seed 0 \
  --images results/A2_full_sr --out results/A2_full.csv 2>&1 | tee "$EV/task-baseline-A2-eval.log"
echo "A2 eval done"

touch "$EV/task-baseline-A2.done"
echo "===== GPU1 LANE COMPLETE (A2) ====="
