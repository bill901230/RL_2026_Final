#!/usr/bin/env bash
# Baseline ladder GPU0 lane: A1 (Direct-SR, null/empty prompt) + A0 (NN interpolation).
# LOCKED PROTOCOL == A3: recursive_multiscale, crop_strategy=center, rec_num=4,
# upscale=4, process_size=512, align_method=nofix, SR LoRA model_20001.pkl +
# VAE vae_encoder_20001.pt, SD3 medium, greedy max_new_tokens=32 (hardcoded).
# Same n=100 valid set data/div2k/valid (0801-0900).
set -eo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
source ./activate.sh
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/tmp/$USER/tmp-baseline-gpu0
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
EV=.sisyphus/evidence
VALID=data/div2k/valid
SD3='stabilityai/stable-diffusion-3-medium-diffusers'

echo "===== A1: Direct-SR null prompt (recursive_multiscale, empty text) ====="
python -u inference_coz.py \
  -i "$VALID" \
  -o results/A1_full_sr \
  --rec_type recursive_multiscale \
  --prompt_type null \
  --prompt "" \
  --lora_path ckpt/SR_LoRA/model_20001.pkl \
  --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --pretrained_model_name_or_path "$SD3" \
  --process_size 512 --upscale 4 --rec_num 4 \
  --crop_strategy center --align_method nofix --mixed_precision fp16 \
  --save_prompts 2>&1 | tee "$EV/task-baseline-A1-infer.log"
echo "A1 inference done"

python -u evaluate.py --arm A1_full --seed 0 \
  --images results/A1_full_sr --out results/A1_full.csv 2>&1 | tee "$EV/task-baseline-A1-eval.log"
echo "A1 eval done"

echo "===== A0: NN interpolation (rec_type=nearest, no SR, no VLM) ====="
python -u inference_coz.py \
  -i "$VALID" \
  -o results/A0_full_sr \
  --rec_type nearest \
  --process_size 512 --upscale 4 --rec_num 4 \
  --crop_strategy center 2>&1 | tee "$EV/task-baseline-A0-infer.log"
echo "A0 inference done"

python -u evaluate.py --arm A0_full --seed 0 \
  --images results/A0_full_sr --out results/A0_full.csv 2>&1 | tee "$EV/task-baseline-A0-eval.log"
echo "A0 eval done"

touch "$EV/task-baseline-A1A0.done"
echo "===== GPU0 LANE COMPLETE (A1 + A0) ====="
