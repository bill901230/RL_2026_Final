#!/bin/bash
set -u
ROOT=/work/seanma0627/RL_2026_Final
SR="--rec_type recursive_multiscale --prompt_type vlm --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers --ram_ft_path ckpt/DAPE/DAPE.pth --ram_path ckpt/RAM/ram_swin_large_14m.pth --save_prompts"

run () {
  local arm=$1 gpu=$2 log=.sisyphus/evidence/proto-$1-eval.log
  tmux new-session -d -s eval_$arm "cd $ROOT && source activate.sh && CUDA_VISIBLE_DEVICES=$gpu python inference_coz.py -i data/eval_subset -o results/${arm}_sr --vlm_lora_path experience/grpo_vlm/$arm/final $SR > $log 2>&1 && python evaluate.py --arm $arm --seed 0 --images results/${arm}_sr/per-sample --out results/$arm.csv >> $log 2>&1; echo EVAL_EXIT=\$? >> $log"
}

run A4 0
run A5 2
run A6 4
sleep 1
tmux list-sessions 2>&1 | grep eval_
echo LAUNCHED
