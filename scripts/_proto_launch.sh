#!/bin/bash
# Launch the 3 text-only prototype arms (A4/A5/A6) in detached tmux sessions,
# one per GPU, unbuffered. Returns immediately; poll the logs to track progress.
set -u
ROOT=/work/seanma0627/RL_2026_Final
COMMON="--config train/configs/grpo_default.yaml --set lora.continue_from=ckpt/VLM_LoRA/checkpoint-10000 dataset.train_txt=data/manifests/train50.txt optim.max_steps=25 generation.group_size=4 rollout.images_per_step=1 logging.ckpt_every=25 rewards.r_fb.enabled=false rewards.r_crit.enabled=false device_policy=cuda:0 device_critic=cuda:0 logging.wandb=false eval.enabled=false"

launch () {
  local name=$1 gpu=$2 toggles=$3 outdir=$4 log=$5
  tmux new-session -d -s "$name" "cd $ROOT && source activate.sh && CUDA_VISIBLE_DEVICES=$gpu python -u -m train.train_grpo_vlm $COMMON $toggles output_dir=$outdir logging.run_name=$name > $log 2>&1; echo TRAIN_EXIT=\$? >> $log"
}

launch coz_A4 0 "rewards.r_rep.enabled=true rewards.r_anc.enabled=false rewards.r_phr.enabled=false" experience/grpo_vlm/A4 .sisyphus/evidence/proto-A4-train.log
launch coz_A5 2 "rewards.r_anc.enabled=true rewards.r_rep.enabled=false rewards.r_phr.enabled=false" experience/grpo_vlm/A5 .sisyphus/evidence/proto-A5-train.log
launch coz_A6 4 "rewards.r_rep.enabled=true rewards.r_anc.enabled=true rewards.r_phr.enabled=true" experience/grpo_vlm/A6 .sisyphus/evidence/proto-A6-train.log

sleep 1
tmux list-sessions 2>&1 | grep coz_
echo "LAUNCHED_OK"
