#!/usr/bin/env bash
# Detached A6 training launcher with full instrumentation.
cd /work/seanma0627/RL_2026_Final || exit 99
source activate.sh
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
echo "RUN_START $(date) pid=$$ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python=$(which python)"
python -m train.train_grpo_vlm --config train/configs/grpo_default.yaml \
  --set lora.continue_from=ckpt/VLM_LoRA/checkpoint-10000 dataset.train_txt=data/manifests/train50.txt \
        optim.max_steps=150 generation.group_size=4 rollout.images_per_step=2 \
        rewards.r_rep.enabled=true rewards.r_anc.enabled=true rewards.r_phr.enabled=true \
        rewards.r_fb.enabled=false rewards.r_crit.enabled=false \
        device_policy=cuda:0 device_critic=cuda:0 logging.wandb=false eval.enabled=false \
        output_dir=experience/grpo_vlm/A6 logging.run_name=A6
code=$?
echo "RUN_EXIT code=$code $(date)"
echo "$code" > .sisyphus/evidence/task-A6-train.exitcode
