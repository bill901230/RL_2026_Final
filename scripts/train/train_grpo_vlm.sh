# GRPO fine-tuning of the Chain-of-Zoom VLM prompt extractor.
# Requires the DIV2K lists (run scripts/download_div2k.sh first) and ~4x24GB GPUs:
#   cuda:0 policy VLM (trained) | cuda:1 frozen SR backbone | cuda:2 critic VLM + CLIP
#
# Edit train/configs/grpo_default.yaml for hyperparameters / reward weights.

CONFIG="train/configs/grpo_default.yaml"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m train.train_grpo_vlm \
    --config "$CONFIG"

# --- text-only smoke test (no SR backbone / critic): ---
# CUDA_VISIBLE_DEVICES="0" python -m train.train_grpo_vlm --config "$CONFIG" \
#     --set rewards.r_fb.enabled=false rewards.r_crit.enabled=false \
#           rewards.r_phr.enabled=false generation.group_size=2 \
#           rollout.images_per_step=1 optim.max_steps=5 logging.ckpt_every=5 \
#           eval.enabled=false logging.wandb=false device_critic=cuda:0
