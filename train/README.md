# GRPO Training for the VLM Prompt Extractor

Fine-tunes the Chain-of-Zoom VLM prompt extractor (Qwen2.5-VL-3B-Instruct, LoRA)
with **GRPO** and a multi-objective reward, targeting the two failure modes from
the paper: semantic drift and prompt convergence / information stagnation.

The SD3 / OSEDiff SR backbone is **frozen** here — it is only used as part of the
environment to render pixels for the `R_fb` reward. You do **not** co-train it.
The shipped `ckpt/VLM_LoRA/checkpoint-10000` is already a LoRA adapter, so by
default we **continue-train that adapter** (see `lora.continue_from`).

## Layout

```
train/
  train_grpo_vlm.py        entry point (python -m train.train_grpo_vlm --config ...)
  evaluate.py              DIV2K-valid pyiqa eval (NIQE/MUSIQ/MANIQA/CLIPIQA)
  configs/grpo_default.yaml  ALL hyperparameters & reward weights (single knob surface)
  dataset/zoom_dataset.py  yields 512x512 anchor images x_0 from a txt path list
  grpo/
    state.py     expanded state-space message builder (x_0 + x_{i-2} + x_{i-1} + scale)
    rollout.py   online zoom trajectory + per-scale GRPO group sampling
    rewards.py   R_anc / R_rep / R_fb / R_crit / R_phr orchestrator (+ running z-norm)
    trainer.py   GRPO loss (clipped surrogate + KL to frozen init adapter), wandb, ckpt
    sr_env.py    frozen SR backbone wrapper (reuses osediff_sd3.OSEDiff_SD3_TEST)
    critic.py    R_crit critic VLM (Qwen2.5-VL-7B) or CLIPScore fallback
    metrics.py   pyiqa wrappers (reused by R_fb and evaluate.py)
    text_sim.py  shared CLIP embedder + n-gram overlap for R_anc / R_rep
```

## Setup

```bash
pip install -r requirements.txt          # adds wandb
bash scripts/download_div2k.sh           # DIV2K train+valid -> txt lists
```

## Train

```bash
bash scripts/train/train_grpo_vlm.sh
# or directly:
python -m train.train_grpo_vlm --config train/configs/grpo_default.yaml
```

Dotted overrides without editing the yaml:

```bash
python -m train.train_grpo_vlm --config train/configs/grpo_default.yaml \
  --set rewards.r_anc.weight=2.0 grpo.kl_beta=0.02
```

### Text-only smoke test (no SR backbone / critic, single GPU)

```bash
python -m train.train_grpo_vlm --config train/configs/grpo_default.yaml \
  --set rewards.r_fb.enabled=false rewards.r_crit.enabled=false \
        rewards.r_phr.enabled=false generation.group_size=2 \
        rollout.images_per_step=1 optim.max_steps=5 logging.ckpt_every=5 \
        eval.enabled=false logging.wandb=false device_critic=cuda:0
```

## Evaluate / use the trained adapter

```bash
python -m train.evaluate --config train/configs/grpo_default.yaml \
  --adapter experience/grpo_vlm/final
```

The saved adapter plugs straight into inference:

```bash
python inference_coz.py -i samples -o out \
  --rec_type recursive_multiscale --prompt_type vlm \
  --vlm_lora_path experience/grpo_vlm/final \
  --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers'
```

## Reward components (`configs/grpo_default.yaml: rewards`)

| key      | fixes              | signal |
|----------|--------------------|--------|
| `r_anc`  | semantic drift     | cosine(prompt, x_0 caption) |
| `r_rep`  | prompt convergence | −similarity(prompt, earlier-scale prompts) |
| `r_fb`   | SR fidelity        | pyiqa quality(SR output) + CLIP consistency |
| `r_crit` | alignment (CoZ)    | critic VLM / CLIPScore preference |
| `r_phr`  | clean prompts (CoZ)| −#conversational fillers |

Each is z-normalised online, then combined as `R = Σ weight_k · R_k`. Toggle any
component with `enabled` and tune `weight` — this is where the planned reward
improvements go.
