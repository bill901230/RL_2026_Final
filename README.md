# RL_2026_Final — Enhanced Chain-of-Zoom

## Project Goal

Enhance Chain-of-Zoom (CoZ) for extreme super-resolution (up to 256×) by addressing two failure modes of the unoptimized VLM prompt extractor:

- **Fail Case 1 — Semantic Drift.** At deep zoom levels the VLM loses global context and hallucinates unrelated concepts (e.g. animal fur → "neurons / synapses").
- **Fail Case 2 — Prompt Convergence.** Sparse visual evidence collapses the VLM into repetitive, low-entropy prompts, providing no new high-frequency guidance to the SR backbone.

### Approach

- **Expanded state space**: condition the VLM on the previous scale-state `x_{i-2}` and the current scale factor (AR-2 trajectory + scale awareness).
- **Multi-objective reward via GRPO**:
  - `R_anc` — semantic anchor reward, consistency with original `x_0` description (fixes drift).
  - `R_rep` — cross-scale repetition penalty (fixes prompt convergence).
  - `R_fb`  — intermediate SR feedback reward on the previous step's pixels.

### Evaluation

- **No-reference VQA metrics**: NIQE, MUSIQ, MANIQA, CLIPIQA
- **Datasets**: DIV2K (800 imgs), DIV8K (1500 imgs); resize + center-crop to 512×512
- **Baselines**: NN Interpolation, Direct SR, original Chain-of-Zoom

---

## Setup

```bash
conda create -n coz python=3.10
conda activate coz
pip install -r requirements.txt
```

## Models

| Models | Checkpoints |
|:---|:---|
| Stable Diffusion v3 | [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium) |
| Qwen2.5-VL-3B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| RAM | [Hugging Face](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth) |

## Quick Inference

```bash
python inference_coz.py \
  -i samples \
  -o inference_results/coz_vlmprompt \
  --rec_type recursive_multiscale \
  --prompt_type vlm \
  --lora_path ckpt/SR_LoRA/model_20001.pkl \
  --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --vlm_lora_path ckpt/VLM_LoRA/checkpoint-10000 \
  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers' \
  --ram_ft_path ckpt/DAPE/DAPE.pth \
  --ram_path ckpt/RAM/ram_swin_large_14m.pth \
  --save_prompts
```
