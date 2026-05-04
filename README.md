# RL_2026_Final

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
