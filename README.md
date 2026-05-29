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

| Models                 | Checkpoints                                                                                                 |
| :--------------------- | :---------------------------------------------------------------------------------------------------------- |
| Stable Diffusion v3    | [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium)                                |
| Qwen2.5-VL-3B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)                                          |
| RAM                    | [Hugging Face](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth) |

## Usage

### CoZ Inference

Standard inference with VLM prompt generation:

```bash
python inference_coz.py \
  -i samples \
  -o inference_results/coz \
  --rec_type recursive_multiscale \
  --prompt_type vlm \
  --lora_path ckpt/SR_LoRA/model_20001.pkl \
  --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --vlm_lora_path ckpt/VLM_LoRA/checkpoint-10000 \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
  --ram_ft_path ckpt/DAPE/DAPE.pth \
  --ram_path ckpt/RAM/ram_swin_large_14m.pth \
  --save_prompts
```


Use `--crop_x <int> --crop_y <int>` to set a custom zoom centre (512×512 space; default: image centre). 

Use `--prompt_type vlm_base` to run without the fine-tuned VLM LoRA.

Use `--efficient_memory` to reduce memory usage, remove if you have enough memory (>24GB).

### Fixed-Prompt Inference (Upper Bound)

Bypasses the VLM and uses hand-crafted prompts to demonstrate the quality ceiling. Edit `FIXED_PROMPTS` at the top of the script to set per-image prompts and crop positions, then run:

```bash
python inference_fixed_prompt.py \
    -i samples \
    -o inference_results/fixed_prompt \
    --crop_x "" --crop_y "" \
    --prompts "prompt for scale 1" "prompt for scale 2" \
              "prompt for scale 3" "prompt for scale 4" \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
    --lora_path  ckpt/SR_LoRA/model_20001.pkl \
    --vae_path   ckpt/SR_VAE/vae_encoder_20001.pt \
    --save_prompts
```

### Visualize Results

Generates a summary figure (images + prompts + IQA metrics) for a single result folder:

```bash
python visualize.py <result_dir> [output.png]

# examples
python visualize.py inference_results/coz/per-sample/0086
python visualize.py inference_results/fixed_prompt/per-sample/0086 out.png
```

Requires `pyiqa` and `setuptools==67.8.0` (see `requirements.txt`) for metric computation.
