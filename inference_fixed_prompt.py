#!/usr/bin/env python3
"""
Fixed-prompt inference for Chain-of-Zoom (single image, CLI-driven).

All per-image settings (crop position, prompts) are passed as arguments so
the shell script is the single source of truth — no editing needed here.

Usage (called by run_fixed_prompt_inference.sh):
    python inference_fixed_prompt.py \
        -i samples_v2/0086.png \
        -o inference_results/fixed_prompt \
        --crop_x 180 --crop_y 218 \
        --prompts "prompt for scale 1" "prompt for scale 2" \
                  "prompt for scale 3" "prompt for scale 4" \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
        --lora_path  ckpt/SR_LoRA/model_20001.pkl \
        --vae_path   ckpt/SR_VAE/vae_encoder_20001.pt \
        --save_prompts
"""

import os
import sys
sys.path.append(os.getcwd())
import argparse
import torch
from torchvision import transforms
from PIL import Image

tensor_transforms = transforms.Compose([transforms.ToTensor()])


def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image",  type=str, required=True, help="Path to a single input image")
    parser.add_argument("-o", "--output_dir",   type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_path",          type=str, default=None)
    parser.add_argument("--vae_path",           type=str, default=None)
    parser.add_argument("--prompts",            type=str, nargs="+", required=True, help="One prompt per recursion step (must match --rec_num)")
    parser.add_argument("--crop_x",             type=int, default=None, help="X centre of first zoom in 512×512 space (default: centre)")
    parser.add_argument("--crop_y",             type=int, default=None, help="Y centre of first zoom in 512×512 space (default: centre)")
    parser.add_argument("--rec_num",            type=int, default=4)
    parser.add_argument("--upscale",            type=int, default=4)
    parser.add_argument("--process_size",       type=int, default=512)
    parser.add_argument("--align_method",       type=str, choices=["wavelet", "adain", "nofix"], default="nofix")
    parser.add_argument("--mixed_precision",    type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--lora_rank",          type=int, default=4)
    parser.add_argument("--merge_and_unload_lora", action="store_true")
    parser.add_argument("--save_prompts",       action="store_true")
    parser.add_argument("--efficient_memory",   action="store_true")
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=128)
    parser.add_argument("--latent_tiled_size",  type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=16)
    args = parser.parse_args()

    if len(args.prompts) < args.rec_num:
        parser.error(f"--prompts has {len(args.prompts)} entries but --rec_num={args.rec_num}")

    # ── load SR model ─────────────────────────────────────────────────────────
    from osediff_sd3 import OSEDiff_SD3_TEST, OSEDiff_SD3_TEST_efficient, SD3Euler
    from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

    model = SD3Euler()
    if args.efficient_memory:
        model.transformer.to("cuda")
        model.vae.to("cuda")
        model_test = OSEDiff_SD3_TEST_efficient(args, model)
    else:
        model.text_enc_1.to("cuda:0")
        model.text_enc_2.to("cuda:0")
        model.text_enc_3.to("cuda:0")
        model.transformer.to("cuda:0")
        model.vae.to("cuda:0")
        model_test = OSEDiff_SD3_TEST(args, model)
    for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3,
              model.transformer, model.vae]:
        p.requires_grad_(False)

    # ── setup dirs ────────────────────────────────────────────────────────────
    bname = os.path.basename(args.input_image)
    stem  = bname[:-4]

    for sub in ("per-sample", "per-scale", "recursive"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    rec_dir = os.path.join(args.output_dir, "per-sample", stem)
    os.makedirs(rec_dir, exist_ok=True)
    if args.save_prompts:
        os.makedirs(os.path.join(rec_dir, "txt"), exist_ok=True)

    print(f"#### IMAGE: {bname}  crop=({args.crop_x}, {args.crop_y})")

    # ── scale 0: original ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(args.output_dir, "per-scale", "scale0"), exist_ok=True)
    first_image = Image.open(args.input_image).convert("RGB")
    first_image = resize_and_center_crop(first_image, args.process_size)
    first_image.save(f"{rec_dir}/0.png")
    first_image.save(os.path.join(args.output_dir, "per-scale", "scale0", bname))

    # ── recursion ─────────────────────────────────────────────────────────────
    for rec in range(args.rec_num):
        os.makedirs(os.path.join(args.output_dir, "per-scale", f"scale{rec+1}"), exist_ok=True)

        prev_pil     = Image.open(f"{rec_dir}/{rec}.png").convert("RGB")
        w, h         = prev_pil.size
        new_w, new_h = w // args.upscale, h // args.upscale

        cx = (max(new_w // 2, min(args.crop_x, w - new_w // 2))
              if (rec == 0 and args.crop_x is not None) else w // 2)
        cy = (max(new_h // 2, min(args.crop_y, h - new_h // 2))
              if (rec == 0 and args.crop_y is not None) else h // 2)

        cropped  = prev_pil.crop((cx - new_w//2, cy - new_h//2, cx + new_w//2, cy + new_h//2))
        sr_input = cropped.resize((w, h), Image.BICUBIC)
        sr_input.save(f"{rec_dir}/{rec+1}_input.png")

        prompt = args.prompts[rec]
        print(f"  [{rec+1}/{args.rec_num}] {prompt[:90]}")

        if args.save_prompts:
            with open(os.path.join(rec_dir, "txt", f"{rec}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)

        lq = tensor_transforms(sr_input).unsqueeze(0).to("cuda")
        with torch.no_grad():
            output = model_test(lq * 2 - 1, prompt=prompt)
            output = torch.clamp(output[0].cpu().float(), -1.0, 1.0)
            output_pil = transforms.ToPILImage()(output * 0.5 + 0.5)

        if args.align_method == "adain":
            output_pil = adain_color_fix(target=output_pil, source=sr_input)
        elif args.align_method == "wavelet":
            output_pil = wavelet_color_fix(target=output_pil, source=sr_input)

        output_pil.save(f"{rec_dir}/{rec+1}.png")
        output_pil.save(os.path.join(args.output_dir, "per-scale", f"scale{rec+1}", bname))

    # ── concatenate strip ─────────────────────────────────────────────────────
    imgs   = [Image.open(f"{rec_dir}/{i}.png").convert("RGB") for i in range(args.rec_num + 1)]
    concat = Image.new("RGB", (sum(im.width for im in imgs), imgs[0].height))
    x_off  = 0
    for im in imgs:
        concat.paste(im, (x_off, 0))
        x_off += im.width
    concat.save(os.path.join(rec_dir, bname))
    concat.save(os.path.join(args.output_dir, "recursive", bname))
    print(f"  saved → {rec_dir}/{bname}")
