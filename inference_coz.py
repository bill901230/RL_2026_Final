# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportConstantRedefinition=false, reportOperatorIssue=false, reportOptionalCall=false, reportOptionalMemberAccess=false, reportPossiblyUnboundVariable=false
import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from peft import PeftModel

DEFAULT_VLM_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
ANCHOR_MSG = "What is in this image? Give me a set of words."
EXPANDED_VLM_SYSTEM_TEMPLATE = (
    "You are the Chain-of-Zoom prompt extractor for extreme super-resolution. "
    "The original image has this global caption: {x0_caption!r}. "
    "{prev2_sentence}"
    "You are now inspecting a single current crop at zoom factor={zoom_factor}x "
    "(scale step {scale}). Stay semantically consistent with the global caption "
    "and the two-steps-back context when present, avoid hallucinating unrelated "
    "objects, avoid repeating the previous scale, and answer with a concise set "
    "of words describing new fine details visible in the current crop."
)

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def generate_vlm_caption(vlm_model, image_path):
    messages = [
        {"role": "system", "content": ANCHOR_MSG},
        {"role": "user", "content": [{"type": "image", "image": image_path}]},
    ]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def expanded_state_system_text(state_context):
    prev2_caption = (state_context or {}).get("prev2_caption", "")
    prev2_sentence = ""
    if prev2_caption:
        prev2_sentence = f"The two-steps-back context caption is: {prev2_caption!r}. "
    return EXPANDED_VLM_SYSTEM_TEMPLATE.format(
        x0_caption=(state_context or {}).get("x0_caption", ""),
        prev2_sentence=prev2_sentence,
        zoom_factor=(state_context or {}).get("zoom_factor", "unknown"),
        scale=(state_context or {}).get("scale", "unknown"),
    )


def select_crop_top_left(img, crop_w, crop_h, strategy='center', stride=16, n_bins=32):
    # Strategies (all return top-left origin of a crop_w x crop_h window):
    #   center  : legacy geometric center.
    #   entropy : highest Shannon entropy of grayscale-bin counts.
    #             Prefers regions with diverse intensities (good for paintings
    #             with high color variety; weaker on near-monochrome dense
    #             scenes like a tightly packed battle in earthen tones).
    #   edges   : highest sum of Sobel gradient magnitude (edge density).
    #             Empirically the strongest "object density" proxy: dense
    #             figurative content (crowds, war scenes, ornate architecture)
    #             always wins over smooth regions (skies, mountains).
    #             O(1) per window via summed-area table on the gradient map.
    #   detail  : composite of `edges` and `entropy` (each min-max normalized
    #             across all candidate windows then summed) - robust default
    #             when image content is mixed.
    w, h = img.size
    if crop_w >= w and crop_h >= h:
        return 0, 0
    if strategy == 'center':
        return (w - crop_w) // 2, (h - crop_h) // 2

    arr = np.asarray(img.convert('L'), dtype=np.float32)
    H, W = arr.shape
    tops = list(range(0, H - crop_h + 1, stride))
    lefts = list(range(0, W - crop_w + 1, stride))
    if not tops:
        tops = [0]
    if not lefts:
        lefts = [0]

    def edge_scores():
        from scipy.ndimage import sobel
        gx = sobel(arr, axis=1, mode='reflect')
        gy = sobel(arr, axis=0, mode='reflect')
        mag = np.sqrt(gx * gx + gy * gy)
        ii = np.cumsum(np.cumsum(mag, axis=0), axis=1)
        ii = np.pad(ii, ((1, 0), (1, 0)), mode='constant')
        out = np.zeros((len(tops), len(lefts)), dtype=np.float64)
        for ti, t in enumerate(tops):
            for li, l in enumerate(lefts):
                s = (ii[t + crop_h, l + crop_w] - ii[t, l + crop_w]
                     - ii[t + crop_h, l] + ii[t, l])
                out[ti, li] = s
        return out

    def entropy_scores():
        bin_idx = (arr.astype(np.int32) * n_bins) // 256
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        out = np.zeros((len(tops), len(lefts)), dtype=np.float64)
        for ti, t in enumerate(tops):
            for li, l in enumerate(lefts):
                window = bin_idx[t:t + crop_h, l:l + crop_w]
                hist = np.bincount(window.ravel(), minlength=n_bins).astype(np.float64)
                p = hist / hist.sum()
                p_nz = p[p > 0]
                out[ti, li] = float(-np.sum(p_nz * np.log2(p_nz)))
        return out

    if strategy == 'entropy':
        scores = entropy_scores()
    elif strategy == 'edges':
        scores = edge_scores()
    elif strategy == 'detail':
        e = edge_scores()
        h_ = entropy_scores()
        # Min-max normalize each, then sum. Equal weighting.
        def _nrm(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-12)
        scores = _nrm(e) + _nrm(h_)
    else:
        raise ValueError(f"Unknown crop_strategy: {strategy!r}")

    ti, li = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return lefts[li], tops[ti]

def get_validation_prompt(args, image, prompt_image_path, dape_model=None, vlm_model=None, device='cuda', state_context=None):
    # prepare low-res tensor for SR input
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    # select prompt source
    if args.prompt_type == "null":
        prompt_text = args.prompt or ""
    elif args.prompt_type == "dape":
        lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
        captions = inference(lq_ram, dape_model)
        prompt_text = f"{captions[0]}, {args.prompt}," if args.prompt else captions[0]
    elif args.prompt_type in ('vlm','vlm_base'):
        message_text = None
        
        if args.rec_type == "recursive":
            message_text = "What is in this image? Give me a set of words."
            print(f'MESSAGE TEXT: {message_text}')
            messages = [
                {"role": "system", "content": f"{message_text}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": prompt_image_path}
                    ]
                }
            ]
            text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
        elif args.rec_type == "recursive_multiscale":
            start_image_path = prompt_image_path[0]
            input_image_path = prompt_image_path[1]
            message_text = "The second image is a zoom-in of the first image. Based on this knowledge, what is in the second image? Give me a set of words."
            print(f'START IMAGE PATH: {start_image_path}\nINPUT IMAGE PATH: {input_image_path}\nMESSAGE TEXT: {message_text}')
            if args.vlm_state == "expanded_text":
                message_text = expanded_state_system_text(state_context)
                print(f'EXPANDED STATE CONTEXT: {state_context}\nMESSAGE TEXT: {message_text}')
                messages = [
                    {"role": "system", "content": f"{message_text}"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": input_image_path}
                        ]
                    }
                ]
            else:
                messages = [
                    {"role": "system", "content": f"{message_text}"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": start_image_path},
                            {"type": "image", "image": input_image_path}
                        ]
                    }
                ]
            print(f'MESSAGES\n{messages}')

            text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        else:
            raise ValueError(f"VLM prompt generation not implemented for rec_type: {args.rec_type}")

        inputs = inputs.to("cuda")

        original_sr_devices = {}
        if args.efficient_memory and 'model' in globals() and hasattr(model, 'text_enc_1'): # Check if SR model is defined
            print("Moving SR model components to CPU for VLM inference.")
            original_sr_devices['text_enc_1'] = model.text_enc_1.device
            original_sr_devices['text_enc_2'] = model.text_enc_2.device
            original_sr_devices['text_enc_3'] = model.text_enc_3.device
            original_sr_devices['transformer'] = model.transformer.device
            original_sr_devices['vae'] = model.vae.device
            
            model.text_enc_1.to('cpu')
            model.text_enc_2.to('cpu')
            model.text_enc_3.to('cpu')
            model.transformer.to('cpu')
            model.vae.to('cpu')
            vlm_model.to('cuda') # vlm_model should already be on its device_map="auto" device

        generated_ids = vlm_model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        prompt_text = f"{output_text[0]}, {args.prompt}," if args.prompt else output_text[0]

        if args.efficient_memory and 'model' in globals() and hasattr(model, 'text_enc_1'):
            print("Restoring SR model components to original devices.")
            vlm_model.to('cpu') # If vlm_model was moved to a specific cuda device and needs to be offloaded
            model.text_enc_1.to(original_sr_devices['text_enc_1'])
            model.text_enc_2.to(original_sr_devices['text_enc_2'])
            model.text_enc_3.to(original_sr_devices['text_enc_3'])
            model.transformer.to(original_sr_devices['transformer'])
            model.vae.to(original_sr_devices['vae'])
    else:
        raise ValueError(f"Unknown prompt_type: {args.prompt_type}")
    return prompt_text, lq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input', help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--process_size', type=int, default=512)
    parser.add_argument('--upscale', type=int, default=4)
    parser.add_argument('--align_method', type=str, choices=['wavelet', 'adain', 'nofix'], default='nofix')
    parser.add_argument('--lora_path', type=str, default=None, help='for LoRA of SR model')
    parser.add_argument('--vae_path', type=str, default=None)
    parser.add_argument('--vlm_model_path', type=str, default=DEFAULT_VLM_MODEL_PATH, help='Base/full-finetuned VLM model directory or HF id')
    parser.add_argument('--vlm_lora_path', type=str, default=None, help='Path to the VLM LoRA adapter directory')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--prompt_type', type=str, choices=['null','dape','vlm_base','vlm'], default='dape', help='type of prompt to use')
    parser.add_argument('--vlm_state', type=str, choices=['standard','expanded_text'], default='standard', help='VLM state prompt for recursive_multiscale. expanded_text uses x0/x_(i-2) text captions plus one current crop image.')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--merge_and_unload_lora', action='store_true', help='merge lora weights before inference')
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--rec_type', type=str, choices=['nearest', 'bicubic','onestep','recursive','recursive_multiscale'], default='recursive_multiscale', help='type of inference to use')
    parser.add_argument('--rec_num', type=int, default=4)
    parser.add_argument('--crop_strategy', type=str, choices=['center', 'entropy', 'edges', 'detail'], default='center',
                        help='How to choose the recursive zoom crop region. center=legacy geometric center. entropy=Shannon entropy of grayscale histogram. edges=Sobel gradient magnitude sum (highest edge/object density; best for figure-dense scenes). detail=composite of edges+entropy with min-max normalization.')
    parser.add_argument('--crop_x', type=int, default=None, help='X center (px) of first zoom in 512×512 space; default: image center. When set together with --crop_y, overrides --crop_strategy for the first zoom (rec==0).')
    parser.add_argument('--crop_y', type=int, default=None, help='Y center (px) of first zoom in 512×512 space; default: image center. When set together with --crop_x, overrides --crop_strategy for the first zoom (rec==0).')
    
    parser.add_argument('--vae_encoder_tiled_size', type=int, default=1024)
    parser.add_argument('--vae_decoder_tiled_size', type=int, default=128)
    parser.add_argument('--latent_tiled_size', type=int, default=64)
    parser.add_argument('--latent_tiled_overlap', type=int, default=16)
    
    parser.add_argument('--save_prompts', default=False, action='store_true')
    parser.add_argument('--efficient_memory', default=False, action='store_true')
    args = parser.parse_args()

    global weight_dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # initialize SR model
    model = None
    if args.rec_type not in ('nearest', 'bicubic'):
        if not args.efficient_memory:
            from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler
            model = SD3Euler()
            model.text_enc_1.to('cuda:0')
            model.text_enc_2.to('cuda:0')
            model.text_enc_3.to('cuda:0')
            model.transformer.to('cuda:0')
            model.vae.to('cuda:0')
            for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
                p.requires_grad_(False)
            model_test = OSEDiff_SD3_TEST(args, model)
        else:
            # For efficient memory, text encoders are moved to CPU/GPU on demand in get_validation_prompt
            # Only load transformer and VAE initially if they are always on GPU
            from osediff_sd3 import OSEDiff_SD3_TEST_efficient, SD3Euler
            model = SD3Euler()
            model.transformer.to('cuda')
            model.vae.to('cuda')
            for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
                p.requires_grad_(False)
            model_test = OSEDiff_SD3_TEST_efficient(args, model)

    # gather input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # load DAPE if needed
    DAPE = None
    if args.prompt_type == "dape":
        DAPE = ram(pretrained=args.ram_path,
                   pretrained_condition=args.ram_ft_path,
                   image_size=384,
                   vit='swin_l')
        DAPE.eval().to("cuda")
        DAPE = DAPE.to(dtype=weight_dtype)

    # load VLM pipeline if needed
    vlm_model = None
    global vlm_processor
    global process_vision_info
    vlm_processor = None
    if args.prompt_type in ('vlm','vlm_base'):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        vlm_model_name = args.vlm_model_path
        print(f"Loading VLM model: {vlm_model_name}")
        vlm_device_map = "cpu" if args.efficient_memory else "cuda:0"
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype=torch.float16 if args.efficient_memory else "auto",
            device_map=vlm_device_map
        )
        vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
        print('Base VLM LOADING COMPLETE')
        
        if args.prompt_type == "vlm":
            if not args.vlm_lora_path:
                if args.vlm_model_path == DEFAULT_VLM_MODEL_PATH:
                    raise ValueError("Please specify --vlm_lora_path or a full-FT --vlm_model_path when using prompt_type 'vlm'")
                print('Using full-FT VLM model without LoRA adapter')
                vlm_model.eval()
            elif not os.path.isdir(args.vlm_lora_path):
                raise ValueError(f"VLM LoRA path does not exist or is not a directory: {args.vlm_lora_path}")
            else:
                # load the GRPO fine-tuned VLM LoRA adapter
                print(f"Loading VLM LoRA adapter from: {args.vlm_lora_path}")
                vlm_model = PeftModel.from_pretrained(vlm_model, args.vlm_lora_path)
                vlm_model = vlm_model.merge_and_unload()
                vlm_model.eval()
                print('VLM LoRA ADAPTER LOADING COMPLETE')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per-sample'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per-scale'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'recursive'), exist_ok=True)
    print(f'There are {len(image_names)} images.')
    print(f'Align Method Used: {args.align_method}')
    print(f'Prompt Type: {args.prompt_type}')

    # inference loop
    for image_name in image_names:
        bname = os.path.basename(image_name)
        rec_dir = os.path.join(args.output_dir, 'per-sample', bname[:-4])
        os.makedirs(rec_dir, exist_ok=True)
        if args.save_prompts:
            txt_path = os.path.join(rec_dir, 'txt')
            os.makedirs(txt_path, exist_ok=True)
        print(f'#### IMAGE: {bname}')

        # first image
        os.makedirs(os.path.join(args.output_dir, 'per-scale', 'scale0'), exist_ok=True)
        first_image = Image.open(image_name).convert('RGB')
        first_image = resize_and_center_crop(first_image, args.process_size)
        first_image.save(f'{rec_dir}/0.png')
        first_image.save(os.path.join(args.output_dir, 'per-scale', 'scale0', bname))
        state_caption_cache = {}
        x0_caption = ""
        if args.vlm_state == 'expanded_text' and args.prompt_type in ('vlm', 'vlm_base'):
            x0_caption = generate_vlm_caption(vlm_model, f'{rec_dir}/0.png')
            state_caption_cache[f'{rec_dir}/0.png'] = x0_caption
            print(f'X0 CAPTION: {x0_caption}')

        # recursion
        for rec in range(args.rec_num):
            print(f'RECURSION: {rec}')
            os.makedirs(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}'), exist_ok=True)
            start_image_path = None
            input_image_path = None
            prompt_image_path = None    # this will hold the path(s) for prompt extraction
            
            current_sr_input_image_pil = None

            if args.rec_type in ('nearest', 'bicubic', 'onestep'):
                start_image_pil_path = f'{rec_dir}/0.png'
                start_image_pil = Image.open(start_image_pil_path).convert('RGB')
                rscale = pow(args.upscale, rec+1)
                w, h = start_image_pil.size
                new_w, new_h = w // rscale, h // rscale

                # crop from the original highest-res image available for this step
                if rec == 0 and args.crop_x is not None and args.crop_y is not None:
                    cx = max(new_w // 2, min(args.crop_x, w - new_w // 2))
                    cy = max(new_h // 2, min(args.crop_y, h - new_h // 2))
                    _l, _t = cx - new_w // 2, cy - new_h // 2
                    print(f'CROP@rec{rec} explicit center=({cx},{cy}) bbox=({_l},{_t},{_l+new_w},{_t+new_h}) of {w}x{h}')
                else:
                    _l, _t = select_crop_top_left(start_image_pil, new_w, new_h, args.crop_strategy)
                    print(f'CROP@rec{rec} strategy={args.crop_strategy} bbox=({_l},{_t},{_l+new_w},{_t+new_h}) of {w}x{h}')
                cropped_region = start_image_pil.crop((_l, _t, _l + new_w, _t + new_h))
                
                if args.rec_type == 'onestep':
                    current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)
                    prompt_image_path = f'{rec_dir}/0_input_for_{rec+1}.png'
                    current_sr_input_image_pil.save(prompt_image_path)
                elif args.rec_type == 'bicubic':
                    current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)
                    current_sr_input_image_pil.save(f'{rec_dir}/{rec+1}.png')
                    current_sr_input_image_pil.save(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}', bname))
                    continue
                elif args.rec_type == 'nearest':
                    current_sr_input_image_pil = cropped_region.resize((w, h), Image.NEAREST)
                    current_sr_input_image_pil.save(f'{rec_dir}/{rec+1}.png')
                    current_sr_input_image_pil.save(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}', bname))
                    continue

            elif args.rec_type == 'recursive':
                # input for SR is based on the previous SR output, cropped and resized
                prev_sr_output_path = f'{rec_dir}/{rec}.png'
                prev_sr_output_pil = Image.open(prev_sr_output_path).convert('RGB')
                rscale = args.upscale
                w, h = prev_sr_output_pil.size
                new_w, new_h = w // rscale, h // rscale
                if rec == 0 and args.crop_x is not None and args.crop_y is not None:
                    cx = max(new_w // 2, min(args.crop_x, w - new_w // 2))
                    cy = max(new_h // 2, min(args.crop_y, h - new_h // 2))
                    _l, _t = cx - new_w // 2, cy - new_h // 2
                    print(f'CROP@rec{rec} explicit center=({cx},{cy}) bbox=({_l},{_t},{_l+new_w},{_t+new_h}) of {w}x{h}')
                else:
                    _l, _t = select_crop_top_left(prev_sr_output_pil, new_w, new_h, args.crop_strategy)
                    print(f'CROP@rec{rec} strategy={args.crop_strategy} bbox=({_l},{_t},{_l+new_w},{_t+new_h}) of {w}x{h}')
                cropped_region = prev_sr_output_pil.crop((_l, _t, _l + new_w, _t + new_h))
                current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)

                # this resized image is also the input for VLM
                input_image_path = f'{rec_dir}/{rec+1}_input.png'
                current_sr_input_image_pil.save(input_image_path)
                prompt_image_path = input_image_path

            elif args.rec_type == 'recursive_multiscale':
                prev_sr_output_path = f'{rec_dir}/{rec}.png'
                prev_sr_output_pil = Image.open(prev_sr_output_path).convert('RGB')
                rscale = args.upscale
                w, h = prev_sr_output_pil.size
                new_w, new_h = w // rscale, h // rscale
                if rec == 0 and args.crop_x is not None and args.crop_y is not None:
                    cx = max(new_w // 2, min(args.crop_x, w - new_w // 2))
                    cy = max(new_h // 2, min(args.crop_y, h - new_h // 2))
                    _l, _t = cx - new_w // 2, cy - new_h // 2
                    print(f'CROP@rec{rec} explicit center=({cx},{cy}) bbox=({_l},{_t},{_l+new_w},{_t+new_h}) of {w}x{h}')
                else:
                    _l, _t = select_crop_top_left(prev_sr_output_pil, new_w, new_h, args.crop_strategy)
                    print(f'CROP@rec{rec} strategy={args.crop_strategy} bbox=({_l},{_t},{_l+new_w},{_t+new_h}) of {w}x{h}')
                cropped_region = prev_sr_output_pil.crop((_l, _t, _l + new_w, _t + new_h))
                current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)

                # save the SR input image (which is the "zoomed-in" image for VLM)
                zoomed_image_path = f'{rec_dir}/{rec+1}_input.png'
                current_sr_input_image_pil.save(zoomed_image_path)
                prompt_image_path = [prev_sr_output_path, zoomed_image_path]

            else:
                raise ValueError(f"Unknown recursion_type: {args.rec_type}")

            # generate prompts
            state_context = None
            if args.vlm_state == 'expanded_text' and args.rec_type == 'recursive_multiscale' and args.prompt_type in ('vlm', 'vlm_base'):
                scale = rec + 1
                prev2_caption = ""
                prev2_path = ""
                if scale >= 2:
                    prev2_path = f'{rec_dir}/0.png' if scale == 2 else f'{rec_dir}/{scale-2}_input.png'
                    prev2_caption = state_caption_cache.get(prev2_path, "")
                    if not prev2_caption:
                        prev2_caption = generate_vlm_caption(vlm_model, prev2_path)
                        state_caption_cache[prev2_path] = prev2_caption
                    print(f'PREV2 CAPTION scale={scale}: {prev2_caption}')
                state_context = {
                    "x0_caption": x0_caption,
                    "prev2_caption": prev2_caption,
                    "zoom_factor": pow(args.upscale, scale),
                    "scale": scale,
                }
            validation_prompt, lq = get_validation_prompt(args, current_sr_input_image_pil, prompt_image_path, DAPE, vlm_model, state_context=state_context)
            if args.save_prompts:
                with open(os.path.join(txt_path, f'{rec}.txt'), 'w', encoding='utf-8') as f:
                    f.write(validation_prompt)
            print(f'TAG: {validation_prompt}')

            # super-resolution
            with torch.no_grad():
                lq = lq * 2 - 1

                if args.efficient_memory and model is not None:
                    print("Ensuring SR model components are on CUDA for SR inference.")
                    if not isinstance(model_test, OSEDiff_SD3_TEST_efficient):
                        model.text_enc_1.to('cuda:0')
                        model.text_enc_2.to('cuda:0')
                        model.text_enc_3.to('cuda:0')
                    # transformer and VAE should already be on CUDA per initialization
                    model.transformer.to('cuda')
                    model.vae.to('cuda')

                output_image = model_test(lq, prompt=validation_prompt)
                output_image = torch.clamp(output_image[0].cpu().float(), -1.0, 1.0)
                output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)
                if args.align_method == 'adain':
                    output_pil = adain_color_fix(target=output_pil, source=current_sr_input_image_pil)
                elif args.align_method == 'wavelet':
                    output_pil = wavelet_color_fix(target=output_pil, source=current_sr_input_image_pil)

            output_pil.save(f'{rec_dir}/{rec+1}.png')   # this is the SR output
            output_pil.save(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}', bname))

        # concatenate and save
        imgs = [Image.open(os.path.join(rec_dir, f'{i}.png')).convert('RGB') for i in range(args.rec_num+1)]
        concat = Image.new('RGB', (sum(im.width for im in imgs), max(im.height for im in imgs)))
        x_off = 0
        for im in imgs:
            concat.paste(im, (x_off, 0))
            x_off += im.width
        concat.save(os.path.join(rec_dir, bname))
        concat.save(os.path.join(args.output_dir, 'recursive', bname))
