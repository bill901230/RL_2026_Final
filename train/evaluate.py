"""DIV2K-valid evaluation: run the recursive zoom pipeline with the current VLM
adapter and score each scale's SR output with pyiqa (NIQE/MUSIQ/MANIQA/CLIPIQA).

Used both periodically from the trainer (`evaluate_adapter`, live model) and as a
standalone CLI to evaluate a saved adapter.
"""
import argparse
import os
import sys

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset.zoom_dataset import ZoomBaseImageDataset
from train.grpo.state import build_messages, process_state
from train.grpo.rollout import _center_zoom_crop
from train.grpo.metrics import IQAMetrics


@torch.no_grad()
def evaluate_adapter(cfg, model, processor, sr_backbone, metrics=None):
    ecfg = cfg["eval"]
    device = cfg["device_policy"]
    upscale = cfg["rollout"]["upscale"]
    rec_num = ecfg["rec_num"]

    if metrics is None:
        metrics = IQAMetrics(ecfg["metrics"], device=cfg["device_sr"])

    ds = ZoomBaseImageDataset(
        ecfg["valid_txt"], process_size=cfg["rollout"]["process_size"],
        limit=ecfg["num_images"],
    )
    model.eval()

    accum = {m: [] for m in ecfg["metrics"]}
    for i in range(len(ds)):
        x0 = ds[i]["image"]
        prev_sr = x0
        prev_crop = None
        for rec in range(rec_num):
            crop = _center_zoom_crop(prev_sr, upscale)
            messages = build_messages(x0, prev_crop, crop, upscale ** (rec + 1))
            inputs = process_state(processor, messages, device)
            out = model.generate(**inputs, max_new_tokens=cfg["generation"]["max_new_tokens"],
                                 do_sample=False)
            prompt = processor.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            sr_pil, sr01 = sr_backbone.render(crop, prompt)
            for m in ecfg["metrics"]:
                accum[m].append(float(metrics.score(m, sr01.unsqueeze(0)).item()))
            prev_sr, prev_crop = sr_pil, crop

    return {m: (sum(v) / len(v) if v else float("nan")) for m, v in accum.items()}


def _build_standalone(cfg, adapter_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from train.grpo.sr_env import FrozenSRBackbone

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model"]["policy_id"], torch_dtype=torch.float16,
        device_map=cfg["device_policy"],
    )
    processor = AutoProcessor.from_pretrained(cfg["model"]["policy_id"])
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    else:
        model = base
    model.eval()
    sr = FrozenSRBackbone(cfg, device=cfg["device_sr"])
    return model, processor, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None, help="path to a saved VLM LoRA adapter")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model, processor, sr = _build_standalone(cfg, args.adapter)
    metrics = evaluate_adapter(cfg, model, processor, sr)
    print("DIV2K-valid metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
