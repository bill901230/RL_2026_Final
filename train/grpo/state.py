"""Expanded state-space message builder for the VLM prompt extractor.

Original CoZ (inference_coz.py:70-95) conditions the VLM on 2 images
(context x_0 / previous SR output, and the current zoom crop). The project's
contribution expands the state with:
  - x_{i-2} : the previous scale-state (AR-2 trajectory, semantic consistency)
  - the scale factor (so the agent matches description detail to zoom level)

This module builds the Qwen2.5-VL chat messages and processes them into model
inputs. Images are passed as in-memory PIL objects (no disk round-trip).
"""
from qwen_vl_utils import process_vision_info

SYSTEM_TEMPLATE = (
    "You are zooming into an image. The first image is the original scene (global "
    "context). The following image(s) are progressively zoomed-in crops; the last "
    "image is the current crop at an effective zoom factor of {scale}x. Using the "
    "global context to stay consistent and avoid hallucination, describe what is in "
    "the current (last) crop. Give me a set of words, focusing on new fine details "
    "visible at this zoom level."
)


def build_messages(x0, x_prev2, x_prev1, scale_factor):
    """Construct Qwen-VL chat messages for the current zoom state.

    Args:
        x0:       PIL.Image, the original anchor scene (always provided).
        x_prev2:  PIL.Image or None, the scale-state two steps back.
        x_prev1:  PIL.Image, the current zoom crop fed to the SR model.
        scale_factor: int/float, effective cumulative zoom (e.g. upscale**i).
    """
    content = [{"type": "image", "image": x0}]
    if x_prev2 is not None:
        content.append({"type": "image", "image": x_prev2})
    content.append({"type": "image", "image": x_prev1})

    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(scale=scale_factor)},
        {"role": "user", "content": content},
    ]
    return messages


def process_state(processor, messages, device):
    """Turn chat messages into model-ready inputs on `device`."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)
