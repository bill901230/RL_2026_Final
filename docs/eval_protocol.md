# Pre-registered Chain-of-Zoom Evaluation Protocol

This protocol is locked before making any training-arm claim. Training results must not change the dataset split, inference settings, metric definitions, aggregation, or CSV schema below.

## Fixed evaluation set

- Use held-out DIV2K-valid images only.
- Evaluate at least 100 valid images per arm.
- Each input is resized/prepared exactly as the Chain-of-Zoom inference pipeline expects, with a center crop / center zoom location and no arm-specific crop tuning.

## Fixed inference settings

- Use the same frozen SR backbone for every arm:
  - SR LoRA: `ckpt/SR_LoRA/model_20001.pkl`
  - SR VAE: `ckpt/SR_VAE/vae_encoder_20001.pt`
- Run 4 recursive Chain-of-Zoom steps with per-step 4x zoom, yielding a final 256x scale (`4^4`).
- Decode VLM prompts greedily (`do_sample=False`) with `max_new_tokens=32`.
- Run at least 3 random seeds per arm. Seed changes may affect model initialization/adapters or any stochastic inference component, but all non-seed settings above remain fixed.

## Registered per-image metrics

The single-source evaluator is `evaluate.py`. It writes one CSV row per image with:

`arm,seed,image,niqe,musiq,maniqa,clipiqa,consistency,unique_token_ratio`

- `niqe`: inverted raw NIQE (`-raw_NIQE`) so higher is better, matching `train/grpo/metrics.py`'s higher-is-better contract.
- `musiq`, `maniqa`, `clipiqa`: pyiqa no-reference quality scores on the deepest Chain-of-Zoom SR image.
- `consistency`: CLIP image-image cosine between the deepest SR image and scale-0 image, mapped from `[-1, 1]` to `[0, 1]` using `train/grpo/text_sim.py`.
- `unique_token_ratio`: unique lexical tokens divided by total lexical tokens across the saved per-scale prompts in `txt/`.

## Registered aggregation

- Primary report: mean ± standard deviation over seed-level means for each metric and arm.
- Seed-level mean: arithmetic mean of per-image rows for that metric within one arm/seed.
- If an arm evaluates more than the minimum 100 images, report the full evaluated count and use all rows; do not cherry-pick subsets after seeing metric values.

Fixture runs used to validate the evaluator implementation are not training-arm claims.
