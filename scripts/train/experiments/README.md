# GRPO experiment matrix

Each `exp*.sh` is one experiment: it picks the **reward set**; the **hardware profile**
(`HW`) picks device placement + memory knobs. So the *same* script runs on any card:

```bash
bash scripts/train/experiments/exp4_full.sh             # HW=4090 (default)
HW=a6000   bash scripts/train/experiments/exp4_full.sh
HW=pro6000 bash scripts/train/experiments/exp4_full.sh
```

Every run pins its own `output_dir` (`experience/grpo_vlm/<name>`) and wandb `run_name`, and all
share the wandb project `grpo_coz_vlm`, so the runs line up on the same dashboards.

## Hardware profiles (`HW`, set in `_common.sh`)

| `HW`      | card               | placement | memory knobs |
|-----------|--------------------|-----------|--------------|
| `4090` *(default)* | 4× RTX 4090, 24 GB | one model per GPU (`cuda:0/1/2`) | checkpointing on, group 6, micro-batch 2 |
| `a6000`   | 1× RTX A6000, 48 GB | all models on `cuda:0` | checkpointing on, group 6, micro-batch 3 |
| `pro6000` | 1× RTX PRO 6000, 96 GB | all models on `cuda:0` | checkpointing **off**, group 8, micro-batch 8 |

Notes:
- `4090`: the SR experiments (`exp3`, `exp4`) set `NEED_SR=1` and load 3 models → 3 cards
  (`GPUS` defaults to `0,1,2`, critic on `cuda:2`). `exp0/1/2` load only policy+critic → **2
  cards** (`GPUS` defaults to `0,1`, critic on `cuda:1`). So `GPUS=2,3 bash ... exp0_baseline.sh`
  works; you can run a second 2-model experiment in parallel on the other pair.
- `a6000`: full method on one 48 GB card is reasonably tight (policy + SR + 7B critic ≈ 40 GB).
  If it OOMs, append `rewards.r_crit.backend=clipscore` to drop the 7B critic for a light CLIP
  scorer, or lower `generation.group_size`.
- `pro6000`: ~40–45 GB used of 96 GB — lots of headroom; checkpointing is off for speed and you
  can push the batch knobs higher.

## Experiment ladder

`exp0` = original Chain-of-Zoom paper reward (`R_crit` + `R_phr`). Each later run adds one of our
contributions; `exp4` is the full proposed method. The delta vs. `exp0` isolates each reward.

| script | rewards on (vs. baseline) | fixes | models loaded |
|--------|---------------------------|-------|---------------|
| `exp0_baseline.sh`   | `R_crit`, `R_phr` | — (paper) | policy + critic |
| `exp1_anchor.sh`     | + `R_anc` | semantic drift | policy + critic |
| `exp2_repetition.sh` | + `R_rep` | prompt convergence | policy + critic |
| `exp3_feedback.sh`   | + `R_fb` | SR fidelity | policy + SR + critic |
| `exp4_full.sh`       | + `R_anc`+`R_rep`+`R_fb` | all (proposed) | policy + SR + critic |

## Overrides

- **Pick physical GPUs:** `GPUS=2,3 bash ... exp1_anchor.sh` (maps to `cuda:0,1,...` inside).
- **Rename a run:** `NAME=exp1_anchor_w2 bash ... exp1_anchor.sh`.
- **Tweak any knob:** edit the script's `run_grpo` args, or call the module directly:
  ```bash
  CUDA_VISIBLE_DEVICES=0 /project2/cookies/miniconda3/envs/coz/bin/python \
    -m train.train_grpo_vlm --config train/configs/grpo_default.yaml --set \
    rewards.r_anc.enabled=true rewards.r_anc.weight=2.0 \
    device_policy=cuda:0 device_critic=cuda:0 \
    output_dir=experience/grpo_vlm/exp1_anchor_w2 logging.run_name=exp1_anchor_w2
  ```

## Smoke test first (recommended)
Cheap 5-step text-only run, 1 GPU, no wandb — validates the loop before committing GPU/time:

```bash
bash scripts/train/experiments/smoke.sh
```

## Evaluate a finished run
```bash
/project2/cookies/miniconda3/envs/coz/bin/python -m train.evaluate \
  --config train/configs/grpo_default.yaml --adapter experience/grpo_vlm/exp4_full/final
```
