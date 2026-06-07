# Per-image PAIRED ablation analysis (control_s456 vs arms)

Pure data analysis over scored ablation CSVs (input CSVs untouched). Per-image join by image id; NaN dropped per-axis. Deltas are **arm - control_s456** computed PER IMAGE, then aggregated.

Paired test: **scipy.stats.ttest_rel** (two-sided, on the shared image ids). 95% CI: non-parametric percentile bootstrap of the mean paired Δ (B=10000, seed=12345).

## 1. Ladder (mean; base control_s456 has n=100)

| arm | vlm_grounding_deep | vlm_grounding_all | unique_token_ratio | musiq | niqe | clipiqa |
|---|---:|---:|---:|---:|---:|---:|
| control_s456 | 6.840 | 7.228 | 0.609 | 51.160 | -8.559 | 0.599 |
| margin_s456 | 5.750 | 6.497 | 0.511 | 50.823 | -8.654 | 0.603 |

## 2. Per-image PAIRED stats vs control_s456

| arm | axis | n | mean Δ | 95% CI (bootstrap) | std Δ | t | p | win/loss/tie | win-rate |
|---|---|---:|---:|:--:|---:|---:|:--:|:--:|---:|
| margin_s456 | grounding_deep | 100 | -1.090 | [-1.595, -0.580] | 2.609 | -4.18 | 6.3e-05 | 20/55/25 | 0.20 |
| margin_s456 | grounding_all | 100 | -0.730 | [-1.035, -0.420] | 1.575 | -4.63 | 1.1e-05 | 28/58/14 | 0.28 |
| margin_s456 | uniqtok | 100 | -0.099 | [-0.123, -0.074] | 0.127 | -7.75 | 8.1e-12 | 20/80/0 | 0.20 |
