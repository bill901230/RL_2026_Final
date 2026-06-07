# Per-image PAIRED ablation analysis (C0_control_n100 vs arms)

Pure data analysis over scored ablation CSVs (input CSVs untouched). Per-image join by image id; NaN dropped per-axis. Deltas are **arm - C0_control_n100** computed PER IMAGE, then aggregated.

Paired test: **scipy.stats.ttest_rel** (two-sided, on the shared image ids). 95% CI: non-parametric percentile bootstrap of the mean paired Δ (B=10000, seed=12345).

## 1. Ladder (mean; base C0_control_n100 has n=100)

| arm | vlm_grounding_deep | vlm_grounding_all | unique_token_ratio | musiq | niqe | clipiqa |
|---|---:|---:|---:|---:|---:|---:|
| C0_control_n100 | 6.720 | 6.915 | 0.616 | 51.069 | -8.296 | 0.609 |
| margin_s456 | 5.750 | 6.497 | 0.511 | 50.823 | -8.654 | 0.603 |

## 2. Per-image PAIRED stats vs C0_control_n100

| arm | axis | n | mean Δ | 95% CI (bootstrap) | std Δ | t | p | win/loss/tie | win-rate |
|---|---|---:|---:|:--:|---:|---:|:--:|:--:|---:|
| margin_s456 | grounding_deep | 100 | -0.970 | [-1.610, -0.340] | 3.240 | -2.99 | 0.0035 | 29/49/22 | 0.29 |
| margin_s456 | grounding_all | 100 | -0.417 | [-0.848, +0.010] | 2.191 | -1.91 | 0.0596 | 44/49/7 | 0.44 |
| margin_s456 | uniqtok | 100 | -0.105 | [-0.134, -0.078] | 0.144 | -7.30 | 7.4e-11 | 27/73/0 | 0.27 |
