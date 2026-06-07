# Per-image PAIRED ablation analysis (C0_control_n100 vs arms)

Pure data analysis over scored ablation CSVs (input CSVs untouched). Per-image join by image id; NaN dropped per-axis. Deltas are **arm - C0_control_n100** computed PER IMAGE, then aggregated.

Paired test: **scipy.stats.ttest_rel** (two-sided, on the shared image ids). 95% CI: non-parametric percentile bootstrap of the mean paired Δ (B=10000, seed=12345).

## 1. Ladder (mean; base C0_control_n100 has n=100)

| arm | vlm_grounding_deep | vlm_grounding_all | unique_token_ratio | musiq | niqe | clipiqa |
|---|---:|---:|---:|---:|---:|---:|
| C0_control_n100 | 6.720 | 6.915 | 0.616 | 51.069 | -8.296 | 0.609 |
| ancW10margin_n100 | 7.625 | 7.683 | 0.568 | 51.244 | -8.353 | 0.603 |
| A3_base_n100 | 7.410 | 7.697 | 0.618 | 50.107 | -8.456 | 0.612 |

## 2. Per-image PAIRED stats vs C0_control_n100

| arm | axis | n | mean Δ | 95% CI (bootstrap) | std Δ | t | p | win/loss/tie | win-rate |
|---|---|---:|---:|:--:|---:|---:|:--:|:--:|---:|
| ancW10margin_n100 | grounding_deep | 100 | +0.905 | [+0.310, +1.510] | 3.053 | 2.96 | 0.0038 | 48/40/12 | 0.48 |
| ancW10margin_n100 | grounding_all | 100 | +0.767 | [+0.392, +1.140] | 1.947 | 3.94 | 0.0002 | 61/29/10 | 0.61 |
| ancW10margin_n100 | uniqtok | 100 | -0.048 | [-0.069, -0.028] | 0.104 | -4.63 | 1.1e-05 | 35/65/0 | 0.35 |
| A3_base_n100 | grounding_deep | 100 | +0.690 | [+0.165, +1.200] | 2.631 | 2.62 | 0.0101 | 57/25/18 | 0.57 |
| A3_base_n100 | grounding_all | 100 | +0.782 | [+0.465, +1.100] | 1.621 | 4.83 | 5.1e-06 | 71/21/8 | 0.71 |
| A3_base_n100 | uniqtok | 100 | +0.002 | [-0.019, +0.023] | 0.107 | 0.22 | 0.8233 | 50/49/1 | 0.50 |
