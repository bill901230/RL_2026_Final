# Per-image PAIRED ablation analysis (C0_control vs arms)

Pure data analysis over scored ablation CSVs (input CSVs untouched). Per-image join by image id; NaN dropped per-axis. Deltas are **arm - C0_control** computed PER IMAGE, then aggregated.

Paired test: **scipy.stats.ttest_rel** (two-sided, on the shared image ids). 95% CI: non-parametric percentile bootstrap of the mean paired Δ (B=10000, seed=12345).

## 1. Ladder (mean; base C0_control has n=30)

| arm | vlm_grounding_deep | vlm_grounding_all | unique_token_ratio | musiq | niqe | clipiqa |
|---|---:|---:|---:|---:|---:|---:|
| C0_control | 6.700 | 7.150 | 0.617 | 52.717 | -7.002 | 0.605 |
| A_drgrpo | 6.733 | 6.817 | 0.609 | 51.723 | -7.279 | 0.614 |
| B_cliphi | 6.717 | 7.133 | 0.622 | 50.996 | -8.028 | 0.575 |
| ancW05 | 6.300 | 6.950 | 0.619 | 51.051 | -7.369 | 0.590 |
| ancW10 | 5.733 | 6.692 | 0.532 | 51.348 | -7.912 | 0.608 |
| ancW10margin | 7.683 | 7.617 | 0.570 | 52.351 | -7.185 | 0.602 |
| ancW20margin | 6.133 | 6.808 | 0.563 | 51.750 | -7.967 | 0.603 |
| ancW10margin_klUp05 | 5.600 | 6.633 | 0.636 | 51.449 | -7.261 | 0.594 |
| ancW10margin_s60 | 6.300 | 6.800 | 0.554 | 51.495 | -7.318 | 0.597 |

## 2. Per-image PAIRED stats vs C0_control

| arm | axis | n | mean Δ | 95% CI (bootstrap) | std Δ | t | p | win/loss/tie | win-rate |
|---|---|---:|---:|:--:|---:|---:|:--:|:--:|---:|
| A_drgrpo | grounding_deep | 30 | +0.033 | [-0.567, +0.617] | 1.697 | 0.11 | 0.9150 | 11/9/10 | 0.37 |
| A_drgrpo | grounding_all | 30 | -0.333 | [-0.867, +0.208] | 1.520 | -1.20 | 0.2396 | 13/13/4 | 0.43 |
| A_drgrpo | uniqtok | 30 | -0.008 | [-0.043, +0.027] | 0.100 | -0.44 | 0.6599 | 14/16/0 | 0.47 |
| B_cliphi | grounding_deep | 30 | +0.017 | [-0.817, +0.833] | 2.369 | 0.04 | 0.9695 | 14/8/8 | 0.47 |
| B_cliphi | grounding_all | 30 | -0.017 | [-0.525, +0.508] | 1.466 | -0.06 | 0.9508 | 12/14/4 | 0.40 |
| B_cliphi | uniqtok | 30 | +0.005 | [-0.038, +0.049] | 0.122 | 0.23 | 0.8199 | 13/17/0 | 0.43 |
| ancW05 | grounding_deep | 30 | -0.400 | [-1.283, +0.450] | 2.500 | -0.88 | 0.3880 | 12/11/7 | 0.40 |
| ancW05 | grounding_all | 30 | -0.200 | [-0.842, +0.408] | 1.746 | -0.63 | 0.5353 | 14/12/4 | 0.47 |
| ancW05 | uniqtok | 30 | +0.002 | [-0.037, +0.038] | 0.106 | 0.11 | 0.9123 | 17/13/0 | 0.57 |
| ancW10 | grounding_deep | 30 | -0.967 | [-1.900, -0.017] | 2.681 | -1.97 | 0.0579 | 9/15/6 | 0.30 |
| ancW10 | grounding_all | 30 | -0.458 | [-1.267, +0.367] | 2.298 | -1.09 | 0.2836 | 14/14/2 | 0.47 |
| ancW10 | uniqtok | 30 | -0.086 | [-0.115, -0.054] | 0.089 | -5.30 | 1.1e-05 | 4/25/1 | 0.13 |
| ancW10margin | grounding_deep | 30 | +0.983 | [+0.417, +1.617] | 1.709 | 3.15 | 0.0038 | 14/5/11 | 0.47 |
| ancW10margin | grounding_all | 30 | +0.467 | [-0.017, +0.967] | 1.395 | 1.83 | 0.0773 | 15/8/7 | 0.50 |
| ancW10margin | uniqtok | 30 | -0.048 | [-0.072, -0.023] | 0.069 | -3.76 | 0.0008 | 5/25/0 | 0.17 |
| ancW20margin | grounding_deep | 30 | -0.567 | [-1.483, +0.350] | 2.602 | -1.19 | 0.2426 | 8/14/8 | 0.27 |
| ancW20margin | grounding_all | 30 | -0.342 | [-1.050, +0.367] | 2.010 | -0.93 | 0.3596 | 13/15/2 | 0.43 |
| ancW20margin | uniqtok | 30 | -0.054 | [-0.091, -0.017] | 0.104 | -2.86 | 0.0077 | 11/19/0 | 0.37 |
| ancW10margin_klUp05 | grounding_deep | 30 | -1.100 | [-2.117, -0.083] | 2.943 | -2.05 | 0.0498 | 8/17/5 | 0.27 |
| ancW10margin_klUp05 | grounding_all | 30 | -0.517 | [-1.217, +0.208] | 2.037 | -1.39 | 0.1754 | 10/16/4 | 0.33 |
| ancW10margin_klUp05 | uniqtok | 30 | +0.019 | [-0.013, +0.052] | 0.091 | 1.11 | 0.2741 | 17/13/0 | 0.57 |
| ancW10margin_s60 | grounding_deep | 30 | -0.400 | [-1.333, +0.500] | 2.594 | -0.84 | 0.4053 | 7/13/10 | 0.23 |
| ancW10margin_s60 | grounding_all | 30 | -0.350 | [-0.983, +0.250] | 1.741 | -1.10 | 0.2800 | 10/14/6 | 0.33 |
| ancW10margin_s60 | uniqtok | 30 | -0.064 | [-0.097, -0.030] | 0.096 | -3.63 | 0.0011 | 6/24/0 | 0.20 |
