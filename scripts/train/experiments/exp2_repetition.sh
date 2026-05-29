#!/usr/bin/env bash
# exp2 — baseline + R_rep (cross-scale repetition penalty). Targets PROMPT
# CONVERGENCE / information stagnation across zoom steps.
# Device placement / memory knobs come from the HW profile (see _common.sh):
#   bash scripts/train/experiments/exp2_repetition.sh          # HW=4090 (default)
#   HW=a6000   bash scripts/train/experiments/exp2_repetition.sh
#   HW=pro6000 bash scripts/train/experiments/exp2_repetition.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME="${NAME:-exp2_repetition}"

run_grpo \
  rewards.r_anc.enabled=false \
  rewards.r_rep.enabled=true \
  rewards.r_fb.enabled=false \
  rewards.r_crit.enabled=true \
  rewards.r_phr.enabled=true
