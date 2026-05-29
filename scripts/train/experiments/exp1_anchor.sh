#!/usr/bin/env bash
# exp1 — baseline + R_anc (semantic anchor). Targets SEMANTIC DRIFT at deep zoom.
# Device placement / memory knobs come from the HW profile (see _common.sh):
#   bash scripts/train/experiments/exp1_anchor.sh              # HW=4090 (default)
#   HW=a6000   bash scripts/train/experiments/exp1_anchor.sh
#   HW=pro6000 bash scripts/train/experiments/exp1_anchor.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME="${NAME:-exp1_anchor}"

run_grpo \
  rewards.r_anc.enabled=true \
  rewards.r_rep.enabled=false \
  rewards.r_fb.enabled=false \
  rewards.r_crit.enabled=true \
  rewards.r_phr.enabled=true
