#!/usr/bin/env bash
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
exec > >(tee .sisyphus/evidence/round4.log) 2>&1
echo "##### ROUND 4 START $(date -Is)"

echo "##### margin_s456 (winner recipe, seed456, eval n=100 robustness) $(date -Is)"
ARM=margin_s456 ANC_MODE=margin ANC_W=1.0 STEPS=30 INPUT=data/div2k/valid \
  EXTRA="++data.seed=456 ++actor_rollout_ref.rollout.seed=456" \
  bash .sisyphus/evidence/run_arm.sh || echo "##### margin_s456 FAILED rc=$?"

echo "##### margin_fb05 (winner + R_fb weight 0.5, n=30 screen) $(date -Is)"
ARM=margin_fb05 ANC_MODE=margin ANC_W=1.0 FB_W=0.5 STEPS=30 INPUT=data/eval_subset \
  bash .sisyphus/evidence/run_arm.sh || echo "##### margin_fb05 FAILED rc=$?"

echo "##### ROUND 4 DONE $(date -Is)"
