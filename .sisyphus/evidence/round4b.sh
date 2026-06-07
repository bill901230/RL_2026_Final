#!/usr/bin/env bash
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
exec > >(tee .sisyphus/evidence/round4b.log) 2>&1
echo "##### ROUND 4B START $(date -Is)"
echo "##### control_s456 (base recipe, seed456, eval n=100; seed-matched control for robustness) $(date -Is)"
ARM=control_s456 ANC_MODE=cosine ANC_W=0.2 STEPS=30 INPUT=data/div2k/valid \
  EXTRA="++data.seed=456 ++actor_rollout_ref.rollout.seed=456" \
  bash .sisyphus/evidence/run_arm.sh || echo "##### control_s456 FAILED rc=$?"
echo "##### ROUND 4B DONE $(date -Is)"
