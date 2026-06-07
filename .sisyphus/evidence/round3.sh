#!/usr/bin/env bash
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
exec > >(tee .sisyphus/evidence/round3.log) 2>&1
echo "##### ROUND 3 START $(date -Is)"

specs=(
"klUp05|actor_rollout_ref.actor.kl_loss_coef=0.05|cosine|0.2|30"
"klUp10|actor_rollout_ref.actor.kl_loss_coef=0.1|cosine|0.2|30"
"steps15||cosine|0.2|15"
"ancW00||cosine|0.0|30"
)
for s in "${specs[@]}"; do
  IFS='|' read -r arm extra anc ancw stp <<< "$s"
  echo "##### ARM ${arm} START $(date -Is)"
  ARM="$arm" EXTRA="$extra" ANC_MODE="$anc" ANC_W="$ancw" STEPS="$stp" INPUT=data/eval_subset bash .sisyphus/evidence/run_arm.sh || echo "##### ARM ${arm} FAILED rc=$?, continuing"
  echo "##### ARM ${arm} END $(date -Is)"
done
echo "##### ROUND 3 DONE $(date -Is)"
