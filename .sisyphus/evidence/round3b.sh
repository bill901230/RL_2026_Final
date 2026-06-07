#!/usr/bin/env bash
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
exec > >(tee .sisyphus/evidence/round3b.log) 2>&1
echo "##### ROUND 3B START $(date -Is)"

ARM=ancW10margin INPUT=data/div2k/valid TAG=ancW10margin_n100 bash .sisyphus/evidence/run_eval.sh || echo "##### confirm winner FAILED"
ARM=C0_control INPUT=data/div2k/valid TAG=C0_control_n100 bash .sisyphus/evidence/run_eval.sh || echo "##### confirm control FAILED"
( source activate.sh
  export VLM_JUDGE_BASE_URL=https://intern-vl3-8b.seanmamasde.me/v1 \
         VLM_JUDGE_API_KEY=sk-internvl3-zpC5lCRvqeRnjxFanKbfttAQsLrz1BD0SVLvbb8i7aA \
         VLM_JUDGE_MODEL=InternVL3-8B VLM_JUDGE_TRANSPORT=curl
  python evaluate.py --arm A3_base_n100 --images results/A3_full_sr/per-sample --vlm-judge --out results/A3_base_n100.csv ) || echo "##### base judge FAILED"
echo "##### n100 confirmation done $(date -Is)"

specs=(
"ancW20margin||margin|2.0|30"
"ancW10margin_klUp05|actor_rollout_ref.actor.kl_loss_coef=0.05|margin|1.0|30"
"ancW10margin_s60||margin|1.0|60"
)
for s in "${specs[@]}"; do
  IFS='|' read -r arm extra anc ancw stp <<< "$s"
  echo "##### ARM ${arm} START $(date -Is)"
  ARM="$arm" EXTRA="$extra" ANC_MODE="$anc" ANC_W="$ancw" STEPS="$stp" INPUT=data/eval_subset bash .sisyphus/evidence/run_arm.sh || echo "##### ARM ${arm} FAILED, continuing"
  echo "##### ARM ${arm} END $(date -Is)"
done
echo "##### ROUND 3B DONE $(date -Is)"
