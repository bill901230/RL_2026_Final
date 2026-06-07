#!/usr/bin/env bash
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
exec > >(tee .sisyphus/evidence/round2.log) 2>&1
echo "##### ROUND 2 START $(date -Is)"

( source activate.sh
  export VLM_JUDGE_BASE_URL=https://intern-vl3-8b.seanmamasde.me/v1 \
         VLM_JUDGE_API_KEY=sk-internvl3-zpC5lCRvqeRnjxFanKbfttAQsLrz1BD0SVLvbb8i7aA \
         VLM_JUDGE_MODEL=InternVL3-8B VLM_JUDGE_TRANSPORT=curl
  python evaluate.py --arm control_r1 --images results/abl_all_sr/per-sample --vlm-judge --out results/control_r1.csv ) || echo "##### control_r1 FAILED"
echo "##### control_r1 done $(date -Is)"

specs=(
"C0_control||cosine|0.2"
"ancW10||cosine|1.0"
"ancW10margin||margin|1.0"
"ancW05||cosine|0.5"
)
for s in "${specs[@]}"; do
  IFS='|' read -r arm extra anc ancw <<< "$s"
  echo "##### ARM ${arm} START $(date -Is)"
  ARM="$arm" EXTRA="$extra" ANC_MODE="$anc" ANC_W="$ancw" STEPS=30 INPUT=data/eval_subset bash .sisyphus/evidence/run_arm.sh || echo "##### ARM ${arm} FAILED rc=$?, continuing"
  echo "##### ARM ${arm} END $(date -Is)"
done
echo "##### ROUND 2 DONE $(date -Is)"
