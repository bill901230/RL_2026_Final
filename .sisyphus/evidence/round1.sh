#!/usr/bin/env bash
set -uo pipefail
umask 022
cd /work/seanma0627/RL_2026_Final
RLOG=".sisyphus/evidence/round1.log"
exec > >(tee "$RLOG") 2>&1
echo "##### ROUND 1 START $(date -Is)"

specs=(
"A_drgrpo|algorithm.norm_adv_by_std_in_grpo=False actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm|cosine"
"B_cliphi|actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28|cosine"
"E_ancmargin||margin"
)
for s in "${specs[@]}"; do
  IFS='|' read -r arm extra anc <<< "$s"
  echo "##### ARM ${arm} START $(date -Is)"
  ARM="$arm" EXTRA="$extra" ANC_MODE="$anc" STEPS=30 INPUT=data/eval_subset bash .sisyphus/evidence/run_arm.sh || echo "##### ARM ${arm} FAILED rc=$?, continuing"
  echo "##### ARM ${arm} END $(date -Is)"
done

( source activate.sh
  export VLM_JUDGE_BASE_URL=https://intern-vl3-8b.seanmamasde.me/v1 \
         VLM_JUDGE_API_KEY=sk-internvl3-zpC5lCRvqeRnjxFanKbfttAQsLrz1BD0SVLvbb8i7aA \
         VLM_JUDGE_MODEL=InternVL3-8B VLM_JUDGE_TRANSPORT=curl
  python evaluate.py --arm control_r1 --images results/abl_all_sr/per-sample --vlm-judge --out results/control_r1.csv )
echo "##### control_r1 re-judged $(date -Is)"
echo "##### ROUND 1 DONE $(date -Is)"
