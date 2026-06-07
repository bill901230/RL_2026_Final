#!/usr/bin/env bash
set -u
cd /work/seanma0627/RL_2026_Final || exit 99
source activate.sh
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

EVID=.sisyphus/evidence
TRAIN_DONE="$EVID/task-A6-train.done"
TRAIN_LOG="$EVID/task-A6-train.log"
ADAPTER=experience/grpo_vlm/A6/final/adapter_model.safetensors
PERSAMPLE=results/A6_sr/per-sample
INFER_LOG="$EVID/task-A6-infer.log"
EVAL_CSV=results/A6.csv
EVAL_RUN_LOG="$EVID/task-A6-eval-run.log"
EVAL_TXT="$EVID/task-A6-eval.txt"
STATUS="$EVID/task-A6-pipeline.status"
PLOG="$EVID/task-A6-pipeline.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$PLOG"; }
setphase() { echo "$1" > "$STATUS"; }

log "PIPELINE_START pid=$$"
setphase "WAIT_TRAIN"

log "waiting for training done sentinel: $TRAIN_DONE"
while [ ! -f "$TRAIN_DONE" ]; do sleep 30; done
log "training done: $(cat "$TRAIN_DONE")"
LASTSTEP=$(grep -E '^step ' "$TRAIN_LOG" | tail -1)
log "final train step line: $LASTSTEP"

if [ ! -f "$ADAPTER" ]; then
  setphase "FAIL_NO_ADAPTER"
  log "ABORT: adapter missing at $ADAPTER"
  exit 2
fi
log "ADAPTER_OK $ADAPTER"

NDIRS=$(find "$PERSAMPLE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r d; do [ -f "$d/4.png" ] && echo x; done | wc -l)
if [ "$NDIRS" -ge 30 ] && [ -f "$EVAL_CSV" ]; then
  log "inference+eval already complete ($NDIRS dirs, csv present); skipping to verify"
else
  setphase "INFER"
  log "launching inference -> results/A6_sr"
  python inference_coz.py -i data/eval_subset -o results/A6_sr \
    --rec_type recursive_multiscale --prompt_type vlm \
    --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
    --vlm_lora_path experience/grpo_vlm/A6/final \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
    --ram_ft_path ckpt/DAPE/DAPE.pth --ram_path ckpt/RAM/ram_swin_large_14m.pth --save_prompts \
    > "$INFER_LOG" 2>&1 < /dev/null
  ICODE=$?
  log "inference exited code=$ICODE"
  echo "$ICODE" > "$EVID/task-A6-infer.exitcode"
  NDIRS=$(find "$PERSAMPLE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r d; do [ -f "$d/4.png" ] && echo x; done | wc -l)
  log "per-sample completed dirs (with 4.png): $NDIRS"
  if [ "$ICODE" != "0" ] || [ "$NDIRS" -lt 30 ]; then
    setphase "FAIL_INFER"
    log "ABORT: inference incomplete (code=$ICODE dirs=$NDIRS)"
    exit 3
  fi
fi

setphase "EVAL"
log "running evaluate.py -> $EVAL_CSV"
python evaluate.py --arm A6 --seed 0 --images "$PERSAMPLE" --out "$EVAL_CSV" \
  > "$EVAL_RUN_LOG" 2>&1 < /dev/null
ECODE=$?
log "evaluate exited code=$ECODE"
if [ "$ECODE" != "0" ] || [ ! -f "$EVAL_CSV" ]; then
  setphase "FAIL_EVAL"
  log "ABORT: evaluate failed (code=$ECODE)"
  exit 4
fi

setphase "ACCEPT"
test -f "$ADAPTER" && echo ADAPTER_OK | tee -a "$PLOG"
python -c "import csv,statistics as st;r=list(csv.DictReader(open('results/A6.csv')));assert len(r)>=30;[print(c,round(st.mean(float(x[c]) for x in r),4)) for c in ['niqe','musiq','maniqa','clipiqa','consistency','unique_token_ratio']]" | tee "$EVAL_TXT"
ACODE=${PIPESTATUS[0]}
if [ "$ACODE" != "0" ]; then
  setphase "FAIL_ACCEPT"
  log "ABORT: acceptance one-liner failed (code=$ACODE)"
  exit 5
fi

setphase "DONE"
log "PIPELINE_DONE final_step=[$LASTSTEP]"
echo "done $(date)" > "$EVID/task-A6-pipeline.done"
