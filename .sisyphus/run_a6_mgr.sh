#!/usr/bin/env bash
set -u
cd /work/seanma0627/RL_2026_Final || exit 99
source activate.sh
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

EVID=.sisyphus/evidence
OUT=experience/a6_out
FINAL=experience/grpo_vlm/A6/final
LOG=$EVID/task-A6-train.log
MLOG=$EVID/task-A6-mgr.log
DONE=$EVID/task-A6-train.done
READY=$EVID/task-A6-train.ready
TARGET=150
DEADLINE=$(( $(date +%s) + 9000 ))

mlog(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MLOG"; }
latest_n(){ ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1; }
promote(){ rm -rf "$FINAL.tmp"; cp -r "$1" "$FINAL.tmp"; rm -rf "$FINAL"; mv "$FINAL.tmp" "$FINAL"; }
launch_chunk(){
  setsid nohup python .sisyphus/zzz_a6.py --config .sisyphus/a6_cfg.yaml \
    --set lora.continue_from="$1" dataset.train_txt=data/manifests/train50.txt \
      optim.max_steps="$2" optim.warmup_steps=3 logging.ckpt_every=3 \
      generation.group_size=4 rollout.images_per_step=2 \
      rewards.r_rep.enabled=true rewards.r_anc.enabled=true rewards.r_phr.enabled=true \
      rewards.r_fb.enabled=false rewards.r_crit.enabled=false \
      device_policy=cuda:0 device_critic=cuda:0 logging.wandb=false eval.enabled=false \
      output_dir="$OUT" logging.run_name=A6 >> "$LOG" 2>&1 < /dev/null &
  echo $!
}

mkdir -p "$(dirname "$FINAL")"
cum_base=0; last_prom=0
mlog "MGR_START pid=$$ target=$TARGET deadline=2.5h"
rm -rf "$OUT"
if [ -f "$FINAL/adapter_model.safetensors" ]; then CF="$FINAL"; else CF=ckpt/VLM_LoRA/checkpoint-10000; fi
rem=$(( TARGET - cum_base )); [ "$rem" -lt 1 ] && rem=1
TPID=$(launch_chunk "$CF" "$rem")
mlog "chunk pid=$TPID cf=$CF rem=$rem"

while : ; do
  sleep 30
  n=$(latest_n)
  if [ -n "${n:-}" ] && [ "$n" -gt "$last_prom" ]; then
    promote "$OUT/checkpoint-$n"; last_prom=$n
    cum=$(( cum_base + n )); echo "$cum" > "$EVID/task-A6-cumsteps.txt"
    [ ! -f "$READY" ] && echo "$cum" > "$READY"
    mlog "promote checkpoint-$n cum~$cum last_step=[$(grep -E '^step ' "$LOG" | tail -1)]"
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    mlog "deadline; stop chunk $TPID"; kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null; break
  fi
  if ! ps -o pid= -p "$TPID" >/dev/null 2>&1; then
    if [ -f "$OUT/final/adapter_model.safetensors" ]; then promote "$OUT/final"; mlog "chunk completed cleanly"; break; fi
    cum_base=$(( cum_base + last_prom )); mlog "chunk died (reaper?) resume cum_base=$cum_base"
    [ "$cum_base" -ge "$TARGET" ] && break
    rm -rf "$OUT"; last_prom=0
    if [ -f "$FINAL/adapter_model.safetensors" ]; then CF="$FINAL"; else CF=ckpt/VLM_LoRA/checkpoint-10000; fi
    rem=$(( TARGET - cum_base )); [ "$rem" -lt 1 ] && rem=1
    TPID=$(launch_chunk "$CF" "$rem"); mlog "relaunch pid=$TPID cf=$CF rem=$rem"
  fi
done

n=$(latest_n); [ -n "${n:-}" ] && [ "$n" -gt "$last_prom" ] && promote "$OUT/checkpoint-$n"
echo "cum~$(( cum_base + last_prom )) $(date)" > "$DONE"
mlog "MGR_DONE cum~$(( cum_base + last_prom )) final=$FINAL"
