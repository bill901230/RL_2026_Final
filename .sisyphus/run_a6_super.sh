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
SLOG=$EVID/task-A6-super.log
READY=$EVID/task-A6-train.ready
DONE=$EVID/task-A6-train.done
TARGET=150
DEADLINE=$(( $(date +%s) + 10800 ))

slog(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SLOG"; }
latest_n(){ ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1; }
promote(){ rm -rf "$FINAL.tmp"; cp -r "$1" "$FINAL.tmp"; rm -rf "$FINAL"; mv "$FINAL.tmp" "$FINAL"; slog "promoted $1 -> $FINAL"; }

mkdir -p "$(dirname "$FINAL")"
cum=0
slog "SUPER_START pid=$$ target=$TARGET deadline=3h"

while : ; do
  if [ -d "$FINAL" ] && [ -f "$FINAL/adapter_model.safetensors" ]; then CF="$FINAL"; else CF=ckpt/VLM_LoRA/checkpoint-10000; fi
  remaining=$(( TARGET - cum )); [ "$remaining" -lt 1 ] && remaining=1
  rm -rf "$OUT"
  slog "CHUNK continue_from=$CF cum=$cum remaining=$remaining"
  setsid nohup python .sisyphus/zzz_a6.py --config .sisyphus/a6_cfg.yaml \
    --set lora.continue_from="$CF" dataset.train_txt=data/manifests/train50.txt \
      optim.max_steps="$remaining" optim.warmup_steps=3 logging.ckpt_every=3 \
      generation.group_size=4 rollout.images_per_step=2 \
      rewards.r_rep.enabled=true rewards.r_anc.enabled=true rewards.r_phr.enabled=true \
      rewards.r_fb.enabled=false rewards.r_crit.enabled=false \
      device_policy=cuda:0 device_critic=cuda:0 logging.wandb=false eval.enabled=false \
      output_dir="$OUT" logging.run_name=A6 \
    >> "$LOG" 2>&1 < /dev/null &
  TPID=$!
  slog "launched chunk pid=$TPID"
  wait "$TPID"; code=$?
  slog "chunk exited code=$code last_step=[$(grep -E '^step ' "$LOG" | tail -1)]"

  n=$(latest_n)
  if [ -n "${n:-}" ]; then promote "$OUT/checkpoint-$n"; cum=$(( cum + n )); fi
  if [ -d "$OUT/final" ] && [ -f "$OUT/final/adapter_model.safetensors" ]; then promote "$OUT/final"; cum=$TARGET; fi
  echo "$cum" > "$EVID/task-A6-cumsteps.txt"
  if [ "$cum" -ge 12 ] && [ ! -f "$READY" ]; then echo "$cum" > "$READY"; slog "READY cum=$cum"; fi

  if [ "$cum" -ge "$TARGET" ]; then slog "DONE target reached cum=$cum"; break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then slog "DONE deadline reached cum=$cum"; break; fi
  if [ "$code" = "0" ]; then slog "DONE chunk completed cleanly cum=$cum"; break; fi
  sleep 3
done

if [ ! -f "$FINAL/adapter_model.safetensors" ]; then slog "WARN no adapter promoted; cum=$cum"; fi
echo "cum=$cum $(date)" > "$DONE"
slog "SUPER_DONE cum=$cum final=$FINAL"
