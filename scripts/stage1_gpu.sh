#!/bin/bash
# Stage-1 training on this host's idle CUDA GPUs: wait for them, train, retry only on early failures.
#
#   W=/path/to/worktree TAG=run-tag POOL=/path/to/designer-plans.jsonl bash scripts/stage1_gpu.sh
#
# The first host whose waiter finds enough idle GPUs takes the lock (a shared directory) and runs;
# the others exit. "Idle" is not a reservation, so a run that dies within RETRY_WITHIN seconds
# (another user took the GPUs while it started) is retried on fresh GPUs with a new run tag, at most
# MAXTRY times. A later failure is a real failure and is not retried.
set -u
: "${W:?worktree}" "${TAG:?run tag}" "${POOL:?merged designer-plans.jsonl}"
MAXTRY=${MAXTRY:-3}; MAXWAIT=${MAXWAIT:-43200}; RETRY_WITHIN=${RETRY_WITHIN:-1500}
ROUNDS=${ROUNDS:-15}; TASKS_PER_ROUND=${TASKS_PER_ROUND:-8}; MIX=${MIX:-0.25}
RUNS=/home/hejingyu/runs; LOCK=$RUNS/.lock-$TAG; LOG=$RUNS/$TAG.supervisor.log
log() { echo "[$(TZ=Asia/Seoul date +%F\ %T) KST $(hostname)] $*" >> "$LOG"; }

idle_gpus() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
              | awk -F', ' '$2<2000 && $3<5 {print $1}'; }

[ -e "$LOCK" ] && { log "lock exists, another waiter owns $TAG - exit"; exit 0; }
log "waiting for >=2 idle GPUs (max ${MAXWAIT}s)"
waited=0
while true; do
  mapfile -t ids < <(idle_gpus)
  if [ "${#ids[@]}" -ge 2 ]; then
    world=2; [ "${#ids[@]}" -ge 4 ] && world=4
    pick=$(printf '%s\n' "${ids[@]:0:$world}" | paste -sd, -)
    sleep 20   # the same GPUs must still be idle 20 s later
    ok=1; for g in ${pick//,/ }; do idle_gpus | grep -qx "$g" || ok=0; done
    if [ "$ok" = 1 ]; then
      mkdir "$LOCK" 2>/dev/null || { log "another waiter took the lock first - exit"; exit 0; }
      break
    fi
  fi
  sleep 30; waited=$((waited + 30))
  [ "$waited" -ge "$MAXWAIT" ] && { log "no idle GPUs after ${MAXWAIT}s - giving up"; exit 3; }
done

attempt=1
while [ "$attempt" -le "$MAXTRY" ]; do
  mapfile -t ids < <(idle_gpus)
  if [ "${#ids[@]}" -lt 2 ]; then log "attempt $attempt: GPUs went away, waiting"; sleep 60; waited=$((waited + 60))
    [ "$waited" -ge "$MAXWAIT" ] && { log "gave up waiting for GPUs"; rmdir "$LOCK"; exit 3; }; continue; fi
  world=2; [ "${#ids[@]}" -ge 4 ] && world=4
  pick=$(printf '%s\n' "${ids[@]:0:$world}" | paste -sd, -)
  RUN=$RUNS/$TAG-t$attempt
  [ -e "$RUN" ] && { log "run root $RUN exists - abort"; exit 2; }
  log "attempt $attempt: $world GPUs [$pick] -> $RUN"
  start=$(date +%s)
  setsid bash -c "
    source /home/hejingyu/miniconda3/etc/profile.d/conda.sh && conda activate rift-gpu-train
    export PYTHONPATH='$W/src:$W/third_party/usami'
    export CUDA_DEVICE_ORDER=PCI_BUS_ID RIFT_PLATFORM=cuda RIFT_WORLD_SIZE=$world CUDA_VISIBLE_DEVICES=$pick
    export RIFT_VLLM_MEMORY_UTILIZATION=0.3 RIFT_AGENT_WORKERS=8 NCCL_P2P_DISABLE=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd '$W' || exit 7
    exec python -m rift.runtime.curriculum_run train-pool \
      --service-model /home/shenyl/hf/model/Qwen/Qwen3-4B --run-root '$RUN' --verl-config-dir '$W/configs/verl' \
      --designer-plans '$POOL' --designer-selected-only --drop-references delivery_age \
      --train-specs configs/experiments/designer_mix.json --mix-fraction $MIX \
      --heldout-specs configs/experiments/designer_heldout.json --heldout-groups 2 \
      --rounds $ROUNDS --tasks-per-round $TASKS_PER_ROUND --group-size 4 --mini-batch-size 8 \
      --persist-every 0 --seed 11
  " > "$RUNS/$TAG-t$attempt.log" 2>&1 < /dev/null &
  pid=$!
  log "attempt $attempt started, session leader PID $pid"
  wait "$pid"; code=$?
  took=$(( $(date +%s) - start ))
  # leave nothing behind: everything the run started is in its own session
  left=$(ps -eo pid,sid,user --no-headers | awk -v s="$pid" '$2==s && $3=="hejingyu" {print $1}')
  [ -n "$left" ] && { kill -TERM $left 2>/dev/null; sleep 15; }
  log "attempt $attempt exited with code $code after ${took}s"
  if [ "$code" = 0 ]; then log "DONE: $RUN"; exit 0; fi
  if [ "$took" -gt "$RETRY_WITHIN" ]; then log "failed after ${took}s - a real failure, not retrying"; exit 1; fi
  attempt=$((attempt + 1))
done
log "all $MAXTRY attempts failed early"; rmdir "$LOCK" 2>/dev/null; exit 1
