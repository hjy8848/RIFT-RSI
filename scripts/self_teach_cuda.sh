#!/bin/bash
# Launch a self-teach (or power-check) run on this host's CUDA GPUs, detached from the caller.
#
#   W=/path/to/worktree TAG=run-tag GPUS=3,5,6,7 MODEL=/path/to/Qwen3-4B \
#     [COMMAND=self-teach] [REGISTRY=/path/to/task_registry] [INIT_FROM=/path/to/run] \
#     bash scripts/self_teach_cuda.sh [extra curriculum_run arguments ...]
#
# Refuses to start if a GPU is busy (more than 2000 MiB in use) or the run directory exists.
# Output: $RUNS/$TAG.log; the run itself is in $RUNS/$TAG.
set -u
: "${W:?worktree}" "${TAG:?run tag}" "${GPUS:?comma-separated GPU ids in nvidia-smi order}" "${MODEL:?Service model dir}"
RUNS=${RUNS:-$HOME/runs}; RUN=$RUNS/$TAG; COMMAND=${COMMAND:-self-teach}
[ -e "$RUN" ] && { echo "run directory $RUN exists" >&2; exit 2; }
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 2000 ] || { echo "GPU $g is busy ($used MiB)" >&2; exit 3; }
done
WORLD=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$RUNS"
export W RUN MODEL
extra=()
[ -n "${REGISTRY:-}" ] && extra+=(--task-registry-from "$REGISTRY")
[ -n "${INIT_FROM:-}" ] && extra+=(--init-from-run "$INIT_FROM")
setsid nohup bash -c '
  source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate rift-gpu-train
  export PYTHONPATH="$W/src:$W/third_party/usami" TMPDIR="$HOME/tmp"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID RIFT_PLATFORM=cuda RIFT_WORLD_SIZE='"$WORLD"' CUDA_VISIBLE_DEVICES='"$GPUS"'
  export RIFT_VLLM_MEMORY_UTILIZATION=${RIFT_VLLM_MEMORY_UTILIZATION:-0.3} RIFT_AGENT_WORKERS=${RIFT_AGENT_WORKERS:-8}
  # P2P stays off by default (it deadlocked between two A6000s on zju-47); NVLink hosts can pass NCCL_P2P_DISABLE=0.
  export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export RIFT_PPO_MICRO_BATCH=${RIFT_PPO_MICRO_BATCH:-1}
  cd "$W" || exit 7
  python -m rift.runtime.curriculum_run '"$COMMAND"' --service-model "$MODEL" --run-root "$RUN" \
    --verl-config-dir "$W/configs/verl" "$@"
  echo "exit $?"
' _ "${extra[@]}" "$@" > "$RUNS/$TAG.log" 2>&1 < /dev/null &
echo "started $! -> $RUN"
