#!/bin/bash
# Launch a self-teach run on some of this host's Ascend NPUs, detached from the caller.
#
#   W=/path/to/worktree TAG=run-tag NPUS=4,5,6,7 MODEL=/path/to/Qwen3-4B \
#     bash scripts/self_teach_npu.sh [curriculum_run self-teach arguments ...]
#
# Refuses to start if a chosen NPU has a process (npu-smi) or the run directory exists.
# Output: $RUNS/$TAG.log; the run itself is in $RUNS/$TAG. CANN's set_env scripts read unset
# variables, so this script never uses `set -u`, and PYTHONPATH is extended, never replaced.
: "${W:?worktree}" "${TAG:?run tag}" "${NPUS:?comma-separated NPU ids}" "${MODEL:?Service model dir}"
RUNS=${RUNS:-$HOME/runs}; RUN=$RUNS/$TAG
[ -e "$RUN" ] && { echo "run directory $RUN exists" >&2; exit 2; }
busy=$(npu-smi info | awk -v want=",$NPUS," '/^\| [0-9]+ +[0-9]+ +\| [0-9]+ /{ if (index(want, "," $2 ",")) print $2 }' | sort -u)
[ -z "$busy" ] || { echo "NPU(s) busy: $busy" >&2; exit 3; }
WORLD=$(echo "$NPUS" | tr ',' '\n' | wc -l)
mkdir -p "$RUNS" "$HOME/tmp"
export W RUN MODEL NPUS WORLD
setsid nohup bash -c '
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  source /home/hejingyu/miniconda3_arm/etc/profile.d/conda.sh
  conda activate verl-v091-cann91-py312
  source /usr/local/Ascend/nnal/atb/set_env.sh
  export PYTHONPATH="$W/src:$W/third_party/usami${PYTHONPATH:+:$PYTHONPATH}" TMPDIR="$HOME/tmp"
  export ASCEND_RT_VISIBLE_DEVICES="$NPUS" RIFT_PLATFORM=ascend RIFT_WORLD_SIZE="$WORLD"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export RIFT_VLLM_MEMORY_UTILIZATION=${RIFT_VLLM_MEMORY_UTILIZATION:-0.4} RIFT_AGENT_WORKERS=${RIFT_AGENT_WORKERS:-8}
  export RIFT_PPO_MICRO_BATCH=${RIFT_PPO_MICRO_BATCH:-1} RIFT_GRAD_CHECKPOINT=${RIFT_GRAD_CHECKPOINT:-1}
  cd "$W" || exit 7
  python -m rift.runtime.curriculum_run self-teach --service-model "$MODEL" --run-root "$RUN" \
    --verl-config-dir "$W/configs/verl" "$@"
  echo "exit $?"
' _ "$@" > "$RUNS/$TAG.log" 2>&1 < /dev/null &
echo "started $! -> $RUN"
