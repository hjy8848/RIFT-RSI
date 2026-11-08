#!/bin/bash
# Serve one model on ONE Ascend NPU with vLLM (OpenAI-compatible, tool calling on), for evaluation runs.
#
#   NPU=4 MODEL=/home/shenyl/hf/model/Qwen/Qwen3-4B TAG=retail-base bash scripts/vllm_server_npu.sh start
#   NPU=4 TAG=retail-base bash scripts/vllm_server_npu.sh stop|status
#
# Settings: NPU (device id, required), MODEL (required for start), TAG (names the pid/log files), PORT (8200),
# NAME (served model name, "solver"), UTIL (0.85), RUNS ($HOME/runs). Listens on 127.0.0.1 only.
# Check with `npu-smi info` first: the card must show no process. `stop` ends only the session this script
# started (same user, same session id), never anything else.
set -u
: "${TAG:?TAG}"
PORT=${PORT:-8200}; NAME=${NAME:-solver}; UTIL=${UTIL:-0.85}; RUNS=${RUNS:-$HOME/runs}
PIDFILE=$RUNS/vllm-$TAG.pid; LOG=$RUNS/vllm-$TAG.log
url="http://127.0.0.1:$PORT/v1"
ready() { curl -sf -m 3 "$url/models" > /dev/null 2>&1; }

case "${1:-}" in
start)
  : "${NPU:?NPU id}" "${MODEL:?model dir}"
  ready && { echo "already serving on $url"; exit 0; }
  mkdir -p "$RUNS"
  setsid nohup bash -c '
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    source /home/hejingyu/miniconda3_arm/etc/profile.d/conda.sh
    conda activate verl-v091-cann91-py312
    set +e; source /usr/local/Ascend/nnal/atb/set_env.sh; set -e
    export ASCEND_RT_VISIBLE_DEVICES='"$NPU"' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$HOME/tmp"
    mkdir -p "$TMPDIR"
    exec vllm serve "'"$MODEL"'" --served-model-name '"$NAME"' --host 127.0.0.1 --port '"$PORT"' \
      --dtype bfloat16 --max-model-len 16384 --max-num-seqs 128 --gpu-memory-utilization '"$UTIL"' \
      --enable-auto-tool-choice --tool-call-parser hermes --seed 1
  ' > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 120); do ready && { echo "serving $NAME on $url (NPU $NPU)"; exit 0; }; sleep 5; done
  echo "not ready after 600 s; see $LOG" >&2; exit 4 ;;
stop)
  [ -f "$PIDFILE" ] || { echo "no pid file $PIDFILE"; exit 0; }
  pid=$(cat "$PIDFILE"); sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "$sid" ] || { echo "not running"; rm -f "$PIDFILE"; exit 0; }
  ids=$(ps -eo pid,sid,user --no-headers | awk -v s="$sid" -v u="$(id -un)" '$2==s && $3==u {print $1}')
  kill -TERM $ids 2>/dev/null; sleep 10; rm -f "$PIDFILE"; echo "stopped" ;;
status)
  if ready; then echo "serving on $url"; else echo "not serving on $url"; exit 1; fi ;;
*) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
