#!/bin/bash
# Serve the frozen model that plays the customer (RIFT_CUSTOMER_URL points at it).
#
#   GPU=3 MODEL=/path/to/Qwen3-4B bash scripts/customer_server.sh start
#   bash scripts/customer_server.sh stop|status
#
# Settings (environment): GPU (nvidia-smi id, required for start), MODEL (required for start),
# PORT (8100), NAME (customer), UTIL (0.15 of the GPU's memory), RUNS ($HOME/runs).
# The server listens on 127.0.0.1 only. It shares its GPU with training: a small memory share, and the
# training's own vLLM is asleep while the policy updates. Never serve the model that is being trained.
set -u
PORT=${PORT:-8100}; NAME=${NAME:-customer}; UTIL=${UTIL:-0.15}; RUNS=${RUNS:-$HOME/runs}
PIDFILE=$RUNS/customer-server-$PORT.pid; LOG=$RUNS/customer-server-$PORT.log
url="http://127.0.0.1:$PORT/v1"
ready() { curl -sf -m 3 "$url/models" > /dev/null 2>&1; }

case "${1:-}" in
start)
  : "${GPU:?GPU id}" "${MODEL:?model dir}"
  ready && { echo "already serving on $url"; exit 0; }
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) || exit 3
  mkdir -p "$RUNS"
  setsid nohup bash -c '
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate rift-gpu-train
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='"$GPU"' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$HOME/tmp"
    exec vllm serve "'"$MODEL"'" --served-model-name '"$NAME"' --host 127.0.0.1 --port '"$PORT"' \
      --dtype bfloat16 --max-model-len 8192 --max-num-seqs 128 --gpu-memory-utilization '"$UTIL"' \
      --enable-auto-tool-choice --tool-call-parser hermes --seed 1
  ' > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 90); do ready && { echo "serving $NAME on $url (GPU $GPU had $used MiB in use)"; exit 0; }; sleep 5; done
  echo "not ready after 450 s; see $LOG" >&2; exit 4 ;;
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
