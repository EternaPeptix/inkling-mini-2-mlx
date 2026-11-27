#!/bin/zsh
set -euo pipefail

ROOT="${LOOPLING_OPT_ROOT:-$HOME/loopling-opt}"
SRC="${LOOPLING_SRC_DIR:-$ROOT/src-main}"
VENV_PY="${LOOPLING_VENV_PY:-$ROOT/.venv/bin/python}"
MODEL_PATH="${LOOPLING_MODEL_PATH:-$HOME/loopling/models/loopling-p3-tess}"
SERVED_MODEL_NAME="${LOOPLING_SERVED_MODEL_NAME:-$HOME/loopling/models/loopling-p3}"
MODEL_TAG="$(basename "$MODEL_PATH")"
HOST="${LOOPLING_HOST:-0.0.0.0}"
PORT="${LOOPLING_PORT:-8080}"
SSD_CACHE_DIR="${LOOPLING_SSD_CACHE_DIR:-$ROOT/vllm_ssd_cache/$MODEL_TAG}"
SSD_CACHE_GB="${LOOPLING_SSD_CACHE_GB:-1500}"
CACHE_MEM_PCT="${LOOPLING_CACHE_MEM_PCT:-0.70}"
REQUEST_TIMEOUT="${LOOPLING_TIMEOUT:-7200}"
LOG_FILE="${LOOPLING_LOG_FILE:-$ROOT/vllm_mlx_p3_${PORT}.log}"
PID_FILE="${LOOPLING_PID_FILE:-$ROOT/vllm_mlx_p3_${PORT}.pid}"

export PATH="$HOME/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPATH="$SRC/vendor:$SRC"
export HF_HUB_OFFLINE=1
export LOOPLING_SERVER_BATCHDECODE=1
export LOOPLING_LOOPCD="${LOOPLING_LOOPCD:-off}"
export LOOPLING_MTP="${LOOPLING_MTP:-dflash2}"

mkdir -p "$SSD_CACHE_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
fi

nohup "$VENV_PY" -m vllm_mlx_inkling2.serve serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --continuous-batching \
    --max-tokens 32768 \
    --max-request-tokens 1048576 \
    --timeout "$REQUEST_TIMEOUT" \
    --default-temperature 0.7 \
    --default-top-p 0.95 \
    --prefill-step-size "${LOOPLING_PREFILL_STEP_SIZE:-2048}" \
    --cache-memory-percent "$CACHE_MEM_PCT" \
    --ssd-cache-dir "$SSD_CACHE_DIR" \
    --ssd-cache-max-gb "$SSD_CACHE_GB" \
    --reasoning-parser inkling2 \
    --enable-auto-tool-choice \
    --tool-call-parser inkling2 \
    > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started vllm_mlx_inkling2 PID=$! port=$PORT log=$LOG_FILE"
