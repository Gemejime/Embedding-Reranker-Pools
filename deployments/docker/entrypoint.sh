#!/usr/bin/env bash
# ==============================================================================
# Dual Process Supervisor for rag-inference-serving
# Supervises:
#   1. NVIDIA Triton Inference Server (ports 8000 HTTP, 8001 gRPC, 8002 Metrics)
#   2. Out-of-band Operational ASGI Server (port 8080 Probe/Doc API)
# ==============================================================================

set -euo pipefail

MODEL_REPOSITORY="${MODEL_REPOSITORY:-/models}"
TRITON_ARGS="${TRITON_ARGS:---strict-model-config=true --log-verbose=0}"
if [[ " ${TRITON_ARGS} " != *" --model-repository="* ]]; then
    TRITON_ARGS="--model-repository=${MODEL_REPOSITORY} ${TRITON_ARGS}"
fi
PROBE_SERVER_HOST="${RAG_SERVING_SERVER_HOST:-0.0.0.0}"
PROBE_SERVER_PORT="${RAG_SERVING_SERVER_PORT:-8080}"

echo "[SUPERVISOR] Starting NVIDIA Triton Inference Server..."
tritonserver ${TRITON_ARGS} &
TRITON_PID=$!

echo "[SUPERVISOR] Starting Out-of-Band Operational Probe Server on ${PROBE_SERVER_HOST}:${PROBE_SERVER_PORT}..."
python3 -m uvicorn internal.server.probe_server:app \
    --host "${PROBE_SERVER_HOST}" \
    --port "${PROBE_SERVER_PORT}" \
    --log-level info &
PROBE_PID=$!

# Graceful drain signal handler
cleanup() {
    local exit_code="${1:-0}"
    echo "[SUPERVISOR] Termination signal or exit received. Draining child processes (status: ${exit_code})..."

    if kill -0 "${PROBE_PID}" 2>/dev/null; then
        echo "[SUPERVISOR] Stopping operational probe server (PID ${PROBE_PID})..."
        kill -TERM "${PROBE_PID}"
    fi

    if kill -0 "${TRITON_PID}" 2>/dev/null; then
        echo "[SUPERVISOR] Stopping Triton inference server (PID ${TRITON_PID})..."
        kill -TERM "${TRITON_PID}"
    fi

    wait "${PROBE_PID}" 2>/dev/null || true
    wait "${TRITON_PID}" 2>/dev/null || true
    echo "[SUPERVISOR] All child processes drained cleanly. Exiting with status ${exit_code}."
    exit "${exit_code}"
}

trap 'cleanup 0' SIGTERM SIGINT

# Wait for any process to exit without set -e aborting
EXIT_STATUS=0
wait -n "${TRITON_PID}" "${PROBE_PID}" || EXIT_STATUS=$?

if [ "${EXIT_STATUS}" -eq 0 ]; then
    # Neither daemon should exit normally on its own; treat unexpected exit as failure
    EXIT_STATUS=1
fi

echo "[SUPERVISOR] A supervised process exited with status ${EXIT_STATUS}. Initiating shutdown."
cleanup "${EXIT_STATUS}"
