#!/usr/bin/env bash
# One-shot: boot the RLDX-1-PT DROID ZeroMQ server, wait until it is loaded,
# run the smoke test against it, then shut it down. Prints PASS/FAIL.
#
# Use this to verify the server end-to-end on this machine. For a long-running
# server that sim-evals connects to, use run_rldx_pt_droid_server.sh directly.
#
# Usage:
#   run_scripts/serve/verify_droid_server.sh
#   PORT=6000 run_scripts/serve/verify_droid_server.sh
set -uo pipefail

cd "$(dirname "$0")/../.."

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-600}"   # seconds to wait for the model to load
LOG="${LOG:-/tmp/rldx_pt_droid_server.log}"

echo "Starting server in background (log: ${LOG}) ..."
HOST="${HOST}" PORT="${PORT}" run_scripts/serve/run_rldx_pt_droid_server.sh >"${LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "Stopping server (pid ${SERVER_PID}) ..."
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting up to ${LOAD_TIMEOUT}s for the server to come up ..."
deadline=$((SECONDS + LOAD_TIMEOUT))
until "${PYTHON}" - "${HOST}" "${PORT}" <<'PY' 2>/dev/null
import sys
from rldx.policy.server_client import PolicyClient
c = PolicyClient(host=sys.argv[1], port=int(sys.argv[2]), timeout_ms=2000)
sys.exit(0 if c.ping() else 1)
PY
do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "FAIL: server process died during load. Last log lines:"
    tail -n 40 "${LOG}"
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "FAIL: server did not respond to ping within ${LOAD_TIMEOUT}s. Last log lines:"
    tail -n 40 "${LOG}"
    exit 1
  fi
  sleep 3
done

echo "Server is up. Running smoke test ..."
"${PYTHON}" run_scripts/serve/smoke_test_droid_server.py --host "${HOST}" --port "${PORT}"
RESULT=$?

exit "${RESULT}"
