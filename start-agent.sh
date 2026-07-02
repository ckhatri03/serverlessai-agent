#!/usr/bin/env sh
set -eu

export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Detect ComfyUI and favor its model directory for persistence
if [ -d "/workspace/ComfyUI/models" ] && [ -z "${MODELS_DIR:-}" ]; then
  export MODELS_DIR="/workspace/ComfyUI/models"
else
  export MODELS_DIR="${MODELS_DIR:-$WORKSPACE_DIR/models}"
fi

export PORT="${PORT:-8000}"
export OUTPUTS_DIR="${OUTPUTS_DIR:-$WORKSPACE_DIR/output}"
export WORKFLOWS_DIR="${WORKFLOWS_DIR:-$WORKSPACE_DIR/workflows}"

mkdir -p "$MODELS_DIR" "$OUTPUTS_DIR" "$WORKFLOWS_DIR"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

AGENT_LOG_FILE="${SERVERLESSAI_AGENT_LOG_FILE:-$WORKSPACE_DIR/agent.log}"

# Ensure log file exists and is writable
touch "$AGENT_LOG_FILE"

echo "Starting Serverless AI Agent, logging to $AGENT_LOG_FILE"
# Use the venv's python if it exists, otherwise fall back to system python
PYTHON_EXEC="$SCRIPT_DIR/venv/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
  PYTHON_EXEC="python"
fi

while true; do
  "$PYTHON_EXEC" -u "$SCRIPT_DIR/main.py" 2>&1 | tee -a "$AGENT_LOG_FILE"
  echo "Agent process exited. Restarting in 2 seconds..." | tee -a "$AGENT_LOG_FILE"
  sleep 2
done
