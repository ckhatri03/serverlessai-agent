#!/usr/bin/env sh
set -eu

export PORT="${PORT:-8000}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export MODELS_DIR="${MODELS_DIR:-$WORKSPACE_DIR/models}"
export OUTPUTS_DIR="${OUTPUTS_DIR:-$WORKSPACE_DIR/output}"
export WORKFLOWS_DIR="${WORKFLOWS_DIR:-$WORKSPACE_DIR/workflows}"

mkdir -p "$MODELS_DIR" "$OUTPUTS_DIR" "$WORKFLOWS_DIR"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

AGENT_LOG_FILE="${SERVERLESSAI_AGENT_LOG_FILE:-$WORKSPACE_DIR/agent.log}"

# Ensure log file exists and is writable
touch "$AGENT_LOG_FILE"

echo "Starting Serverless AI Agent, logging to $AGENT_LOG_FILE"
# Use tee to send logs to both stdout (for RunPod) and the log file
python "$SCRIPT_DIR/main.py" 2>&1 | tee "$AGENT_LOG_FILE"
