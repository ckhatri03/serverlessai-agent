#!/usr/bin/env sh
set -eu

export PORT="${PORT:-8000}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export MODELS_DIR="${MODELS_DIR:-$WORKSPACE_DIR/models}"
export OUTPUTS_DIR="${OUTPUTS_DIR:-$WORKSPACE_DIR/output}"
export WORKFLOWS_DIR="${WORKFLOWS_DIR:-$WORKSPACE_DIR/workflows}"
export COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"

mkdir -p "$MODELS_DIR" "$OUTPUTS_DIR" "$WORKFLOWS_DIR"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

exec python "$SCRIPT_DIR/main.py"
