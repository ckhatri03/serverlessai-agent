#!/usr/bin/env sh
set -eu

RAW_BASE="${SERVERLESSAI_AGENT_REPO_RAW_BASE:-https://raw.githubusercontent.com/ckhatri03/serverlessai-agent/main}"
INSTALL_DIR="${SERVERLESSAI_AGENT_INSTALL_DIR:-/workspace/serverlessai-agent}"
REQUIRE_CUDA="${SERVERLESSAI_REQUIRE_CUDA:-true}"
REQUIRE_PYTORCH="${SERVERLESSAI_REQUIRE_PYTORCH:-true}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
  printf '%s\n' "serverlessai-agent: $*"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

install_system_packages() {
  if have apt-get; then
    log "updating apt package index"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl git python3 python3-pip
    return
  fi

  log "apt-get not found; skipping OS package update"
}

check_python() {
  if ! have "$PYTHON_BIN"; then
    if have python; then
      PYTHON_BIN="python"
    else
      log "python is not available"
      exit 1
    fi
  fi

  "$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || {
    log "pip is not available for $PYTHON_BIN"
    exit 1
  }
}

check_cuda() {
  if [ "$REQUIRE_CUDA" != "true" ]; then
    return
  fi

  if have nvidia-smi; then
    nvidia-smi >/dev/null
    return
  fi

  if have nvcc; then
    nvcc --version >/dev/null
    return
  fi

  log "CUDA runtime check failed; nvidia-smi or nvcc is required before registration"
  exit 1
}

check_pytorch() {
  if [ "$REQUIRE_PYTORCH" != "true" ]; then
    return
  fi

  "$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")

if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is not available")

print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
PY
}

mkdir -p "$INSTALL_DIR"

install_system_packages
check_python
check_cuda
check_pytorch

log "downloading agent files"
curl -fsSL "$RAW_BASE/main.py" -o "$INSTALL_DIR/main.py"
curl -fsSL "$RAW_BASE/start-agent.sh" -o "$INSTALL_DIR/start-agent.sh"
curl -fsSL "$RAW_BASE/requirements.txt" -o "$INSTALL_DIR/requirements.txt"
chmod +x "$INSTALL_DIR/start-agent.sh"

log "installing agent Python dependencies"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$INSTALL_DIR/requirements.txt"

log "runtime ready; starting agent registration and API"
exec "$INSTALL_DIR/start-agent.sh"
