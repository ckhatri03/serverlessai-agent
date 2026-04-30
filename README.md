# serverlessai-agent

The Python-based execution agent for RunPod / ComfyUI workers.

## Overview
`serverlessai-agent` is the lightweight worker that runs inside GPU pods. It acts as the bridge between the `serverlessai` control plane and the AI execution environment (ComfyUI).

## Responsibilities
- **Model Management:** Downloading weights from S3 or Hugging Face.
- **Workflow Execution:** Triggering ComfyUI API with provided JSON graphs.
- **Output Handling:** Uploading generated images/videos to secure storage.
- **Health Monitoring:** Reporting pod status and resource utilization.

## Tech Stack
- **Language:** Python 3.10+
- **Framework:** FastAPI / Uvicorn
- **AI Tooling:** ComfyUI, PyTorch
- **Communication:** REST API

## Runtime Contract

The agent is designed to run inside the RunPod instance provisioned by the `serverlessai` control plane. The control plane owns pod lifecycle and calls the agent over HTTP after the pod is reachable.

### Environment Variables

| Name | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | Agent HTTP port. |
| `SERVERLESSAI_AGENT_TOKEN` | empty | Optional bearer token required by all non-health endpoints. |
| `SERVERLESSAI_AGENT_TOKEN_FILE` | `/workspace/.serverlessai-agent-token` | File used to persist the permanent agent token after registration. |
| `SERVERLESSAI_AGENT_REGISTER_TOKEN` | empty | One-time control-plane registration token injected by the generated RunPod template. |
| `SERVERLESSAI_CONTROL_PLANE_URL` | empty | Future callback URL for job/status reporting. |
| `SERVERLESSAI_AGENT_PUBLIC_URL` | empty | Public URL assigned to this pod/agent by RunPod. |
| `SERVERLESSAI_REQUIRE_CUDA` | `true` | Installer fails before registration unless CUDA is visible through `nvidia-smi` or `nvcc`. |
| `SERVERLESSAI_REQUIRE_PYTORCH` | `true` | Installer fails before registration unless PyTorch imports and CUDA is available to PyTorch. |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API base URL inside the pod. |
| `WORKSPACE_DIR` | `/workspace` | Base persistent workspace path. |
| `MODELS_DIR` | `/workspace/models` | Safe root for model downloads. |
| `OUTPUTS_DIR` | `/workspace/output` | Safe root for generated outputs. |
| `WORKFLOWS_DIR` | `/workspace/workflows` | Safe root for saved workflow JSON. |
| `RUNPOD_POD_ID` | empty | RunPod pod id when provided by the runtime/template. |

### Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | no | Readiness check for RunPod/control plane. |
| `GET` | `/status` | yes | Agent, workspace, and ComfyUI status. |
| `POST` | `/download-model` | yes | Download a model file under `MODELS_DIR`. |
| `POST` | `/install-workflow` | yes | Save a ComfyUI workflow graph under `WORKFLOWS_DIR`. |
| `POST` | `/run-workflow` | yes | Submit a workflow graph to ComfyUI `/prompt`. |
| `GET` | `/status/{prompt_id}` | yes | Fetch ComfyUI history for a prompt. |
| `POST` | `/upload-output` | yes | Upload an output file to a signed destination URL. |
| `POST` | `/shutdown` | yes | Stop the agent process; pod termination remains control-plane owned. |

When `SERVERLESSAI_AGENT_TOKEN` is set, send:

```http
Authorization: Bearer <token>
```

## RunPod Usage

The public repository can be used in a generated RunPod template by injecting `SERVERLESSAI_AGENT_REGISTER_TOKEN`,
`SERVERLESSAI_CONTROL_PLANE_URL`, and an install script URL. The template startup command can install and run the agent
without building a custom image:

```bash
curl -fsSL "$SERVERLESSAI_AGENT_INSTALL_URL" | sh
```

The installer updates Ubuntu packages when `apt-get` is available, verifies Python, pip, curl, CUDA, and PyTorch before
starting the agent. Registration happens only after these readiness checks pass.

The agent expects ComfyUI to be reachable from inside the same pod at `COMFYUI_URL`.

## Documentation
- [Disclaimer](./DISCLAIMER.md)
- [License](./LICENSE)

---
© 2026 Chirayu Khatri / Rectrix. All rights reserved.
