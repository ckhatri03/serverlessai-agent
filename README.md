# Serverless AI Agent

The Python-based execution agent for RunPod workers.

`serverlessai-agent` is the lightweight worker that runs inside GPU pods. It acts as the bridge between the `serverlessai` control plane and the native Python AI inference environment using `diffusers`, `transformers`, and other core libraries.

## Features

- **Automated Registration:** Self-registers with the Go control plane on startup.
- **Native Inference:** Supports `txt2img`, `img2img`, `txt2vid`, `img2vid`, `faceswap`, and `controlnet` using standard Python pipelines.
- **Optimized Execution:** Uses PyTorch, CUDA, and `xformers` for high-performance inference.
- **Memory Management:** Implements dynamic model offloading to support large models (like Flux.1) on consumer-grade hardware.
- **Model Management:** Native support for downloading models directly from Hugging Face and Civitai.
- **Secure Commands:** Restricted `/exec` endpoint for administrative tasks and dependency management.

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port the agent API listens on. |
| `SERVERLESSAI_AGENT_TOKEN` | | Optional permanent token (bypass registration). |
| `SERVERLESSAI_AGENT_REGISTER_TOKEN` | | Token used to register with the control plane. |
| `SERVERLESSAI_CONTROL_PLANE_URL` | | URL of the Go control plane. |
| `WORKSPACE_DIR` | `/workspace` | Root workspace directory. |
| `MODELS_DIR` | `/workspace/models` | Where model weights are stored. |
| `OUTPUTS_DIR` | `/workspace/output` | Where generated media is saved. |
| `HF_TOKEN` | | Hugging Face token for private models. |

## API Reference (Partial)

| Method | Path | Auth | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | no | Basic liveness check. |
| `GET` | `/status` | yes | Agent and workspace status. |
| `POST` | `/api/v1/txt2img` | yes | Generate image from text. |
| `POST` | `/api/v1/img2img` | yes | Generate image from image. |
| `POST` | `/api/v1/txt2vid` | yes | Generate video from text (e.g. Wan 2.1). |
| `POST` | `/api/v1/faceswap` | yes | Swap faces between images. |
| `POST` | `/download-model` | yes | Trigger a model download to the pod. |
| `POST` | `/exec` | yes | Run arbitrary shell commands. |

## Installation

The agent is typically installed via the `install.sh` script provided by the control plane when a pod is provisioned.

```bash
# Manual start
./start-agent.sh
```
