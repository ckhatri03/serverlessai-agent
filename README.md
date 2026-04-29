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

## Documentation
- [Disclaimer](./DISCLAIMER.md)
- [License](./LICENSE)

---
© 2026 Chirayu Khatri / Rectrix. All rights reserved.
