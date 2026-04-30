import os
import json
import shutil
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Any

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field


APP_VERSION = "0.1.0"


class Settings(BaseModel):
    agent_token: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_AGENT_TOKEN", ""))
    agent_token_file: Path = Field(default_factory=lambda: Path(os.getenv("SERVERLESSAI_AGENT_TOKEN_FILE", "/workspace/.serverlessai-agent-token")))
    register_token: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_AGENT_REGISTER_TOKEN", ""))
    control_plane_url: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_CONTROL_PLANE_URL", ""))
    comfyui_url: str = Field(default_factory=lambda: os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"))
    workspace_dir: Path = Field(default_factory=lambda: Path(os.getenv("WORKSPACE_DIR", "/workspace")))
    models_dir: Path = Field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "/workspace/models")))
    outputs_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUTS_DIR", "/workspace/output")))
    workflows_dir: Path = Field(default_factory=lambda: Path(os.getenv("WORKFLOWS_DIR", "/workspace/workflows")))
    pod_id: str = Field(default_factory=lambda: os.getenv("RUNPOD_POD_ID", os.getenv("RUNPOD_POD_HOSTNAME", "")))
    public_url: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_AGENT_PUBLIC_URL", ""))


settings = Settings()
app = FastAPI(title="Serverless AI Agent", version=APP_VERSION)


class HealthResponse(BaseModel):
    status: str
    version: str
    podId: str
    comfyui: str


class AgentStatusResponse(BaseModel):
    status: str
    version: str
    podId: str
    publicUrl: str
    workspaceDir: str
    modelsDir: str
    outputsDir: str
    workflowsDir: str
    comfyuiUrl: str
    comfyuiReachable: bool


class DownloadModelRequest(BaseModel):
    url: str
    destination: str
    overwrite: bool = False


class DownloadModelResponse(BaseModel):
    path: str
    bytes: int


class InstallWorkflowRequest(BaseModel):
    name: str
    graph: dict[str, Any]


class InstallWorkflowResponse(BaseModel):
    workflowPath: str


class RunWorkflowRequest(BaseModel):
    graph: dict[str, Any]
    clientId: str | None = None


class RunWorkflowResponse(BaseModel):
    promptId: str
    clientId: str


class WorkflowStatusResponse(BaseModel):
    promptId: str
    history: dict[str, Any]


class UploadOutputRequest(BaseModel):
    path: str
    destinationUrl: str | None = None


class UploadOutputResponse(BaseModel):
    path: str
    destinationUrl: str | None = None
    uploaded: bool


class ShutdownRequest(BaseModel):
    terminatePod: bool = False


def require_agent_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.agent_token:
        return

    expected = f"Bearer {settings.agent_token}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent token")


def ensure_child_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    target = (root / relative_path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Path escapes allowed directory")
    return target


def comfyui_get(path: str) -> requests.Response:
    try:
        return requests.get(f"{settings.comfyui_url.rstrip('/')}{path}", timeout=5)
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"ComfyUI unavailable: {exc}") from exc


def comfyui_post(path: str, payload: dict[str, Any]) -> requests.Response:
    try:
        return requests.post(f"{settings.comfyui_url.rstrip('/')}{path}", json=payload, timeout=30)
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"ComfyUI unavailable: {exc}") from exc


def comfyui_reachable() -> bool:
    try:
        response = requests.get(f"{settings.comfyui_url.rstrip('/')}/system_stats", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def load_persisted_agent_token() -> str:
    if settings.agent_token:
        return settings.agent_token
    if not settings.agent_token_file.exists():
        return ""
    return settings.agent_token_file.read_text().strip()


def persist_agent_token(token: str) -> None:
    settings.agent_token_file.parent.mkdir(parents=True, exist_ok=True)
    settings.agent_token_file.write_text(token)
    settings.agent_token_file.chmod(0o600)


def register_with_control_plane() -> None:
    persisted_token = load_persisted_agent_token()
    if persisted_token:
        settings.agent_token = persisted_token
        return

    if not settings.register_token or not settings.control_plane_url:
        return

    register_url = f"{settings.control_plane_url.rstrip('/')}/agents/register"
    payload = {
        "registerToken": settings.register_token,
        "podId": settings.pod_id,
        "publicUrl": settings.public_url,
        "version": APP_VERSION,
    }
    try:
        response = requests.post(register_url, json=payload, timeout=20)
        response.raise_for_status()
        permanent_token = response.json().get("permanentToken", "")
    except requests.RequestException as exc:
        raise RuntimeError(f"Agent registration failed: {exc}") from exc

    if not permanent_token:
        raise RuntimeError("Agent registration did not return a permanent token")

    persist_agent_token(permanent_token)
    settings.agent_token = permanent_token


@app.on_event("startup")
def prepare_workspace() -> None:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.workflows_dir.mkdir(parents=True, exist_ok=True)
    register_with_control_plane()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        podId=settings.pod_id,
        comfyui="ok" if comfyui_reachable() else "unreachable",
    )


@app.get("/status", response_model=AgentStatusResponse, dependencies=[Depends(require_agent_auth)])
async def status_view() -> AgentStatusResponse:
    return AgentStatusResponse(
        status="ready",
        version=APP_VERSION,
        podId=settings.pod_id,
        publicUrl=settings.public_url,
        workspaceDir=str(settings.workspace_dir),
        modelsDir=str(settings.models_dir),
        outputsDir=str(settings.outputs_dir),
        workflowsDir=str(settings.workflows_dir),
        comfyuiUrl=settings.comfyui_url,
        comfyuiReachable=comfyui_reachable(),
    )


@app.post("/download-model", response_model=DownloadModelResponse, dependencies=[Depends(require_agent_auth)])
async def download_model(request: DownloadModelRequest) -> DownloadModelResponse:
    target = ensure_child_path(settings.models_dir, request.destination)
    if target.exists() and not request.overwrite:
        return DownloadModelResponse(path=str(target), bytes=target.stat().st_size)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_suffix(target.suffix + ".download")

    try:
        with requests.get(request.url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with temporary_target.open("wb") as output:
                shutil.copyfileobj(response.raw, output)
        temporary_target.replace(target)
    except requests.RequestException as exc:
        if temporary_target.exists():
            temporary_target.unlink()
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Download failed: {exc}") from exc

    return DownloadModelResponse(path=str(target), bytes=target.stat().st_size)


@app.post("/install-workflow", response_model=InstallWorkflowResponse, dependencies=[Depends(require_agent_auth)])
async def install_workflow(request: InstallWorkflowRequest) -> InstallWorkflowResponse:
    workflow_path = ensure_child_path(settings.workflows_dir, f"{request.name}.json")
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(json.dumps(request.graph))
    return InstallWorkflowResponse(workflowPath=str(workflow_path))


@app.post("/run-workflow", response_model=RunWorkflowResponse, dependencies=[Depends(require_agent_auth)])
async def run_workflow(request: RunWorkflowRequest) -> RunWorkflowResponse:
    client_id = request.clientId or str(uuid.uuid4())
    response = comfyui_post("/prompt", {"prompt": request.graph, "client_id": client_id})
    if not response.ok:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=response.text)

    payload = response.json()
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="ComfyUI did not return prompt_id")

    return RunWorkflowResponse(promptId=prompt_id, clientId=client_id)


@app.get("/status/{prompt_id}", response_model=WorkflowStatusResponse, dependencies=[Depends(require_agent_auth)])
async def workflow_status(prompt_id: str) -> WorkflowStatusResponse:
    response = comfyui_get(f"/history/{prompt_id}")
    if not response.ok:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=response.text)
    return WorkflowStatusResponse(promptId=prompt_id, history=response.json())


@app.post("/upload-output", response_model=UploadOutputResponse, dependencies=[Depends(require_agent_auth)])
async def upload_output(request: UploadOutputRequest) -> UploadOutputResponse:
    output_path = ensure_child_path(settings.outputs_dir, request.path)
    if not output_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Output file not found")

    if not request.destinationUrl:
        return UploadOutputResponse(path=str(output_path), destinationUrl=None, uploaded=False)

    with output_path.open("rb") as data:
        response = requests.put(request.destinationUrl, data=data, timeout=60)
    if not response.ok:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Upload failed: {response.text}")

    return UploadOutputResponse(path=str(output_path), destinationUrl=request.destinationUrl, uploaded=True)


@app.post("/shutdown", dependencies=[Depends(require_agent_auth)])
async def shutdown(request: ShutdownRequest) -> dict[str, str | bool]:
    if request.terminatePod:
        return {"accepted": False, "reason": "pod termination is owned by the control plane"}

    subprocess.Popen(["/bin/sh", "-c", "sleep 1 && kill -TERM 1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"accepted": True}


if __name__ == "__main__":
    import uvicorn

    def handle_sigterm(*_: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
