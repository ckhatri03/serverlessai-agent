import asyncio
import os
import json
import mimetypes
import shutil
import signal
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import torch
from PIL import Image
from diffusers import (
    AutoPipelineForText2Image, 
    AutoPipelineForImage2Image, 
    ControlNetModel, 
    StableDiffusionControlNetPipeline,
    WanPipeline, 
    WanImageToVideoPipeline,
    FluxPipeline,
    FluxImg2ImgPipeline,
    # Schedulers
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler,
    # Specific Pipelines for single file loading
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from diffusers.utils import export_to_video
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field, HttpUrl


APP_VERSION = "0.1.1"


class Settings(BaseModel):
    agent_id: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_AGENT_ID", ""))
    agent_token: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_AGENT_TOKEN", ""))
    agent_token_file: Path = Field(default_factory=lambda: Path(os.getenv("SERVERLESSAI_AGENT_TOKEN_FILE", "/workspace/.serverlessai-agent-token")))
    agent_id_file: Path = Field(default_factory=lambda: Path(os.getenv("SERVERLESSAI_AGENT_ID_FILE", "/workspace/.serverlessai-agent-id")))
    register_token: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_AGENT_REGISTER_TOKEN", ""))
    control_plane_url: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_CONTROL_PLANE_URL", ""))
    workspace_dir: Path = Field(default_factory=lambda: Path(os.getenv("WORKSPACE_DIR", "/workspace")))
    models_dir: Path = Field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "/workspace/models")))
    outputs_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUTS_DIR", "/workspace/output")))
    workflows_dir: Path = Field(default_factory=lambda: Path(os.getenv("WORKFLOWS_DIR", "/workspace/workflows")))
    pod_id: str = Field(default_factory=lambda: os.getenv("RUNPOD_POD_ID", os.getenv("RUNPOD_POD_HOSTNAME", "")))
    public_url: str = Field(default_factory=lambda: os.getenv("SERVERLESSAI_AGENT_PUBLIC_URL", ""))
    agent_log_file: Path = Field(default_factory=lambda: Path(os.getenv("SERVERLESSAI_AGENT_LOG_FILE", "/workspace/agent.log")))
    hf_token: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    hf_home: Path = Field(default_factory=lambda: Path(os.getenv("HF_HOME", "/workspace/huggingface")))

    @property
    def effective_public_url(self) -> str:
        if self.public_url:
            return self.public_url
        if self.pod_id:
            # Construct RunPod proxy URL
            return f"https://{self.pod_id}-8000.proxy.runpod.net"
        return ""


settings = Settings()

# Global state for guardrails
active_downloads = set()
download_lock = asyncio.Lock()
active_generation_job = None
job_lock = asyncio.Lock()

@asynccontextmanager
async def generation_session(job_id: str):
    """Guardrail to ensure only one generation job runs at a time."""
    global active_generation_job
    if active_generation_job is not None and active_generation_job != job_id:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Another job is already running: {active_generation_job}. Please wait for it to complete."
        )
    
    # If it's already set to the same job_id, we are in a nested call (e.g. from run_generate_job)
    is_nested = (active_generation_job == job_id)
    
    if not is_nested:
        active_generation_job = job_id
        log("info", f"Started generation session for job: {job_id}")
        
    try:
        yield
    finally:
        if not is_nested:
            if active_generation_job == job_id:
                active_generation_job = None
                log("info", f"Finished generation session for job: {job_id}")

# Force HF and Transformers cache to /workspace to avoid filling container disk
os.environ["HF_HOME"] = str(settings.hf_home)
os.environ["TRANSFORMERS_CACHE"] = str(settings.hf_home / "transformers")
os.environ["HF_HUB_CACHE"] = str(settings.hf_home / "hub")
os.environ["XDG_CACHE_HOME"] = str(settings.workspace_dir / ".cache")


def log(level: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{timestamp} [{level.upper()}] {message}"
    print(line, flush=True)


def control_plane_root_url() -> str:
    url = settings.control_plane_url.rstrip("/")
    for suffix in ("/api/v1", "/api"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def control_plane_v1_url() -> str:
    url = settings.control_plane_url.rstrip("/")
    if url.endswith("/api/v1"):
        return url
    if url.endswith("/api"):
        return f"{url}/v1"
    return f"{url}/api/v1"


def report_workflow_event(user_id: str, workflow_id: str, event: str, success: bool, 
                          error: str = "", output: str = "", agent_id: str | None = None,
                          current_step: int = 0, total_steps: int = 0, 
                          batch_index: int = 0, batch_size: int = 0, 
                          percentage: float = 0.0):
    if not settings.control_plane_url:
        return

    try:
        # Use a dedicated public endpoint prefix to avoid Cognito Authorizer issues
        url = f"{control_plane_root_url()}/api/agent/events"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Token": settings.agent_token
        }
        payload = {
            "userId": user_id,
            "agentId": agent_id or settings.agent_id,
            "workflowId": workflow_id,
            "event": event,
            "success": success,
            "error": error,
            "output": output,
            "currentStep": current_step,
            "totalSteps": total_steps,
            "batchIndex": batch_index,
            "batchSize": batch_size,
            "percentage": percentage
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code != 200:
            log("error", f"Failed to report workflow event: {resp.status_code} {resp.text}")
    except Exception as exc:
        log("error", f"Error reporting workflow event: {exc}")


def apply_scheduler(pipe, sampler_name: str | None, scheduler_type: str | None):
    if not sampler_name:
        return

    # Map generic names to diffusers scheduler classes
    mapping = {
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpmpp_2m": DPMSolverMultistepScheduler,
        "dpmpp_2m_sde": DPMSolverMultistepScheduler,
        "dpmpp_sde": DPMSolverSDEScheduler,
        "lms": LMSDiscreteScheduler,
        "pndm": PNDMScheduler,
        "ddim": DDIMScheduler,
        "flow_match": FlowMatchEulerDiscreteScheduler,
    }

    scheduler_class = mapping.get(sampler_name.lower())
    if not scheduler_class:
        log("warn", f"Unsupported sampler: {sampler_name}, skipping scheduler override")
        return

    if isinstance(pipe, (FluxPipeline, FluxImg2ImgPipeline)) and scheduler_class is not FlowMatchEulerDiscreteScheduler:
        log("info", f"Keeping native Flux scheduler; sampler {sampler_name} is not compatible with Flux timesteps")
        return

    # Handle scheduler types (karras, etc.)
    kwargs = {}
    if scheduler_type == "karras":
        kwargs["use_karras_sigmas"] = True
    elif scheduler_type == "exponential":
        kwargs["use_exponential_sigmas"] = True

    try:
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config, **kwargs)
        log("info", f"Applied scheduler: {sampler_name} (type={scheduler_type or 'default'})")
    except Exception as exc:
        log("error", f"Failed to apply scheduler {sampler_name}: {exc}")


async def ensure_git_repo() -> None:
    try:
        # Check if .git exists
        if not (Path.cwd() / ".git").exists():
            print("Initializing git repository...")
            process = await asyncio.create_subprocess_exec(
                "git", "init",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            process = await asyncio.create_subprocess_exec(
                "git", "remote", "add", "origin", "https://github.com/ckhatri03/serverlessai-agent.git",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            process = await asyncio.create_subprocess_exec(
                "git", "fetch",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            # Force current files to match origin/main without deleting anything important
            process = await asyncio.create_subprocess_exec(
                "git", "checkout", "-t", "origin/main", "-f",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
    except Exception as exc:
        print(f"Failed to initialize git repo: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_git_repo()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.workflows_dir.mkdir(parents=True, exist_ok=True)
    
    register_with_control_plane()
    yield


app = FastAPI(title="Serverless AI Agent", version=APP_VERSION, lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    error_msg = f"Unhandled error: {str(exc)}\n{traceback.format_exc()}"
    log("error", error_msg)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()},
    )

from fastapi.responses import JSONResponse


class HealthResponse(BaseModel):
    status: str
    version: str
    podId: str


class AgentStatusResponse(BaseModel):
    status: str
    version: str
    podId: str
    publicUrl: str
    workspaceDir: str
    modelsDir: str
    outputsDir: str
    workflowsDir: str


class SystemInfoResponse(BaseModel):
    pytorch: str
    cuda: str
    nvidia_driver: str


class DownloadModelRequest(BaseModel):
    url: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    destination: str
    overwrite: bool = False
    hf_token: str | None = None


class DownloadModelResponse(BaseModel):
    path: str
    bytes: int


class ExecRequest(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    background: bool = False


class ExecResponse(BaseModel):
    stdout: str
    stderr: str
    exitCode: int


class OutputImage(BaseModel):
    name: str
    path: str
    root: str
    subfolder: str
    sizeBytes: int
    modifiedAt: str
    type: str = "output"


class OutputsResponse(BaseModel):
    images: list[OutputImage]
    total: int
    page: int
    pageSize: int
    totalPages: int
    outputsDir: str


class LogsResponse(BaseModel):
    logs: str
    path: str


class UploadOutputRequest(BaseModel):
    path: str
    destinationUrl: str | None = None


class UploadOutputResponse(BaseModel):
    path: str
    destinationUrl: str | None = None
    uploaded: bool


class ShutdownRequest(BaseModel):
    terminatePod: bool = False


class Text2ImageRequest(BaseModel):
    user_id: str | None = None
    job_id: str | None = None
    model_id: str
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = -1
    sampler_name: str | None = None
    scheduler: str | None = None
    batch_size: int = 1
    loras: list[dict] = Field(default_factory=list)
    embeddings: list[dict] = Field(default_factory=list)
    hf_token: str | None = None


class Image2ImageRequest(BaseModel):
    user_id: str | None = None
    job_id: str | None = None
    model_id: str
    image: str  # File path relative to input root or URL
    prompt: str
    negative_prompt: str | None = None
    strength: float = 0.8
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = -1
    sampler_name: str | None = None
    scheduler: str | None = None
    batch_size: int = 1
    loras: list[dict] = Field(default_factory=list)
    embeddings: list[dict] = Field(default_factory=list)
    hf_token: str | None = None


class InferenceResponse(BaseModel):
    image_path: str
    seed: int
    duration: float
    image_paths: list[str] = Field(default_factory=list)


class GenerateJobRequest(BaseModel):
    job_id: str | None = None
    endpoint: str
    payload: dict[str, Any]


class GenerateJobResponse(BaseModel):
    job_id: str
    status: str


class ControlNetRequest(BaseModel):
    user_id: str | None = None
    job_id: str | None = None
    model_id: str
    controlnet_model_id: str
    image: str  # Control image (path or URL)
    prompt: str
    negative_prompt: str | None = None
    controlnet_conditioning_scale: float = 1.0
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = -1
    loras: list[dict] = Field(default_factory=list)
    embeddings: list[dict] = Field(default_factory=list)
    hf_token: str | None = None


class FaceSwapRequest(BaseModel):
    user_id: str | None = None
    job_id: str | None = None
    source_image: str  # Face to use (path or URL)
    target_image: str  # Image to swap face into (path or URL)
    model_path: str | None = None  # Path to inswapper model


class OpenPoseRequest(BaseModel):
    image: str  # Source image (path or URL)
    include_body: bool = True
    include_hand: bool = False
    include_face: bool = False


class Text2VideoRequest(BaseModel):
    user_id: str | None = None
    job_id: str | None = None
    model_id: str
    prompt: str
    negative_prompt: str | None = None
    width: int = 832
    height: int = 480
    num_frames: int = 81
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    seed: int = -1
    fps: int = 16
    loras: list[dict] = Field(default_factory=list)
    embeddings: list[dict] = Field(default_factory=list)
    hf_token: str | None = None


class Image2VideoRequest(BaseModel):
    user_id: str | None = None
    job_id: str | None = None
    model_id: str
    image: str  # Source image (path or URL)
    prompt: str
    negative_prompt: str | None = None
    width: int = 1280
    height: int = 720
    num_frames: int = 81
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    seed: int = -1
    fps: int = 16
    loras: list[dict] = Field(default_factory=list)
    embeddings: list[dict] = Field(default_factory=list)
    hf_token: str | None = None


class UpscaleRequest(BaseModel):
    image: str  # Source image (path or URL)
    upscale_model_id: str | None = None  # e.g. 4x-UltraSharp.pth
    upscale_factor: float = 4.0


class UpscaleResponse(BaseModel):
    image_path: str
    duration: float


def require_agent_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.agent_token:
        return

    expected = f"Bearer {settings.agent_token}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent token")


class SystemCheckResponse(BaseModel):
    status: str
    families: dict[str, bool]
    workflows: dict[str, bool]
    storage: dict[str, Any]


@app.get("/api/v1/system/check", response_model=SystemCheckResponse, dependencies=[Depends(require_agent_auth)])
async def system_check():
    # Check for model families
    wan_diffusers_dir = settings.models_dir / "diffusers"
    families = {
        "sdxl": (settings.models_dir / "checkpoints/sd_xl_base_1.0.safetensors").exists() or any((settings.models_dir / "checkpoints").glob("*xl*")),
        "flux": (settings.models_dir / "checkpoints/flux1-schnell.safetensors").exists() or (settings.models_dir / "checkpoints/flux1-dev.safetensors").exists() or any((settings.models_dir / "checkpoints").glob("*flux*")),
        "qwen": any((settings.models_dir / "checkpoints").glob("*qwen*")),
        "wan": any((settings.models_dir / "checkpoints").glob("*wan*")) or any(wan_diffusers_dir.glob("*Wan*")) or any(wan_diffusers_dir.glob("*wan*")),
        "zit": any((settings.models_dir / "checkpoints").glob("*zit*")),
        "illustrious": any((settings.models_dir / "checkpoints").glob("*illustrious*")),
        "pony": any((settings.models_dir / "checkpoints").glob("*pony*")),
    }

    # Check for workflows
    workflows = {
        "txt2image": families["sdxl"] or families["flux"] or families["zit"] or families["illustrious"] or families["pony"],
        "img2image": families["sdxl"] or families["flux"] or families["zit"] or families["illustrious"] or families["pony"],
        "txt2vid": families["wan"] or families["qwen"],
        "img2vid": families["wan"] or families["qwen"],
        "faceswap": (settings.models_dir / "insightface/inswapper_128.onnx").exists(),
        "controlnet": any((settings.models_dir / "controlnet").glob("*")),
        "openpose": (settings.models_dir / "controlnet/control_v11p_sd15_openpose.pth").exists() or any((settings.models_dir / "controlnet").glob("*openpose*")),
    }

    # Storage info
    import shutil
    total, used, free = shutil.disk_usage(settings.workspace_dir)

    return SystemCheckResponse(
        status="ok",
        families=families,
        workflows=workflows,
        storage={
            "total": total,
            "used": used,
            "free": free,
            "percent": round((used / total) * 100, 2)
        }
    )


def ensure_child_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    target = (root / relative_path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Path escapes allowed directory")
    return target


def ensure_output_file(relative_path: str) -> Path:
    output_path = ensure_child_path(settings.outputs_dir, relative_path)
    if not output_path.exists() or not output_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Output file not found")
    return output_path


def get_pytorch_version() -> str:
    try:
        result = subprocess.run(
            ["python3", "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "not installed"


def get_cuda_version() -> str:
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line:
                    return line.split("release")[-1].strip().split(",")[0]
    except Exception:
        pass
    return "not detected"


def get_nvidia_driver_version() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "not detected"


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


def load_persisted_agent_id() -> str:
    if settings.agent_id:
        return settings.agent_id
    if not settings.agent_id_file.exists():
        return ""
    return settings.agent_id_file.read_text().strip()


def persist_agent_id(agent_id: str) -> None:
    settings.agent_id_file.parent.mkdir(parents=True, exist_ok=True)
    settings.agent_id_file.write_text(agent_id)
    settings.agent_id_file.chmod(0o600)


def register_with_control_plane() -> None:
    persisted_token = load_persisted_agent_token()
    persisted_id = load_persisted_agent_id()
    if persisted_token and persisted_id:
        settings.agent_token = persisted_token
        settings.agent_id = persisted_id
        return

    if not settings.register_token or not settings.control_plane_url:
        return

    register_url = f"{control_plane_v1_url()}/agents/register"
    payload = {
        "registerToken": settings.register_token,
        "podId": settings.pod_id,
        "publicUrl": settings.effective_public_url,
        "version": APP_VERSION,
        "systemInfo": {
            "pytorch": get_pytorch_version(),
            "cuda": get_cuda_version(),
            "nvidia_driver": get_nvidia_driver_version(),
        },
    }
    try:
        response = requests.post(register_url, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        permanent_token = data.get("permanentToken", "")
        agent_id = data.get("agentId", "")
    except requests.RequestException as exc:
        raise RuntimeError(f"Agent registration failed: {exc}") from exc

    if not permanent_token or not agent_id:
        raise RuntimeError("Agent registration did not return a permanent token or agent ID")

    persist_agent_token(permanent_token)
    persist_agent_id(agent_id)
    settings.agent_token = permanent_token
    settings.agent_id = agent_id


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        podId=settings.pod_id,
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
    )


@app.get("/system-info", response_model=SystemInfoResponse, dependencies=[Depends(require_agent_auth)])
async def system_info() -> SystemInfoResponse:
    return SystemInfoResponse(
        pytorch=get_pytorch_version(),
        cuda=get_cuda_version(),
        nvidia_driver=get_nvidia_driver_version(),
    )


import uvicorn
import logging

# Suppress verbose uvicorn access logs for health and status polling
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/logs") == -1 and \
               record.getMessage().find("/health") == -1 and \
               record.getMessage().find("/status") == -1

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


@app.post("/download-model", response_model=DownloadModelResponse, dependencies=[Depends(require_agent_auth)])
async def download_model(request: DownloadModelRequest) -> DownloadModelResponse:
    target = ensure_child_path(settings.models_dir, request.destination)
    
    # Guardrail 1: check if already exists
    if target.exists() and not request.overwrite:
        log("info", f"Download skipped, file already exists: {target}")
        return DownloadModelResponse(path=str(target), bytes=target.stat().st_size)

    # Guardrail 2: only one download at a time
    if download_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
            detail="Another download is already in progress. Please wait for it to complete."
        )

    async with download_lock:
        return await asyncio.to_thread(_download_model_sync, request)


def _download_model_sync(request: DownloadModelRequest) -> DownloadModelResponse:
    target = ensure_child_path(settings.models_dir, request.destination)
    # Double check inside sync just in case
    if target.exists() and not request.overwrite:
        return DownloadModelResponse(path=str(target), bytes=target.stat().st_size)

    target.parent.mkdir(parents=True, exist_ok=True)
    
    effective_hf_token = request.hf_token or settings.hf_token or None
    temporary_target = target.with_suffix(target.suffix + ".download")

    # Guardrail 3: remove partial file if it exists from a previous failed/interrupted attempt
    if temporary_target.exists():
        log("info", f"Removing partial download file: {temporary_target}")
        temporary_target.unlink()

    if request.repo_id:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(
                repo_id=request.repo_id,
                filename=request.filename,
                local_dir=target.parent,
                token=effective_hf_token
            )
            downloaded_path = Path(path)
            if downloaded_path.resolve() != target.resolve():
                if target.exists():
                    target.unlink()
                downloaded_path.replace(target)
            return DownloadModelResponse(path=str(target), bytes=target.stat().st_size)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"HF Download failed: {exc}")

    if not request.url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either url or repo_id must be provided")

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


def get_progress_callback(user_id: str, workflow_id: str, batch_index: int, total_batch: int, total_steps: int):
    """Returns a callback for Diffusers to log progress."""
    def callback(pipe, step_index, timestep, callback_kwargs):
        # step_index is 0-indexed
        progress = (step_index + 1) / total_steps * 100
        if (step_index + 1) % 5 == 0 or (step_index + 1) == total_steps:
            log("info", f"[PROGRESS] Image {batch_index + 1}/{total_batch} Step {step_index + 1}/{total_steps} ({progress:.1f}%)")
            # Reporting to control plane for UI progress bars
            report_workflow_event(
                user_id, workflow_id, "generation-progress", True, 
                current_step=step_index + 1, total_steps=total_steps,
                batch_index=batch_index, # Number of fully completed images so far
                batch_size=total_batch,
                percentage=progress
            )
        return callback_kwargs
    return callback


class PipelineManager:
    def __init__(self):
        self.current_model_id = None
        self.current_controlnet_id = None
        self.pipeline = None
        self.type = None # "t2i", "i2i", "controlnet"
        self.has_loras = False

    def resolve_model_id(self, model_id: str) -> str:
        mappings = {
            "sdxl-base": "stabilityai/stable-diffusion-xl-base-1.0",
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            "flux-schnell": "black-forest-labs/FLUX.1-schnell",
            "flux-dev": "black-forest-labs/FLUX.1-dev",
            "flux": "black-forest-labs/FLUX.1-schnell",
            "wan-2.1-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "wan-2.1-14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "zit": "Alibaba-ALP/Z-Image-Turbo-Diffusers",
        }
        return mappings.get(model_id.lower(), model_id)

    def load_pipeline(self, model_id: str, task: str = "t2i", controlnet_id: str | None = None, hf_token: str | None = None):
        model_id = self.resolve_model_id(model_id)
        # If we had LoRAs, we must reload to clear them (or unload if diffusers supports it well)
        if self.current_model_id == model_id and self.type == task and self.current_controlnet_id == controlnet_id and self.pipeline is not None and not self.has_loras:
            return self.pipeline

        # Clear memory
        self.pipeline = None
        self.has_loras = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        effective_hf_token = hf_token or settings.hf_token or None
        
        # Check if model_id is a local file
        is_single_file = model_id.endswith((".safetensors", ".ckpt", ".pt"))
        if not is_single_file and not os.path.sep in model_id and not "/" in model_id:
             # Try to find it in models/checkpoints
             local_path = settings.models_dir / "checkpoints" / model_id
             if local_path.exists():
                 model_id = str(local_path)
                 is_single_file = model_id.endswith((".safetensors", ".ckpt", ".pt"))
        
        try:
            log("info", f"Loading pipeline model={model_id} task={task}")
            if task == "t2i":
                if is_single_file:
                    # SDXL models often need specific classes and can be finicky with CLIP sub-models
                    if "xl" in model_id.lower() or "illustrious" in model_id.lower() or "pony" in model_id.lower():
                        from diffusers import StableDiffusionXLPipeline
                        self.pipeline = StableDiffusionXLPipeline.from_single_file(
                            model_id, 
                            torch_dtype=dtype, 
                            token=effective_hf_token,
                            local_files_only=False
                        )
                    elif "flux" in model_id.lower():
                        from diffusers import FluxPipeline
                        self.pipeline = FluxPipeline.from_single_file(
                            model_id, 
                            torch_dtype=dtype, 
                            token=effective_hf_token,
                            local_files_only=False
                        )
                    else:
                        from diffusers import StableDiffusionPipeline
                        self.pipeline = StableDiffusionPipeline.from_single_file(
                            model_id, 
                            torch_dtype=dtype, 
                            token=effective_hf_token,
                            local_files_only=False
                        )
                else:
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(
                        model_id, 
                        torch_dtype=dtype, 
                        use_safetensors=True,
                        token=effective_hf_token
                    )
            elif task == "i2i":
                if is_single_file:
                    if "xl" in model_id.lower():
                        from diffusers import StableDiffusionXLImg2ImgPipeline
                        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                            model_id, 
                            torch_dtype=dtype, 
                            token=effective_hf_token,
                            local_files_only=False
                        )
                    elif "flux" in model_id.lower():
                        from diffusers import FluxImg2ImgPipeline
                        self.pipeline = FluxImg2ImgPipeline.from_single_file(
                            model_id, 
                            torch_dtype=dtype, 
                            token=effective_hf_token,
                            local_files_only=False
                        )
                    else:
                        from diffusers import StableDiffusionImg2ImgPipeline
                        self.pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                            model_id, 
                            torch_dtype=dtype, 
                            token=effective_hf_token,
                            local_files_only=False
                        )
                else:
                    self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                        model_id, 
                        torch_dtype=dtype, 
                        use_safetensors=True,
                        token=effective_hf_token
                    )
            elif task == "controlnet":
                controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype, token=effective_hf_token)
                from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
                if "xl" in model_id.lower():
                    self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                        model_id, controlnet=controlnet, torch_dtype=dtype, use_safetensors=True, token=effective_hf_token
                    )
                else:
                    self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                        model_id, controlnet=controlnet, torch_dtype=dtype, use_safetensors=True, token=effective_hf_token
                    )
            elif task == "t2v":
                if is_single_file:
                    raise HTTPException(status_code=400, detail="Text2Video requires a Diffusers model directory or Hugging Face repo, not a single checkpoint file")
                if "wan" not in model_id.lower():
                    raise HTTPException(status_code=400, detail="Text2Video currently supports Wan Diffusers models only")
                # Primarily Wan 2.1 support
                self.pipeline = WanPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, use_safetensors=True, token=effective_hf_token
                )
            elif task == "i2v":
                # Primarily Wan 2.1 support
                self.pipeline = WanImageToVideoPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, use_safetensors=True, token=effective_hf_token
                )
            
            self.current_model_id = model_id
            self.current_controlnet_id = controlnet_id
            self.type = task
            log("info", f"Pipeline ready model={model_id} task={task}")
            return self.pipeline
        except HTTPException:
            raise
        except Exception as exc:
            msg = str(exc)
            if "403" in msg:
                log("error", f"403 Forbidden error detected for model {model_id}. This usually means you need to set HF_TOKEN and accept the license on Hugging Face.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, 
                    detail=f"Hugging Face 403 Forbidden for {model_id}. Ensure you have set HF_TOKEN and accepted the model license on Hugging Face."
                ) from exc
            log("error", f"Failed to load pipeline model={model_id} task={task}: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to load pipeline: {exc}") from exc


pipeline_manager = PipelineManager()


async def run_generate_job(job_id: str, endpoint: str, payload: dict[str, Any]) -> None:
    import traceback

    started = time.time()
    user_id = payload.get("user_id", "unknown")
    agent_id = payload.get("agent_id") or settings.agent_id
    payload["job_id"] = job_id
    log("info", f"Generation job {job_id} started endpoint={endpoint} user={user_id}")
    try:
        if endpoint == "/api/v1/txt2img":
            response = await txt2img(Text2ImageRequest(**payload))
            output_path = response.image_path
        elif endpoint == "/api/v1/img2img":
            response = await img2img(Image2ImageRequest(**payload))
            output_path = response.image_path
        elif endpoint == "/api/v1/txt2vid":
            response = await txt2vid(Text2VideoRequest(**payload))
            output_path = response["video_path"]
        elif endpoint == "/api/v1/img2vid":
            response = await img2vid(Image2VideoRequest(**payload))
            output_path = response["video_path"]
        elif endpoint == "/api/v1/upscale":
            response = await upscale_image(UpscaleRequest(**payload))
            output_path = response.image_path
        elif endpoint == "/api/v1/faceswap":
            response = await faceswap(FaceSwapRequest(**payload))
            output_path = response.image_path
        else:
            raise ValueError(f"Unsupported generation endpoint: {endpoint}")

        log("info", f"Generation job {job_id} complete output={output_path} duration={time.time() - started:.2f}s")
        await asyncio.to_thread(report_workflow_event, user_id, job_id, "generation-complete", True, output=output_path, agent_id=agent_id)
    except Exception as exc:
        err_msg = f"{exc}\n{traceback.format_exc()}"
        log("error", f"Generation job {job_id} failed: {err_msg}")
        await asyncio.to_thread(report_workflow_event, user_id, job_id, "generation-failed", False, error=str(exc), agent_id=agent_id)


@app.post("/api/v1/jobs/generate", response_model=GenerateJobResponse, dependencies=[Depends(require_agent_auth)])
async def start_generate_job(request: GenerateJobRequest, background_tasks: BackgroundTasks) -> GenerateJobResponse:
    global active_generation_job
    
    if active_generation_job is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Another job is already running: {active_generation_job}. Please wait for it to complete."
        )

    job_id = request.job_id or f"job-{uuid.uuid4()}"
    active_generation_job = job_id
    
    background_tasks.add_task(run_generate_job_wrapper, job_id, request.endpoint, request.payload)
    log("info", f"Queued generation job {job_id} endpoint={request.endpoint}")
    return GenerateJobResponse(job_id=job_id, status="queued")


async def run_generate_job_wrapper(job_id: str, endpoint: str, payload: dict[str, Any]) -> None:
    global active_generation_job
    try:
        await run_generate_job(job_id, endpoint, payload)
    finally:
        if active_generation_job == job_id:
            active_generation_job = None


@app.post("/api/v1/upscale", response_model=UpscaleResponse, dependencies=[Depends(require_agent_auth)])
async def upscale_image(request: UpscaleRequest) -> UpscaleResponse:
    async with generation_session("upscale"):
        return await asyncio.to_thread(_upscale_image_sync, request)


def _upscale_image_sync(request: UpscaleRequest) -> UpscaleResponse:
    start_time = time.time()
    from spandrel import ImageModelDescriptor, ModelLoader
    
    img = load_image_any(request.image)
    
    model_path = None
    if request.upscale_model_id:
        # Check in models/upscale_models
        check_path = settings.models_dir / "upscale_models" / request.upscale_model_id
        if check_path.exists():
            model_path = check_path
    
    if not model_path:
        # Default fallback or error
        # In a production app, we'd have a default ESRGAN model downloaded
        raise HTTPException(status_code=400, detail="Upscale model not found. Please download 4x-UltraSharp.pth to /workspace/models/upscale_models/")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = ModelLoader()
    model = loader.load_from_file(str(model_path)).to(device).eval()
    
    import numpy as np
    # Convert PIL to tensor
    img_np = np.array(img).transpose(2, 0, 1) / 255.0
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(img_tensor)
    
    # Convert back to PIL
    output_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_np = (output_np.clip(0, 1) * 255.0).astype(np.uint8)
    output_img = Image.fromarray(output_np)
    
    # If a specific factor is requested and model doesn't match, resize
    if request.upscale_factor != 4.0: # Assuming most models are 4x
         new_w = int(img.width * request.upscale_factor)
         new_h = int(img.height * request.upscale_factor)
         output_img = output_img.resize((new_w, new_h), Image.LANCZOS)

    filename = f"upscale_{uuid.uuid4()}.png"
    save_path = settings.outputs_dir / filename
    output_img.save(save_path)
    
    return UpscaleResponse(
        image_path=str(save_path),
        duration=time.time() - start_time
    )


def load_image_any(image_src: str) -> Image.Image:
    if image_src.startswith(("http://", "https://")):
        from diffusers.utils import load_image
        return load_image(image_src).convert("RGB")
    
    image_path = Path(image_src)
    if not image_path.is_absolute():
        candidates = [
            settings.workspace_dir / "input" / image_src,
            settings.outputs_dir / image_src,
            settings.workspace_dir / image_src,
        ]
        image_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    
    if not image_path.exists():
         raise HTTPException(status_code=404, detail=f"Image not found: {image_src}")
    return Image.open(image_path).convert("RGB")


def apply_loras_and_embeddings(pipe, loras: list[dict], embeddings: list[dict]) -> tuple[list[str], list[str]]:
    """Loads LoRAs and embeddings, returning (positive_tokens, negative_tokens) for prompt injection."""
    # Load selected LoRAs...
    if loras:
        if not hasattr(pipe, "load_lora_weights"):
            raise HTTPException(status_code=400, detail="Selected model pipeline does not support LoRA weights")

        adapter_names: list[str] = []
        adapter_weights: list[float] = []
        named_adapters_supported = True
        for index, lora in enumerate(loras):
            path = lora.get("path")
            if path:
                lora_path = Path(path)
                if not lora_path.is_absolute():
                    lora_path = settings.models_dir / "loras" / path

                if not lora_path.exists():
                    raise HTTPException(status_code=404, detail=f"LoRA not found: {path}")

                adapter_name = f"lora_{index}"
                try:
                    pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                except TypeError:
                    if len(loras) > 1:
                        raise HTTPException(status_code=400, detail="This pipeline cannot load multiple named LoRA adapters")
                    named_adapters_supported = False
                    pipe.load_lora_weights(str(lora_path))
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Failed to load LoRA {lora_path.name}: {exc}") from exc

                if named_adapters_supported:
                    adapter_names.append(adapter_name)
                adapter_weights.append(float(lora.get("scale", 1.0)))
                log("info", f"Loaded LoRA: {lora_path.name}")

        if len(adapter_names) > 0 and hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to activate LoRA adapters: {exc}") from exc

    pos_tokens: list[str] = []
    neg_tokens: list[str] = []

    # Load selected textual inversion embeddings.
    if embeddings:
        if not hasattr(pipe, "load_textual_inversion"):
            raise HTTPException(status_code=400, detail="Selected model pipeline does not support textual inversion embeddings")
        
        from safetensors.torch import load_file

        for emb in embeddings:
            path = emb.get("path")
            is_neg = emb.get("is_negative", False)
            if path:
                emb_path = Path(path)
                if not emb_path.is_absolute():
                    emb_path = settings.models_dir / "embeddings" / path
                
                if not emb_path.exists():
                    raise HTTPException(status_code=404, detail=f"Embedding not found: {path}")

                token = emb_path.stem
                # Try default loading first
                try:
                    pipe.load_textual_inversion(str(emb_path), token=token)
                    log("info", f"Loaded embedding: {emb_path.name} with token: {token}")
                    if is_neg:
                        neg_tokens.append(token)
                    else:
                        pos_tokens.append(token)
                    continue
                except Exception:
                    pass

                # Fallback for SDXL multi-encoder embeddings (e.g. Kohya style with clip_l/clip_g)
                if emb_path.suffix == ".safetensors" and hasattr(pipe, "text_encoder_2"):
                    state_dict = load_file(str(emb_path))
                    # If it has clip_l and clip_g, we can load them individually
                    if "clip_l" in state_dict and "clip_g" in state_dict:
                        pipe.load_textual_inversion(
                            state_dict["clip_l"], 
                            token=token, 
                            text_encoder=pipe.text_encoder, 
                            tokenizer=pipe.tokenizer
                        )
                        pipe.load_textual_inversion(
                            state_dict["clip_g"], 
                            token=token, 
                            text_encoder=pipe.text_encoder_2, 
                            tokenizer=pipe.tokenizer_2
                        )
                        log("info", f"Loaded SDXL multi-encoder embedding (manual): {emb_path.name} with token: {token}")
                        if is_neg:
                            neg_tokens.append(token)
                        else:
                            pos_tokens.append(token)
                        continue
                    else:
                        # Try loading the whole state dict on the pipeline
                        pipe.load_textual_inversion(state_dict, token=token)
                        log("info", f"Loaded SDXL state_dict embedding: {emb_path.name} with token: {token}")
                        if is_neg:
                            neg_tokens.append(token)
                        else:
                            pos_tokens.append(token)
                        continue
                
                # If we got here, it's a non-SDXL model or a different format, and the first attempt failed
                pipe.load_textual_inversion(str(emb_path))
                log("info", f"Loaded embedding (fallback): {emb_path.name}")
                if is_neg:
                    neg_tokens.append(token)
                else:
                    pos_tokens.append(token)
    
    return pos_tokens, neg_tokens


def safe_batch_size(value: int) -> int:
    return min(max(value, 1), 20)


@app.post("/api/v1/txt2img", response_model=InferenceResponse, dependencies=[Depends(require_agent_auth)])
async def txt2img(request: Text2ImageRequest) -> InferenceResponse:
    async with generation_session(request.job_id or "txt2img"):
        return await asyncio.to_thread(_txt2img_sync, request)


def _txt2img_sync(request: Text2ImageRequest) -> InferenceResponse:
    start_time = time.time()
    pipe = pipeline_manager.load_pipeline(request.model_id, "t2i", hf_token=request.hf_token)
    
    # Apply LoRAs and Embeddings
    pos_tokens, neg_tokens = [], []
    if request.loras or request.embeddings:
        pipeline_manager.has_loras = True
        pos_tokens, neg_tokens = apply_loras_and_embeddings(pipe, request.loras, request.embeddings)
    
    apply_scheduler(pipe, request.sampler_name, request.scheduler)

    # Inject embedding tokens into prompts
    final_prompt = request.prompt
    if pos_tokens:
        final_prompt = f"{final_prompt}, {', '.join(pos_tokens)}"

    final_negative_prompt = request.negative_prompt or ""
    if neg_tokens:
        if final_negative_prompt:
            final_negative_prompt = f"{final_negative_prompt}, {', '.join(neg_tokens)}"
        else:
            final_negative_prompt = ", ".join(neg_tokens)

    log("info", f"Inference parameters: steps={request.num_inference_steps} cfg={request.guidance_scale} size={request.width}x{request.height}")
    log("info", f"Final prompt: {final_prompt}")
    log("info", f"Final negative prompt: {final_negative_prompt}")

    # Finalize pipeline before inference (enable offload after all adaptations)
    if torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    batch_size = safe_batch_size(request.batch_size)

    base_seed = request.seed if request.seed != -1 else torch.Generator().seed()
    image_paths: list[str] = []
    save_path: Path | None = None

    for index in range(batch_size):
        seed = base_seed + index
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        callback = get_progress_callback(
            request.user_id or "unknown", 
            request.job_id or "unknown", 
            index, 
            batch_size, 
            request.num_inference_steps
        )
        
        # Check if pipeline supports negative_prompt
        pipe_kwargs = {
            "prompt": final_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
            "callback_on_step_end": callback,
        }
        
        import inspect
        sig = inspect.signature(pipe.__call__)
        if "negative_prompt" in sig.parameters:
            pipe_kwargs["negative_prompt"] = final_negative_prompt
        
        output = pipe(**pipe_kwargs).images[0]

        filename = f"{uuid.uuid4()}.png"
        save_path = settings.outputs_dir / filename
        output.save(save_path)
        image_paths.append(str(save_path))
        log("info", f"Generated batch image {index + 1}/{batch_size} seed={seed} output={save_path}")
        report_workflow_event(
            request.user_id or "unknown", 
            request.job_id or "unknown", 
            "batch-progress", 
            True, 
            batch_index=index + 1, 
            batch_size=batch_size
        )
    
    return InferenceResponse(
        image_path=str(save_path),
        seed=base_seed,
        duration=time.time() - start_time,
        image_paths=image_paths,
    )


@app.post("/api/v1/img2img", response_model=InferenceResponse, dependencies=[Depends(require_agent_auth)])
async def img2img(request: Image2ImageRequest) -> InferenceResponse:
    async with generation_session(request.job_id or "img2img"):
        return await asyncio.to_thread(_img2img_sync, request)


def _img2img_sync(request: Image2ImageRequest) -> InferenceResponse:
    start_time = time.time()
    pipe = pipeline_manager.load_pipeline(request.model_id, "i2i", hf_token=request.hf_token)
    
    # Apply LoRAs and Embeddings
    pos_tokens, neg_tokens = [], []
    if request.loras or request.embeddings:
        pipeline_manager.has_loras = True
        pos_tokens, neg_tokens = apply_loras_and_embeddings(pipe, request.loras, request.embeddings)
    
    apply_scheduler(pipe, request.sampler_name, request.scheduler)

    # Inject embedding tokens into prompts
    final_prompt = request.prompt
    if pos_tokens:
        final_prompt = f"{final_prompt}, {', '.join(pos_tokens)}"

    final_negative_prompt = request.negative_prompt or ""
    if neg_tokens:
        if final_negative_prompt:
            final_negative_prompt = f"{final_negative_prompt}, {', '.join(neg_tokens)}"
        else:
            final_negative_prompt = ", ".join(neg_tokens)

    log("info", f"Inference parameters: steps={request.num_inference_steps} cfg={request.guidance_scale} strength={request.strength}")
    log("info", f"Final prompt: {final_prompt}")
    log("info", f"Final negative prompt: {final_negative_prompt}")

    # Finalize pipeline before inference
    if torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    init_image = load_image_any(request.image)
    batch_size = safe_batch_size(request.batch_size)
    base_seed = request.seed if request.seed != -1 else torch.Generator().seed()
    image_paths: list[str] = []
    save_path: Path | None = None

    for index in range(batch_size):
        seed = base_seed + index
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        callback = get_progress_callback(
            request.user_id or "unknown", 
            request.job_id or "unknown", 
            index, 
            batch_size, 
            request.num_inference_steps
        )
        
        # Check if pipeline supports negative_prompt
        pipe_kwargs = {
            "prompt": final_prompt,
            "image": init_image,
            "strength": request.strength,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
            "callback_on_step_end": callback,
        }
        
        import inspect
        sig = inspect.signature(pipe.__call__)
        if "negative_prompt" in sig.parameters:
            pipe_kwargs["negative_prompt"] = final_negative_prompt
        
        output = pipe(**pipe_kwargs).images[0]

        filename = f"{uuid.uuid4()}.png"
        save_path = settings.outputs_dir / filename
        output.save(save_path)
        image_paths.append(str(save_path))
        log("info", f"Generated batch image {index + 1}/{batch_size} seed={seed} output={save_path}")
        report_workflow_event(
            request.user_id or "unknown", 
            request.job_id or "unknown", 
            "batch-progress", 
            True, 
            batch_index=index + 1, 
            batch_size=batch_size
        )
    
    return InferenceResponse(
        image_path=str(save_path),
        seed=base_seed,
        duration=time.time() - start_time,
        image_paths=image_paths,
    )


@app.post("/api/v1/controlnet", response_model=InferenceResponse, dependencies=[Depends(require_agent_auth)])
async def controlnet_inference(request: ControlNetRequest) -> InferenceResponse:
    async with generation_session(request.job_id or "controlnet"):
        return await asyncio.to_thread(_controlnet_sync, request)


def _controlnet_sync(request: ControlNetRequest) -> InferenceResponse:
    start_time = time.time()
    pipe = pipeline_manager.load_pipeline(request.model_id, "controlnet", request.controlnet_model_id, hf_token=request.hf_token)
    
    # Apply LoRAs and Embeddings
    pos_tokens, neg_tokens = [], []
    if request.loras or request.embeddings:
        pipeline_manager.has_loras = True
        pos_tokens, neg_tokens = apply_loras_and_embeddings(pipe, request.loras, request.embeddings)
    
    # Inject embedding tokens into prompts
    final_prompt = request.prompt
    if pos_tokens:
        final_prompt = f"{final_prompt}, {', '.join(pos_tokens)}"
    
    final_negative_prompt = request.negative_prompt or ""
    if neg_tokens:
        if final_negative_prompt:
            final_negative_prompt = f"{final_negative_prompt}, {', '.join(neg_tokens)}"
        else:
            final_negative_prompt = ", ".join(neg_tokens)
    
    log("info", f"Inference parameters: steps={request.num_inference_steps} cfg={request.guidance_scale} size={request.width}x{request.height} scale={request.controlnet_conditioning_scale}")
    log("info", f"Final prompt: {final_prompt}")
    log("info", f"Final negative prompt: {final_negative_prompt}")

    # Finalize pipeline before inference
    if torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    control_image = load_image_any(request.image)

    seed = request.seed if request.seed != -1 else torch.Generator().seed()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    callback = get_progress_callback(
        request.user_id or "unknown", 
        request.job_id or "unknown", 
        0, 
        1, 
        request.num_inference_steps
    )
    
    # Check if pipeline supports negative_prompt
    pipe_kwargs = {
        "prompt": final_prompt,
        "image": control_image,
        "controlnet_conditioning_scale": request.controlnet_conditioning_scale,
        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "generator": generator,
        "callback_on_step_end": callback,
    }
    
    import inspect
    sig = inspect.signature(pipe.__call__)
    if "negative_prompt" in sig.parameters:
        pipe_kwargs["negative_prompt"] = final_negative_prompt
    
    output = pipe(**pipe_kwargs).images[0]
    
    filename = f"{uuid.uuid4()}.png"
    save_path = settings.outputs_dir / filename
    output.save(save_path)
    
    report_workflow_event(
        request.user_id or "unknown", 
        request.job_id or "unknown", 
        "batch-progress", 
        True, 
        batch_index=1, 
        batch_size=1
    )
    
    return InferenceResponse(
        image_path=str(save_path),
        seed=seed,
        duration=time.time() - start_time
    )


@app.post("/api/v1/faceswap", response_model=InferenceResponse, dependencies=[Depends(require_agent_auth)])
async def faceswap(request: FaceSwapRequest) -> InferenceResponse:
    async with generation_session(request.job_id or "faceswap"):
        return await asyncio.to_thread(_faceswap_sync, request)


def _faceswap_sync(request: FaceSwapRequest) -> InferenceResponse:
    start_time = time.time()
    import cv2
    import numpy as np
    import insightface
    from insightface.app import FaceAnalysis

    source_img = cv2.cvtColor(np.array(load_image_any(request.source_image)), cv2.COLOR_RGB2BGR)
    target_img = cv2.cvtColor(np.array(load_image_any(request.target_image)), cv2.COLOR_RGB2BGR)

    app = FaceAnalysis(name='antelopev2', root=str(settings.hf_home))
    app.prepare(ctx_id=0, det_size=(640, 640))

    source_faces = app.get(source_img)
    target_faces = app.get(target_img)

    if not source_faces:
        raise HTTPException(status_code=400, detail="No face detected in source image")
    if not target_faces:
        raise HTTPException(status_code=400, detail="No face detected in target image")

    # Load swapper model
    model_path = request.model_path
    if not model_path:
        # Default location or download
        model_path = settings.models_dir / "inswapper_128.onnx"
        if not model_path.exists():
            # In a real app we might auto-download here
            raise HTTPException(status_code=400, detail="inswapper_128.onnx model not found in models directory")
    
    swapper = insightface.model_zoo.get_model(str(model_path), download=False, check_flags=False)
    
    res = target_img.copy()
    # Swap first face found in both
    res = swapper.get(res, target_faces[0], source_faces[0], paste_back=True)

    filename = f"{uuid.uuid4()}.png"
    save_path = settings.outputs_dir / filename
    cv2.imwrite(str(save_path), res)
    
    return InferenceResponse(
        image_path=str(save_path),
        seed=0,
        duration=time.time() - start_time
    )


@app.post("/api/v1/txt2vid", response_model=dict[str, Any], dependencies=[Depends(require_agent_auth)])
async def txt2vid(request: Text2VideoRequest) -> dict[str, Any]:
    async with generation_session(request.job_id or "txt2vid"):
        return await asyncio.to_thread(_txt2vid_sync, request)


def _txt2vid_sync(request: Text2VideoRequest) -> dict[str, Any]:
    start_time = time.time()
    pipe = pipeline_manager.load_pipeline(request.model_id, "t2v", hf_token=request.hf_token)

    # Apply LoRAs and Embeddings
    pos_tokens, neg_tokens = [], []
    if request.loras or request.embeddings:
        pipeline_manager.has_loras = True
        pos_tokens, neg_tokens = apply_loras_and_embeddings(pipe, request.loras, request.embeddings)

    # Inject embedding tokens into prompts
    final_prompt = request.prompt
    if pos_tokens:
        final_prompt = f"{final_prompt}, {', '.join(pos_tokens)}"

    final_negative_prompt = request.negative_prompt or ""
    if neg_tokens:
        if final_negative_prompt:
            final_negative_prompt = f"{final_negative_prompt}, {', '.join(neg_tokens)}"
        else:
            final_negative_prompt = ", ".join(neg_tokens)
    
    log("info", f"Final prompt: {final_prompt}")
    log("info", f"Final negative prompt: {final_negative_prompt}")

    seed = request.seed if request.seed != -1 else torch.Generator().seed()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    callback = get_progress_callback(
        request.user_id or "unknown", 
        request.job_id or "unknown", 
        0, 
        1, 
        request.num_inference_steps
    )

    video = pipe(
        prompt=final_prompt,
        negative_prompt=final_negative_prompt,
        width=request.width,

        height=request.height,
        num_frames=request.num_frames,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        generator=generator,
        callback_on_step_end=callback,
    ).frames[0]
    
    filename = f"{uuid.uuid4()}.mp4"
    save_path = settings.outputs_dir / filename
    export_to_video(video, str(save_path), fps=request.fps)
    
    report_workflow_event(
        request.user_id or "unknown", 
        request.job_id or "unknown", 
        "batch-progress", 
        True, 
        batch_index=1, 
        batch_size=1
    )
    
    return {
        "video_path": str(save_path),
        "seed": seed,
        "duration": time.time() - start_time
    }


@app.post("/api/v1/img2vid", response_model=dict[str, Any], dependencies=[Depends(require_agent_auth)])
async def img2vid(request: Image2VideoRequest) -> dict[str, Any]:
    async with generation_session(request.job_id or "img2vid"):
        return await asyncio.to_thread(_img2vid_sync, request)


def _img2vid_sync(request: Image2VideoRequest) -> dict[str, Any]:
    start_time = time.time()
    pipe = pipeline_manager.load_pipeline(request.model_id, "i2v", hf_token=request.hf_token)
    init_image = load_image_any(request.image)
    
    # Apply LoRAs and Embeddings
    pos_tokens, neg_tokens = [], []
    if request.loras or request.embeddings:
        pipeline_manager.has_loras = True
        pos_tokens, neg_tokens = apply_loras_and_embeddings(pipe, request.loras, request.embeddings)
    
    # Inject embedding tokens into prompts
    final_prompt = request.prompt
    if pos_tokens:
        final_prompt = f"{final_prompt}, {', '.join(pos_tokens)}"
    
    final_negative_prompt = request.negative_prompt or ""
    if neg_tokens:
        if final_negative_prompt:
            final_negative_prompt = f"{final_negative_prompt}, {', '.join(neg_tokens)}"
        else:
            final_negative_prompt = ", ".join(neg_tokens)
    
    log("info", f"Final prompt: {final_prompt}")
    log("info", f"Final negative prompt: {final_negative_prompt}")

    seed = request.seed if request.seed != -1 else torch.Generator().seed()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    callback = get_progress_callback(
        request.user_id or "unknown", 
        request.job_id or "unknown", 
        0, 
        1, 
        request.num_inference_steps
    )
    
    video = pipe(
        image=init_image,
        prompt=final_prompt,
        negative_prompt=final_negative_prompt,
        width=request.width,
        height=request.height,
        num_frames=request.num_frames,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        generator=generator,
        callback_on_step_end=callback,
    ).frames[0]
    
    filename = f"{uuid.uuid4()}.mp4"
    save_path = settings.outputs_dir / filename
    export_to_video(video, str(save_path), fps=request.fps)
    
    report_workflow_event(
        request.user_id or "unknown", 
        request.job_id or "unknown", 
        "batch-progress", 
        True, 
        batch_index=1, 
        batch_size=1
    )
    
    return {
        "video_path": str(save_path),
        "seed": seed,
        "duration": time.time() - start_time
    }


@app.post("/api/v1/preprocess/openpose", response_model=dict[str, str], dependencies=[Depends(require_agent_auth)])
async def preprocess_openpose(request: OpenPoseRequest) -> dict[str, str]:
    async with generation_session("openpose"):
        return await asyncio.to_thread(_preprocess_openpose_sync, request)


def _preprocess_openpose_sync(request: OpenPoseRequest) -> dict[str, str]:
    from controlnet_aux import OpenposeDetector
    
    img = load_image_any(request.image)
    processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
    processed_image = processor(
        img, 
        include_body=request.include_body, 
        include_hand=request.include_hand, 
        include_face=request.include_face
    )
    
    filename = f"pose_{uuid.uuid4()}.png"
    save_path = settings.outputs_dir / filename
    processed_image.save(save_path)
    
    return {"image_path": str(save_path)}


@app.get("/outputs", response_model=OutputsResponse, dependencies=[Depends(require_agent_auth)])
async def list_outputs(page: int = 1, pageSize: int = 10) -> OutputsResponse:
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    safe_page = max(page, 1)
    safe_page_size = min(max(pageSize, 1), 100)
    files: list[OutputImage] = []
    
    root = settings.outputs_dir.resolve()
    if root.exists():
        for path in root.rglob("*"):
            resolved = path.resolve()
            if root != resolved and root not in resolved.parents:
                continue
            if not resolved.is_file() or resolved.suffix.lower() not in allowed_suffixes:
                continue

            stat = resolved.stat()
            relative_path = resolved.relative_to(root).as_posix()
            subfolder = resolved.parent.relative_to(root).as_posix()
            files.append(OutputImage(
                name=resolved.name,
                path=relative_path,
                root="outputs",
                subfolder="" if subfolder == "." else subfolder,
                sizeBytes=stat.st_size,
                modifiedAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            ))

    files.sort(key=lambda item: item.modifiedAt, reverse=True)
    total = len(files)
    total_pages = max((total + safe_page_size - 1) // safe_page_size, 1)
    start = (safe_page - 1) * safe_page_size
    return OutputsResponse(
        images=files[start:start + safe_page_size],
        total=total,
        page=safe_page,
        pageSize=safe_page_size,
        totalPages=total_pages,
        outputsDir=str(settings.outputs_dir),
    )


@app.get("/outputs/file", dependencies=[Depends(require_agent_auth)])
async def output_file(path: str, max_side: int | None = None, quality: int = 85) -> Any:
    from fastapi.responses import FileResponse, Response

    output_path = ensure_output_file(path)
    media_type = mimetypes.guess_type(output_path.name)[0] or "application/octet-stream"
    if max_side and max_side > 0 and media_type.startswith("image/"):
        with Image.open(output_path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((max_side, max_side))
            buffer = BytesIO()
            safe_quality = min(max(quality, 40), 95)
            preview.save(buffer, format="JPEG", quality=safe_quality, optimize=True)
            return Response(
                content=buffer.getvalue(),
                media_type="image/jpeg",
                headers={"Content-Disposition": f'inline; filename="{output_path.stem}_preview.jpg"'},
            )
    return FileResponse(output_path, media_type=media_type, filename=output_path.name)


@app.delete("/outputs/file", dependencies=[Depends(require_agent_auth)])
async def delete_output_file(path: str) -> dict[str, str | bool]:
    output_path = ensure_output_file(path)
    output_path.unlink()
    return {"deleted": True, "path": path}


@app.post("/upload", dependencies=[Depends(require_agent_auth)])
async def upload_file(
    file: UploadFile = File(...),
    subfolder: str = "input"
) -> dict[str, str]:
    return await asyncio.to_thread(_upload_file_sync, file, subfolder)


def _upload_file_sync(file: UploadFile, subfolder: str) -> dict[str, str]:
    target_root = settings.workspace_dir / "input"
    target_root.mkdir(parents=True, exist_ok=True)

    target_dir = ensure_child_path(target_root, subfolder)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = target_dir / file.filename
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "filename": file.filename,
        "subfolder": subfolder,
        "path": target_path.relative_to(target_root).as_posix()
    }


@app.get("/logs", response_model=LogsResponse, dependencies=[Depends(require_agent_auth)])
async def get_logs(path: str | None = None, lines: int = 100) -> LogsResponse:
    # Default to agent log file if no path provided
    target_path = Path(path) if path else settings.agent_log_file
    
    # Security: Ensure path is within workspace_dir
    try:
        target_path = ensure_child_path(settings.workspace_dir, str(target_path))
    except HTTPException:
        # If it's the agent log file itself, it might be outside if configured so, 
        # but by default it is in /workspace/agent.log
        if path: # Only raise if the user provided an escaping path
            raise
    
    if not target_path.exists():
        if not path: # Default agent log might not exist yet
            return LogsResponse(logs="", path=str(target_path))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")

    try:
        # Use tail command for efficiency
        result = subprocess.run(
            ["tail", "-n", str(lines), str(target_path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return LogsResponse(logs=result.stdout, path=str(target_path))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/exec", response_model=ExecResponse, dependencies=[Depends(require_agent_auth)])
async def execute_command(request: ExecRequest) -> ExecResponse:
    full_env = {**os.environ, **(request.env or {})}
    
    if request.background:
        try:
            # For background execution, we use subprocess.Popen and don't wait
            # We use start_new_session=True to ensure it keeps running if the agent restarts
            subprocess.Popen(
                request.command,
                shell=True,
                cwd=request.cwd,
                env=full_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return ExecResponse(
                stdout="Command started in background",
                stderr="",
                exitCode=0
            )
        except Exception as exc:
            return ExecResponse(
                stdout="",
                stderr=f"Failed to start background process: {exc}",
                exitCode=1
            )

    try:
        # For synchronous execution, use asyncio to avoid blocking the event loop
        process = await asyncio.create_subprocess_shell(
            request.command,
            cwd=request.cwd,
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            return ExecResponse(
                stdout=stdout.decode() if stdout else "",
                stderr=stderr.decode() if stderr else "",
                exitCode=process.returncode or 0,
            )
        except asyncio.TimeoutExpired:
            try:
                process.terminate()
            except:
                pass
            return ExecResponse(
                stdout="",
                stderr="Command timed out after 300 seconds",
                exitCode=124,
            )
            
    except Exception as exc:
        return ExecResponse(
            stdout="",
            stderr=str(exc),
            exitCode=1,
        )


@app.post("/upload-output", response_model=UploadOutputResponse, dependencies=[Depends(require_agent_auth)])
async def upload_output(request: UploadOutputRequest) -> UploadOutputResponse:
    return await asyncio.to_thread(_upload_output_sync, request)


def _upload_output_sync(request: UploadOutputRequest) -> UploadOutputResponse:
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
