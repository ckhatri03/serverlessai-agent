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
from pathlib import Path
from typing import Any

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field, HttpUrl


APP_VERSION = "0.1.1"


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
    agent_log_file: Path = Field(default_factory=lambda: Path(os.getenv("SERVERLESSAI_AGENT_LOG_FILE", "/workspace/agent.log")))
    auto_start_comfyui: bool = Field(default_factory=lambda: os.getenv("SERVERLESSAI_AUTO_START_COMFYUI", "true").lower() == "true")

    @property
    def effective_public_url(self) -> str:
        if self.public_url:
            return self.public_url
        if self.pod_id:
            # Construct RunPod proxy URL
            return f"https://{self.pod_id}-8000.proxy.runpod.net"
        return ""


settings = Settings()


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


async def trim_comfyui_logs() -> None:
    log_path = comfyui_log_file()
    while True:
        try:
            if log_path.exists():
                # Use tail to get the last 100 lines and overwrite the file
                process = await asyncio.create_subprocess_exec(
                    "tail", "-n", "100", str(log_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                if process.returncode == 0:
                    log_path.write_bytes(stdout)
        except Exception as exc:
            print(f"Failed to trim ComfyUI logs: {exc}")
        await asyncio.sleep(60) # Trim every minute


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_git_repo()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Start log trimming task
    trim_task = asyncio.create_task(trim_comfyui_logs())
    
    if settings.auto_start_comfyui and get_comfyui_path():
        try:
            start_comfyui_process()
        except Exception as exc:
            print(f"Failed to auto-start ComfyUI: {exc}")
    register_with_control_plane()
    yield
    trim_task.cancel()
    try:
        await trim_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Serverless AI Agent", version=APP_VERSION, lifespan=lifespan)


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


class SystemInfoResponse(BaseModel):
    pytorch: str
    cuda: str
    nvidia_driver: str
    comfyui: str


class ComfyUIInfoResponse(BaseModel):
    installed: bool
    version: str
    running: bool
    managerInstalled: bool
    customNodes: list[str]


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


class ExecRequest(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    background: bool = False


class ExecResponse(BaseModel):
    stdout: str
    stderr: str
    exitCode: int


class ComfyUIStartResponse(BaseModel):
    started: bool
    running: bool
    pid: int | None = None
    logPath: str


class ComfyUIStopResponse(BaseModel):
    stopped: bool
    running: bool


class InstallCustomNodeRequest(BaseModel):
    repoUrl: HttpUrl
    name: str | None = None
    branch: str | None = None
    installRequirements: bool = True


class InstallCustomNodeResponse(BaseModel):
    name: str
    path: str
    installed: bool
    requirementsInstalled: bool


class WorkflowStatusResponse(BaseModel):
    promptId: str
    history: dict[str, Any]


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


def output_roots() -> dict[str, Path]:
    roots = {"outputs": settings.outputs_dir}
    comfy_path = get_comfyui_path()
    if comfy_path:
        comfy_output = comfy_path / "output"
        if comfy_output.resolve() != settings.outputs_dir.resolve():
            roots["comfyui"] = comfy_output
    return roots


def ensure_output_file(relative_path: str, root_name: str = "outputs") -> Path:
    roots = output_roots()
    if root_name not in roots:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown output root")
    output_path = ensure_child_path(roots[root_name], relative_path)
    if not output_path.exists() or not output_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Output file not found")
    return output_path


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


def get_comfyui_path() -> Path | None:
    # Common locations
    locations = [
        settings.workspace_dir / "ComfyUI",
        Path("/workspace/ComfyUI"),
        Path("/app/ComfyUI"),
    ]
    for loc in locations:
        # A valid installation requires both the main entry point AND the sentinel file
        if (loc / "main.py").exists() and (loc / ".install-complete").exists():
            return loc
    return None


def get_comfyui_path_for_management() -> Path:
    comfy_path = get_comfyui_path()
    if not comfy_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ComfyUI is not installed")
    return comfy_path


def comfyui_pid_file() -> Path:
    return settings.workspace_dir / "comfyui.pid"


def comfyui_log_file() -> Path:
    return settings.workspace_dir / "comfyui.log"


def ensure_comfyui_runtime(comfy_path: Path, log_path: Path) -> None:
    python_bin = comfy_path / "venv" / "bin" / "python3"
    if not python_bin.exists():
        return

    check = subprocess.run(
        [str(python_bin), "-c", "import torch; raise SystemExit(0 if hasattr(torch.library, 'custom_op') else 1)"],
        cwd=comfy_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check.returncode == 0:
        return

    with log_path.open("ab") as log_file:
        log_file.write(b"Installing CUDA 11.8 Torch runtime for current ComfyUI...\n")
        subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "torch==2.4.1+cu118",
                "torchvision==0.19.1+cu118",
                "torchaudio==2.4.1+cu118",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu118",
            ],
            cwd=comfy_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=True,
        )
        subprocess.run(
            [str(python_bin), "-m", "pip", "install", "numpy>=1.25,<2"],
            cwd=comfy_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=True,
        )


def get_comfyui_pid() -> int | None:
    pid_path = comfyui_pid_file()
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except ValueError:
        return None


def process_running(pid: int | None) -> bool:
    if not pid:
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            command = cmdline.read_text(errors="ignore")
            if "ComfyUI" not in command and "main.py" not in command:
                return False
        except Exception:
            pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def comfyui_running() -> bool:
    return comfyui_reachable() or process_running(get_comfyui_pid())


def start_comfyui_process() -> ComfyUIStartResponse:
    comfy_path = get_comfyui_path_for_management()
    pid = get_comfyui_pid()
    log_path = comfyui_log_file()

    if comfyui_running():
        return ComfyUIStartResponse(started=False, running=True, pid=pid, logPath=str(log_path))

    python_bin = comfy_path / "venv" / "bin" / "python3"
    if not python_bin.exists():
        python_bin = Path("python3")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_comfyui_runtime(comfy_path, log_path)
    log_file = log_path.open("ab")
    process = subprocess.Popen(
        [str(python_bin), "main.py", "--listen", "0.0.0.0", "--port", "8188", "--output-directory", str(settings.outputs_dir)],
        cwd=comfy_path,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    comfyui_pid_file().write_text(str(process.pid))
    time.sleep(5)
    return ComfyUIStartResponse(started=True, running=comfyui_reachable() or process.poll() is None, pid=process.pid, logPath=str(log_path))


def stop_comfyui_process() -> ComfyUIStopResponse:
    pid = get_comfyui_pid()
    if not process_running(pid):
        return ComfyUIStopResponse(stopped=False, running=False)

    assert pid is not None
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return ComfyUIStopResponse(stopped=False, running=False)
    return ComfyUIStopResponse(stopped=True, running=False)


def get_comfyui_version(comfy_path: Path) -> str:
    try:
        # Try to get git version
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=comfy_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_comfyui_custom_nodes(comfy_path: Path) -> list[str]:
    nodes_dir = comfy_path / "custom_nodes"
    if not nodes_dir.exists():
        return []
    
    nodes = []
    try:
        for item in nodes_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                nodes.append(item.name)
    except Exception:
        pass
    return sorted(nodes)


def is_comfyui_manager_installed(comfy_path: Path) -> bool:
    manager_path = comfy_path / "custom_nodes" / "ComfyUI-Manager"
    return (manager_path / ".install-complete").exists()


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
        "publicUrl": settings.effective_public_url,
        "version": APP_VERSION,
        "systemInfo": {
            "pytorch": get_pytorch_version(),
            "cuda": get_cuda_version(),
            "nvidia_driver": get_nvidia_driver_version(),
            "comfyui": "ok" if comfyui_reachable() else "not reachable",
        },
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


@app.get("/system-info", response_model=SystemInfoResponse, dependencies=[Depends(require_agent_auth)])
async def system_info() -> SystemInfoResponse:
    return SystemInfoResponse(
        pytorch=get_pytorch_version(),
        cuda=get_cuda_version(),
        nvidia_driver=get_nvidia_driver_version(),
        comfyui="ok" if comfyui_reachable() else "not reachable",
    )


@app.get("/comfyui-info", response_model=ComfyUIInfoResponse, dependencies=[Depends(require_agent_auth)])
async def comfyui_info() -> ComfyUIInfoResponse:
    path = get_comfyui_path()
    if not path:
        return ComfyUIInfoResponse(installed=False, version="N/A", running=False, managerInstalled=False, customNodes=[])
    
    return ComfyUIInfoResponse(
        installed=True,
        version=get_comfyui_version(path),
        running=comfyui_running(),
        managerInstalled=is_comfyui_manager_installed(path),
        customNodes=get_comfyui_custom_nodes(path),
    )


@app.post("/comfyui/start", response_model=ComfyUIStartResponse, dependencies=[Depends(require_agent_auth)])
async def start_comfyui() -> ComfyUIStartResponse:
    return start_comfyui_process()


@app.post("/comfyui/stop", response_model=ComfyUIStopResponse, dependencies=[Depends(require_agent_auth)])
async def stop_comfyui() -> ComfyUIStopResponse:
    return stop_comfyui_process()


@app.post("/custom-nodes/install", response_model=InstallCustomNodeResponse, dependencies=[Depends(require_agent_auth)])
async def install_custom_node(request: InstallCustomNodeRequest) -> InstallCustomNodeResponse:
    comfy_path = get_comfyui_path_for_management()
    nodes_dir = comfy_path / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    repo_url = str(request.repoUrl)
    node_name = request.name or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    if not node_name or "/" in node_name or node_name.startswith("."):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid custom node name")

    target = nodes_dir / node_name
    if target.exists() and not (target / ".git").exists():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Target custom node directory exists but is not a git repository")

    try:
        if target.exists():
            subprocess.run(["git", "pull", "--ff-only"], cwd=target, check=True, capture_output=True, text=True, timeout=120)
        else:
            clone_cmd = ["git", "clone", repo_url, str(target)]
            if request.branch:
                clone_cmd = ["git", "clone", "--branch", request.branch, "--single-branch", repo_url, str(target)]
            subprocess.run(clone_cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=exc.stderr or exc.stdout or "Git operation failed") from exc

    requirements_installed = False
    requirements = target / "requirements.txt"
    python_bin = comfy_path / "venv" / "bin" / "python3"
    if request.installRequirements and requirements.exists():
        try:
            subprocess.run([str(python_bin), "-m", "pip", "install", "-r", str(requirements)], check=True, capture_output=True, text=True, timeout=600)
            requirements_installed = True
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=exc.stderr or exc.stdout or "Requirements installation failed") from exc

    return InstallCustomNodeResponse(
        name=node_name,
        path=str(target),
        installed=True,
        requirementsInstalled=requirements_installed,
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


@app.get("/view", dependencies=[Depends(require_agent_auth)])
async def workflow_view(filename: str, subfolder: str = "", type: str = "output") -> Any:
    from fastapi.responses import StreamingResponse
    import io
    
    params = {"filename": filename, "subfolder": subfolder, "type": type}
    try:
        response = requests.get(f"{settings.comfyui_url.rstrip('/')}/view", params=params, timeout=30)
        if not response.ok:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return StreamingResponse(
            io.BytesIO(response.content),
            media_type=response.headers.get("Content-Type", "image/png")
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"ComfyUI unavailable: {exc}")


@app.get("/outputs", response_model=OutputsResponse, dependencies=[Depends(require_agent_auth)])
async def list_outputs(page: int = 1, pageSize: int = 10) -> OutputsResponse:
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    safe_page = max(page, 1)
    safe_page_size = min(max(pageSize, 1), 100)
    files: list[OutputImage] = []
    for root_name, root_path in output_roots().items():
        root = root_path.resolve()
        if not root.exists():
            continue
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
                root=root_name,
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
async def output_file(path: str, root: str = "outputs") -> Any:
    from fastapi.responses import FileResponse

    output_path = ensure_output_file(path, root)
    media_type = mimetypes.guess_type(output_path.name)[0] or "application/octet-stream"
    return FileResponse(output_path, media_type=media_type, filename=output_path.name)


@app.delete("/outputs/file", dependencies=[Depends(require_agent_auth)])
async def delete_output_file(path: str, root: str = "outputs") -> dict[str, str | bool]:
    output_path = ensure_output_file(path, root)
    output_path.unlink()
    return {"deleted": True, "path": path, "root": root}


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


@app.get("/comfyui/object-info", dependencies=[Depends(require_agent_auth)])
async def comfyui_object_info() -> Any:
    response = comfyui_get("/object_info")
    if not response.ok:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=response.text)
    return response.json()


@app.get("/comfyui/list-models", dependencies=[Depends(require_agent_auth)])
async def list_models() -> dict[str, list[str]]:
    comfy_path = get_comfyui_path()
    if not comfy_path:
        return {}
    
    models_root = comfy_path / "models"
    if not models_root.exists():
        return {}
    
    result = {}
    for folder in models_root.iterdir():
        if folder.is_dir():
            files = []
            for f in folder.rglob("*"):
                if f.is_file() and f.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}:
                    files.append(f.relative_to(folder).as_posix())
            result[folder.name] = sorted(files)
    return result


@app.post("/upload", dependencies=[Depends(require_agent_auth)])
async def upload_file(
    file: UploadFile = File(...),
    subfolder: str = "input",
    root: str = "comfyui"
) -> dict[str, str]:
    roots = output_roots()
    if root == "comfyui":
        comfy_path = get_comfyui_path()
        if not comfy_path:
            raise HTTPException(status_code=400, detail="ComfyUI not installed")
        target_root = comfy_path / "input"
    elif root in roots:
        target_root = roots[root]
    else:
        raise HTTPException(status_code=400, detail="Invalid root")

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


if __name__ == "__main__":
    import uvicorn

    def handle_sigterm(*_: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
