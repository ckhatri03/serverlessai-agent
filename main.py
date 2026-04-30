import asyncio
import os
import json
import shutil
import signal
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, status
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_git_repo()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.workflows_dir.mkdir(parents=True, exist_ok=True)
    if settings.auto_start_comfyui and get_comfyui_path():
        try:
            start_comfyui_process()
        except Exception as exc:
            print(f"Failed to auto-start ComfyUI: {exc}")
    register_with_control_plane()
    yield


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
    log_file = log_path.open("ab")
    process = subprocess.Popen(
        [str(python_bin), "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        cwd=comfy_path,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    comfyui_pid_file().write_text(str(process.pid))
    return ComfyUIStartResponse(started=True, running=True, pid=process.pid, logPath=str(log_path))


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


if __name__ == "__main__":
    import uvicorn

    def handle_sigterm(*_: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
