FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WORKSPACE_DIR=/workspace \
    MODELS_DIR=/workspace/models \
    OUTPUTS_DIR=/workspace/output \
    WORKFLOWS_DIR=/workspace/workflows \
    COMFYUI_URL=http://127.0.0.1:8188

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY main.py start-agent.sh ./
RUN chmod +x /app/start-agent.sh

EXPOSE 8000

CMD ["/app/start-agent.sh"]
