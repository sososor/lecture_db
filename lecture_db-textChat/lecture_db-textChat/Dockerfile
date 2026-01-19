FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# システム依存をできるだけ減らす。numpy などは wheels を利用。
# ここで codex-cli などコンテナ内で使う CLI も合わせてインストールする
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        nodejs \
        npm \
    && npm install -g @openai/codex \
    && rm -rf /var/lib/apt/lists/*
    
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:${PATH}"

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ソースを配置（FastAPI は backend/main.py をエントリに frontend を静的配信する）
COPY backend /app/backend
COPY frontend /app/frontend

WORKDIR /app/backend

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
