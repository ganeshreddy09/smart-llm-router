FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# sentence-transformers pulls in torch, and torch's default Linux wheel is
# the CUDA/GPU build — several GB of NVIDIA libraries we don't need just to
# run MiniLM on CPU. Installing the CPU-only build first means the later
# `pip install -r requirements.txt` sees torch already satisfied and skips
# the GPU download entirely. --default-timeout/--retries make the download
# resilient on a slow connection instead of failing outright.
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
COPY sql ./sql
COPY static ./static

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
