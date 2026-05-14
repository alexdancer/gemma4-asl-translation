FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/hf-endpoint/requirements.runtime.txt ./requirements.runtime.txt
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.runtime.txt

COPY src ./src
COPY scripts/runtime/serve_hf_custom_endpoint_service.py ./scripts/runtime/serve_hf_custom_endpoint_service.py

EXPOSE 8080

CMD ["python", "scripts/runtime/serve_hf_custom_endpoint_service.py", "--host", "0.0.0.0", "--port", "8080"]
