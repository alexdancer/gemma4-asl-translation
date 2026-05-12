FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY src ./src
COPY scripts/runtime/serve_hf_custom_endpoint_service.py ./scripts/runtime/serve_hf_custom_endpoint_service.py

EXPOSE 8080

CMD ["python", "scripts/runtime/serve_hf_custom_endpoint_service.py", "--host", "0.0.0.0", "--port", "8080"]
