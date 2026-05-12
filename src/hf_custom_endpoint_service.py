"""HF custom endpoint service exposing minimal OpenAI chat-completions contract.

This adapter intentionally keeps API surface minimal for first green path:
- GET /healthz
- POST /v1/chat/completions
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class InferenceBackend(Protocol):
    def generate(self, *, messages: list[dict[str, str]], model: str) -> str: ...


class StubBackend:
    """Deterministic stub backend for local/dev contract bring-up."""

    def generate(self, *, messages: list[dict[str, str]], model: str) -> str:
        _ = model
        user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if not user_messages:
            return ""
        final = user_messages[-1].strip()
        if final.startswith("{"):
            try:
                decoded = json.loads(final)
                input_payload = decoded.get("input") if isinstance(decoded, dict) else None
                if isinstance(input_payload, dict):
                    filename = str(input_payload.get("filename") or "").strip()
                    if filename:
                        return f"ASL_TRANSLATION_FOR_{filename}"
            except json.JSONDecodeError:
                pass
        return final[:200]


@dataclass
class RuntimeState:
    ready: bool
    model_id: str
    model_version: str
    backend_name: str
    backend: InferenceBackend | None = None
    load_error: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None


def _load_runtime() -> RuntimeState:
    model_id = os.environ.get("ASL_HF_ENDPOINT_MODEL_ID", "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit").strip()
    model_version = os.environ.get("ASL_HF_ENDPOINT_MODEL_VERSION", "dev").strip() or "dev"
    backend_name = os.environ.get("ASL_HF_ENDPOINT_BACKEND", "stub").strip().lower() or "stub"

    if backend_name == "stub":
        return RuntimeState(
            ready=True,
            model_id=model_id,
            model_version=model_version,
            backend_name="stub",
            backend=StubBackend(),
        )

    return RuntimeState(
        ready=False,
        model_id=model_id,
        model_version=model_version,
        backend_name=backend_name,
        backend=None,
        load_error=f"unsupported backend: {backend_name}",
    )


def _error(status_code: int, *, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


def create_hf_custom_endpoint_app(*, runtime_loader: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader = runtime_loader or _load_runtime
        app.state.runtime = loader()
        yield

    app = FastAPI(title="ASL HF Custom Endpoint", version="1", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(400, code="INVALID_REQUEST", message=f"invalid request payload: {exc.errors()[0].get('msg', 'validation error')}")

    @app.exception_handler(404)
    async def _handle_not_found(_: Request, __: Any) -> JSONResponse:
        return _error(404, code="NOT_FOUND", message="path not found")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        runtime: RuntimeState = app.state.runtime
        return {
            "ok": runtime.ready,
            "ready": runtime.ready,
            "model_id": runtime.model_id,
            "model_version": runtime.model_version,
            "backend": runtime.backend_name,
            "load_error": runtime.load_error,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest) -> dict[str, Any]:
        runtime: RuntimeState = app.state.runtime
        if not runtime.ready or runtime.backend is None:
            return _error(503, code="MODEL_NOT_READY", message="model runtime is not ready")

        content = runtime.backend.generate(
            model=payload.model,
            messages=[{"role": m.role, "content": m.content} for m in payload.messages],
        )
        if not content:
            return _error(502, code="INFERENCE_FAILED", message="empty model output")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [{"message": {"content": content}}],
        }

    return app


hf_custom_endpoint_app = create_hf_custom_endpoint_app()
