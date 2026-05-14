"""HF custom endpoint service exposing minimal OpenAI chat-completions contract.

This adapter intentionally keeps API surface minimal for first green path:
- GET /healthz
- POST /v1/chat/completions
"""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from huggingface_hub import InferenceClient

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from src.cloud_translate_api import CloudInferError, translate_sign_wsgi_app


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


class RealBackend:
    """Real backend that calls local mounted model for target id, else HF client fallbacks."""

    _DIRECT_TEXTGEN_MODEL_ID = "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit"
    _FILENAME_ECHO_RE = re.compile(r"^(?:IMG_\d+|[^\s]+\.(?:mov|mp4|m4v|avi|mkv|webm))$", re.IGNORECASE)
    _STRUCTURED_NOISE_RE = re.compile(r"(?:\"pose_summary\"|\"frame_count\"|\"first_ts_ms\"|\"last_ts_ms\")")
    _PREDICTION_PREFIX_RE = re.compile(r"^(?:translation|output|answer)\s*:\s*", re.IGNORECASE)

    def __init__(self, *, token: str = "") -> None:
        self._client = InferenceClient(api_key=token or None)
        self._model_dir = os.environ.get("HF_MODEL_DIR", "/repository").strip() or "/repository"
        self._local_tokenizer: Any | None = None
        self._local_model: Any | None = None
        self._local_device: str = "cpu"

    @staticmethod
    def _normalize_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role") or "user").strip() or "user"
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                try:
                    decoded = json.loads(content)
                    if isinstance(decoded, dict):
                        instruction = str(decoded.get("instruction") or "Translate ASL payload into short text.").strip()
                        input_payload = decoded.get("input")
                        if isinstance(input_payload, dict):
                            sanitized_input = dict(input_payload)
                            sanitized_input.pop("filename", None)
                            content = (
                                f"{instruction}\n"
                                "Return only translated text, no metadata.\n"
                                f"Input payload JSON:\n{json.dumps(sanitized_input, ensure_ascii=False)}"
                            )
                except json.JSONDecodeError:
                    pass
            normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _build_textgen_prompt(prepared: list[dict[str, str]]) -> str:
        body = "\n".join(f"{m['role']}: {m['content']}" for m in prepared)
        return (
            "You translate ASL recognition context into concise English output.\n"
            "Return exactly one plain-English translation sentence.\n"
            "Do not return JSON, keys, timestamps, frame counts, filenames, analysis, or markdown.\n"
            "Output must be only the translation text.\n\n"
            f"Context:\n{body}\n\n"
            "Assistant:"
        )

    @staticmethod
    def _normalize_prediction_text(text: str) -> str:
        line = str(text or "").strip()
        line = RealBackend._PREDICTION_PREFIX_RE.sub("", line).strip()
        return line

    @staticmethod
    def _salvage_translation_from_structured_text(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        decoded: Any | None = None
        if raw.startswith("{") or raw.startswith("["):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                decoded = None

        if isinstance(decoded, dict):
            candidate = decoded.get("translation")
            if isinstance(candidate, str):
                cleaned = RealBackend._normalize_prediction_text(candidate)
                invalid, _ = RealBackend._is_invalid_prediction(cleaned)
                return "" if invalid else cleaned
        return ""

    @staticmethod
    def _is_invalid_prediction(prediction: str) -> tuple[bool, str]:
        if RealBackend._FILENAME_ECHO_RE.fullmatch(prediction):
            return True, "filename_echo"
        if RealBackend._STRUCTURED_NOISE_RE.search(prediction):
            return True, "structured_echo"

        stripped = prediction.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
                if isinstance(decoded, (dict, list)):
                    return True, "json_echo"
            except json.JSONDecodeError:
                pass
        return False, ""

    def _ensure_local_model_loaded(self) -> None:
        if self._local_tokenizer is not None and self._local_model is not None:
            return

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"local model deps unavailable: {_format_exception(exc)}") from exc

        model_dir = self._model_dir
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"HF model directory missing: {model_dir}")

        try:
            self._local_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self._local_model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map="auto",
            )
            self._local_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:
            raise RuntimeError(f"local model init failed: {_format_exception(exc)}") from exc

    def _local_text_generation(self, *, prepared: list[dict[str, str]]) -> str:
        self._ensure_local_model_loaded()
        assert self._local_tokenizer is not None and self._local_model is not None

        prompt = self._build_textgen_prompt(prepared)
        try:
            encoded = self._local_tokenizer(prompt, return_tensors="pt")
            if self._local_device == "cuda":
                encoded = {k: v.to("cuda") for k, v in encoded.items()}

            input_ids = encoded.get("input_ids")
            input_len = int(input_ids.shape[-1]) if input_ids is not None else 0

            def _generate(*, min_new_tokens: int | None = None):
                kwargs: dict[str, Any] = {
                    **encoded,
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "do_sample": False,
                }
                if min_new_tokens is not None:
                    kwargs["min_new_tokens"] = min_new_tokens
                return self._local_model.generate(**kwargs)

            outputs = _generate()
            generated_ids = outputs[0][input_len:] if input_len > 0 else outputs[0]
            text = self._local_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if not text:
                outputs = _generate(min_new_tokens=4)
                generated_ids = outputs[0][input_len:] if input_len > 0 else outputs[0]
                text = self._local_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            raise RuntimeError(f"local model generate failed: {_format_exception(exc)}") from exc

        if not text:
            raise RuntimeError("local model empty output")
        first_line = text.splitlines()[0].strip()
        normalized = RealBackend._normalize_prediction_text(first_line or text)
        if not normalized:
            raise RuntimeError("local model empty output")
        return normalized

    def generate(self, *, messages: list[dict[str, str]], model: str) -> str:
        prepared = self._normalize_prompt(messages)
        if not prepared:
            return ""

        if model == self._DIRECT_TEXTGEN_MODEL_ID:
            return self._local_text_generation(prepared=prepared)

        failures: list[str] = []

        def _extract_chat_content(completion: Any) -> str:
            choices = getattr(completion, "choices", None)
            if not choices:
                return ""
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", "")
            if isinstance(content, list):
                content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
            return str(content or "").strip()

        try:
            completion = self._client.chat.completions.create(
                model=model,
                messages=prepared,
                temperature=0,
                max_tokens=128,
            )
            content = _extract_chat_content(completion)
            if content:
                return content
            failures.append("chat.completions.create: empty output")
        except Exception as exc:
            failures.append(f"chat.completions.create: {exc.__class__.__name__}: {_format_exception(exc)}")

        try:
            completion = self._client.chat_completion(
                model=model,
                messages=prepared,
                temperature=0,
                max_tokens=128,
            )
            content = _extract_chat_content(completion)
            if content:
                return content
            failures.append("chat_completion: empty output")
        except Exception as exc:
            failures.append(f"chat_completion: {exc.__class__.__name__}: {_format_exception(exc)}")

        try:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in prepared)
            generated = self._client.text_generation(
                prompt=prompt,
                model=model,
                max_new_tokens=128,
                temperature=0,
                return_full_text=False,
            )
            content = str(generated or "").strip()
            if content:
                return content
            failures.append("text_generation: empty output")
        except Exception as exc:
            failures.append(f"text_generation: {exc.__class__.__name__}: {_format_exception(exc)}")

        raise RuntimeError("all inference methods failed: " + " | ".join(failures))


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


def _format_exception(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    args = getattr(exc, "args", ())
    if args:
        return f"{exc.__class__.__name__} args={args!r}"
    return repr(exc)


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

    if backend_name == "real":
        try:
            token = os.environ.get("ASL_HF_TOKEN", "").strip()
            return RuntimeState(
                ready=True,
                model_id=model_id,
                model_version=model_version,
                backend_name="real",
                backend=RealBackend(token=token),
            )
        except Exception as exc:  # pragma: no cover - defensive load guard
            return RuntimeState(
                ready=False,
                model_id=model_id,
                model_version=model_version,
                backend_name="real",
                backend=None,
                load_error=f"real backend init failed: {exc}",
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


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _status_code_from_wsgi(status: str) -> int:
    return int(status.split(" ", 1)[0])


async def _build_wsgi_environ(request: Request) -> dict[str, Any]:
    body = await request.body()
    environ: dict[str, Any] = {
        "REQUEST_METHOD": request.method,
        "PATH_INFO": request.url.path,
        "CONTENT_TYPE": request.headers.get("content-type", ""),
        "wsgi.input_body": body,
    }
    for key, value in request.headers.items():
        header_key = f"HTTP_{key.upper().replace('-', '_')}"
        environ[header_key] = value
    if request.state is not None and getattr(request.state, "injection", None):
        environ.update(request.state.injection)
    return environ


def _build_fastapi_response(status: str, headers: list[tuple[str, str]], body: bytes) -> Response:
    response = Response(content=body, status_code=_status_code_from_wsgi(status))
    for key, value in headers:
        key_lower = key.lower()
        if key_lower in _HOP_BY_HOP_HEADERS or key_lower == "content-length":
            continue
        if key_lower == "set-cookie":
            response.headers.append(key, value)
            continue
        response.headers[key] = value
    return response


def _build_local_cloud_infer(runtime: RuntimeState):
    def _local_cloud_infer(
        *,
        video_bytes: bytes,
        filename: str,
        request_id: str,
        timeout_seconds: float,
        pose_handoff: Any | None = None,
    ) -> dict[str, Any]:
        _ = (video_bytes, timeout_seconds)
        if runtime.backend is None:
            raise RuntimeError("model runtime is not ready")

        input_payload: dict[str, Any] = {"filename": filename}
        if pose_handoff is not None:
            input_payload["pose_summary"] = {
                "frame_count": int(getattr(pose_handoff, "frame_count", 0) or 0),
                "first_ts_ms": int(getattr(pose_handoff, "first_ts_ms", 0) or 0),
                "last_ts_ms": int(getattr(pose_handoff, "last_ts_ms", 0) or 0),
            }

        try:
            started = time.monotonic()
            content = runtime.backend.generate(
                model=runtime.model_id,
                messages=[
                    {"role": "system", "content": "You are an ASL translator."},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "instruction": "Translate ASL payload into short text.",
                                "input": input_payload,
                            }
                        ),
                    },
                ],
            )
            latency_ms = max(int((time.monotonic() - started) * 1000), 1)
        except Exception as exc:
            raise CloudInferError(
                "UPSTREAM_FAILURE",
                f"backend inference failed: {exc.__class__.__name__}: {_format_exception(exc)}",
                retryable=True,
                status="503 Service Unavailable",
                details={"stage": "backend_generate", "exception_class": exc.__class__.__name__},
            ) from exc

        prediction = RealBackend._normalize_prediction_text(str(content or "").strip())
        if not prediction:
            raise CloudInferError(
                "INFERENCE_FAILED",
                "model returned empty output",
                retryable=True,
                status="502 Bad Gateway",
                details={"stage": "backend_generate", "raw_output": str(content)[:200]},
            )

        invalid_output, reason = RealBackend._is_invalid_prediction(prediction)
        if invalid_output:
            salvaged = RealBackend._salvage_translation_from_structured_text(str(content or ""))
            if salvaged:
                prediction = salvaged
                invalid_output = False
            else:
                raise CloudInferError(
                    "INFERENCE_FAILED",
                    "invalid model output",
                    retryable=False,
                    status="502 Bad Gateway",
                    details={"stage": "backend_generate", "reason": reason, "raw_output": prediction[:200]},
                )


        return {
            "request_id": request_id,
            "prediction": prediction,
            "confidence": 0.5,
            "alternatives": [],
            "transcript_words": prediction.split(),
            "sequence_confidence": 0.5,
            "low_confidence": False,
            "runtime_mode": "cactus_engine",
            "cloud_handoff": False,
            "model_id": runtime.model_id,
            "model_version": runtime.model_version,
            "latency_ms": latency_ms,
        }

    return _local_cloud_infer


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
    async def chat_completions(payload: ChatCompletionRequest) -> Any:
        runtime: RuntimeState = app.state.runtime
        if not runtime.ready or runtime.backend is None:
            return _error(503, code="MODEL_NOT_READY", message="model runtime is not ready")

        try:
            content = runtime.backend.generate(
                model=payload.model,
                messages=[{"role": m.role, "content": m.content} for m in payload.messages],
            )
        except Exception as exc:
            print(json.dumps({"event": "hf_endpoint_inference_error", "error": _format_exception(exc)}), flush=True)
            return _error(502, code="INFERENCE_FAILED", message="backend inference failed")
        if not content:
            return _error(502, code="INFERENCE_FAILED", message="empty model output")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    @app.post("/v1/translate-sign")
    async def translate_sign(request: Request) -> Response:
        runtime: RuntimeState = app.state.runtime
        if not runtime.ready or runtime.backend is None:
            return _error(503, code="MODEL_NOT_READY", message="model runtime is not ready")

        environ = await _build_wsgi_environ(request)
        environ["cloud_infer_callable"] = _build_local_cloud_infer(runtime)
        try:
            status, headers, raw = translate_sign_wsgi_app(environ)
        except Exception as exc:
            print(json.dumps({"event": "hf_endpoint_translate_sign_error", "error": _format_exception(exc)}), flush=True)
            return _error(502, code="INFERENCE_FAILED", message="translate-sign pipeline failed")
        return _build_fastapi_response(status, headers, raw)

    return app


hf_custom_endpoint_app = create_hf_custom_endpoint_app()
