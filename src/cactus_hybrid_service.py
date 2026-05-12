"""WSGI service for Arch-hosted Cactus hybrid inference with HF OpenAI handoff.

Endpoint contract (for cloud_translate_api upstream call):
- POST /
- Authorization: Bearer <ASL_CACTUS_SERVICE_API_KEY>
- JSON: {request_id, model, input{filename, pose_summary?, pose_sequence?, video_base64?}}

Response contract:
- request_id, prediction, confidence
- runtime_mode, cloud_handoff, model_id, model_version
- latency_ms

Fail-closed rule: if cloud handoff fails or proof fields cannot be produced, return non-200.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest


class CloudHandoffError(RuntimeError):
    """Structured upstream handoff error."""

    def __init__(self, message: str, *, code: str = "CLOUD_HANDOFF_FAILED") -> None:
        super().__init__(message)
        self.code = code

Response = tuple[str, list[tuple[str, str]], bytes]
HybridInferCallable = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    method: str
    path: str
    body: bytes
    auth_header: str


def _json_response(status: str, payload: dict[str, Any]) -> Response:
    return status, [("Content-Type", "application/json")], json.dumps(payload).encode("utf-8")


def _error(code: str, message: str, request_id: str, status: str, retryable: bool) -> Response:
    return _json_response(
        status,
        {
            "error_code": code,
            "message": message,
            "request_id": request_id,
            "retryable": retryable,
        },
    )


def _build_context(environ: dict[str, Any]) -> RequestContext:
    body = environ.get("wsgi.input_body")
    if not isinstance(body, (bytes, bytearray)):
        stream = environ.get("wsgi.input")
        content_length_raw = environ.get("CONTENT_LENGTH")
        try:
            content_length = int(content_length_raw or 0)
        except (TypeError, ValueError):
            content_length = 0
        if hasattr(stream, "read") and content_length > 0:
            body = stream.read(content_length)
        else:
            body = b""
    return RequestContext(
        request_id=str(environ.get("HTTP_X_REQUEST_ID") or ""),
        method=str(environ.get("REQUEST_METHOD") or ""),
        path=str(environ.get("PATH_INFO") or ""),
        body=bytes(body),
        auth_header=str(environ.get("HTTP_AUTHORIZATION") or ""),
    )


def _require_auth(context: RequestContext) -> bool:
    expected = os.environ.get("ASL_CACTUS_SERVICE_API_KEY", "").strip()
    if not expected:
        return False
    if not context.auth_header.lower().startswith("bearer "):
        return False
    got = context.auth_header.split(" ", 1)[1].strip()
    return hmac.compare_digest(got, expected)


def _require_hf_config() -> tuple[str, str, str]:
    base_url = os.environ.get("ASL_HF_OPENAI_BASE_URL", "").strip()
    hf_token = os.environ.get("ASL_HF_TOKEN", "").strip()
    model_version = os.environ.get("ASL_CACTUS_MODEL_VERSION", "v1")
    if not base_url or not hf_token:
        raise CloudHandoffError("HF handoff is not configured")
    return base_url, hf_token, model_version


def _hf_post_json(*, endpoint: str, body: dict[str, Any], hf_token: str, request_id: str, timeout_seconds: float) -> dict[str, Any]:
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {hf_token}",
            "X-Request-ID": request_id,
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except TimeoutError as exc:
        raise CloudHandoffError("cloud handoff timeout") from exc
    except urlerror.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        try:
            decoded = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            decoded = {}
        err = decoded.get("error") if isinstance(decoded, dict) else None
        code = ""
        message = ""
        if isinstance(err, dict):
            code = str(err.get("code") or "").strip()
            message = str(err.get("message") or "").strip()
        detail = message or raw or str(exc.reason)
        raise CloudHandoffError(f"cloud handoff http {exc.code}: {detail}", code=code or "CLOUD_HANDOFF_FAILED") from exc
    except urlerror.URLError as exc:
        raise CloudHandoffError(f"cloud handoff failed: {exc.reason}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CloudHandoffError("cloud handoff returned malformed json") from exc


def _extract_prediction_from_chat(decoded: dict[str, Any]) -> str:
    choices = decoded.get("choices") if isinstance(decoded, dict) else None
    if not isinstance(choices, list) or not choices:
        raise CloudHandoffError("cloud handoff missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not content:
        raise CloudHandoffError("cloud handoff missing completion content")
    return str(content).strip()


def _extract_prediction_from_completion(decoded: dict[str, Any]) -> str:
    choices = decoded.get("choices") if isinstance(decoded, dict) else None
    if isinstance(choices, list) and choices:
        text = choices[0].get("text") if isinstance(choices[0], dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    generated_text = decoded.get("generated_text") if isinstance(decoded, dict) else None
    if isinstance(generated_text, str) and generated_text.strip():
        return generated_text.strip()
    raise CloudHandoffError("cloud handoff missing generated text")


def _cloud_handoff_chat(*, request_id: str, model: str, input_payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    base_url, hf_token, model_version = _require_hf_config()
    endpoint = base_url.rstrip("/") + "/chat/completions"
    prompt = {
        "instruction": "Translate ASL payload into short text.",
        "input": input_payload,
    }
    decoded = _hf_post_json(
        endpoint=endpoint,
        body={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an ASL translator."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "temperature": 0,
        },
        hf_token=hf_token,
        request_id=request_id,
        timeout_seconds=timeout_seconds,
    )
    prediction = _extract_prediction_from_chat(decoded)
    return {
        "request_id": request_id,
        "prediction": prediction,
        "confidence": 0.5,
        "runtime_mode": "cactus_engine",
        "cloud_handoff": True,
        "model_id": model,
        "model_version": model_version,
    }


def _cloud_handoff_completion(*, request_id: str, model: str, input_payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    base_url, hf_token, model_version = _require_hf_config()
    endpoint = base_url.rstrip("/") + "/completions"
    prompt = json.dumps(
        {
            "instruction": "Translate ASL payload into short text.",
            "input": input_payload,
        }
    )
    decoded = _hf_post_json(
        endpoint=endpoint,
        body={
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 128,
        },
        hf_token=hf_token,
        request_id=request_id,
        timeout_seconds=timeout_seconds,
    )
    prediction = _extract_prediction_from_completion(decoded)
    return {
        "request_id": request_id,
        "prediction": prediction,
        "confidence": 0.5,
        "runtime_mode": "cactus_engine",
        "cloud_handoff": True,
        "model_id": model,
        "model_version": model_version,
    }


def _run_hybrid(*, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    request_id = str(payload.get("request_id") or "")
    if not request_id:
        raise ValueError("missing request_id")
    model_id = str(payload.get("model") or "cactus-asl-v2")
    input_payload = payload.get("input")
    if not isinstance(input_payload, dict):
        raise ValueError("missing input payload")

    route_mode = os.environ.get("ASL_HF_ROUTE_MODE", "auto").strip().lower() or "auto"
    if route_mode not in {"auto", "chat", "completion"}:
        raise ValueError("ASL_HF_ROUTE_MODE must be one of: auto, chat, completion")

    if route_mode == "chat":
        return _cloud_handoff_chat(
            request_id=request_id,
            model=model_id,
            input_payload=input_payload,
            timeout_seconds=timeout_seconds,
        )
    if route_mode == "completion":
        return _cloud_handoff_completion(
            request_id=request_id,
            model=model_id,
            input_payload=input_payload,
            timeout_seconds=timeout_seconds,
        )

    # auto: prefer chat, fall back to completion when model is not chat-capable.
    try:
        return _cloud_handoff_chat(
            request_id=request_id,
            model=model_id,
            input_payload=input_payload,
            timeout_seconds=timeout_seconds,
        )
    except CloudHandoffError as exc:
        if exc.code == "model_not_supported":
            return _cloud_handoff_completion(
                request_id=request_id,
                model=model_id,
                input_payload=input_payload,
                timeout_seconds=timeout_seconds,
            )
        raise


def _normalize_proof_response(*, request_id: str, result: Any, started: float) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("inference result must be an object")

    required_fields = {"runtime_mode", "cloud_handoff", "model_id", "model_version", "prediction", "confidence"}
    missing = [field for field in sorted(required_fields) if field not in result]
    if missing:
        raise KeyError(f"Missing required proof fields: {', '.join(missing)}")

    if not isinstance(result["cloud_handoff"], bool):
        raise ValueError("cloud_handoff must be a boolean")

    try:
        confidence = float(result["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    prediction = str(result["prediction"]).strip()
    if not prediction:
        raise ValueError("prediction must be non-empty")

    runtime_mode = result["runtime_mode"]
    model_id = result["model_id"]
    model_version = result["model_version"]
    if not isinstance(runtime_mode, str) or not runtime_mode.strip():
        raise ValueError("runtime_mode must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string")

    result_request_id = str(result.get("request_id") or "").strip()
    if result_request_id and result_request_id != request_id:
        raise ValueError("request_id mismatch between request and proof response")

    return {
        "request_id": request_id,
        "prediction": prediction,
        "confidence": confidence,
        "runtime_mode": runtime_mode,
        "cloud_handoff": result["cloud_handoff"],
        "model_id": model_id,
        "model_version": model_version,
        "latency_ms": max(int((time.monotonic() - started) * 1000), 1),
    }


def hybrid_wsgi_app(environ: dict[str, Any], timeout_seconds: float = 20.0) -> Response:
    started = time.monotonic()
    context = _build_context(environ)

    if context.method != "POST" or context.path != "/":
        return _error("NOT_FOUND", "Endpoint not found", context.request_id or "unknown", "404 Not Found", False)

    if not _require_auth(context):
        return _error("UNAUTHORIZED", "Missing or invalid bearer token", context.request_id or "unknown", "401 Unauthorized", False)

    try:
        payload = json.loads(context.body.decode("utf-8"))
    except Exception:
        return _error("INVALID_JSON", "Malformed JSON body", context.request_id or "unknown", "400 Bad Request", False)

    if not isinstance(payload, dict):
        return _error("INVALID_REQUEST", "JSON body must be an object", context.request_id or "unknown", "400 Bad Request", False)

    request_id = str(payload.get("request_id") or context.request_id or "unknown")

    infer_callable = environ.get("hybrid_infer_callable") or _run_hybrid
    try:
        result = infer_callable(payload=payload, timeout_seconds=timeout_seconds)
    except ValueError as exc:
        return _error("INVALID_REQUEST", str(exc), request_id, "400 Bad Request", False)
    except Exception:
        # Fail closed on handoff failure.
        return _error(
            "CLOUD_HANDOFF_FAILED",
            "Cloud handoff request failed",
            request_id,
            "503 Service Unavailable",
            True,
        )

    try:
        response = _normalize_proof_response(request_id=request_id, result=result, started=started)
    except KeyError as exc:
        return _error("PROOF_FIELDS_MISSING", str(exc), request_id, "502 Bad Gateway", True)
    except ValueError as exc:
        return _error("PROOF_FIELDS_INVALID", str(exc), request_id, "502 Bad Gateway", True)

    return _json_response("200 OK", response)


__all__ = ["hybrid_wsgi_app"]
