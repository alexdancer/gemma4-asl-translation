"""ASL v2 cloud translation API skeleton.

This module provides a minimal WSGI-compatible app with a single endpoint:
POST /v1/translate-sign

Behavior for slice #51:
- Accept multipart/form-data with a `video` file part
- Return mock success payload with PRD fields
- Return standardized error payload on invalid requests
- Guard with a 12-second timeout scaffold
"""

from __future__ import annotations

import json
import os
import time
import uuid
from email.parser import BytesParser
from email.policy import default
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest

Response = tuple[str, list[tuple[str, str]], bytes]
CloudInferCallable = Callable[..., dict[str, Any]]


def _json_response(status: str, payload: dict[str, Any]) -> Response:
    return status, [("Content-Type", "application/json")], json.dumps(payload).encode("utf-8")


def _error(error_code: str, message: str, request_id: str, retryable: bool, status: str = "400 Bad Request") -> Response:
    return _json_response(
        status,
        {
            "error_code": error_code,
            "message": message,
            "request_id": request_id,
            "retryable": retryable,
        },
    )


def _extract_video_part(content_type: str, body: bytes) -> tuple[bytes, str] | None:
    if "multipart/form-data" not in content_type:
        return None

    envelope = b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    message = BytesParser(policy=default).parsebytes(envelope)

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        if name != "video":
            continue

        filename = part.get_param("filename", header="content-disposition") or "upload.bin"
        payload = part.get_payload(decode=True) or b""
        return payload, filename

    return None


def _default_cloud_infer(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float) -> dict[str, Any]:
    endpoint = os.environ.get("ASL_CLOUD_INFER_URL")
    api_key = os.environ.get("ASL_CLOUD_API_KEY")
    model_name = os.environ.get("ASL_CLOUD_MODEL", "cactus-asl-v2")

    if not endpoint:
        raise RuntimeError("ASL_CLOUD_INFER_URL is not configured")
    if not api_key:
        raise RuntimeError("ASL_CLOUD_API_KEY is not configured")

    payload = {
        "request_id": request_id,
        "model": model_name,
        "input": {
            "filename": filename,
            "video_base64": video_bytes.hex(),
            "encoding": "hex",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Request-ID": request_id,
        },
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except TimeoutError as exc:
        raise TimeoutError("upstream timeout") from exc
    except urlerror.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError("upstream timeout") from exc
        raise RuntimeError(f"upstream request failed: {exc.reason}") from exc

    decoded = json.loads(raw)
    gloss = decoded.get("gloss") or decoded.get("output", {}).get("gloss")
    translation = decoded.get("translation") or decoded.get("output", {}).get("translation")
    confidence = decoded.get("confidence") or decoded.get("output", {}).get("confidence")

    if gloss is None or translation is None or confidence is None:
        raise RuntimeError("upstream response missing required fields")

    latency_ms = int(decoded.get("latency_ms") or decoded.get("timing", {}).get("latency_ms") or 0)
    return {
        "request_id": decoded.get("request_id", request_id),
        "gloss": str(gloss),
        "translation": str(translation),
        "confidence": float(confidence),
        "latency_ms": latency_ms,
    }


def translate_sign_wsgi_app(
    environ: dict[str, Any],
    _start_response: Callable[[str, list[tuple[str, str]]], None] | None = None,
    timeout_seconds: float = 12.0,
) -> Response:
    """Handle POST /v1/translate-sign and return JSON response."""

    request_id = environ.get("HTTP_X_REQUEST_ID") or str(uuid.uuid4())
    started = time.monotonic()

    method = environ.get("REQUEST_METHOD", "")
    path = environ.get("PATH_INFO", "")

    if method != "POST" or path != "/v1/translate-sign":
        return _error("not_found", "Endpoint not found", request_id, retryable=False, status="404 Not Found")

    content_type = environ.get("CONTENT_TYPE", "")
    body = environ.get("wsgi.input_body", b"")

    # 12-second timeout guard scaffold
    elapsed = time.monotonic() - started
    if elapsed > timeout_seconds:
        return _error("TIMEOUT", "Request exceeded timeout", request_id, retryable=True, status="504 Gateway Timeout")

    video_part = _extract_video_part(content_type, body)
    if video_part is None:
        return _error(
            "INVALID_REQUEST",
            "Expected multipart/form-data with a video part named 'video'",
            request_id,
            retryable=False,
        )

    video_bytes, filename = video_part
    if not video_bytes:
        return _error("INVALID_VIDEO", "Uploaded video is empty", request_id, retryable=False)

    cloud_infer = environ.get("cloud_infer_callable") or _default_cloud_infer

    try:
        result = cloud_infer(
            video_bytes=video_bytes,
            filename=filename,
            request_id=request_id,
            timeout_seconds=timeout_seconds,
        )
    except TimeoutError:
        return _error(
            "TIMEOUT",
            "Cloud inference timed out",
            request_id,
            retryable=True,
            status="504 Gateway Timeout",
        )
    except Exception as exc:
        return _error(
            "UPSTREAM_FAILURE",
            f"Cloud inference failed: {exc}",
            request_id,
            retryable=True,
            status="503 Service Unavailable",
        )

    payload = {
        "request_id": result.get("request_id", request_id),
        "gloss": result["gloss"],
        "translation": result["translation"],
        "confidence": float(result["confidence"]),
        "latency_ms": int(result.get("latency_ms", (time.monotonic() - started) * 1000)),
    }

    print(
        json.dumps(
            {
                "event": "translate_sign",
                "request_id": payload["request_id"],
                "filename": filename,
                "bytes_received": len(video_bytes),
                "latency_ms": payload["latency_ms"],
                "status": "ok",
            }
        )
    )

    return _json_response("200 OK", payload)


__all__ = ["translate_sign_wsgi_app"]
