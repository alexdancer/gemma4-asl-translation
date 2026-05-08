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
import time
import uuid
from email.parser import BytesParser
from email.policy import default
from typing import Any, Callable

Response = tuple[str, list[tuple[str, str]], bytes]


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
        return _error("timeout", "Request exceeded timeout", request_id, retryable=True, status="504 Gateway Timeout")

    video_part = _extract_video_part(content_type, body)
    if video_part is None:
        return _error(
            "invalid_request",
            "Expected multipart/form-data with a video part named 'video'",
            request_id,
            retryable=False,
        )

    video_bytes, filename = video_part
    if not video_bytes:
        return _error("invalid_video", "Uploaded video is empty", request_id, retryable=False)

    payload = {
        "request_id": request_id,
        "gloss": "HELLO",
        "translation": "Hello",
        "confidence": 0.93,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "mock": True,
        "filename": filename,
        "bytes_received": len(video_bytes),
    }
    return _json_response("200 OK", payload)


__all__ = ["translate_sign_wsgi_app"]
