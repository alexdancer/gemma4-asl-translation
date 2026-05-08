from __future__ import annotations

import json
import uuid

from src.cloud_translate_api import translate_sign_wsgi_app


BOUNDARY = "----WebKitFormBoundary7MA4YWxkTrZu0gW"


def _multipart_body(field_name: str = "video", filename: str = "clip.mov", payload: bytes = b"video-bytes") -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
        "Content-Type: video/quicktime\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{BOUNDARY}--\r\n".encode("utf-8")


def _call_app(method: str, path: str, content_type: str, body: bytes, request_id: str | None = None):
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_TYPE": content_type,
        "wsgi.input_body": body,
    }
    if request_id:
        environ["HTTP_X_REQUEST_ID"] = request_id
    status, headers, raw = translate_sign_wsgi_app(environ)
    return status, dict(headers), json.loads(raw.decode("utf-8"))


def test_translate_sign_accepts_multipart_and_returns_mock_schema() -> None:
    rid = str(uuid.uuid4())
    status, headers, payload = _call_app(
        method="POST",
        path="/v1/translate-sign",
        content_type=f"multipart/form-data; boundary={BOUNDARY}",
        body=_multipart_body(),
        request_id=rid,
    )

    assert status == "200 OK"
    assert headers["Content-Type"] == "application/json"

    assert payload["request_id"] == rid
    assert isinstance(payload["gloss"], str)
    assert isinstance(payload["translation"], str)
    assert isinstance(payload["confidence"], float)
    assert isinstance(payload["latency_ms"], int)


def test_translate_sign_requires_multipart_video_field() -> None:
    status, _headers, payload = _call_app(
        method="POST",
        path="/v1/translate-sign",
        content_type=f"multipart/form-data; boundary={BOUNDARY}",
        body=_multipart_body(field_name="not_video"),
    )

    assert status == "400 Bad Request"
    assert payload["error_code"] == "invalid_request"
    assert payload["retryable"] is False
    assert "request_id" in payload


def test_translate_sign_unknown_path_uses_standard_error_schema() -> None:
    status, _headers, payload = _call_app(
        method="POST",
        path="/v1/unknown",
        content_type=f"multipart/form-data; boundary={BOUNDARY}",
        body=_multipart_body(),
    )

    assert status == "404 Not Found"
    assert payload["error_code"] == "not_found"
    assert isinstance(payload["message"], str)
    assert isinstance(payload["request_id"], str)
    assert isinstance(payload["retryable"], bool)
