from __future__ import annotations

import json
import uuid
from pathlib import Path

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


def test_translate_sign_accepts_multipart_and_returns_real_schema(tmp_path: Path, monkeypatch) -> None:
    telemetry_file = tmp_path / "cloud_telemetry.jsonl"
    monkeypatch.setenv("ASL_TELEMETRY_PATH", str(telemetry_file))
    rid = str(uuid.uuid4())

    def fake_cloud_infer(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float):
        assert video_bytes == b"video-bytes"
        assert filename == "clip.mov"
        assert request_id == rid
        assert timeout_seconds == 12.0
        return {
            "request_id": request_id,
            "gloss": "HELLO",
            "translation": "Hello",
            "confidence": 0.93,
            "latency_ms": 123,
        }

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "HTTP_X_REQUEST_ID": rid,
        "cloud_infer_callable": fake_cloud_infer,
    }
    status, headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert dict(headers)["Content-Type"] == "application/json"
    assert payload == {
        "request_id": rid,
        "gloss": "HELLO",
        "translation": "Hello",
        "confidence": 0.93,
        "latency_ms": 123,
    }

    telemetry_lines = telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(telemetry_lines) == 1
    event = json.loads(telemetry_lines[0])
    assert event["request_id"] == rid
    assert event["outcome"] == "success"
    assert "video" not in event
    assert "video_bytes" not in event


def test_translate_sign_requires_multipart_video_field() -> None:
    status, _headers, payload = _call_app(
        method="POST",
        path="/v1/translate-sign",
        content_type=f"multipart/form-data; boundary={BOUNDARY}",
        body=_multipart_body(field_name="not_video"),
    )

    assert status == "400 Bad Request"
    assert payload["error_code"] == "INVALID_REQUEST"
    assert payload["retryable"] is False
    assert "request_id" in payload


def test_translate_sign_returns_retryable_timeout_error_when_cloud_times_out() -> None:
    def fake_timeout(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float):
        raise TimeoutError("simulated timeout")

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "cloud_infer_callable": fake_timeout,
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "504 Gateway Timeout"
    assert payload["error_code"] == "TIMEOUT"
    assert payload["retryable"] is True
    assert isinstance(payload["request_id"], str)


def test_translate_sign_returns_service_unavailable_and_request_id_on_cloud_failure() -> None:
    def fake_failure(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float):
        raise RuntimeError("upstream unavailable")

    rid = str(uuid.uuid4())
    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "HTTP_X_REQUEST_ID": rid,
        "cloud_infer_callable": fake_failure,
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "503 Service Unavailable"
    assert payload["error_code"] == "UPSTREAM_FAILURE"
    assert payload["request_id"] == rid
    assert payload["retryable"] is True


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
