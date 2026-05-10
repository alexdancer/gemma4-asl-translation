from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

from src.cloud_translate_api import _default_cloud_infer, translate_sign_wsgi_app
from src.video_ingest import CanonicalVideoProfile, VideoProbeResult


BOUNDARY = "----WebKitFormBoundary7MA4YWxkTrZu0gW"


def _canonical_probe() -> VideoProbeResult:
    return VideoProbeResult(duration_seconds=1.0, fps=30.0, width=1280, height=720)


def _default_video_ingest(video_bytes: bytes, _filename: str):
    probe = _canonical_probe()
    return probe, probe, video_bytes, False, CanonicalVideoProfile()


def _base_environ(*, method: str = "POST", path: str = "/v1/translate-sign", content_type: str | None = None, body: bytes | None = None, request_id: str | None = None) -> dict:
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_TYPE": content_type or f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": body if body is not None else _multipart_body(),
        "video_ingest_callable": _default_video_ingest,
    }
    if request_id:
        environ["HTTP_X_REQUEST_ID"] = request_id
    return environ


def _multipart_body(field_name: str = "video", filename: str = "clip.mov", payload: bytes = b"video-bytes") -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
        "Content-Type: video/quicktime\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{BOUNDARY}--\r\n".encode("utf-8")


def _call_app(method: str, path: str, content_type: str, body: bytes, request_id: str | None = None):
    environ = _base_environ(method=method, path=path, content_type=content_type, body=body, request_id=request_id)
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
        "video_ingest_callable": lambda video_bytes, _filename: (
            _canonical_probe(),
            _canonical_probe(),
            video_bytes,
            False,
            CanonicalVideoProfile(),
        ),
    }
    status, headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert dict(headers)["Content-Type"] == "application/json"
    assert payload["request_id"] == rid
    assert payload["gloss"] == "HELLO"
    assert payload["translation"] == "Hello"
    assert payload["confidence"] == 0.93
    assert payload["latency_ms"] == 123
    assert payload["video_ingest"]["normalization_applied"] is False

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
        "video_ingest_callable": lambda video_bytes, _filename: (
            _canonical_probe(),
            _canonical_probe(),
            video_bytes,
            False,
            CanonicalVideoProfile(),
        ),
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
        "video_ingest_callable": lambda video_bytes, _filename: (
            _canonical_probe(),
            _canonical_probe(),
            video_bytes,
            False,
            CanonicalVideoProfile(),
        ),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "503 Service Unavailable"
    assert payload["error_code"] == "UPSTREAM_FAILURE"
    assert payload["request_id"] == rid
    assert payload["retryable"] is True


def test_translate_sign_rejects_oversized_upload() -> None:
    huge = b"x" * 60_000_001
    status, _headers, payload = _call_app(
        method="POST",
        path="/v1/translate-sign",
        content_type=f"multipart/form-data; boundary={BOUNDARY}",
        body=_multipart_body(payload=huge),
    )

    assert status == "413 Payload Too Large"
    assert payload["error_code"] == "PAYLOAD_TOO_LARGE"
    assert payload["retryable"] is False


def test_translate_sign_telemetry_failure_is_fail_open(monkeypatch) -> None:
    def fake_cloud_infer(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float):
        return {
            "request_id": request_id,
            "gloss": "HELLO",
            "translation": "Hello",
            "confidence": 0.93,
            "latency_ms": 42,
        }

    def explode_append_event(_event):
        raise OSError("disk full")

    monkeypatch.setattr("src.cloud_translate_api.append_event", explode_append_event)

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "cloud_infer_callable": fake_cloud_infer,
        "video_ingest_callable": lambda video_bytes, _filename: (
            _canonical_probe(),
            _canonical_probe(),
            video_bytes,
            False,
            CanonicalVideoProfile(),
        ),
    }
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert payload["gloss"] == "HELLO"


def test_default_cloud_infer_sends_real_base64(monkeypatch) -> None:
    captured = {}

    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"gloss":"HELLO","translation":"Hello","confidence":0.9,"latency_ms":12}'

    def fake_urlopen(req, timeout):
        captured["body"] = req.data
        captured["timeout"] = timeout
        return DummyResp()

    monkeypatch.setenv("ASL_CLOUD_INFER_URL", "https://example.test/infer")
    monkeypatch.setenv("ASL_CLOUD_API_KEY", "k")
    monkeypatch.setattr("src.cloud_translate_api.urlrequest.urlopen", fake_urlopen)

    _default_cloud_infer(video_bytes=b"abc", filename="clip.mov", request_id="rid-1", timeout_seconds=3.0)
    body = json.loads(captured["body"].decode("utf-8"))

    assert body["input"]["encoding"] == "base64"
    assert body["input"]["video_base64"] == base64.b64encode(b"abc").decode("ascii")


def test_translate_sign_unknown_path_uses_standard_error_schema() -> None:
    status, _headers, payload = _call_app(
        method="POST",
        path="/v1/unknown",
        content_type=f"multipart/form-data; boundary={BOUNDARY}",
        body=_multipart_body(),
    )

    assert status == "404 Not Found"
    assert payload["error_code"] == "NOT_FOUND"
    assert isinstance(payload["message"], str)
    assert isinstance(payload["request_id"], str)
    assert isinstance(payload["retryable"], bool)


def test_translate_sign_rejects_video_longer_than_10_seconds() -> None:
    def long_duration(_video_bytes: bytes, _filename: str):
        return (
            VideoProbeResult(duration_seconds=10.8, fps=30.0, width=1280, height=720),
            VideoProbeResult(duration_seconds=10.8, fps=30.0, width=1280, height=720),
            b"canonical",
            False,
            CanonicalVideoProfile(),
        )

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "video_ingest_callable": long_duration,
    }
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "400 Bad Request"
    assert payload["error_code"] == "VIDEO_DURATION_EXCEEDED"
    assert payload["retryable"] is False


def test_translate_sign_emits_video_ingest_metadata_and_uses_canonical_bytes() -> None:
    captured = {}

    def fake_process(video_bytes: bytes, filename: str):
        assert video_bytes == b"video-bytes"
        assert filename == "clip.mov"
        return (
            VideoProbeResult(duration_seconds=4.2, fps=24.0, width=1920, height=1080),
            VideoProbeResult(duration_seconds=4.2, fps=30.0, width=1280, height=720),
            b"canonical-bytes",
            True,
            CanonicalVideoProfile(),
        )

    def fake_cloud_infer(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float):
        captured["video_bytes"] = video_bytes
        return {
            "request_id": request_id,
            "gloss": "HELLO",
            "translation": "Hello",
            "confidence": 0.88,
            "latency_ms": 111,
        }

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "cloud_infer_callable": fake_cloud_infer,
        "video_ingest_callable": fake_process,
    }
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert captured["video_bytes"] == b"canonical-bytes"
    assert payload["video_ingest"]["normalization_applied"] is True
    assert payload["video_ingest"]["original"]["fps"] == 24.0
    assert payload["video_ingest"]["normalized"]["fps"] == 30.0
    assert payload["video_ingest"]["target"]["width"] == 1280


def test_translate_sign_returns_structured_error_for_unreadable_video() -> None:
    def fake_process(_video_bytes: bytes, _filename: str):
        raise RuntimeError("ffprobe failed")

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "video_ingest_callable": fake_process,
    }
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "400 Bad Request"
    assert payload["error_code"] == "INVALID_VIDEO"
    assert payload["retryable"] is False

