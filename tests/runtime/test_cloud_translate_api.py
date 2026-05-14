from __future__ import annotations

import base64
import io
import json
import os
from types import SimpleNamespace
import uuid
from pathlib import Path
from urllib import error as urlerror

from src import cloud_translate_api as cloud_translate_api_module
from src.cloud_translate_api import CloudInferError, _default_cloud_infer, translate_sign_wsgi_app
from src.frame_extraction import FrameExtractionError
from src.video_ingest import CanonicalVideoProfile, VideoProbeResult


BOUNDARY = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
TEST_API_KEY = "test-key"
os.environ.setdefault("ASL_V1_API_KEYS", TEST_API_KEY)


def setup_function(_function) -> None:
    cloud_translate_api_module._RATE_LIMIT_HITS_BY_KEY.clear()


def _canonical_probe() -> VideoProbeResult:
    return VideoProbeResult(duration_seconds=1.0, fps=30.0, width=1280, height=720)


def _default_video_ingest(video_bytes: bytes, _filename: str):
    probe = _canonical_probe()
    return probe, probe, video_bytes, False, CanonicalVideoProfile()


def _default_frame_extractor(_video_bytes: bytes, *, probe: VideoProbeResult):
    assert probe.fps == 30.0
    return SimpleNamespace(
        frame_count=30,
        first_ts_ms=0,
        last_ts_ms=967,
        effective_fps=30.0,
        cadence="fixed_fps",
    )


def _default_pose_pipeline(*, request_id: str, frame_extraction):
    return SimpleNamespace(
        request_id=request_id,
        frame_count=frame_extraction.frame_count,
        first_ts_ms=frame_extraction.first_ts_ms,
        last_ts_ms=frame_extraction.last_ts_ms,
    )


def _proof_fields(*, model_id: str = "cactus-asl-v2", model_version: str = "v1", runtime_mode: str = "cactus_engine", cloud_handoff: bool = True) -> dict:
    return {
        "runtime_mode": runtime_mode,
        "cloud_handoff": cloud_handoff,
        "model_id": model_id,
        "model_version": model_version,
    }


def _base_environ(*, method: str = "POST", path: str = "/v1/translate-sign", content_type: str | None = None, body: bytes | None = None, request_id: str | None = None) -> dict:
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_TYPE": content_type or f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": body if body is not None else _multipart_body(),
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "video_ingest_callable": _default_video_ingest,
        "frame_extractor_callable": _default_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
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
            **_proof_fields(),
        }

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "HTTP_X_REQUEST_ID": rid,
        "cloud_infer_callable": fake_cloud_infer,
        "video_ingest_callable": lambda video_bytes, _filename: (
            _canonical_probe(),
            _canonical_probe(),
            video_bytes,
            False,
            CanonicalVideoProfile(),
        ),
        "frame_extractor_callable": _default_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
    }
    status, headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert dict(headers)["Content-Type"] == "application/json"
    assert payload["request_id"] == rid
    assert payload["prediction"] == "Hello"
    assert payload["translation"] == "Hello"
    assert payload["confidence"] == 0.93
    assert payload["latency_ms"] == 123
    assert payload["runtime_mode"] == "cactus_engine"
    assert payload["cloud_handoff"] is True
    assert payload["model_id"] == "cactus-asl-v2"
    assert payload["model_version"] == "v1"
    assert payload["video_ingest"]["normalization_applied"] is False
    assert payload["frame_extraction"]["frame_count"] == 30
    assert payload["frame_extraction"]["effective_fps"] == 30.0
    assert payload["frame_extraction"]["cadence"] == "fixed_fps"
    assert payload["pose_extraction"]["frame_count"] == 30
    assert payload["pose_extraction"]["first_ts_ms"] == 0
    assert payload["pose_extraction"]["last_ts_ms"] == 967
    assert payload["pose_extraction"]["aligned_with_frame_timestamps"] is True

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
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "cloud_infer_callable": fake_timeout,
        "video_ingest_callable": lambda video_bytes, _filename: (
            _canonical_probe(),
            _canonical_probe(),
            video_bytes,
            False,
            CanonicalVideoProfile(),
        ),
        "frame_extractor_callable": _default_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
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
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "HTTP_X_REQUEST_ID": rid,
        "cloud_infer_callable": fake_failure,
        "video_ingest_callable": lambda video_bytes, _filename: (
            _canonical_probe(),
            _canonical_probe(),
            video_bytes,
            False,
            CanonicalVideoProfile(),
        ),
        "frame_extractor_callable": _default_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
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
            **_proof_fields(),
        }

    def explode_append_event(_event):
        raise OSError("disk full")

    monkeypatch.setattr("src.cloud_translate_api.append_event", explode_append_event)

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "cloud_infer_callable": fake_cloud_infer,
        "video_ingest_callable": lambda video_bytes, _filename: (
            _canonical_probe(),
            _canonical_probe(),
            video_bytes,
            False,
            CanonicalVideoProfile(),
        ),
        "frame_extractor_callable": _default_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
    }
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert payload["prediction"] == "Hello"


def test_default_cloud_infer_sends_real_base64(monkeypatch) -> None:
    captured = {}

    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"gloss":"HELLO","translation":"Hello","confidence":0.9,"latency_ms":12,"runtime_mode":"cactus_engine","cloud_handoff":true,"model_id":"cactus-asl-v2","model_version":"v1"}'

    def fake_urlopen(req, timeout):
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
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
    assert "x-api-key" not in {k.lower(): v for k, v in captured.get("headers", {}).items()}


def test_default_cloud_infer_uses_multipart_for_translate_sign_endpoint(monkeypatch) -> None:
    captured = {}

    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"translation":"Hello","confidence":0.9,"latency_ms":12,"runtime_mode":"cactus_engine","cloud_handoff":true,"model_id":"cactus-asl-v2","model_version":"v1"}'

    def fake_urlopen(req, timeout):
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = req.data
        captured["timeout"] = timeout
        return DummyResp()

    monkeypatch.setenv("ASL_CLOUD_INFER_URL", "https://example.test/v1/translate-sign")
    monkeypatch.setenv("ASL_CLOUD_API_KEY", "k")
    monkeypatch.setattr("src.cloud_translate_api.urlrequest.urlopen", fake_urlopen)

    _default_cloud_infer(video_bytes=b"abc", filename="clip.mov", request_id="rid-1", timeout_seconds=3.0)

    assert captured["headers"]["content-type"].startswith("multipart/form-data; boundary=asl-boundary-rid-1")
    assert b"name=\"video\"; filename=\"clip.mov\"" in captured["body"]
    assert captured["body"].endswith(b"\r\n--asl-boundary-rid-1--\r\n")
    assert b"abc" in captured["body"]


def test_default_cloud_infer_sets_upstream_x_api_key_when_configured(monkeypatch) -> None:
    captured = {}

    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"translation":"Hello","confidence":0.9,"latency_ms":12,"runtime_mode":"cactus_engine","cloud_handoff":true,"model_id":"cactus-asl-v2","model_version":"v1"}'

    def fake_urlopen(req, timeout):
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["timeout"] = timeout
        return DummyResp()

    monkeypatch.setenv("ASL_CLOUD_INFER_URL", "https://example.test/v1/translate-sign")
    monkeypatch.setenv("ASL_CLOUD_API_KEY", "hf-token")
    monkeypatch.setenv("ASL_CLOUD_UPSTREAM_APP_KEY", "app-key")
    monkeypatch.setattr("src.cloud_translate_api.urlrequest.urlopen", fake_urlopen)

    _default_cloud_infer(video_bytes=b"abc", filename="clip.mov", request_id="rid-1", timeout_seconds=3.0)

    assert captured["headers"]["authorization"] == "Bearer hf-token"
    assert captured["headers"]["x-api-key"] == "app-key"


def test_default_cloud_infer_includes_pose_summary_when_pose_handoff_present(monkeypatch) -> None:
    captured = {}

    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"translation":"Hello","confidence":0.9,"latency_ms":12,"runtime_mode":"cactus_engine","cloud_handoff":true,"model_id":"cactus-asl-v2","model_version":"v1"}'

    def fake_urlopen(req, timeout):
        captured["body"] = req.data
        captured["timeout"] = timeout
        return DummyResp()

    monkeypatch.setenv("ASL_CLOUD_INFER_URL", "https://example.test/infer")
    monkeypatch.setenv("ASL_CLOUD_API_KEY", "k")
    monkeypatch.setattr("src.cloud_translate_api.urlrequest.urlopen", fake_urlopen)

    pose_handoff = SimpleNamespace(
        frame_count=30,
        first_ts_ms=0,
        last_ts_ms=967,
        frames=[SimpleNamespace(index=0, timestamp_ms=0, landmarks={"body": []})],
    )

    _default_cloud_infer(
        video_bytes=b"abc",
        filename="clip.mov",
        request_id="rid-1",
        timeout_seconds=3.0,
        pose_handoff=pose_handoff,
    )
    body = json.loads(captured["body"].decode("utf-8"))
    assert body["input"]["pose_summary"] == {"frame_count": 30, "first_ts_ms": 0, "last_ts_ms": 967}


def test_default_cloud_infer_includes_pose_sequence_when_enabled(monkeypatch) -> None:
    captured = {}

    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"translation":"Hello","confidence":0.9,"latency_ms":12,"runtime_mode":"cactus_engine","cloud_handoff":true,"model_id":"cactus-asl-v2","model_version":"v1"}'

    def fake_urlopen(req, timeout):
        captured["body"] = req.data
        return DummyResp()

    monkeypatch.setenv("ASL_CLOUD_INFER_URL", "https://example.test/infer")
    monkeypatch.setenv("ASL_CLOUD_API_KEY", "k")
    monkeypatch.setenv("ASL_INFER_INCLUDE_POSE_SEQUENCE", "1")
    monkeypatch.setattr("src.cloud_translate_api.urlrequest.urlopen", fake_urlopen)

    pose_handoff = SimpleNamespace(
        frame_count=1,
        first_ts_ms=0,
        last_ts_ms=0,
        frames=[SimpleNamespace(index=0, timestamp_ms=0, landmarks={"body": [[0.0, 0.0, 0.0, 1.0]]})],
    )

    _default_cloud_infer(
        video_bytes=b"abc",
        filename="clip.mov",
        request_id="rid-1",
        timeout_seconds=3.0,
        pose_handoff=pose_handoff,
    )
    body = json.loads(captured["body"].decode("utf-8"))
    assert body["input"]["pose_sequence"][0]["index"] == 0
    assert body["input"]["pose_sequence"][0]["timestamp_ms"] == 0


def test_default_cloud_infer_accepts_zero_confidence(monkeypatch) -> None:
    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"translation":"Hello","confidence":0,"latency_ms":12,"runtime_mode":"cactus_engine","cloud_handoff":true,"model_id":"cactus-asl-v2","model_version":"v1"}'

    def fake_urlopen(req, timeout):
        return DummyResp()

    monkeypatch.setenv("ASL_CLOUD_INFER_URL", "https://example.test/infer")
    monkeypatch.setenv("ASL_CLOUD_API_KEY", "k")
    monkeypatch.setattr("src.cloud_translate_api.urlrequest.urlopen", fake_urlopen)

    result = _default_cloud_infer(video_bytes=b"abc", filename="clip.mov", request_id="rid-1", timeout_seconds=3.0)
    assert result["confidence"] == 0.0


def test_default_cloud_infer_propagates_upstream_error_payload(monkeypatch) -> None:
    def fake_urlopen(req, timeout):
        _ = (req, timeout)
        payload = json.dumps(
            {
                "error_code": "INFERENCE_FAILED",
                "message": "invalid model output",
                "retryable": False,
                "details": {"stage": "backend_generate", "reason": "structured_echo"},
            }
        ).encode("utf-8")
        raise urlerror.HTTPError(
            url="https://example.test/v1/translate-sign",
            code=502,
            msg="Bad Gateway",
            hdrs=None,
            fp=io.BytesIO(payload),
        )

    monkeypatch.setenv("ASL_CLOUD_INFER_URL", "https://example.test/v1/translate-sign")
    monkeypatch.setenv("ASL_CLOUD_API_KEY", "k")
    monkeypatch.setattr("src.cloud_translate_api.urlrequest.urlopen", fake_urlopen)

    try:
        _default_cloud_infer(video_bytes=b"abc", filename="clip.mov", request_id="rid-1", timeout_seconds=3.0)
    except CloudInferError as exc:
        assert exc.code == "INFERENCE_FAILED"
        assert str(exc) == "invalid model output"
        assert exc.retryable is False
        assert exc.status == "502 Bad Gateway"
        assert exc.details.get("stage") == "backend_generate"
        assert exc.details.get("reason") == "structured_echo"
        assert exc.details.get("http_status") == 502
    else:
        raise AssertionError("expected CloudInferError")


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
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "video_ingest_callable": long_duration,
        "frame_extractor_callable": _default_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
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
            **_proof_fields(),
        }

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "cloud_infer_callable": fake_cloud_infer,
        "video_ingest_callable": fake_process,
        "frame_extractor_callable": _default_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
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
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "video_ingest_callable": fake_process,
        "frame_extractor_callable": _default_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
    }
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "400 Bad Request"
    assert payload["error_code"] == "INVALID_VIDEO"
    assert payload["retryable"] is False


def test_translate_sign_returns_422_frame_count_exceeded_details() -> None:
    def fake_frame_extractor(_video_bytes: bytes, *, probe: VideoProbeResult):
        raise FrameExtractionError(
            "FRAME_COUNT_EXCEEDED",
            "Frame extraction would exceed the configured maximum frame count",
            details={"fps": 30.0, "max_frames": 300, "video_duration_s": 10.0},
        )

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/translate-sign",
        "CONTENT_TYPE": f"multipart/form-data; boundary={BOUNDARY}",
        "wsgi.input_body": _multipart_body(),
        "HTTP_AUTHORIZATION": f"Bearer {TEST_API_KEY}",
        "video_ingest_callable": _default_video_ingest,
        "frame_extractor_callable": fake_frame_extractor,
        "pose_pipeline_callable": _default_pose_pipeline,
    }
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "422 Unprocessable Entity"
    assert payload["error_code"] == "FRAME_COUNT_EXCEEDED"
    assert payload["retryable"] is False
    assert payload["details"] == {"fps": 30.0, "max_frames": 300, "video_duration_s": 10.0}


def test_translate_sign_returns_503_when_pose_dependencies_unavailable() -> None:
    def unavailable_pose_pipeline(*, request_id: str, frame_extraction):
        from src.pose_handoff import PosePipelineError

        raise PosePipelineError(
            "POSE_EXTRACTION_UNAVAILABLE",
            "Pose extraction dependencies are unavailable",
            retryable=True,
            status="503 Service Unavailable",
        )

    environ = _base_environ()
    environ["pose_pipeline_callable"] = unavailable_pose_pipeline
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "503 Service Unavailable"
    assert payload["error_code"] == "POSE_EXTRACTION_UNAVAILABLE"
    assert payload["retryable"] is True


def test_translate_sign_returns_422_for_invalid_inference_response() -> None:
    def invalid_infer(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float):
        raise CloudInferError(
            "INFERENCE_INVALID_RESPONSE",
            "Model provider returned empty prediction",
            retryable=False,
            status="422 Unprocessable Entity",
        )

    environ = _base_environ()
    environ["cloud_infer_callable"] = invalid_infer
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "422 Unprocessable Entity"
    assert payload["error_code"] == "INFERENCE_INVALID_RESPONSE"
    assert payload["retryable"] is False


def test_translate_sign_returns_502_for_malformed_upstream_payload() -> None:
    def malformed_infer(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float):
        raise CloudInferError(
            "INFERENCE_UPSTREAM_MALFORMED",
            "Model provider returned malformed JSON",
            retryable=True,
            status="502 Bad Gateway",
        )

    environ = _base_environ()
    environ["cloud_infer_callable"] = malformed_infer
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "INFERENCE_UPSTREAM_MALFORMED"
    assert payload["retryable"] is True


def test_translate_sign_rejects_missing_or_invalid_api_key(monkeypatch) -> None:
    monkeypatch.setenv("ASL_V1_API_KEYS", "valid-key")
    monkeypatch.delenv("ASL_V1_API_KEYS_NEXT", raising=False)

    environ = _base_environ()
    environ.pop("HTTP_AUTHORIZATION", None)
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))
    assert status == "401 Unauthorized"
    assert payload["error_code"] == "UNAUTHORIZED"
    assert payload["retryable"] is False

    environ = _base_environ()
    environ["HTTP_AUTHORIZATION"] = "Bearer wrong-key"
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))
    assert status == "401 Unauthorized"
    assert payload["error_code"] == "UNAUTHORIZED"


def test_translate_sign_accepts_next_rotation_key(monkeypatch) -> None:
    monkeypatch.setenv("ASL_V1_API_KEYS", "current-key")
    monkeypatch.setenv("ASL_V1_API_KEYS_NEXT", "next-key")

    environ = _base_environ()
    environ["HTTP_AUTHORIZATION"] = "Bearer next-key"
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "translation": "Hello",
        "confidence": 0.9,
        "latency_ms": 5,
        **_proof_fields(),
    }
    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert payload["prediction"] == "Hello"


def test_translate_sign_rate_limits_per_key(monkeypatch) -> None:
    monkeypatch.setenv("ASL_V1_API_KEYS", "rl-key")
    monkeypatch.setenv("ASL_V1_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("ASL_V1_RATE_LIMIT_WINDOW_SECONDS", "60")

    environ = _base_environ()
    environ["HTTP_AUTHORIZATION"] = "Bearer rl-key"
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "translation": "Hello",
        "confidence": 0.9,
        "latency_ms": 5,
        **_proof_fields(),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))
    assert status == "200 OK"
    assert payload["prediction"] == "Hello"

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))
    assert status == "429 Too Many Requests"
    assert payload["error_code"] == "RATE_LIMITED"
    assert payload["retryable"] is True
    assert payload["details"]["limit"] == 1
    assert payload["details"]["window_seconds"] == 60
    assert payload["details"]["retry_after_seconds"] >= 1


def test_translate_sign_accepts_x_api_key_header(monkeypatch) -> None:
    monkeypatch.setenv("ASL_V1_API_KEYS", "x-key")

    environ = _base_environ()
    environ.pop("HTTP_AUTHORIZATION", None)
    environ["HTTP_X_API_KEY"] = "x-key"
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "translation": "Hello",
        "confidence": 0.9,
        "latency_ms": 5,
        **_proof_fields(),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))
    assert status == "200 OK"
    assert payload["prediction"] == "Hello"


def test_translate_sign_maps_multiword_contract_fields() -> None:
    environ = _base_environ(request_id="rid-mw-1")
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "transcript_words": [
            {"word": "HELLO", "start_ms": 0, "end_ms": 420, "confidence": 0.91},
            {"word": "WORLD", "start_ms": 430, "end_ms": 900, "confidence": 0.89},
        ],
        "sequence_confidence": 0.9,
        "low_confidence": False,
        "confidence": 0.9,
        "latency_ms": 12,
        **_proof_fields(),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert payload["translation"] == "HELLO WORLD"
    assert payload["prediction"] == "HELLO WORLD"
    assert payload["status"] == "completed"
    assert payload["sequence_confidence"] == 0.9
    assert payload["low_confidence"] is False
    assert payload["transcript_words"] == [
        {"word": "HELLO", "start_ms": 0, "end_ms": 420, "confidence": 0.91},
        {"word": "WORLD", "start_ms": 430, "end_ms": 900, "confidence": 0.89},
    ]


def test_translate_sign_defaults_multiword_fields_when_provider_returns_translation_only() -> None:
    environ = _base_environ(request_id="rid-mw-2")
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "translation": "Hello",
        "confidence": 0.77,
        "latency_ms": 7,
        **_proof_fields(),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert payload["sequence_confidence"] == 0.77
    assert payload["low_confidence"] is False
    assert payload["transcript_words"] == [
        {"word": "Hello", "start_ms": 0, "end_ms": 0, "confidence": 0.77}
    ]


def test_translate_sign_returns_422_when_transcript_word_timestamp_invalid() -> None:
    environ = _base_environ(request_id="rid-mw-bad-ts")
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "transcript_words": [{"word": "HELLO", "start_ms": "bad", "end_ms": 100, "confidence": 0.9}],
        "confidence": 0.9,
        **_proof_fields(),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))
    assert status == "422 Unprocessable Entity"
    assert payload["error_code"] == "INFERENCE_INVALID_RESPONSE"
    assert payload["retryable"] is False


def test_translate_sign_parses_string_low_confidence_false() -> None:
    environ = _base_environ(request_id="rid-mw-low")
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "translation": "Hello",
        "confidence": 0.77,
        "low_confidence": "false",
        "latency_ms": 7,
        **_proof_fields(),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "200 OK"
    assert payload["low_confidence"] is False


def test_translate_sign_rejects_missing_upstream_proof_fields() -> None:
    environ = _base_environ(request_id="rid-proof-missing")
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "translation": "Hello",
        "confidence": 0.88,
        "latency_ms": 9,
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "INFERENCE_PROOF_MISSING"
    assert payload["retryable"] is False


def test_translate_sign_rejects_invalid_upstream_proof_field_types() -> None:
    environ = _base_environ(request_id="rid-proof-invalid")
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": kwargs["request_id"],
        "translation": "Hello",
        "confidence": 0.88,
        "latency_ms": 9,
        **_proof_fields(cloud_handoff="true"),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "INFERENCE_PROOF_INVALID"
    assert payload["retryable"] is False


def test_translate_sign_rejects_upstream_request_id_mismatch() -> None:
    environ = _base_environ(request_id="rid-proof-match")
    environ["cloud_infer_callable"] = lambda **kwargs: {
        "request_id": "different-request-id",
        "translation": "Hello",
        "confidence": 0.88,
        "latency_ms": 9,
        **_proof_fields(),
    }

    status, _headers, raw = translate_sign_wsgi_app(environ)
    payload = json.loads(raw.decode("utf-8"))

    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "INFERENCE_PROOF_INVALID"
    assert payload["retryable"] is False

