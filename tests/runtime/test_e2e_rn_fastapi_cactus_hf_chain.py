from __future__ import annotations

import io
import json

from fastapi import Request
from fastapi.testclient import TestClient

from src.cactus_hybrid_service import hybrid_wsgi_app
from src.fastapi_apps import create_cloud_translate_app
from src.video_ingest import CanonicalVideoProfile, VideoProbeResult


class _FrameExtractionStub:
    frame_count = 2
    first_ts_ms = 0
    last_ts_ms = 33
    effective_fps = 30.0
    cadence = 1


class _PoseExtractionStub:
    frame_count = 2
    first_ts_ms = 0
    last_ts_ms = 33
    frames = []


def test_reproducible_rn_fastapi_to_cactus_to_hf_happy_path(monkeypatch):
    monkeypatch.setenv("ASL_V1_API_KEYS", "dev-local-key-1")
    monkeypatch.setenv("ASL_CLOUD_API_KEY", "dev-cactus-secret")
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "dev-cactus-secret")
    monkeypatch.setenv("ASL_HF_OPENAI_BASE_URL", "https://dev-endpoint.example/v1")
    monkeypatch.setenv("ASL_HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("ASL_HF_ROUTE_MODE", "chat")
    monkeypatch.setenv("ASL_CACTUS_MODEL_VERSION", "dev-2026-05-12")

    class _HttpOk:
        def __init__(self, payload: dict):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data.decode("utf-8"))
        assert req.full_url.endswith("/chat/completions")
        assert body["model"] == "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit"
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        return _HttpOk({"choices": [{"message": {"content": "HELLO FRIEND"}}]})

    monkeypatch.setattr("src.cactus_hybrid_service.urlrequest.urlopen", fake_urlopen)

    def video_ingest_stub(video_bytes: bytes, _filename: str):
        probe = VideoProbeResult(
            duration_seconds=1.0,
            fps=30.0,
            width=640,
            height=480,
            codec="h264",
            pixel_format="yuv420p",
            has_audio=False,
        )
        return probe, probe, video_bytes, False, CanonicalVideoProfile()

    def frame_extractor_stub(_video_bytes: bytes, *, probe):
        _ = probe
        return _FrameExtractionStub()

    def pose_pipeline_stub(*, request_id: str, frame_extraction):
        _ = request_id
        _ = frame_extraction
        return _PoseExtractionStub()

    def cloud_infer_stub(**kwargs):
        request_id = kwargs["request_id"]
        filename = kwargs["filename"]
        model = "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit"
        pose_extraction = kwargs.get("pose_handoff")
        assert pose_extraction is not None
        payload = {
            "request_id": request_id,
            "model": model,
            "input": {
                "filename": filename,
                "pose_summary": {
                    "frame_count": pose_extraction.frame_count,
                    "first_ts_ms": pose_extraction.first_ts_ms,
                    "last_ts_ms": pose_extraction.last_ts_ms,
                },
            },
        }
        encoded = json.dumps(payload).encode("utf-8")
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/",
            "CONTENT_TYPE": "application/json",
            "CONTENT_LENGTH": str(len(encoded)),
            "wsgi.input": io.BytesIO(encoded),
            "wsgi.input_body": encoded,
            "HTTP_AUTHORIZATION": "Bearer dev-cactus-secret",
            "HTTP_X_REQUEST_ID": request_id,
        }
        status, _headers, raw = hybrid_wsgi_app(environ, timeout_seconds=20.0)
        assert status == "200 OK"
        return json.loads(raw.decode("utf-8"))

    app = create_cloud_translate_app()

    @app.middleware("http")
    async def inject_runtime_seams(request: Request, call_next):
        request.state.injection = {
            "video_ingest_callable": video_ingest_stub,
            "frame_extractor_callable": frame_extractor_stub,
            "pose_pipeline_callable": pose_pipeline_stub,
            "cloud_infer_callable": cloud_infer_stub,
        }
        return await call_next(request)

    with TestClient(app) as client:
        response = client.post(
            "/v1/translate-sign",
            headers={"x-api-key": "dev-local-key-1", "x-request-id": "rid-e2e-84"},
            files={"video": ("clip.mov", b"video-bytes", "video/quicktime")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "rid-e2e-84"
    assert payload["prediction"] == "HELLO FRIEND"
    assert payload["translation"] == "HELLO FRIEND"
    assert payload["runtime_mode"] == "cactus_engine"
    assert payload["cloud_handoff"] is True
    assert payload["model_id"] == "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit"
    assert payload["model_version"] == "dev-2026-05-12"
