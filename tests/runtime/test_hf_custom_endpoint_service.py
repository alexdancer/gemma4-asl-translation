from __future__ import annotations

from fastapi.testclient import TestClient

from src.hf_custom_endpoint_service import RealBackend, RuntimeState, StubBackend, create_hf_custom_endpoint_app
from src.video_ingest import CanonicalVideoProfile, VideoProbeResult


def test_healthz_reports_ready_runtime_state() -> None:
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
            model_version="dev-2026-05-12",
            backend_name="stub",
            backend=StubBackend(),
        )
    )
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["ready"] is True
    assert payload["model_id"] == "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit"


def test_chat_completions_returns_minimal_openai_fields() -> None:
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
            model_version="dev-2026-05-12",
            backend_name="stub",
            backend=StubBackend(),
        )
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
                "messages": [
                    {"role": "system", "content": "You are an ASL translator."},
                    {"role": "user", "content": "Translate this clip"},
                ],
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload.keys()) == {"id", "object", "created", "model", "choices", "usage"}
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit"
    assert isinstance(payload["choices"], list) and payload["choices"]
    assert payload["choices"][0]["index"] == 0
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"]
    assert payload["usage"]["total_tokens"] == 0


def test_chat_completions_rejects_invalid_payload_with_deterministic_error() -> None:
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="m",
            model_version="v",
            backend_name="stub",
            backend=StubBackend(),
        )
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": []},
        )

    assert resp.status_code == 400
    payload = resp.json()
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_chat_completions_returns_503_when_model_not_ready() -> None:
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=False,
            model_id="m",
            model_version="v",
            backend_name="transformers",
            backend=None,
            load_error="failed to load model",
        )
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert resp.status_code == 503
    payload = resp.json()
    assert payload["error"]["code"] == "MODEL_NOT_READY"


def test_unknown_path_returns_deterministic_not_found_error() -> None:
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="m",
            model_version="v",
            backend_name="stub",
            backend=StubBackend(),
        )
    )
    with TestClient(app) as client:
        resp = client.get("/bad-path")

    assert resp.status_code == 404
    payload = resp.json()
    assert payload["error"]["code"] == "NOT_FOUND"


def test_real_backend_normalizes_json_user_prompt() -> None:
    backend = RealBackend(token="")
    messages = [
        {"role": "system", "content": "You are an ASL translator."},
        {
            "role": "user",
            "content": '{"instruction":"Translate ASL payload into short text.","input":{"filename":"IMG_1.MOV"}}',
        },
    ]
    normalized = backend._normalize_prompt(messages)
    assert len(normalized) == 2
    assert normalized[1]["role"] == "user"
    assert "Return only translated text" in normalized[1]["content"]
    assert "IMG_1.MOV" not in normalized[1]["content"]


def test_chat_completion_returns_502_when_backend_raises() -> None:
    class RaisingBackend:
        def generate(self, *, messages: list[dict[str, str]], model: str) -> str:
            _ = (messages, model)
            raise RuntimeError("boom")

    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="m",
            model_version="v",
            backend_name="real",
            backend=RaisingBackend(),
        )
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert resp.status_code == 502
    payload = resp.json()
    assert payload["error"]["code"] == "INFERENCE_FAILED"
    assert payload["error"]["message"] == "backend inference failed"


def test_translate_sign_route_runs_full_pipeline_via_runtime_backend(monkeypatch) -> None:
    class FakeBackend:
        def generate(self, *, messages: list[dict[str, str]], model: str) -> str:
            _ = model
            return "HELLO FROM B"

    class DummyFrameExtraction:
        frame_count = 2
        first_ts_ms = 0
        last_ts_ms = 33
        effective_fps = 30.0
        cadence = "uniform"

    class DummyPoseExtraction:
        frame_count = 2
        first_ts_ms = 0
        last_ts_ms = 33
        frames = [{"ts_ms": 0, "pose": [0.1, 0.2]}, {"ts_ms": 33, "pose": [0.3, 0.4]}]

    def fake_video_ingest(video_bytes: bytes, filename: str):
        probe = VideoProbeResult(duration_seconds=1.0, fps=30.0, width=1280, height=720, codec="h264", pixel_format="yuv420p", has_audio=False)
        return probe, probe, video_bytes, False, CanonicalVideoProfile()

    monkeypatch.setenv("ASL_V1_API_KEYS", "test-key")
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
            model_version="gemma4-e2b-q64-top50",
            backend_name="real",
            backend=FakeBackend(),
        )
    )

    @app.middleware("http")
    async def _inject_runtime_seams(request, call_next):
        request.state.injection = {
            "video_ingest_callable": fake_video_ingest,
            "frame_extractor_callable": lambda _video, probe=None: DummyFrameExtraction(),
            "pose_pipeline_callable": lambda request_id, frame_extraction: DummyPoseExtraction(),
        }
        return await call_next(request)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/translate-sign",
            files={"video": ("sample.mp4", b"fake-bytes", "video/mp4")},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["translation"] == "HELLO FROM B"
    assert payload["runtime_mode"] == "cactus_engine"
    assert payload["cloud_handoff"] is False
    assert payload["model_id"] == "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit"
    assert payload["model_version"] == "gemma4-e2b-q64-top50"
    assert payload["latency_ms"] >= 1


def test_real_backend_target_model_uses_local_model_path(monkeypatch) -> None:
    backend = RealBackend(token="")

    captured = {}

    def fake_local(prepared):
        captured["prepared"] = prepared
        return "HELLO WORLD"

    monkeypatch.setattr(backend, "_local_text_generation", fake_local)

    out = backend.generate(
        messages=[{"role": "user", "content": "hi"}],
        model="AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
    )

    assert out == "HELLO WORLD"
    assert captured["prepared"][0]["role"] == "user"


def test_real_backend_normalizes_translation_prefix() -> None:
    assert RealBackend._normalize_prediction_text("Translation: hello there") == "hello there"
    assert RealBackend._normalize_prediction_text("output: good morning") == "good morning"


def test_real_backend_target_model_fail_closed_no_client_fallback(monkeypatch) -> None:
    class NeverClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    _ = kwargs
                    raise AssertionError("chat fallback must not run")

            completions = _Completions()

        chat = _Chat()

        def chat_completion(self, **kwargs):
            _ = kwargs
            raise AssertionError("chat_completion fallback must not run")

        def text_generation(self, **kwargs):
            _ = kwargs
            raise AssertionError("text_generation fallback must not run")

    backend = RealBackend(token="")
    backend._client = NeverClient()  # type: ignore[attr-defined]

    def fail_local(prepared):
        _ = prepared
        raise RuntimeError("local model generate failed: boom")

    monkeypatch.setattr(backend, "_local_text_generation", fail_local)

    try:
        backend.generate(
            messages=[{"role": "user", "content": "hi"}],
            model="AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
        )
    except RuntimeError as exc:
        assert "local model generate failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_real_backend_local_text_generation_retries_when_first_decode_empty(monkeypatch) -> None:
    backend = RealBackend(token="")

    class FakeTensor:
        def __init__(self, values):
            self._vals = list(values)
            self.shape = (1, len(self._vals))

        def __getitem__(self, key):
            if isinstance(key, slice):
                return FakeTensor(self._vals[key])
            return self._vals[key]

        def to(self, _device):
            return self

    class FakeTokenizer:
        def __call__(self, _prompt, return_tensors="pt"):
            _ = return_tensors
            return {"input_ids": FakeTensor([11, 22, 33])}

        def decode(self, token_ids, skip_special_tokens=True):
            _ = skip_special_tokens
            vals = token_ids._vals if hasattr(token_ids, "_vals") else list(token_ids)
            if vals == []:
                return ""
            if vals == [99, 100]:
                return "thank you"
            return ""

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return [FakeTensor([11, 22, 33])]
            return [FakeTensor([11, 22, 33, 99, 100])]

    monkeypatch.setattr(backend, "_local_tokenizer", FakeTokenizer())
    fake_model = FakeModel()
    monkeypatch.setattr(backend, "_local_model", fake_model)
    monkeypatch.setattr(backend, "_local_device", "cpu")

    out = backend._local_text_generation(prepared=[{"role": "user", "content": "translate"}])

    assert out == "thank you"
    assert len(fake_model.calls) == 2
    assert "min_new_tokens" not in fake_model.calls[0]
    assert fake_model.calls[1]["min_new_tokens"] == 4


def test_real_backend_returns_structured_failure_when_all_paths_fail() -> None:
    class FakePrimary:
        def create(self, **kwargs):
            _ = kwargs
            raise StopIteration()

    class FakeChat:
        completions = FakePrimary()

    class FakeClient:
        chat = FakeChat()

        def chat_completion(self, **kwargs):
            _ = kwargs
            raise RuntimeError("chat failed")

        def text_generation(self, **kwargs):
            _ = kwargs
            raise RuntimeError("text failed")

    backend = RealBackend(token="")
    backend._client = FakeClient()  # type: ignore[attr-defined]

    try:
        backend.generate(messages=[{"role": "user", "content": "hi"}], model="m")
    except RuntimeError as exc:
        message = str(exc)
        assert "all inference methods failed" in message
        assert "chat.completions.create" in message
        assert "chat_completion" in message
        assert "text_generation" in message
    else:
        raise AssertionError("expected RuntimeError")


def test_load_runtime_allows_missing_token_for_local_real_backend(monkeypatch) -> None:
    monkeypatch.setenv("ASL_HF_ENDPOINT_BACKEND", "real")
    monkeypatch.setenv("ASL_HF_ENDPOINT_MODEL_ID", "m")
    monkeypatch.setenv("ASL_HF_ENDPOINT_MODEL_VERSION", "v")
    monkeypatch.delenv("ASL_HF_TOKEN", raising=False)

    app = create_hf_custom_endpoint_app()
    with TestClient(app) as client:
        resp = client.get("/healthz")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ready"] is True
    assert payload["backend"] == "real"


def test_translate_sign_returns_502_when_model_echoes_filename(monkeypatch) -> None:
    class FakeBackend:
        def generate(self, *, messages: list[dict[str, str]], model: str) -> str:
            _ = (messages, model)
            return "IMG_4790.MOV"

    class DummyFrameExtraction:
        frame_count = 2
        first_ts_ms = 0
        last_ts_ms = 33
        effective_fps = 30.0
        cadence = "uniform"

    class DummyPoseExtraction:
        frame_count = 2
        first_ts_ms = 0
        last_ts_ms = 33
        frames = [{"ts_ms": 0, "pose": [0.1, 0.2]}, {"ts_ms": 33, "pose": [0.3, 0.4]}]

    def fake_video_ingest(video_bytes: bytes, filename: str):
        probe = VideoProbeResult(duration_seconds=1.0, fps=30.0, width=1280, height=720, codec="h264", pixel_format="yuv420p", has_audio=False)
        return probe, probe, video_bytes, False, CanonicalVideoProfile()

    monkeypatch.setenv("ASL_V1_API_KEYS", "test-key")
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
            model_version="gemma4-e2b-q64-top50",
            backend_name="real",
            backend=FakeBackend(),
        )
    )

    @app.middleware("http")
    async def _inject_runtime_seams(request, call_next):
        request.state.injection = {
            "video_ingest_callable": fake_video_ingest,
            "frame_extractor_callable": lambda _video, probe=None: DummyFrameExtraction(),
            "pose_pipeline_callable": lambda request_id, frame_extraction: DummyPoseExtraction(),
        }
        return await call_next(request)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/translate-sign",
            files={"video": ("sample.mov", b"fake-bytes", "video/quicktime")},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 502
    payload = resp.json()
    assert payload["error_code"] == "INFERENCE_FAILED"
    assert "invalid model output" in payload["message"]


def test_translate_sign_returns_502_when_model_echoes_structured_payload(monkeypatch) -> None:
    class FakeBackend:
        def generate(self, *, messages: list[dict[str, str]], model: str) -> str:
            _ = (messages, model)
            return '{"pose_summary":{"frame_count":57,"first_ts_ms":0,"last_ts_ms":1867}}'

    class DummyFrameExtraction:
        frame_count = 2
        first_ts_ms = 0
        last_ts_ms = 33
        effective_fps = 30.0
        cadence = "uniform"

    class DummyPoseExtraction:
        frame_count = 2
        first_ts_ms = 0
        last_ts_ms = 33
        frames = [{"ts_ms": 0, "pose": [0.1, 0.2]}, {"ts_ms": 33, "pose": [0.3, 0.4]}]

    def fake_video_ingest(video_bytes: bytes, filename: str):
        probe = VideoProbeResult(duration_seconds=1.0, fps=30.0, width=1280, height=720, codec="h264", pixel_format="yuv420p", has_audio=False)
        return probe, probe, video_bytes, False, CanonicalVideoProfile()

    monkeypatch.setenv("ASL_V1_API_KEYS", "test-key")
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
            model_version="gemma4-e2b-q64-top50",
            backend_name="real",
            backend=FakeBackend(),
        )
    )

    @app.middleware("http")
    async def _inject_runtime_seams(request, call_next):
        request.state.injection = {
            "video_ingest_callable": fake_video_ingest,
            "frame_extractor_callable": lambda _video, probe=None: DummyFrameExtraction(),
            "pose_pipeline_callable": lambda request_id, frame_extraction: DummyPoseExtraction(),
        }
        return await call_next(request)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/translate-sign",
            files={"video": ("sample.mov", b"fake-bytes", "video/quicktime")},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 502
    payload = resp.json()
    assert payload["error_code"] == "INFERENCE_FAILED"
    assert "invalid model output" in payload["message"]


def test_translate_sign_salvages_translation_field_from_structured_output(monkeypatch) -> None:
    class FakeBackend:
        def generate(self, *, messages: list[dict[str, str]], model: str) -> str:
            _ = (messages, model)
            return '{"translation":"Hello there"}'

    class DummyFrameExtraction:
        frame_count = 2
        first_ts_ms = 0
        last_ts_ms = 33
        effective_fps = 30.0
        cadence = "uniform"

    class DummyPoseExtraction:
        frame_count = 2
        first_ts_ms = 0
        last_ts_ms = 33
        frames = [{"ts_ms": 0, "pose": [0.1, 0.2]}, {"ts_ms": 33, "pose": [0.3, 0.4]}]

    def fake_video_ingest(video_bytes: bytes, filename: str):
        probe = VideoProbeResult(duration_seconds=1.0, fps=30.0, width=1280, height=720, codec="h264", pixel_format="yuv420p", has_audio=False)
        return probe, probe, video_bytes, False, CanonicalVideoProfile()

    monkeypatch.setenv("ASL_V1_API_KEYS", "test-key")
    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
            model_version="gemma4-e2b-q64-top50",
            backend_name="real",
            backend=FakeBackend(),
        )
    )

    @app.middleware("http")
    async def _inject_runtime_seams(request, call_next):
        request.state.injection = {
            "video_ingest_callable": fake_video_ingest,
            "frame_extractor_callable": lambda _video, probe=None: DummyFrameExtraction(),
            "pose_pipeline_callable": lambda request_id, frame_extraction: DummyPoseExtraction(),
        }
        return await call_next(request)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/translate-sign",
            files={"video": ("sample.mov", b"fake-bytes", "video/quicktime")},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["translation"] == "Hello there"


def test_translate_sign_returns_502_when_wsgi_adapter_raises(monkeypatch) -> None:
    class FakeBackend:
        def generate(self, *, messages: list[dict[str, str]], model: str) -> str:
            _ = (messages, model)
            return "HELLO"

    monkeypatch.setenv("ASL_V1_API_KEYS", "test-key")

    def _raise_wsgi(_environ):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.hf_custom_endpoint_service.translate_sign_wsgi_app", _raise_wsgi)

    app = create_hf_custom_endpoint_app(
        runtime_loader=lambda: RuntimeState(
            ready=True,
            model_id="m",
            model_version="v",
            backend_name="real",
            backend=FakeBackend(),
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/v1/translate-sign",
            files={"video": ("sample.mp4", b"fake-bytes", "video/mp4")},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 502
    payload = resp.json()
    assert payload["error"]["code"] == "INFERENCE_FAILED"
    assert payload["error"]["message"] == "translate-sign pipeline failed"
