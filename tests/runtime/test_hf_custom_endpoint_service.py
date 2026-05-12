from __future__ import annotations

from fastapi.testclient import TestClient

from src.hf_custom_endpoint_service import RuntimeState, StubBackend, create_hf_custom_endpoint_app


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
    assert set(payload.keys()) == {"id", "object", "created", "model", "choices"}
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit"
    assert isinstance(payload["choices"], list) and payload["choices"]
    assert payload["choices"][0]["message"]["content"]


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
