from __future__ import annotations

import io
import json

from urllib import error as urlerror

from src.cactus_hybrid_service import hybrid_wsgi_app, _run_hybrid


def _environ(*, body: dict, auth: str = "Bearer svc-key") -> dict:
    encoded = json.dumps(body).encode("utf-8")
    return {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/",
        "CONTENT_LENGTH": str(len(encoded)),
        "wsgi.input": io.BytesIO(encoded),
        "wsgi.input_body": encoded,
        "HTTP_AUTHORIZATION": auth,
        "HTTP_X_REQUEST_ID": "req-123",
    }


def _decode(response):
    status, _headers, raw = response
    return status, json.loads(raw.decode("utf-8"))


def test_hybrid_service_returns_proof_fields(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")

    def fake_infer(*, payload, timeout_seconds):
        assert timeout_seconds == 20.0
        assert payload["model"] == "cactus-asl-v2"
        return {
            "request_id": payload["request_id"],
            "prediction": "HELLO",
            "confidence": 0.91,
            "runtime_mode": "cactus_engine",
            "cloud_handoff": True,
            "model_id": payload["model"],
            "model_version": "2026-05-11",
        }

    env = _environ(body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}})
    env["hybrid_infer_callable"] = fake_infer

    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "200 OK"
    assert payload["runtime_mode"] == "cactus_engine"
    assert payload["cloud_handoff"] is True
    assert payload["model_id"] == "cactus-asl-v2"
    assert payload["model_version"] == "2026-05-11"


def test_hybrid_service_fails_closed_when_handoff_fails(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")

    def explode(*, payload, timeout_seconds):
        raise RuntimeError("hf unavailable")

    env = _environ(body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}})
    env["hybrid_infer_callable"] = explode

    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "503 Service Unavailable"
    assert payload["error_code"] == "CLOUD_HANDOFF_FAILED"


def test_hybrid_service_rejects_missing_proof_fields(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")

    def incomplete(*, payload, timeout_seconds):
        return {
            "request_id": payload["request_id"],
            "prediction": "HELLO",
            "confidence": 0.7,
        }

    env = _environ(body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}})
    env["hybrid_infer_callable"] = incomplete

    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "PROOF_FIELDS_MISSING"


def test_hybrid_service_rejects_invalid_auth(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")
    env = _environ(
        body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}},
        auth="Bearer wrong-key",
    )
    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "401 Unauthorized"
    assert payload["error_code"] == "UNAUTHORIZED"


def test_hybrid_service_rejects_wrong_path(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")
    env = _environ(body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}})
    env["PATH_INFO"] = "/bad-path"
    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "404 Not Found"
    assert payload["error_code"] == "NOT_FOUND"


def test_hybrid_service_rejects_malformed_json(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")
    bad = b"{"
    env = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/",
        "CONTENT_LENGTH": str(len(bad)),
        "wsgi.input": io.BytesIO(bad),
        "wsgi.input_body": bad,
        "HTTP_AUTHORIZATION": "Bearer svc-key",
        "HTTP_X_REQUEST_ID": "req-123",
    }
    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "400 Bad Request"
    assert payload["error_code"] == "INVALID_JSON"


def test_hybrid_service_rejects_invalid_proof_field_types(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")

    def invalid_types(*, payload, timeout_seconds):
        return {
            "request_id": payload["request_id"],
            "prediction": "HELLO",
            "confidence": "abc",
            "runtime_mode": "cactus_engine",
            "cloud_handoff": "false",
            "model_id": payload["model"],
            "model_version": "2026-05-11",
        }

    env = _environ(body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}})
    env["hybrid_infer_callable"] = invalid_types

    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "PROOF_FIELDS_INVALID"


def test_hybrid_service_reads_wsgi_input_when_input_body_missing(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")

    body = {
        "request_id": "req-123",
        "model": "cactus-asl-v2",
        "input": {"filename": "a.mp4"},
    }
    encoded = json.dumps(body).encode("utf-8")
    env = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/",
        "CONTENT_LENGTH": str(len(encoded)),
        "wsgi.input": io.BytesIO(encoded),
        "HTTP_AUTHORIZATION": "Bearer svc-key",
        "HTTP_X_REQUEST_ID": "req-123",
    }

    def fake_infer(*, payload, timeout_seconds):
        return {
            "request_id": payload["request_id"],
            "prediction": "HELLO",
            "confidence": 0.91,
            "runtime_mode": "cactus_engine",
            "cloud_handoff": True,
            "model_id": payload["model"],
            "model_version": "2026-05-11",
        }

    env["hybrid_infer_callable"] = fake_infer
    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "200 OK"
    assert payload["prediction"] == "HELLO"


def test_hybrid_service_rejects_request_id_mismatch(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")

    def mismatch(*, payload, timeout_seconds):
        return {
            "request_id": "different-id",
            "prediction": "HELLO",
            "confidence": 0.5,
            "runtime_mode": "cactus_engine",
            "cloud_handoff": True,
            "model_id": payload["model"],
            "model_version": "2026-05-11",
        }

    env = _environ(body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}})
    env["hybrid_infer_callable"] = mismatch
    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "PROOF_FIELDS_INVALID"


def test_hybrid_service_rejects_empty_runtime_mode(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")

    def invalid_runtime(*, payload, timeout_seconds):
        return {
            "request_id": payload["request_id"],
            "prediction": "HELLO",
            "confidence": 0.5,
            "runtime_mode": "",
            "cloud_handoff": True,
            "model_id": payload["model"],
            "model_version": "2026-05-11",
        }

    env = _environ(body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}})
    env["hybrid_infer_callable"] = invalid_runtime
    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "PROOF_FIELDS_INVALID"


def test_hybrid_service_rejects_confidence_out_of_range(monkeypatch):
    monkeypatch.setenv("ASL_CACTUS_SERVICE_API_KEY", "svc-key")

    def invalid_confidence(*, payload, timeout_seconds):
        return {
            "request_id": payload["request_id"],
            "prediction": "HELLO",
            "confidence": 1.2,
            "runtime_mode": "cactus_engine",
            "cloud_handoff": True,
            "model_id": payload["model"],
            "model_version": "2026-05-11",
        }

    env = _environ(body={"request_id": "req-123", "model": "cactus-asl-v2", "input": {"filename": "a.mp4"}})
    env["hybrid_infer_callable"] = invalid_confidence
    status, payload = _decode(hybrid_wsgi_app(env))
    assert status == "502 Bad Gateway"
    assert payload["error_code"] == "PROOF_FIELDS_INVALID"


def test_run_hybrid_auto_falls_back_to_completion_when_chat_model_not_supported(monkeypatch):
    monkeypatch.setenv("ASL_HF_ROUTE_MODE", "auto")

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
        if req.full_url.endswith("/chat/completions"):
            body = json.dumps(
                {
                    "error": {
                        "message": "The requested model is not a chat model.",
                        "code": "model_not_supported",
                    }
                }
            ).encode("utf-8")
            raise urlerror.HTTPError(req.full_url, 400, "Bad Request", hdrs=None, fp=io.BytesIO(body))
        if req.full_url.endswith("/completions"):
            return _HttpOk({"choices": [{"text": "HELLO"}]})
        raise AssertionError("unexpected url")

    monkeypatch.setenv("ASL_HF_OPENAI_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("ASL_HF_TOKEN", "hf_test")
    monkeypatch.setattr("src.cactus_hybrid_service.urlrequest.urlopen", fake_urlopen)

    result = _run_hybrid(
        payload={"request_id": "req-1", "model": "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit", "input": {"filename": "a.mp4"}},
        timeout_seconds=20.0,
    )
    assert result["prediction"] == "HELLO"
    assert result["cloud_handoff"] is True


def test_run_hybrid_completion_mode_uses_completions(monkeypatch):
    monkeypatch.setenv("ASL_HF_ROUTE_MODE", "completion")

    class _HttpOk:
        def __init__(self, payload: dict):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        return _HttpOk({"choices": [{"text": "WORLD"}]})

    monkeypatch.setenv("ASL_HF_OPENAI_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("ASL_HF_TOKEN", "hf_test")
    monkeypatch.setattr("src.cactus_hybrid_service.urlrequest.urlopen", fake_urlopen)

    result = _run_hybrid(
        payload={"request_id": "req-2", "model": "m", "input": {"filename": "a.mp4"}},
        timeout_seconds=20.0,
    )
    assert calls and calls[0].endswith("/completions")
    assert result["prediction"] == "WORLD"
