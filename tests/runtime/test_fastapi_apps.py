import json

from fastapi.testclient import TestClient

from src.fastapi_apps import create_cactus_hybrid_app, create_cloud_translate_app


def _multipart_body(boundary: str = "testboundary") -> bytes:
    payload = b"VIDEO-BYTES"
    return (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"video\"; filename=\"clip.mp4\"\r\n"
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")


def test_cloud_translate_fastapi_adapter_routes_to_wsgi_logic(monkeypatch):
    app = create_cloud_translate_app()

    def fake_translate(environ):
        assert environ["REQUEST_METHOD"] == "POST"
        assert environ["PATH_INFO"] == "/v1/translate-sign"
        assert environ["HTTP_X_API_KEY"] == "k1"
        assert environ["CONTENT_TYPE"].startswith("multipart/form-data")
        assert environ["wsgi.input_body"]
        return (
            "200 OK",
            [("Content-Type", "application/json")],
            json.dumps({"status": "ok", "request_id": "r1"}).encode("utf-8"),
        )

    monkeypatch.setattr("src.fastapi_apps.translate_sign_wsgi_app", fake_translate)

    client = TestClient(app)
    boundary = "testboundary"
    body = _multipart_body(boundary)
    resp = client.post(
        "/v1/translate-sign",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}", "x-api-key": "k1"},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_cactus_hybrid_fastapi_adapter_routes_to_wsgi_logic(monkeypatch):
    app = create_cactus_hybrid_app()

    def fake_hybrid(environ):
        assert environ["REQUEST_METHOD"] == "POST"
        assert environ["PATH_INFO"] == "/"
        assert environ["HTTP_AUTHORIZATION"] == "Bearer svc"
        payload = json.loads(environ["wsgi.input_body"].decode("utf-8"))
        assert payload["request_id"] == "req-1"
        return (
            "200 OK",
            [("Content-Type", "application/json")],
            json.dumps({"request_id": "req-1", "status": "ok"}).encode("utf-8"),
        )

    monkeypatch.setattr("src.fastapi_apps.hybrid_wsgi_app", fake_hybrid)

    client = TestClient(app)
    resp = client.post(
        "/",
        json={"request_id": "req-1", "model": "m", "input": {"filename": "f.mp4", "pose_summary": "x"}},
        headers={"authorization": "Bearer svc"},
    )

    assert resp.status_code == 200
    assert resp.json()["request_id"] == "req-1"


def test_adapter_preserves_wsgi_status_code_and_headers(monkeypatch):
    app = create_cloud_translate_app()

    def fake_translate(_environ):
        return (
            "429 Too Many Requests",
            [("Content-Type", "application/json"), ("Retry-After", "9")],
            json.dumps({"error_code": "RATE_LIMITED"}).encode("utf-8"),
        )

    monkeypatch.setattr("src.fastapi_apps.translate_sign_wsgi_app", fake_translate)
    client = TestClient(app)
    resp = client.get("/any")

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "9"
    assert resp.json()["error_code"] == "RATE_LIMITED"
