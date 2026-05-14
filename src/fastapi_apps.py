"""FastAPI adapters that preserve existing WSGI business logic contracts."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request, Response

from src.cactus_hybrid_service import hybrid_wsgi_app
from src.cloud_translate_api import translate_sign_wsgi_app


_HOP_BY_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}


def _status_code_from_wsgi(status: str) -> int:
    return int(status.split(" ", 1)[0])


async def _build_wsgi_environ(request: Request) -> dict[str, Any]:
    body = await request.body()
    environ: dict[str, Any] = {
        "REQUEST_METHOD": request.method,
        "PATH_INFO": request.url.path,
        "CONTENT_TYPE": request.headers.get("content-type", ""),
        "wsgi.input_body": body,
    }

    for key, value in request.headers.items():
        header_key = f"HTTP_{key.upper().replace('-', '_')}"
        environ[header_key] = value

    if request.state is not None and getattr(request.state, "injection", None):
        environ.update(request.state.injection)
    return environ


def _build_fastapi_response(status: str, headers: list[tuple[str, str]], body: bytes) -> Response:
    response = Response(content=body, status_code=_status_code_from_wsgi(status))
    for key, value in headers:
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        response.headers.append(key, value)
    return response


def _resolve_cloud_timeout_seconds() -> float:
    raw = os.environ.get("ASL_CLOUD_TIMEOUT_SECONDS", "12").strip()
    try:
        value = float(raw)
    except ValueError:
        return 12.0
    return value if value > 0 else 12.0


def create_cloud_translate_app() -> FastAPI:
    app = FastAPI(title="ASL Cloud Translate API", version="1")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def route_all(request: Request, path: str) -> Response:
        _ = path
        environ = await _build_wsgi_environ(request)
        status, headers, raw = translate_sign_wsgi_app(environ, timeout_seconds=_resolve_cloud_timeout_seconds())
        return _build_fastapi_response(status, headers, raw)

    return app


def create_cactus_hybrid_app() -> FastAPI:
    app = FastAPI(title="ASL Cactus Hybrid Inference Service", version="1")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def route_all(request: Request, path: str) -> Response:
        _ = path
        environ = await _build_wsgi_environ(request)
        status, headers, raw = hybrid_wsgi_app(environ)
        return _build_fastapi_response(status, headers, raw)

    return app


cloud_translate_app = create_cloud_translate_app()
cactus_hybrid_app = create_cactus_hybrid_app()
