#!/usr/bin/env python3
"""Serve Cactus hybrid inference WSGI service.

Usage:
  python3 scripts/runtime/serve_cactus_hybrid_service.py --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from wsgiref.simple_server import make_server

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cactus_hybrid_service import hybrid_wsgi_app


def wsgi_app(environ, start_response):
    body = environ["wsgi.input"].read(int(environ.get("CONTENT_LENGTH") or 0))
    environ["wsgi.input_body"] = body
    status, headers, raw = hybrid_wsgi_app(environ)
    start_response(status, headers)
    return [raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cactus hybrid inference service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    print(f"serving Cactus hybrid inference on http://{args.host}:{args.port}/", flush=True)
    make_server(args.host, args.port, wsgi_app).serve_forever()


if __name__ == "__main__":
    main()
