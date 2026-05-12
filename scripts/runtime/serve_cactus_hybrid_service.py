#!/usr/bin/env python3
"""Serve Cactus hybrid inference FastAPI service.

Usage:
  python3 scripts/runtime/serve_cactus_hybrid_service.py --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fastapi_apps import cactus_hybrid_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cactus hybrid inference service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    print(f"serving Cactus hybrid inference on http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(cactus_hybrid_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
