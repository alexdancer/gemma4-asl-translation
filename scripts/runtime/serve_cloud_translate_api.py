#!/usr/bin/env python3
"""Run the ASL cloud translate FastAPI app locally.

Usage:
  python scripts/runtime/serve_cloud_translate_api.py
  python scripts/runtime/serve_cloud_translate_api.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fastapi_apps import cloud_translate_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ASL cloud translate API locally")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    print(f"serving /v1/translate-sign on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(cloud_translate_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
