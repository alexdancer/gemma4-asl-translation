#!/usr/bin/env python3
"""Serve HF custom endpoint service.

Usage:
  python3 scripts/runtime/serve_hf_custom_endpoint_service.py --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hf_custom_endpoint_service import hf_custom_endpoint_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve HF custom endpoint service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"serving HF custom endpoint on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(hf_custom_endpoint_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
