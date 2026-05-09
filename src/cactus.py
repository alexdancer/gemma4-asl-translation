"""Compatibility shim for Cactus Python SDK bindings.

Tries import from a normal `cactus` module first; if unavailable, falls back to
loading the SDK module file from a local Cactus repo checkout.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_symbols():
    # Preferred: standard module import (if published/importable as `cactus`).
    try:
        import cactus as mod  # type: ignore

        return mod.cactus_init, mod.cactus_complete, mod.cactus_destroy
    except Exception:
        pass

    # Fallback: local checkout path (defaulting to ~/cactus/python/src/cactus.py)
    candidate = os.environ.get("CACTUS_PY_SDK_FILE", str(Path.home() / "cactus/python/src/cactus.py"))
    sdk_file = Path(candidate).expanduser().resolve()
    if not sdk_file.exists():
        raise ImportError(
            "Could not import Cactus Python SDK. Set CACTUS_PY_SDK_FILE to cactus.py from the Cactus repo "
            f"(current candidate missing: {sdk_file})"
        )

    spec = importlib.util.spec_from_file_location("cactus_sdk_file", sdk_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Cactus SDK module spec from {sdk_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return module.cactus_init, module.cactus_complete, module.cactus_destroy
    except AttributeError as exc:
        raise ImportError(
            f"Cactus SDK module at {sdk_file} does not expose cactus_init/cactus_complete/cactus_destroy"
        ) from exc


cactus_init, cactus_complete, cactus_destroy = _load_symbols()

__all__ = ["cactus_init", "cactus_complete", "cactus_destroy"]
