#!/usr/bin/env python3
"""Run Python modules with the repository virtualenv when available."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def select_project_python(repo_root: Path = REPO_ROOT) -> Path:
    """Return the interpreter that should run project Python commands."""

    venv_python = Path(repo_root) / "venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-python", action="store_true", help="Print the selected interpreter and exit.")
    parser.add_argument("module", nargs="?", help="Python module to execute with -m.")
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    python = select_project_python()
    if args.print_python:
        print(python)
        return 0
    if not args.module:
        parser.error("module is required unless --print-python is used")

    os.execv(str(python), [str(python), "-m", args.module, *args.module_args])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
