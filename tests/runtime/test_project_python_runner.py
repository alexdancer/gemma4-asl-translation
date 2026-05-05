"""Behavior tests for the project Python command runner."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.project_python import select_project_python


def test_project_python_prefers_repo_virtualenv(tmp_path: Path) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    assert select_project_python(tmp_path) == venv_python


def test_project_python_falls_back_to_current_interpreter_without_virtualenv(tmp_path: Path) -> None:
    assert select_project_python(tmp_path) == Path(sys.executable)
