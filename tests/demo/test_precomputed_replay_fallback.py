"""Behavior tests for fallback B precomputed output replay mode."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.demo.run_precomputed_replay import main as run_replay_main
from src.demo.fallback_b import ReplayRunConfig, run_precomputed_replay


def _write_replay(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "scenario": "judge-demo-thanks",
                "steps": [
                    {
                        "at_ms": 0,
                        "prediction": "hello",
                        "confidence": 0.91,
                        "latency_ms": 0.0,
                    },
                    {
                        "at_ms": 1200,
                        "prediction": "thanks",
                        "confidence": 0.94,
                        "latency_ms": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_precomputed_replay_runs_without_live_inference_dependencies(tmp_path: Path) -> None:
    replay_path = tmp_path / "demo_replay.json"
    _write_replay(replay_path)

    result = run_precomputed_replay(ReplayRunConfig(replay_path=replay_path))

    assert result.mode == "replay"
    assert result.scenario == "judge-demo-thanks"
    assert result.elapsed_ms < 10_000
    assert [step.output.display_text for step in result.steps] == ["hello", "thanks"]
    assert all(step.output.status == "ok" for step in result.steps)
    assert "mode=replay" in result.observation


def test_precomputed_replay_demo_script_executes_end_to_end(tmp_path: Path, capsys) -> None:
    replay_path = tmp_path / "demo_replay.json"
    _write_replay(replay_path)

    exit_code = run_replay_main(["--replay-path", str(replay_path), "--no-sleep"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"mode": "replay"' in captured.out
    assert '"scenario": "judge-demo-thanks"' in captured.out
    assert '"display_text": "thanks"' in captured.out
