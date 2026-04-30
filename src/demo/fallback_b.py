"""Precomputed output replay fallback for emergency demo completion."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.demo.output_contract import DemoOutput, DemoOutputConfig, format_demo_output

ReplayMode = "replay"


@dataclass(frozen=True)
class ReplayRunConfig:
    """Emergency replay settings."""

    replay_path: Path | str
    output_config: DemoOutputConfig = DemoOutputConfig()
    sleep: bool = False


@dataclass(frozen=True)
class ReplayPrediction:
    """Prediction result shape consumed by the shared demo output contract."""

    ok: bool
    prediction: Optional[str]
    confidence: float
    latency_ms: float
    latency_target_ms: float = 800.0
    error: Optional[str] = None


@dataclass(frozen=True)
class ReplayStep:
    """One scripted replay output."""

    at_ms: int
    output: DemoOutput


@dataclass(frozen=True)
class ReplayRunResult:
    """Observable replay result for UI/logs and demo scripts."""

    mode: str
    scenario: str
    replay_path: str
    observation: str
    elapsed_ms: float
    steps: tuple[ReplayStep, ...]


def run_precomputed_replay(config: ReplayRunConfig) -> ReplayRunResult:
    """Load and play a precomputed demo output script."""

    started = time.perf_counter()
    replay_path = Path(config.replay_path)
    payload = _load_replay_payload(replay_path)
    scenario = str(payload.get("scenario") or replay_path.stem)
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("replay script must contain a non-empty steps list.")

    steps: list[ReplayStep] = []
    previous_at_ms = 0
    for step_payload in steps_payload:
        step = _parse_step(step_payload, config.output_config)
        if step.at_ms < previous_at_ms:
            raise ValueError("replay step at_ms values must be monotonic.")
        if config.sleep:
            time.sleep((step.at_ms - previous_at_ms) / 1000.0)
        previous_at_ms = step.at_ms
        steps.append(step)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return ReplayRunResult(
        mode=ReplayMode,
        scenario=scenario,
        replay_path=str(replay_path),
        observation=f"mode={ReplayMode} scenario={scenario} replay_path={replay_path}",
        elapsed_ms=elapsed_ms,
        steps=tuple(steps),
    )


def _load_replay_payload(replay_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"replay script not found: {replay_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"replay script is not valid JSON: {replay_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("replay script must be a JSON object.")
    return payload


def _parse_step(payload: Any, output_config: DemoOutputConfig) -> ReplayStep:
    if not isinstance(payload, dict):
        raise ValueError("each replay step must be a JSON object.")
    prediction = ReplayPrediction(
        ok=bool(payload.get("ok", True)),
        prediction=payload.get("prediction"),
        confidence=float(payload.get("confidence", 1.0)),
        latency_ms=float(payload.get("latency_ms", 0.0)),
        latency_target_ms=float(payload.get("latency_target_ms", 800.0)),
        error=payload.get("error"),
    )
    return ReplayStep(
        at_ms=int(payload.get("at_ms", 0)),
        output=format_demo_output(prediction, output_config),
    )
