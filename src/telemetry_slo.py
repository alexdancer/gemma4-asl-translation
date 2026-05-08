from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TelemetryEvent:
    request_id: str
    latency_ms: int
    outcome: str
    confidence: float
    model_tag: str

    def to_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "latency_ms": int(self.latency_ms),
            "outcome": self.outcome,
            "confidence": float(self.confidence),
            "model_tag": self.model_tag,
        }


def telemetry_path() -> Path:
    raw = os.environ.get("ASL_TELEMETRY_PATH", "evaluation/results/runtime/cloud_telemetry.jsonl")
    return Path(raw)


def append_event(event: TelemetryEvent) -> Path:
    path = telemetry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event.to_record()) + "\n")
    return path


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, int(len(ordered) * 0.95) - 1)
    return int(ordered[idx])


def summarize(path: Path | None = None) -> dict[str, Any]:
    target = path or telemetry_path()
    if not target.exists():
        return {
            "count": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "p95_latency_ms": 0,
            "telemetry_path": str(target),
        }

    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    latencies = [int(r.get("latency_ms", 0)) for r in records]
    success_count = sum(1 for r in records if r.get("outcome") == "success")
    total = len(records)
    success_rate = (success_count / total) if total else 0.0

    return {
        "count": total,
        "success_count": success_count,
        "success_rate": round(success_rate, 4),
        "p95_latency_ms": _p95(latencies),
        "telemetry_path": str(target),
    }
