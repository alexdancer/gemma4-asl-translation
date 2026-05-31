from __future__ import annotations

import json
from pathlib import Path

from src.telemetry_slo import TelemetryEvent, append_event, summarize


def test_telemetry_append_and_summary(tmp_path: Path, monkeypatch) -> None:
    telemetry_file = tmp_path / "cloud_telemetry.jsonl"
    monkeypatch.setenv("ASL_TELEMETRY_PATH", str(telemetry_file))

    append_event(TelemetryEvent(request_id="a", latency_ms=120, outcome="success", confidence=0.91, model_tag="m1"))
    append_event(TelemetryEvent(request_id="b", latency_ms=900, outcome="timeout", confidence=0.0, model_tag="m1"))
    append_event(TelemetryEvent(request_id="c", latency_ms=300, outcome="success", confidence=0.87, model_tag="m1"))

    lines = telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["request_id"] == "a"
    assert "video" not in first
    assert "video_bytes" not in first

    report = summarize()
    assert report["count"] == 3
    assert report["success_count"] == 2
    assert report["success_rate"] == 0.6667
    assert report["p95_latency_ms"] == 300
