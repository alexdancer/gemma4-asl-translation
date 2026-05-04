"""Python video -> q64 -> prompt-control smoke path.

This module is intentionally scoped to one known Top-50 sample. It validates
the Python orchestration path from a prerecorded video through q64 extraction
and prompt-control free generation, but it is not a production ASL recognizer.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from src.data.video_pose_q64_smoke import (
    VideoPoseExtractor,
    VideoPoseQ64SmokeConfig,
    run_video_pose_q64_smoke,
)
from src.evaluation.unsloth_asl import (
    GlossPredictor,
    RealUnslothASLGlossPredictor,
    build_prompt_control_prompt,
    infer_q64_record,
    load_manifest_labels,
    normalize_gloss,
)


SMOKE_SCOPE = "python_video_prompt_control_smoke"
SMOKE_CLAIMS = (
    "Python smoke path for video -> q64 -> prompt-control free generation on "
    "one known video sample only; not production ASL recognition."
)
PROMPT_CONTROL_CONTRACT = "q64_prompt_control_free_generation"


@dataclass(frozen=True)
class PythonVideoPromptControlSmokeConfig:
    """Configuration for one known-video prompt-control smoke run."""

    video_path: Path | str
    checkpoint_path: Path | str
    sample_id: str
    expected_gloss: str
    manifest_path: Path | str
    records_path: Path | str
    out_dir: Path | str
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip():
            raise ValueError("sample_id is required.")
        if not str(self.expected_gloss).strip():
            raise ValueError("expected_gloss is required.")


@dataclass(frozen=True)
class PythonVideoPromptControlSmokeResult:
    """Observable result and readiness artifact path for the smoke run."""

    scope: str
    model_path: str
    input_record_id: str
    video_path: str
    q64_jsonl_path: Path
    artifact_path: Path
    raw_model_output: str
    normalized_gloss: str | None
    expected_gloss: str
    valid_label: bool
    correct: bool
    inference_mode: str


def run_python_video_prompt_control_smoke(
    config: PythonVideoPromptControlSmokeConfig,
    *,
    extractor_factory: Callable[[], VideoPoseExtractor] | None = None,
    predictor: GlossPredictor | None = None,
) -> PythonVideoPromptControlSmokeResult:
    """Run one video through q64 extraction and prompt-control q64 inference."""

    sample_id = str(config.sample_id).strip()
    expected_gloss = normalize_gloss(str(config.expected_gloss))
    checkpoint_path = Path(config.checkpoint_path)
    out_dir = Path(config.out_dir)
    q64_out_dir = out_dir / "video_pose_q64"

    total_start = time.perf_counter()
    q64_start = time.perf_counter()
    q64_result = run_video_pose_q64_smoke(
        VideoPoseQ64SmokeConfig(
            video_path=config.video_path,
            sample_id=sample_id,
            expected_gloss=expected_gloss,
            manifest_path=config.manifest_path,
            records_path=config.records_path,
            out_dir=q64_out_dir,
            max_frames=config.max_frames,
        ),
        extractor_factory=extractor_factory,
    )
    video_to_q64_ms = _elapsed_ms(q64_start)

    labels = load_manifest_labels(config.manifest_path)
    checkpoint_predictor = predictor or RealUnslothASLGlossPredictor(
        checkpoint_path,
        prompt_builder=lambda record: build_prompt_control_prompt(record, labels),
    )

    inference_start = time.perf_counter()
    inference = infer_q64_record(q64_result.record, checkpoint_predictor, labels)
    prompt_control_inference_ms = _elapsed_ms(inference_start)
    correct = bool(
        inference.valid_label
        and inference.predicted_gloss is not None
        and inference.predicted_gloss == expected_gloss
    )
    total_ms = _elapsed_ms(total_start)

    artifact_path = _write_readiness_artifact(
        out_dir=out_dir,
        model_path=checkpoint_path,
        input_record_id=sample_id,
        video_path=Path(config.video_path),
        q64_jsonl_path=q64_result.jsonl_path,
        raw_model_output=inference.raw_model_output,
        normalized_gloss=inference.predicted_gloss,
        expected_gloss=expected_gloss,
        valid_label=inference.valid_label,
        correct=correct,
        inference_mode=inference.mode,
        labels=labels,
        timing_ms={
            "total": total_ms,
            "video_to_q64": video_to_q64_ms,
            "prompt_control_inference": prompt_control_inference_ms,
        },
        diagnostics={
            "video_pose_q64_scope": q64_result.scope,
            "q64_encoding": q64_result.encoding,
            "q64_frames": q64_result.frames,
            "q64_features_per_frame": q64_result.features_per_frame,
            "video_pose_q64_report_path": str(q64_result.report_path),
        },
    )

    return PythonVideoPromptControlSmokeResult(
        scope=SMOKE_SCOPE,
        model_path=str(checkpoint_path),
        input_record_id=sample_id,
        video_path=str(config.video_path),
        q64_jsonl_path=q64_result.jsonl_path,
        artifact_path=artifact_path,
        raw_model_output=inference.raw_model_output,
        normalized_gloss=inference.predicted_gloss,
        expected_gloss=expected_gloss,
        valid_label=inference.valid_label,
        correct=correct,
        inference_mode=inference.mode,
    )


def result_to_dict(result: PythonVideoPromptControlSmokeResult) -> dict[str, object]:
    """Serialize a smoke result for CLI output."""

    payload = asdict(result)
    payload["q64_jsonl_path"] = str(result.q64_jsonl_path)
    payload["artifact_path"] = str(result.artifact_path)
    return payload


def _write_readiness_artifact(
    *,
    out_dir: Path,
    model_path: Path,
    input_record_id: str,
    video_path: Path,
    q64_jsonl_path: Path,
    raw_model_output: str,
    normalized_gloss: str | None,
    expected_gloss: str,
    valid_label: bool,
    correct: bool,
    inference_mode: str,
    labels: tuple[str, ...],
    timing_ms: dict[str, float],
    diagnostics: dict[str, object],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "python_video_prompt_control_smoke_readiness.json"
    payload = {
        "scope": SMOKE_SCOPE,
        "claims": SMOKE_CLAIMS,
        "model_path": str(model_path),
        "input_record_id": input_record_id,
        "video_path": str(video_path),
        "q64_jsonl_path": str(q64_jsonl_path),
        "inference_mode": inference_mode,
        "raw_model_output": raw_model_output,
        "normalized_gloss": normalized_gloss,
        "expected_gloss": expected_gloss,
        "valid_label": valid_label,
        "correct": correct,
        "prompt_control": {
            "contract": PROMPT_CONTROL_CONTRACT,
            "generation": "free_generation",
            "valid_labels": list(labels),
        },
        "timing_ms": {key: round(value, 3) for key, value in timing_ms.items()},
        "diagnostics": diagnostics,
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
