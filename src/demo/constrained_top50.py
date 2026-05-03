"""Demo-safe constrained Top-50 q64 inference path.

This module is diagnostic/demo-scoped. It chooses from canonical manifest
labels at inference time and must remain separate from free-generation proof
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.demo.output_contract import DemoOutput, DemoOutputConfig, format_demo_output
from src.demo.readiness_artifacts import (
    DEMO_CLAIMS_CONSTRAINED_TOP50,
    DEMO_SCOPE_CONSTRAINED_TOP50,
    write_constrained_top50_readiness_artifact,
)
from src.evaluation.unsloth_asl import (
    ConstrainedGlossScorer,
    RealUnslothASLGlossPredictor,
    load_manifest_labels,
    load_q64_jsonl,
    score_q64_record_constrained,
)

DEMO_SCOPE = DEMO_SCOPE_CONSTRAINED_TOP50
DEMO_CLAIMS = DEMO_CLAIMS_CONSTRAINED_TOP50
INFERENCE_MODE = "demo_safe_constrained_top50"


@dataclass(frozen=True)
class ConstrainedTop50DemoConfig:
    """Configuration for one demo-safe constrained Top-50 q64 run."""

    checkpoint_path: Path | str
    records_path: Path | str
    manifest_path: Path | str
    record_id: str
    out_dir: Path | str
    top_k: int = 5
    output_config: DemoOutputConfig = DemoOutputConfig()

    def __post_init__(self) -> None:
        if not str(self.record_id).strip():
            raise ValueError("record_id is required.")
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")


@dataclass(frozen=True)
class ConstrainedTop50Candidate:
    """One constrained Top-50 candidate score exposed for demo explanation."""

    label: str
    score: float


@dataclass(frozen=True)
class ConstrainedTop50DemoResult:
    """Observable constrained Top-50 demo result for UI, logs, and artifacts."""

    model_path: str
    input_record_id: str
    selected_label: str
    best_label: str
    expected_gloss: str | None
    correct: bool | None
    top_candidates: tuple[ConstrainedTop50Candidate, ...]
    inference_mode: str
    scope: str
    claims: str
    constrained_metadata: dict[str, Any]
    output: DemoOutput
    artifact_path: Path


@dataclass(frozen=True)
class _ConstrainedDemoPrediction:
    ok: bool
    prediction: str | None
    confidence: float
    latency_ms: float = 0.0
    latency_target_ms: float = 800.0
    error: str | None = None


def run_constrained_top50_demo(
    config: ConstrainedTop50DemoConfig,
    *,
    scorer: ConstrainedGlossScorer | None = None,
) -> ConstrainedTop50DemoResult:
    """Run one q64 record through diagnostic constrained Top-50 scoring."""

    checkpoint_path = Path(config.checkpoint_path)
    records_path = Path(config.records_path)
    manifest_path = Path(config.manifest_path)
    out_dir = Path(config.out_dir)

    labels = load_manifest_labels(manifest_path)
    record = select_q64_record_by_sample_id(load_q64_jsonl(records_path), config.record_id)
    constrained_scorer = scorer or RealUnslothASLGlossPredictor(checkpoint_path)
    scoring = score_q64_record_constrained(record, constrained_scorer, labels)
    top_candidates = tuple(
        ConstrainedTop50Candidate(label=item.label, score=item.score)
        for item in scoring.ranked_scores[: config.top_k]
    )
    confidence_proxy = 1.0
    output = format_demo_output(
        _ConstrainedDemoPrediction(
            ok=True,
            prediction=scoring.best_label,
            confidence=confidence_proxy,
        ),
        config.output_config,
    )
    metadata = {
        "constrained": True,
        "scope": DEMO_SCOPE,
        "canonical_label_count": len(labels),
        "top_k": config.top_k,
        "selected_from_canonical_top50": True,
        "confidence_available": False,
        "confidence_note": "Candidate scores are ranking scores, not calibrated probabilities.",
        "activation_evidence": {
            "issue_21": {
                "useful_constrained_signal": True,
                "summary": "Constrained diagnostic signal improved top-1 on the referenced run.",
            },
            "issue_22": {
                "prompt_control_sufficient": True,
                "summary": "Prompt-control report showed strong free-generation control, so this is optional fallback.",
            },
        },
        "metric_boundary": "Diagnostic/demo-safe fallback only; not primary free-generation proof metric.",
    }
    candidate_payload = [
        {"label": candidate.label, "score": candidate.score}
        for candidate in top_candidates
    ]
    artifact_path = write_constrained_top50_readiness_artifact(
        out_dir=out_dir,
        model_path=checkpoint_path,
        input_record_id=config.record_id,
        inference_mode=scoring.mode,
        selected_label=scoring.best_label,
        expected_gloss=scoring.expected_gloss,
        correct=scoring.correct,
        visible_gloss=output.display_text,
        status=output.status,
        top_candidates=candidate_payload,
        constrained_metadata=metadata,
        confidence_proxy_used_for_display=confidence_proxy,
    )
    return ConstrainedTop50DemoResult(
        model_path=str(checkpoint_path),
        input_record_id=config.record_id,
        selected_label=scoring.best_label,
        best_label=scoring.best_label,
        expected_gloss=scoring.expected_gloss,
        correct=scoring.correct,
        top_candidates=top_candidates,
        inference_mode=scoring.mode,
        scope=DEMO_SCOPE,
        claims=DEMO_CLAIMS,
        constrained_metadata=metadata,
        output=output,
        artifact_path=artifact_path,
    )


def select_q64_record_by_sample_id(
    records: list[dict[str, Any]],
    record_id: str,
) -> Mapping[str, Any]:
    """Select a q64 JSONL record by sample_id embedded in the input text."""

    for record in records:
        input_text = str(record.get("input", ""))
        if any(line == f"sample_id={record_id}" for line in input_text.splitlines()):
            return record
    raise ValueError(f"record_id not found in q64 records: {record_id}")
