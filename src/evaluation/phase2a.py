"""Phase 2A validation report for the ASL 943-pose decision gate."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

Phase2ADecision = Literal["proceed_to_phase2b", "adjust_strategy"]


@dataclass(frozen=True)
class Phase2AConfig:
    """Thresholds for the 943-pose validation gate."""

    min_macro_f1_for_phase2b: float = 0.75
    weak_class_f1_threshold: float = 0.65

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_macro_f1_for_phase2b <= 1.0:
            raise ValueError("min_macro_f1_for_phase2b must be between 0.0 and 1.0.")
        if not 0.0 <= self.weak_class_f1_threshold <= 1.0:
            raise ValueError("weak_class_f1_threshold must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall, F1, and support for one gloss."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate test-set metrics."""

    macro_f1: float
    accuracy: float
    sample_count: int
    class_count: int


@dataclass(frozen=True)
class SplitSummary:
    """Split composition and signer leakage check."""

    train_samples: int
    val_samples: int
    test_samples: int
    train_signers: int
    val_signers: int
    test_signers: int
    train_val_signer_overlap: tuple[str, ...]
    train_test_signer_overlap: tuple[str, ...]
    val_test_signer_overlap: tuple[str, ...]
    has_signer_leakage: bool


@dataclass(frozen=True)
class LossCurve:
    """Training and validation loss history."""

    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]


@dataclass(frozen=True)
class Phase2AReport:
    """Decision artifact for proceeding from Phase 2A to Phase 2B."""

    metric_summary: MetricSummary
    per_class: dict[str, ClassMetrics]
    weak_classes: tuple[str, ...]
    split_summary: SplitSummary
    loss_curve: LossCurve
    decision: Phase2ADecision
    blockers: tuple[str, ...]
    metadata: Mapping[str, str] | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def build_phase2a_report(
    *,
    truth: Sequence[str],
    predictions: Sequence[str],
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    test_split: pd.DataFrame,
    training_history: Mapping[str, Sequence[float]],
    config: Phase2AConfig | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Phase2AReport:
    """Build the Phase 2A validation decision report."""

    config = config or Phase2AConfig()
    per_class = _per_class_metrics(truth, predictions)
    macro_f1 = _macro_f1(truth, predictions)
    accuracy = round(sum(1 for actual, predicted in zip(truth, predictions) if actual == predicted) / len(truth), 3)
    weak_classes = tuple(
        sorted(label for label, metrics in per_class.items() if metrics.f1 < config.weak_class_f1_threshold)
    )
    split_summary = _split_summary(train_split, val_split, test_split)
    loss_curve = LossCurve(
        train_loss=tuple(float(value) for value in training_history.get("train_loss", ())),
        val_loss=tuple(float(value) for value in training_history.get("val_loss", ())),
    )

    blockers: list[str] = []
    if macro_f1 < config.min_macro_f1_for_phase2b:
        blockers.append(
            f"macro_f1 {macro_f1:.3f} is below Phase 2B threshold {config.min_macro_f1_for_phase2b:.3f}"
        )
    if weak_classes:
        blockers.append(
            f"{len(weak_classes)} weak classes below F1 {config.weak_class_f1_threshold:.2f}: "
            f"{', '.join(weak_classes[:10])}"
        )
    if split_summary.has_signer_leakage:
        blockers.append("signer leakage detected across train/val/test splits")

    return Phase2AReport(
        metric_summary=MetricSummary(
            macro_f1=macro_f1,
            accuracy=accuracy,
            sample_count=len(truth),
            class_count=len(per_class),
        ),
        per_class=per_class,
        weak_classes=weak_classes,
        split_summary=split_summary,
        loss_curve=loss_curve,
        decision="adjust_strategy" if blockers else "proceed_to_phase2b",
        blockers=tuple(blockers),
        metadata=dict(metadata) if metadata is not None else None,
    )


def write_phase2a_artifacts(report: Phase2AReport, output_dir: Path) -> dict[str, Path]:
    """Write JSON, markdown, and loss-curve CSV artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase2a_report.json"
    markdown_path = output_dir / "phase2a_decision.md"
    loss_curve_path = output_dir / "loss_curve.csv"

    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_loss_curve_csv(report.loss_curve, loss_curve_path)

    return {"json": json_path, "markdown": markdown_path, "loss_curve_csv": loss_curve_path}


def run_label_prior_phase2a(
    *,
    train_csv: Path | str,
    val_csv: Path | str,
    test_csv: Path | str,
    output_dir: Path | str,
    max_samples: int | None = None,
) -> Phase2AReport:
    """Run a deterministic label-prior baseline for the Phase 2A gate.

    This keeps the decision-report path runnable without the old placeholder
    Gemma/Unsloth training script.
    """

    train_df = _read_phase2a_split(Path(train_csv), max_samples=max_samples)
    val_df = _read_phase2a_split(Path(val_csv), max_samples=max_samples // 2 if max_samples else None)
    test_df = _read_phase2a_split(Path(test_csv), max_samples=max_samples // 2 if max_samples else None)
    predictions = _label_prior_predictions(train_df, test_df)
    truth = [str(value) for value in test_df["gloss"].tolist()]
    report = build_phase2a_report(
        truth=truth,
        predictions=predictions,
        train_split=train_df,
        val_split=val_df,
        test_split=test_df,
        training_history={
            "train_loss": [_label_prior_loss(train_df, train_df)],
            "val_loss": [_label_prior_loss(train_df, val_df)],
        },
        config=Phase2AConfig(min_macro_f1_for_phase2b=0.75, weak_class_f1_threshold=0.65),
        metadata={
            "backend": "label_prior",
            "note": "Deterministic fallback used when Gemma/Unsloth is unavailable.",
        },
    )
    write_phase2a_artifacts(report, Path(output_dir))
    return report


def _read_phase2a_split(csv_path: Path, max_samples: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if max_samples is not None:
        frame = frame.head(max_samples)
    return frame


def _label_prior_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    counts = train_df["gloss"].value_counts()
    if counts.empty:
        raise ValueError("train split must contain at least one gloss.")
    max_count = counts.max()
    majority_gloss = sorted(str(gloss) for gloss, count in counts.items() if count == max_count)[0]
    return [majority_gloss] * len(test_df)


def _label_prior_loss(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> float:
    counts = train_df["gloss"].value_counts()
    labels = sorted(set(str(label) for label in train_df["gloss"].tolist()) | set(str(label) for label in eval_df["gloss"].tolist()))
    denominator = float(counts.sum() + len(labels))
    losses = []
    for label in eval_df["gloss"].tolist():
        probability = (float(counts.get(label, 0) or 0) + 1.0) / denominator
        losses.append(-math.log(probability))
    return round(sum(losses) / len(losses), 6) if losses else 0.0


def _per_class_metrics(truth: Sequence[str], predictions: Sequence[str]) -> dict[str, ClassMetrics]:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length.")
    if not truth:
        raise ValueError("truth and predictions must not be empty.")

    labels = sorted(set(truth) | set(predictions))
    metrics: dict[str, ClassMetrics] = {}
    for label in labels:
        true_positive = sum(1 for actual, predicted in zip(truth, predictions) if actual == label and predicted == label)
        false_positive = sum(1 for actual, predicted in zip(truth, predictions) if actual != label and predicted == label)
        false_negative = sum(1 for actual, predicted in zip(truth, predictions) if actual == label and predicted != label)
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = 0.0 if precision_denominator == 0 else true_positive / precision_denominator
        recall = 0.0 if recall_denominator == 0 else true_positive / recall_denominator
        f1_denominator = precision + recall
        f1 = 0.0 if f1_denominator == 0.0 else 2 * precision * recall / f1_denominator
        metrics[label] = ClassMetrics(
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1=round(f1, 3),
            support=sum(1 for actual in truth if actual == label),
        )
    return metrics


def _macro_f1(truth: Sequence[str], predictions: Sequence[str]) -> float:
    labels = sorted(set(truth) | set(predictions))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(1 for actual, predicted in zip(truth, predictions) if actual == label and predicted == label)
        false_positive = sum(1 for actual, predicted in zip(truth, predictions) if actual != label and predicted == label)
        false_negative = sum(1 for actual, predicted in zip(truth, predictions) if actual == label and predicted != label)
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return round(sum(scores) / len(scores), 3)


def _split_summary(train_split: pd.DataFrame, val_split: pd.DataFrame, test_split: pd.DataFrame) -> SplitSummary:
    for name, frame in {"train": train_split, "val": val_split, "test": test_split}.items():
        if "signer_id" not in frame.columns:
            raise KeyError(f"{name}_split must contain a 'signer_id' column.")

    train_signers = _signer_set(train_split)
    val_signers = _signer_set(val_split)
    test_signers = _signer_set(test_split)
    train_val_overlap = tuple(sorted(train_signers & val_signers))
    train_test_overlap = tuple(sorted(train_signers & test_signers))
    val_test_overlap = tuple(sorted(val_signers & test_signers))

    return SplitSummary(
        train_samples=len(train_split),
        val_samples=len(val_split),
        test_samples=len(test_split),
        train_signers=len(train_signers),
        val_signers=len(val_signers),
        test_signers=len(test_signers),
        train_val_signer_overlap=train_val_overlap,
        train_test_signer_overlap=train_test_overlap,
        val_test_signer_overlap=val_test_overlap,
        has_signer_leakage=bool(train_val_overlap or train_test_overlap or val_test_overlap),
    )


def _signer_set(frame: pd.DataFrame) -> set[str]:
    return {str(value) for value in frame["signer_id"].dropna().tolist()}


def _write_loss_curve_csv(loss_curve: LossCurve, output_path: Path) -> None:
    max_epochs = max(len(loss_curve.train_loss), len(loss_curve.val_loss))
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for index in range(max_epochs):
            train_loss = loss_curve.train_loss[index] if index < len(loss_curve.train_loss) else ""
            val_loss = loss_curve.val_loss[index] if index < len(loss_curve.val_loss) else ""
            writer.writerow([index + 1, train_loss, val_loss])


def _render_markdown(report: Phase2AReport) -> str:
    decision_text = "Proceed to Phase 2B" if report.decision == "proceed_to_phase2b" else "Adjust strategy before Phase 2B"
    blockers = "\n".join(f"- {blocker}" for blocker in report.blockers) if report.blockers else "- None"
    weak_classes = ", ".join(report.weak_classes) if report.weak_classes else "None"
    metadata_lines = []
    if report.metadata:
        metadata_lines = [
            "## Run Metadata",
            "",
            *[f"- {key.replace('_', ' ').title()}: {value}" for key, value in sorted(report.metadata.items())],
            "",
        ]

    return "\n".join(
        [
            "# Phase 2A Validation Decision",
            "",
            f"**Decision:** {decision_text}",
            "",
            "## Metrics",
            "",
            f"- Macro F1: {report.metric_summary.macro_f1:.3f}",
            f"- Accuracy: {report.metric_summary.accuracy:.3f}",
            f"- Test samples: {report.metric_summary.sample_count}",
            f"- Classes evaluated: {report.metric_summary.class_count}",
            f"- Weak classes: {weak_classes}",
            "",
            "## Split Check",
            "",
            f"- Train/val/test samples: {report.split_summary.train_samples}/"
            f"{report.split_summary.val_samples}/{report.split_summary.test_samples}",
            f"- Signer leakage: {'yes' if report.split_summary.has_signer_leakage else 'no'}",
            "",
            *metadata_lines,
            "## Blockers",
            "",
            blockers,
            "",
        ]
    )
