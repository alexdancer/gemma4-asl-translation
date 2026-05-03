"""Evaluation utilities for Unsloth-trained ASL q64 JSONL classifiers."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence


InferenceMode = Literal["mock", "real"]
INVALID_PREDICTION = "__invalid__"


class GlossPredictor(Protocol):
    """Callable model wrapper used by q64 record evaluation."""

    mode: InferenceMode

    def predict_raw(self, record: Mapping[str, Any]) -> str:
        """Return the raw text generated for a q64 JSONL record."""


class ConstrainedGlossScorer(Protocol):
    """Scores canonical labels as diagnostic continuations for one q64 record."""

    mode: InferenceMode

    def score_candidate_labels(
        self,
        record: Mapping[str, Any],
        labels: Sequence[str],
    ) -> Sequence["CandidateLabelScore"]:
        """Return one score for each canonical candidate label."""


@dataclass(frozen=True)
class Q64InferenceResult:
    """Stable inference result for one q64 Unsloth ASL JSONL record."""

    predicted_gloss: str | None
    raw_model_output: str
    valid_label: bool
    expected_gloss: str | None
    mode: InferenceMode


@dataclass(frozen=True)
class CandidateLabelScore:
    """Score for one constrained Top-50 candidate label continuation."""

    label: str
    score: float


@dataclass(frozen=True)
class Q64ConstrainedScoringResult:
    """Diagnostic-only constrained Top-50 score result for one q64 record.

    This is intentionally separate from strict free-generation proof metrics:
    constrained scoring ranks manifest labels under the q64 prompt contract, but
    it does not prove the model can freely generate the correct gloss.
    """

    best_label: str
    ranked_scores: tuple[CandidateLabelScore, ...]
    expected_gloss: str | None
    correct: bool | None
    mode: InferenceMode


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths written by an evaluator run."""

    predictions_csv: Path
    metrics_json: Path


def load_q64_jsonl(path: Path | str, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load instruction/input/output q64 records from JSONL."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_samples is not None and len(records) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            records.append(record)
    return records


def load_manifest_labels(path: Path | str) -> tuple[str, ...]:
    """Load canonical ASL labels from the Top-50 manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{path} must contain a non-empty 'labels' list.")
    return tuple(normalize_gloss(str(label)) for label in labels)


def normalize_gloss(text: str) -> str:
    """Normalize a gloss for strict label comparison."""

    normalized = text.strip().lower()
    normalized = re.sub(r"^`+|`+$", "", normalized)
    normalized = re.sub(r"[^a-z0-9_ -]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_model_output(raw_output: str, labels: Sequence[str]) -> tuple[str | None, bool]:
    """Extract and validate a single gloss from model output."""

    normalized_labels = {normalize_gloss(label) for label in labels}
    candidates = [
        normalize_gloss(piece)
        for piece in re.split(r"[\n,;:]+", raw_output)
        if normalize_gloss(piece)
    ]
    if not candidates:
        return None, False

    first = candidates[0]
    if first in normalized_labels:
        return first, True

    words = first.split()
    if len(words) == 1 and words[0] in normalized_labels:
        return words[0], True
    return first, False


def infer_q64_record(
    record: Mapping[str, Any],
    predictor: GlossPredictor,
    labels: Sequence[str],
) -> Q64InferenceResult:
    """Run a reusable q64 JSONL inference contract for one record."""

    raw_output = predictor.predict_raw(record)
    predicted_gloss, valid_label = normalize_model_output(raw_output, labels)
    expected = record.get("output")
    expected_gloss = normalize_gloss(str(expected)) if expected is not None else None
    return Q64InferenceResult(
        predicted_gloss=predicted_gloss,
        raw_model_output=raw_output,
        valid_label=valid_label,
        expected_gloss=expected_gloss,
        mode=predictor.mode,
    )


def score_q64_record_constrained(
    record: Mapping[str, Any],
    scorer: ConstrainedGlossScorer,
    labels: Sequence[str],
) -> Q64ConstrainedScoringResult:
    """Rank every canonical Top-50 label for one q64 record.

    This constrained path is diagnostic only. It is intentionally separate from
    strict free-generation metrics because it does not prove the model can
    freely generate the correct gloss.
    """

    normalized_labels = tuple(normalize_gloss(label) for label in labels)
    if not normalized_labels:
        raise ValueError("labels must not be empty.")

    raw_scores = scorer.score_candidate_labels(record, normalized_labels)
    scores_by_label: dict[str, CandidateLabelScore] = {}
    for score in raw_scores:
        normalized_label = normalize_gloss(score.label)
        if normalized_label in scores_by_label:
            raise ValueError(f"Duplicate constrained score for label: {normalized_label}")
        scores_by_label[normalized_label] = CandidateLabelScore(
            label=normalized_label,
            score=float(score.score),
        )

    label_set = set(normalized_labels)
    missing = [label for label in normalized_labels if label not in scores_by_label]
    extra = sorted(label for label in scores_by_label if label not in label_set)
    if missing:
        raise ValueError(f"Missing constrained scores for labels: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected constrained scores for labels: {', '.join(extra)}")

    label_order = {label: index for index, label in enumerate(normalized_labels)}
    ranked_scores = tuple(
        sorted(
            scores_by_label.values(),
            key=lambda candidate: (-candidate.score, label_order[candidate.label]),
        )
    )
    expected = record.get("output")
    expected_gloss = normalize_gloss(str(expected)) if expected is not None else None
    best_label = ranked_scores[0].label
    correct = best_label == expected_gloss if expected_gloss is not None else None
    return Q64ConstrainedScoringResult(
        best_label=best_label,
        ranked_scores=ranked_scores,
        expected_gloss=expected_gloss,
        correct=correct,
        mode=scorer.mode,
    )


class MockASLGlossPredictor:
    """Deterministic no-model predictor for CI and local contract checks."""

    mode: InferenceMode = "mock"

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = tuple(labels)
        if not self.labels:
            raise ValueError("labels must not be empty.")

    def predict_raw(self, record: Mapping[str, Any]) -> str:
        sample_key = _record_sample_key(record)
        digest = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(self.labels)
        return self.labels[index]


class MockConstrainedGlossScorer:
    """Deterministic no-model constrained scorer for unit tests."""

    mode: InferenceMode = "mock"

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = {normalize_gloss(label): float(score) for label, score in scores.items()}

    def score_candidate_labels(
        self,
        record: Mapping[str, Any],
        labels: Sequence[str],
    ) -> Sequence[CandidateLabelScore]:
        del record
        return tuple(
            CandidateLabelScore(
                label=normalize_gloss(label),
                score=self.scores.get(normalize_gloss(label), 0.0),
            )
            for label in labels
        )


class RealUnslothASLGlossPredictor:
    """Lazy real model wrapper for Unsloth/PEFT LoRA checkpoints."""

    mode: InferenceMode = "real"

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        max_new_tokens: int = 8,
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self._load_model(max_seq_length, load_in_4bit)

    def predict_raw(self, record: Mapping[str, Any]) -> str:
        user_prompt = build_unsloth_prompt(record)
        prompt = _build_generation_prompt(self.tokenizer, user_prompt)

        # Gemma4 exports may return a processor whose patched __call__ expects
        # text as a keyword. Plain tokenizers also accept this form.
        tokenized = self.tokenizer(text=prompt, return_tensors="pt")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(_dependency_error_message()) from exc

        device = next(self.model.parameters()).device
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        prompt_tokens = tokenized["input_ids"].shape[-1]
        generated = outputs[0][prompt_tokens:]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()

    def score_candidate_labels(
        self,
        record: Mapping[str, Any],
        labels: Sequence[str],
    ) -> Sequence[CandidateLabelScore]:
        """Score each canonical label as a q64 prompt continuation.

        This constrained Top-50 path is diagnostic only. It uses conditional log
        likelihood under the same frozen checkpoint and q64 prompt contract as
        free generation, but it must not be mixed into strict free-generation
        proof metrics.
        """

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(_dependency_error_message()) from exc

        user_prompt = build_unsloth_prompt(record)
        prompt = _build_generation_prompt(self.tokenizer, user_prompt)
        prompt_tokens = self.tokenizer(text=prompt, return_tensors="pt")
        prompt_length = prompt_tokens["input_ids"].shape[-1]
        device = next(self.model.parameters()).device
        scores: list[CandidateLabelScore] = []

        for label in labels:
            normalized_label = normalize_gloss(label)
            full_tokens = self.tokenizer(text=f"{prompt}{normalized_label}", return_tensors="pt")
            tokenized = {key: value.to(device) for key, value in full_tokens.items()}
            target_ids = tokenized["input_ids"].clone()
            target_ids[:, :prompt_length] = -100
            if target_ids[:, prompt_length:].numel() == 0:
                scores.append(CandidateLabelScore(label=normalized_label, score=float("-inf")))
                continue

            with torch.no_grad():
                logits = self.model(**tokenized).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = target_ids[:, 1:]
            candidate_mask = shift_labels.ne(-100)
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1,
                index=shift_labels.clamp_min(0).unsqueeze(-1),
            ).squeeze(-1)
            candidate_score = token_log_probs.masked_select(candidate_mask).sum().item()
            scores.append(CandidateLabelScore(label=normalized_label, score=float(candidate_score)))

        return tuple(scores)

    def _load_model(self, max_seq_length: int, load_in_4bit: bool) -> tuple[Any, Any]:
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise RuntimeError(_dependency_error_message()) from exc

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.checkpoint),
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
            model = FastLanguageModel.for_inference(model)
        except Exception:
            peft_config = PeftConfig.from_pretrained(str(self.checkpoint))
            base_model_name = peft_config.base_model_name_or_path
            if not base_model_name:
                raise RuntimeError(
                    "PEFT adapter config is missing base_model_name_or_path; "
                    "save the dashboard checkpoint with adapter_config.json metadata."
                )
            tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint))
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="auto",
                load_in_4bit=load_in_4bit,
            )
            model = PeftModel.from_pretrained(model, str(self.checkpoint))
            model.eval()

        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


def build_unsloth_prompt(record: Mapping[str, Any]) -> str:
    """Recreate the instruction/input prompt used by Dashboard JSONL records."""

    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    return f"{instruction}\n\n{input_text}".strip()


def _build_generation_prompt(tokenizer: Any, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    return f"{user_prompt}\n\nAnswer:"


def evaluate_q64_records(
    records: Sequence[Mapping[str, Any]],
    predictor: GlossPredictor,
    labels: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate q64 records and return prediction rows plus metrics."""

    normalized_labels = tuple(normalize_gloss(label) for label in labels)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        result = infer_q64_record(record, predictor, normalized_labels)
        predicted_for_metrics = (
            result.predicted_gloss if result.valid_label and result.predicted_gloss else INVALID_PREDICTION
        )
        correct = (
            result.expected_gloss is not None
            and result.valid_label
            and result.predicted_gloss == result.expected_gloss
        )
        rows.append(
            {
                "index": index,
                "sample_id": _record_sample_key(record),
                "expected_gloss": result.expected_gloss or "",
                "predicted_gloss": result.predicted_gloss or "",
                "raw_model_output": result.raw_model_output,
                "valid_label": result.valid_label,
                "correct": correct,
                "mode": result.mode,
                "prediction_for_metrics": predicted_for_metrics,
            }
        )

    return rows, build_metrics(rows, normalized_labels)


def build_metrics(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    """Build exact-match, invalid-rate, per-class, and confusion metrics."""

    total = len(rows)
    correct = sum(1 for row in rows if bool(row["correct"]))
    invalid = sum(1 for row in rows if not bool(row["valid_label"]))

    per_class: dict[str, dict[str, float | int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for label in labels:
        label_rows = [row for row in rows if row["expected_gloss"] == label]
        label_correct = sum(1 for row in label_rows if bool(row["correct"]))
        support = len(label_rows)
        per_class[label] = {
            "support": support,
            "correct": label_correct,
            "accuracy": round(label_correct / support, 6) if support else 0.0,
        }

    for row in rows:
        expected = str(row["expected_gloss"] or "unknown")
        predicted = str(row["prediction_for_metrics"])
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

    return {
        "sample_count": total,
        "strict_normalized_top1_accuracy": round(correct / total, 6) if total else 0.0,
        "invalid_output_rate": round(invalid / total, 6) if total else 0.0,
        "correct": correct,
        "invalid": invalid,
        "class_count": len(labels),
        "per_class_accuracy": per_class,
        "confusion_matrix_counts": confusion,
    }


def write_evaluation_artifacts(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    out_dir: Path | str,
) -> EvaluationArtifacts:
    """Write predictions CSV and metrics JSON."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"

    fieldnames = [
        "index",
        "sample_id",
        "expected_gloss",
        "predicted_gloss",
        "raw_model_output",
        "valid_label",
        "correct",
        "mode",
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return EvaluationArtifacts(predictions_csv=predictions_path, metrics_json=metrics_path)


def _record_sample_key(record: Mapping[str, Any]) -> str:
    input_text = str(record.get("input", ""))
    for line in input_text.splitlines():
        if line.startswith("sample_id="):
            return line.split("=", 1)[1].strip()
    return hashlib.sha256(input_text.encode("utf-8")).hexdigest()[:12]


def _dependency_error_message() -> str:
    return (
        "Real Unsloth ASL evaluation requires torch, unsloth, transformers, and peft. "
        "Install project dependencies with `pip install -r requirements.txt`, or run "
        "`scripts/evaluate_unsloth_asl.py --mock ...` for fast local contract testing."
    )


def result_to_dict(result: Q64InferenceResult) -> dict[str, Any]:
    """Return a JSON-friendly representation of a q64 inference result."""

    return asdict(result)
