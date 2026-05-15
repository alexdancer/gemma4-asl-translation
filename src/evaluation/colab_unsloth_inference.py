"""Colab-first Unsloth Top-50 single-gloss inference helpers.

Issue #89 scope:
- enforce single Top-50 gloss output contract
- deterministic one-shot OOV retry with stronger constraints
- log first-pass vs retry outcome for each sample
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.evaluation.unsloth_asl import normalize_gloss


PredictFn = Callable[[str, Mapping[str, Any]], str]


@dataclass(frozen=True)
class RetryConfig:
    max_new_tokens: int = 2
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class InferenceAttempt:
    prompt: str
    decode: dict[str, Any]
    raw_output: str
    normalized_output: str
    canonical_gloss: str | None
    valid: bool


@dataclass(frozen=True)
class InferenceResult:
    sample_id: str
    expected_gloss: str
    first_pass: InferenceAttempt
    retry_pass: InferenceAttempt | None
    used_retry: bool
    final_gloss: str | None
    final_valid: bool
    correct: bool


def _alias_lookup(alias_entries: Sequence[Mapping[str, str]] | None) -> dict[str, str]:
    table: dict[str, str] = {}
    for entry in alias_entries or ():
        alias = normalize_gloss(str(entry.get("alias", "")))
        canonical = normalize_gloss(str(entry.get("canonical", "")))
        if not alias or not canonical:
            continue
        table[alias] = canonical
    return table


def _resolve_candidate(
    raw_output: str,
    *,
    allowlist: Sequence[str],
    alias_entries: Sequence[Mapping[str, str]] | None,
) -> tuple[str, str | None, bool]:
    normalized = normalize_gloss(raw_output)
    if not normalized:
        return normalized, None, False

    allow = {normalize_gloss(item) for item in allowlist}
    aliases = _alias_lookup(alias_entries)
    canonical = aliases.get(normalized, normalized)
    valid = canonical in allow
    return normalized, canonical if valid else None, valid


def build_first_pass_prompt(*, record_input: str, allowlist: Sequence[str]) -> str:
    choices = "\n".join(f"- {normalize_gloss(label)}" for label in allowlist)
    return (
        "You are evaluating an ASL pose/frame-derived sample.\n"
        "Return exactly one canonical gloss from the allowlist below.\n"
        "Output must be only the gloss token (no explanation/punctuation).\n\n"
        f"Allowlist:\n{choices}\n\n"
        f"Sample:\n{record_input.strip()}"
    )


def build_retry_prompt(*, record_input: str, allowlist: Sequence[str], first_output: str) -> str:
    choices = ", ".join(normalize_gloss(label) for label in allowlist)
    return (
        "STRICT RETRY (deterministic): your previous output was out-of-vocabulary.\n"
        "Choose EXACTLY one token from this allowlist and output only that token.\n"
        f"Allowlist tokens: {choices}\n"
        f"Previous output: {first_output!r}\n"
        f"Sample: {record_input.strip()}"
    )


def infer_with_single_retry(
    *,
    sample_id: str,
    expected_gloss: str,
    record_input: str,
    allowlist: Sequence[str],
    alias_entries: Sequence[Mapping[str, str]] | None,
    predict_fn: PredictFn,
    first_decode: Mapping[str, Any] | None = None,
    retry_config: RetryConfig = RetryConfig(),
) -> InferenceResult:
    first_decode_cfg = dict(first_decode or {"max_new_tokens": 4, "temperature": 0.0, "top_p": 1.0})
    first_prompt = build_first_pass_prompt(record_input=record_input, allowlist=allowlist)
    first_raw = predict_fn(first_prompt, first_decode_cfg)
    first_norm, first_gloss, first_valid = _resolve_candidate(
        first_raw, allowlist=allowlist, alias_entries=alias_entries
    )
    first_attempt = InferenceAttempt(
        prompt=first_prompt,
        decode=first_decode_cfg,
        raw_output=first_raw,
        normalized_output=first_norm,
        canonical_gloss=first_gloss,
        valid=first_valid,
    )

    retry_attempt: InferenceAttempt | None = None
    final_gloss = first_gloss
    final_valid = first_valid
    used_retry = False

    if not first_valid:
        used_retry = True
        retry_prompt = build_retry_prompt(
            record_input=record_input,
            allowlist=allowlist,
            first_output=first_raw,
        )
        retry_decode_cfg = {
            "max_new_tokens": retry_config.max_new_tokens,
            "temperature": retry_config.temperature,
            "top_p": retry_config.top_p,
        }
        retry_raw = predict_fn(retry_prompt, retry_decode_cfg)
        retry_norm, retry_gloss, retry_valid = _resolve_candidate(
            retry_raw,
            allowlist=allowlist,
            alias_entries=alias_entries,
        )
        retry_attempt = InferenceAttempt(
            prompt=retry_prompt,
            decode=retry_decode_cfg,
            raw_output=retry_raw,
            normalized_output=retry_norm,
            canonical_gloss=retry_gloss,
            valid=retry_valid,
        )
        final_gloss = retry_gloss
        final_valid = retry_valid

    expected = normalize_gloss(expected_gloss)
    return InferenceResult(
        sample_id=sample_id,
        expected_gloss=expected,
        first_pass=first_attempt,
        retry_pass=retry_attempt,
        used_retry=used_retry,
        final_gloss=final_gloss,
        final_valid=final_valid,
        correct=bool(final_valid and final_gloss == expected),
    )


def evaluate_anchor_samples(
    *,
    samples: Sequence[Mapping[str, str]],
    allowlist: Sequence[str],
    alias_entries: Sequence[Mapping[str, str]] | None,
    predict_fn: PredictFn,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        result = infer_with_single_retry(
            sample_id=str(sample["sample_id"]),
            expected_gloss=str(sample["expected_gloss"]),
            record_input=str(sample["input"]),
            allowlist=allowlist,
            alias_entries=alias_entries,
            predict_fn=predict_fn,
        )
        rows.append(
            {
                "sample_id": result.sample_id,
                "expected_gloss": result.expected_gloss,
                "first_pass_raw": result.first_pass.raw_output,
                "first_pass_valid": result.first_pass.valid,
                "retry_used": result.used_retry,
                "retry_raw": result.retry_pass.raw_output if result.retry_pass else "",
                "retry_valid": result.retry_pass.valid if result.retry_pass else False,
                "final_gloss": result.final_gloss or "",
                "final_valid": result.final_valid,
                "correct": result.correct,
            }
        )
    return rows
