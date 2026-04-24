"""Utility helpers for ASL transcription metrics and text decoding."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

import numpy as np


def canonicalize_text(text: str) -> str:
    """Normalize whitespace and casing for fair string comparison."""

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return normalized


def word_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute exact-match word accuracy over two text sequences."""

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not references:
        return 0.0

    correct = sum(
        canonicalize_text(prediction) == canonicalize_text(reference)
        for prediction, reference in zip(predictions, references)
    )
    return correct / len(references)


def top_k_accuracy(
    candidate_predictions: Sequence[Sequence[str]],
    references: Sequence[str],
    k: int = 5,
) -> float:
    """Compute top-k accuracy from ranked candidate prediction lists."""

    if len(candidate_predictions) != len(references):
        raise ValueError("Candidate predictions and references must have the same length.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if not references:
        return 0.0

    correct = 0
    for candidates, reference in zip(candidate_predictions, references):
        normalized_reference = canonicalize_text(reference)
        normalized_candidates = [canonicalize_text(candidate) for candidate in candidates[:k]]
        if normalized_reference in normalized_candidates:
            correct += 1
    return correct / len(references)


def normalize_pose_embeddings(
    pose_embeddings: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize pose features per sequence while preserving padded rows."""

    if pose_embeddings.ndim != 2:
        raise ValueError("Expected pose_embeddings with shape (timesteps, features).")
    if pose_embeddings.size == 0:
        return pose_embeddings.astype(np.float32)

    centered = pose_embeddings.astype(np.float32) - pose_embeddings.mean(axis=0, keepdims=True)
    scale = pose_embeddings.std(axis=0, keepdims=True)
    normalized = centered / np.maximum(scale, epsilon)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def decode_text_batch(
    tokenizer: Any,
    token_ids: Any,
    skip_special_tokens: bool = True,
) -> List[str]:
    """Decode a batch of token ids using a Hugging Face compatible tokenizer."""

    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu().tolist()
    elif hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if not isinstance(token_ids, Iterable):
        raise TypeError("token_ids must be an iterable of token sequences.")

    decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    return [text.strip() for text in decoded]

