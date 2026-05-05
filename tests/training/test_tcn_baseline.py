"""Behavior tests for the Top-50 TCN baseline inference slice."""

from __future__ import annotations

import numpy as np
import torch

from src.models.tcn_baseline import (
    TCNTrainingConfig,
    load_top50_glosses,
    predict_feature_window,
    train_tcn_baseline,
)


class SimpleFeatureWindow:
    def __init__(self, features: np.ndarray) -> None:
        self.features = features.astype(np.float32)


def _feature_window(features: np.ndarray) -> SimpleFeatureWindow:
    return SimpleFeatureWindow(features)


def test_tcn_baseline_trains_and_serves_top50_prediction_from_feature_window() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    glosses = load_top50_glosses()
    feature_dim = 12
    sequence_length = 8

    hello_samples = np.zeros((8, sequence_length, feature_dim), dtype=np.float32)
    thanks_samples = np.ones((8, sequence_length, feature_dim), dtype=np.float32)
    train_features = np.concatenate([hello_samples, thanks_samples], axis=0)
    train_labels = ["hello"] * len(hello_samples) + ["thanks"] * len(thanks_samples)

    model, report = train_tcn_baseline(
        train_features,
        train_labels,
        glosses=glosses,
        config=TCNTrainingConfig(epochs=18, batch_size=4, learning_rate=0.03, hidden_channels=8),
    )
    result = predict_feature_window(model, _feature_window(np.ones((sequence_length, feature_dim))), glosses)

    assert report.trained_samples == 16
    assert report.final_loss < report.initial_loss
    assert result.ok is True
    assert result.prediction == "thanks"
    assert result.confidence > 0.50
    assert result.latency_ms >= 0.0
    assert result.latency_target_ms == 800.0
    assert result.error is None


def test_tcn_baseline_returns_safe_error_for_invalid_feature_window() -> None:
    glosses = load_top50_glosses()
    model, _ = train_tcn_baseline(
        np.zeros((2, 4, 6), dtype=np.float32),
        ["hello", "thanks"],
        glosses=glosses,
        config=TCNTrainingConfig(epochs=1, hidden_channels=4),
    )

    result = predict_feature_window(model, _feature_window(np.zeros((0, 6), dtype=np.float32)), glosses)

    assert result.ok is False
    assert result.prediction is None
    assert result.confidence == 0.0
    assert result.error is not None
    assert "feature window" in result.error.lower()
