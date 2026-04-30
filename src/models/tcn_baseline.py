"""TCN/1D-CNN baseline for Top-50 ASL gloss inference."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_TOP50_CONTRACT_PATH = Path("data/contracts/asl_top50_glosses_v1.json")
DEFAULT_LATENCY_TARGET_MS = 800.0


@dataclass(frozen=True)
class TCNTrainingConfig:
    """Training settings for the demo baseline."""

    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-3
    hidden_channels: int = 64
    kernel_size: int = 3
    latency_target_ms: float = DEFAULT_LATENCY_TARGET_MS
    device: str = "cpu"


@dataclass(frozen=True)
class TCNTrainingReport:
    """Small training summary used by readiness and demo checks."""

    trained_samples: int
    initial_loss: float
    final_loss: float
    epochs: int


@dataclass(frozen=True)
class GlossPrediction:
    """Structured inference result that can safely flow into demo UI."""

    ok: bool
    prediction: Optional[str]
    confidence: float
    latency_ms: float
    latency_target_ms: float = DEFAULT_LATENCY_TARGET_MS
    error: Optional[str] = None


class TCNBaseline(nn.Module):
    """Small temporal convolutional classifier for pose feature windows."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, feature_windows: torch.Tensor) -> torch.Tensor:
        """Classify `(batch, timesteps, features)` windows."""

        if feature_windows.ndim != 3:
            raise ValueError("feature_windows must have shape (batch, timesteps, features).")
        if feature_windows.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {feature_windows.shape[-1]}.")
        return self.network(feature_windows.transpose(1, 2))


def load_top50_glosses(contract_path: Path | str = DEFAULT_TOP50_CONTRACT_PATH) -> Tuple[str, ...]:
    """Load the fixed Top-50 gloss contract in inference order."""

    path = Path(contract_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    glosses = tuple(str(gloss) for gloss in payload["glosses"])
    if len(glosses) != 50:
        raise ValueError(f"Expected 50 glosses in {path}, found {len(glosses)}.")
    return glosses


def train_tcn_baseline(
    train_features: np.ndarray,
    train_labels: Sequence[str],
    glosses: Sequence[str],
    config: Optional[TCNTrainingConfig] = None,
) -> tuple[TCNBaseline, TCNTrainingReport]:
    """Train the TCN baseline on dense `(samples, timesteps, features)` windows."""

    config = config or TCNTrainingConfig()
    features = _coerce_training_features(train_features)
    if len(train_labels) != features.shape[0]:
        raise ValueError("train_labels must align with train_features.")
    if not glosses:
        raise ValueError("glosses must not be empty.")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")

    label_to_index = {gloss: index for index, gloss in enumerate(glosses)}
    try:
        label_indices = torch.tensor([label_to_index[label] for label in train_labels], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"Training label is not in the Top-50 gloss contract: {exc.args[0]}") from exc

    device = torch.device(config.device)
    model = TCNBaseline(
        input_dim=int(features.shape[-1]),
        num_classes=len(glosses),
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
    ).to(device)
    feature_tensor = torch.from_numpy(features).to(device)
    label_tensor = label_indices.to(device)
    dataset = TensorDataset(feature_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    with torch.no_grad():
        initial_loss = float(loss_fn(model(feature_tensor), label_tensor).detach().cpu().item())

    final_loss = initial_loss
    for _ in range(config.epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.no_grad():
        final_loss = float(loss_fn(model(feature_tensor), label_tensor).detach().cpu().item())

    return model.cpu(), TCNTrainingReport(
        trained_samples=int(features.shape[0]),
        initial_loss=initial_loss,
        final_loss=final_loss,
        epochs=config.epochs,
    )


def predict_feature_window(
    model: TCNBaseline,
    feature_window: object,
    glosses: Sequence[str],
    latency_target_ms: float = DEFAULT_LATENCY_TARGET_MS,
) -> GlossPrediction:
    """Serve a Top-50 gloss prediction from a live capture `FeatureWindow`."""

    started = time.perf_counter()
    try:
        features = _coerce_inference_features(getattr(feature_window, "features"))
        if len(glosses) != model.num_classes:
            raise ValueError("glosses must align with the model output classes.")

        with torch.no_grad():
            logits = model(torch.from_numpy(features).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=-1)[0]
            confidence, class_index = torch.max(probabilities, dim=0)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return GlossPrediction(
            ok=True,
            prediction=str(glosses[int(class_index.item())]),
            confidence=float(confidence.item()),
            latency_ms=latency_ms,
            latency_target_ms=latency_target_ms,
            error=None,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return GlossPrediction(
            ok=False,
            prediction=None,
            confidence=0.0,
            latency_ms=latency_ms,
            latency_target_ms=latency_target_ms,
            error=f"Feature window inference failed safely: {exc}",
        )


def _coerce_training_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("train_features must have shape (samples, timesteps, features).")
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError("train_features must have non-empty samples, timesteps, and features.")
    return array


def _coerce_inference_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("feature window must have shape (timesteps, features).")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("feature window must have non-empty timesteps and features.")
    return array
