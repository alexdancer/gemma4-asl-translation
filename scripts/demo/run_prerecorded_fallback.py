"""Run fallback A: prerecorded clip with live TCN inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.live_capture import LiveFeatureStream, PrerecordedVideoSource
from src.demo.fallback_a import DemoInferenceRunConfig, run_demo_inference_once


class DemoFrameExtractor:
    """Deterministic demo extractor that maps frame brightness to hands+pose features."""

    def extract_from_frame(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        value = float(frame_bgr.mean() / 255.0)
        return {
            "body": np.full((17, 4), value, dtype=np.float32),
            "left_hand": np.full((21, 4), value, dtype=np.float32),
            "right_hand": np.full((21, 4), value, dtype=np.float32),
            "face": np.zeros((0, 4), dtype=np.float32),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-path", required=True, help="Prerecorded video file to run through live inference.")
    parser.add_argument("--frame-count", type=int, default=8, help="Number of frames to capture for one prediction.")
    args = parser.parse_args(argv)

    media_path = Path(args.media_path)
    stream = LiveFeatureStream(
        source=PrerecordedVideoSource(media_path),
        extractor=DemoFrameExtractor(),
    )
    from src.models.tcn_baseline import predict_feature_window

    glosses = ("hello", "thanks")
    model = _train_demo_model(feature_dim=(17 + 21 + 21) * 4, sequence_length=args.frame_count, glosses=glosses)

    result = run_demo_inference_once(
        stream=stream,
        predict=lambda window: predict_feature_window(model, window, glosses),
        config=DemoInferenceRunConfig(
            mode="prerecorded",
            frame_count=args.frame_count,
            media_path=str(media_path),
        ),
    )

    print(json.dumps(_serialize_result(result), indent=2, sort_keys=True))
    return 0


def _train_demo_model(feature_dim: int, sequence_length: int, glosses: Sequence[str]):
    from src.models.tcn_baseline import TCNTrainingConfig, train_tcn_baseline
    import torch

    # Keep prerecorded fallback smoke output deterministic across runs.
    torch.manual_seed(0)
    np.random.seed(0)

    low_samples = np.full((6, sequence_length, feature_dim), 0.05, dtype=np.float32)
    high_samples = np.full((6, sequence_length, feature_dim), 0.85, dtype=np.float32)
    train_features = np.concatenate([low_samples, high_samples], axis=0)
    train_labels = ["hello"] * len(low_samples) + ["thanks"] * len(high_samples)
    model, _ = train_tcn_baseline(
        train_features,
        train_labels,
        glosses=glosses,
        config=TCNTrainingConfig(
            epochs=20,
            batch_size=4,
            learning_rate=0.03,
            hidden_channels=8,
        ),
    )
    return model


def _serialize_result(result) -> dict[str, object]:
    return {
        "mode": result.mode,
        "media_path": result.media_path,
        "observation": result.observation,
        "status": result.output.status,
        "display_text": result.output.display_text,
        "prediction": result.output.prediction,
        "confidence": result.output.confidence,
        "is_uncertain": result.output.is_uncertain,
        "latency_ms": result.output.latency_ms,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
