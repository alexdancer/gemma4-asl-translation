"""PyTorch dataset and collation utilities for pose-to-text training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.utils import normalize_pose_embeddings

LOGGER = logging.getLogger(__name__)

POSE_COMPONENTS: Tuple[str, ...] = ("body", "left_hand", "right_hand", "face")


@dataclass(frozen=True)
class PoseAugmentationConfig:
    """Configuration for lightweight geometric and temporal augmentation."""

    rotation_degrees: float = 0.0
    speed_variation: float = 0.0
    enabled: bool = False


def _rotate_xy(pose_array: np.ndarray, radians: float) -> np.ndarray:
    """Rotate XY coordinates for every joint in a sequence."""

    if pose_array.size == 0 or abs(radians) < 1e-8:
        return pose_array

    rotated = pose_array.copy()
    sin_theta = np.sin(radians)
    cos_theta = np.cos(radians)
    xy = rotated[..., :2]
    rotated[..., 0] = xy[..., 0] * cos_theta - xy[..., 1] * sin_theta
    rotated[..., 1] = xy[..., 0] * sin_theta + xy[..., 1] * cos_theta
    return rotated


def _resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a `(T, ...)` sequence to a new temporal length."""

    if sequence.shape[0] == target_length:
        return sequence
    if sequence.shape[0] <= 1 or target_length <= 1:
        return np.repeat(sequence[:1], repeats=max(1, target_length), axis=0)

    original_positions = np.linspace(0.0, 1.0, num=sequence.shape[0], dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    flat = sequence.reshape(sequence.shape[0], -1)
    resampled_columns = [
        np.interp(target_positions, original_positions, flat[:, column_index])
        for column_index in range(flat.shape[1])
    ]
    stacked = np.stack(resampled_columns, axis=1).reshape((target_length, *sequence.shape[1:]))
    return stacked.astype(np.float32)


def augment_pose_sequence(
    pose_components: Mapping[str, np.ndarray],
    config: PoseAugmentationConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Apply small random rotation and speed perturbations to pose data."""

    if not config.enabled:
        return {name: array.copy() for name, array in pose_components.items()}

    rng = rng or np.random.default_rng()
    augmented = {name: array.copy() for name, array in pose_components.items()}

    if config.rotation_degrees > 0.0:
        degrees = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
        radians = np.deg2rad(degrees)
        for name, array in augmented.items():
            if array.ndim == 3 and array.shape[-1] >= 2:
                augmented[name] = _rotate_xy(array, radians)

    if config.speed_variation > 0.0:
        reference_name = next((name for name, value in augmented.items() if value.size > 0), None)
        if reference_name is not None:
            source_length = augmented[reference_name].shape[0]
            speed_scale = rng.uniform(1.0 - config.speed_variation, 1.0 + config.speed_variation)
            target_length = max(1, int(round(source_length / max(speed_scale, 1e-3))))
            for name, array in augmented.items():
                if array.shape[0] == source_length:
                    augmented[name] = _resample_sequence(array, target_length)

    return augmented


def flatten_pose_components(
    pose_components: Mapping[str, np.ndarray],
    include_components: Sequence[str] = POSE_COMPONENTS,
    normalize: bool = True,
) -> np.ndarray:
    """Flatten selected pose components into a dense `(timesteps, features)` tensor."""

    arrays: List[np.ndarray] = []
    timesteps: Optional[int] = None

    for name in include_components:
        component = pose_components.get(name)
        if component is None:
            continue
        if component.ndim != 3:
            raise ValueError(f"Pose component '{name}' must have shape (T, J, C).")
        if timesteps is None:
            timesteps = component.shape[0]
        elif component.shape[0] != timesteps:
            raise ValueError("All pose components must share the same sequence length.")
        arrays.append(component.reshape(component.shape[0], -1))

    if not arrays or timesteps is None:
        raise ValueError("No pose components were available to flatten.")

    flattened = np.concatenate(arrays, axis=1).astype(np.float32)
    return normalize_pose_embeddings(flattened) if normalize else flattened


class PoseToTextDataset(Dataset[Dict[str, Any]]):
    """Load pose archives and labels for fine-tuning or evaluation."""

    def __init__(
        self,
        annotations: pd.DataFrame,
        pose_root: Optional[Path] = None,
        include_face: bool = False,
        augmentation: Optional[PoseAugmentationConfig] = None,
        normalize: bool = True,
    ) -> None:
        if annotations.empty:
            raise ValueError("annotations must not be empty.")
        if "pose_path" not in annotations.columns or "gloss" not in annotations.columns:
            raise KeyError("annotations must contain 'pose_path' and 'gloss' columns.")

        self.annotations = annotations.reset_index(drop=True)
        self.pose_root = None if pose_root is None else Path(pose_root)
        self.include_face = include_face
        self.augmentation = augmentation or PoseAugmentationConfig()
        self.normalize = normalize
        self.components = ("body", "left_hand", "right_hand", "face") if include_face else ("body", "left_hand", "right_hand")

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        pose_root: Optional[Path] = None,
        include_face: bool = False,
        augmentation: Optional[PoseAugmentationConfig] = None,
        normalize: bool = True,
    ) -> "PoseToTextDataset":
        """Build the dataset from a CSV manifest."""

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {csv_path}")

        frame = pd.read_csv(csv_path)
        return cls(
            annotations=frame,
            pose_root=pose_root,
            include_face=include_face,
            augmentation=augmentation,
            normalize=normalize,
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def _resolve_pose_path(self, raw_path: str) -> Path:
        pose_path = Path(raw_path)
        if not pose_path.is_absolute() and self.pose_root is not None:
            pose_path = self.pose_root / pose_path
        return pose_path

    def _load_pose_archive(self, pose_path: Path) -> Dict[str, np.ndarray]:
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose archive not found: {pose_path}")

        try:
            archive = np.load(pose_path, allow_pickle=False)
        except ValueError as exc:
            raise ValueError(f"Pose archive is unreadable or corrupt: {pose_path}") from exc

        components: Dict[str, np.ndarray] = {}
        for component_name in self.components:
            if component_name in archive:
                components[component_name] = archive[component_name].astype(np.float32)

        if "body" not in components or "left_hand" not in components or "right_hand" not in components:
            raise KeyError(f"Pose archive missing required body/hand components: {pose_path}")
        if self.include_face and "face" not in components:
            LOGGER.warning("Face landmarks requested but missing in %s; continuing without face", pose_path)

        return components

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.annotations.iloc[index]
        pose_path = self._resolve_pose_path(str(row["pose_path"]))
        pose_components = self._load_pose_archive(pose_path)
        pose_components = augment_pose_sequence(pose_components, self.augmentation)
        pose_features = flatten_pose_components(
            pose_components,
            include_components=self.components,
            normalize=self.normalize,
        )

        sample_id = str(row["sample_id"]) if "sample_id" in row else pose_path.stem
        return {
            "sample_id": sample_id,
            "gloss": str(row["gloss"]),
            "pose_features": torch.from_numpy(pose_features),
            "sequence_length": int(pose_features.shape[0]),
            "pose_path": str(pose_path),
        }


def collate_pose_text_batch(batch: Sequence[MutableMapping[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length pose sequences and keep labels aligned."""

    if not batch:
        raise ValueError("batch must not be empty.")

    sequence_lengths = torch.tensor([int(item["sequence_length"]) for item in batch], dtype=torch.long)
    feature_dims = {int(item["pose_features"].shape[1]) for item in batch}
    if len(feature_dims) != 1:
        raise ValueError(f"Expected one pose feature dimension per batch, found {sorted(feature_dims)}")

    max_length = int(sequence_lengths.max().item())
    feature_dim = next(iter(feature_dims))
    padded = torch.zeros((len(batch), max_length, feature_dim), dtype=torch.float32)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        features = item["pose_features"]
        length = min(features.shape[0], max_length)
        padded[batch_index, :length] = features[:length]
        attention_mask[batch_index, :length] = True

    return {
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "texts": [str(item["gloss"]) for item in batch],
        "pose_features": padded,
        "pose_attention_mask": attention_mask,
        "sequence_lengths": sequence_lengths,
    }
