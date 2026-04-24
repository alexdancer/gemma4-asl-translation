"""Utilities for downloading and loading the WLASL dataset metadata."""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import pandas as pd

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - fallback for lightweight environments
    class Dataset:  # type: ignore[override]
        """Minimal Dataset fallback used when PyTorch is unavailable."""

        pass

LOGGER = logging.getLogger(__name__)

DEFAULT_WLASL_METADATA_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/main/start_kit/WLASL_v0.3.json"
)


@dataclass(frozen=True)
class WLASLVideoRecord:
    """Structured representation of one WLASL video example."""

    sample_id: str
    gloss: str
    split: str
    signer_id: Optional[int]
    video_id: str
    video_path: Optional[str]
    fps: Optional[float]
    frame_start: Optional[int]
    frame_end: Optional[int]
    raw_metadata: Dict[str, Any]


def download_wlasl_metadata(
    output_path: Path,
    url: str = DEFAULT_WLASL_METADATA_URL,
    overwrite: bool = False,
) -> Path:
    """Download WLASL metadata from GitHub.

    Parameters
    ----------
    output_path:
        Local path where the metadata JSON should be saved.
    url:
        Source URL for the metadata file.
    overwrite:
        Whether to overwrite an existing file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        LOGGER.info("WLASL metadata already exists at %s", output_path)
        return output_path

    LOGGER.info("Downloading WLASL metadata from %s", url)
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as exc:  # pragma: no cover - exercised through integration
        raise RuntimeError(f"Failed to download WLASL metadata from {url}") from exc

    LOGGER.info("Saved WLASL metadata to %s", output_path)
    return output_path


def load_wlasl_metadata(metadata_path: Path, video_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and flatten the WLASL metadata JSON into a DataFrame."""

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"WLASL metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records: List[Dict[str, Any]] = []
    for entry in payload:
        gloss = entry.get("gloss")
        for instance in entry.get("instances", []):
            video_id = str(instance.get("video_id"))
            split = str(instance.get("split", "unspecified"))
            sample_id = f"{gloss}_{video_id}"
            local_path = None
            if video_dir is not None:
                candidate = Path(video_dir) / f"{video_id}.mp4"
                local_path = str(candidate) if candidate.exists() else None

            records.append(
                {
                    "sample_id": sample_id,
                    "gloss": gloss,
                    "split": split,
                    "signer_id": instance.get("signer_id"),
                    "video_id": video_id,
                    "video_path": local_path,
                    "fps": instance.get("fps"),
                    "frame_start": instance.get("frame_start"),
                    "frame_end": instance.get("frame_end"),
                    "metadata": instance,
                }
            )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No video instances found in metadata: {metadata_path}")

    LOGGER.info("Loaded %d WLASL samples covering %d glosses", len(frame), frame["gloss"].nunique())
    return frame


def parse_record(row: pd.Series) -> WLASLVideoRecord:
    """Convert a DataFrame row into a typed WLASL video record."""

    return WLASLVideoRecord(
        sample_id=str(row["sample_id"]),
        gloss=str(row["gloss"]),
        split=str(row["split"]),
        signer_id=None if pd.isna(row["signer_id"]) else int(row["signer_id"]),
        video_id=str(row["video_id"]),
        video_path=None if pd.isna(row["video_path"]) else str(row["video_path"]),
        fps=None if pd.isna(row["fps"]) else float(row["fps"]),
        frame_start=None if pd.isna(row["frame_start"]) else int(row["frame_start"]),
        frame_end=None if pd.isna(row["frame_end"]) else int(row["frame_end"]),
        raw_metadata=dict(row["metadata"]),
    )


def load_video_frames(video_path: Path, max_frames: Optional[int] = None) -> List[Any]:
    """Load a video file into a list of BGR image frames."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    frames: List[Any] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames were decoded from video: {video_path}")

    LOGGER.debug("Loaded %d frames from %s", len(frames), video_path)
    return frames


class WLASLDataset(Dataset):
    """PyTorch dataset wrapper for WLASL metadata and local video files."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        max_frames: Optional[int] = None,
        return_metadata: bool = True,
    ) -> None:
        if metadata.empty:
            raise ValueError("Metadata DataFrame is empty.")
        self.metadata = metadata.reset_index(drop=True)
        self.max_frames = max_frames
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.metadata.iloc[index]
        record = parse_record(row)
        if record.video_path is None:
            raise FileNotFoundError(
                f"Missing local video path for sample {record.sample_id}. "
                "Populate data/raw/wlasl/videos/ before loading frames."
            )

        frames = load_video_frames(record.video_path, max_frames=self.max_frames)
        sample = {
            "sample_id": record.sample_id,
            "gloss": record.gloss,
            "frames": frames,
        }
        if self.return_metadata:
            sample["metadata"] = record
        return sample

    @classmethod
    def from_paths(
        cls,
        metadata_path: Path,
        video_dir: Path,
        max_frames: Optional[int] = None,
        split: Optional[str] = None,
    ) -> "WLASLDataset":
        """Build a dataset from metadata and a local video directory."""

        metadata = load_wlasl_metadata(metadata_path=metadata_path, video_dir=video_dir)
        if split is not None:
            metadata = metadata.loc[metadata["split"] == split].reset_index(drop=True)
            if metadata.empty:
                raise ValueError(f"No samples found for split '{split}'")
        return cls(metadata=metadata, max_frames=max_frames)


def iter_records(metadata: pd.DataFrame) -> Iterable[WLASLVideoRecord]:
    """Yield typed records from a metadata DataFrame."""

    for _, row in metadata.iterrows():
        yield parse_record(row)
