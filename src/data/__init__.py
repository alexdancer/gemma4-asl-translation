"""Data loading, preprocessing, and split generation utilities.

Heavy optional dependencies are imported lazily so lightweight modules such as
live capture can be used without loading pandas, torch, or MediaPipe.
"""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "DEFAULT_WLASL_METADATA_URL": "src.data.wlasl_loader",
    "POSE_COMPONENTS": "src.data.pose_to_text_dataset",
    "PoseAugmentationConfig": "src.data.pose_to_text_dataset",
    "CachedPoseQ64VerificationConfig": "src.data.cached_pose_q64",
    "CachedPoseQ64VerificationResult": "src.data.cached_pose_q64",
    "PoseExtractionError": "src.data.pose_extractor",
    "PoseExtractor": "src.data.pose_extractor",
    "PoseToTextDataset": "src.data.pose_to_text_dataset",
    "WLASLDataset": "src.data.wlasl_loader",
    "WLASLVideoRecord": "src.data.wlasl_loader",
    "augment_pose_sequence": "src.data.pose_to_text_dataset",
    "collate_pose_text_batch": "src.data.pose_to_text_dataset",
    "create_manifest": "src.data.create_splits",
    "download_wlasl_metadata": "src.data.wlasl_loader",
    "flatten_pose_components": "src.data.pose_to_text_dataset",
    "load_video_frames": "src.data.wlasl_loader",
    "load_wlasl_metadata": "src.data.wlasl_loader",
    "normalize_pose_coordinates": "src.data.pose_extractor",
    "parse_record": "src.data.wlasl_loader",
    "save_pose_sequence": "src.data.pose_extractor",
    "save_splits": "src.data.create_splits",
    "stratified_split": "src.data.create_splits",
    "verify_cached_pose_q64": "src.data.cached_pose_q64",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.data' has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
