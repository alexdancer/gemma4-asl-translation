"""Batch pose extraction for WLASL videos."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.pose_extractor import PoseExtractionError, PoseExtractor, save_pose_sequence
from src.data.wlasl_loader import load_wlasl_metadata

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure process-wide logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def extract_pose_archives(
    metadata: pd.DataFrame,
    output_dir: Path,
    include_face: bool = False,
    max_frames: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Extract pose archives for each metadata row, logging recoverable failures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    error_records: List[Dict[str, str]] = []
    written = 0
    skipped = 0

    extractor = PoseExtractor(include_face=include_face)
    try:
        for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Extracting poses"):
            video_path = getattr(row, "video_path", None)
            sample_id = str(getattr(row, "sample_id"))
            if video_path is None or (isinstance(video_path, float) and pd.isna(video_path)) or str(video_path).strip() == "":
                LOGGER.warning("Skipping %s because video_path is missing", sample_id)
                error_records.append({"sample_id": sample_id, "error": "missing_video_path"})
                continue

            pose_path = output_dir / f"{sample_id}.npz"
            if pose_path.exists() and not overwrite:
                skipped += 1
                continue

            try:
                sequence = extractor.extract_from_video(Path(video_path), max_frames=max_frames)
                metadata_blob = {
                    "sample_id": sample_id,
                    "gloss": str(getattr(row, "gloss")),
                    "split": str(getattr(row, "split", "unknown")),
                    "video_id": str(getattr(row, "video_id", "")),
                }
                save_pose_sequence(sequence, pose_path, metadata=metadata_blob)
                written += 1
            except (FileNotFoundError, RuntimeError, ValueError, PoseExtractionError) as exc:
                LOGGER.error("Failed to extract %s: %s", sample_id, exc)
                error_records.append({"sample_id": sample_id, "error": str(exc)})
    finally:
        extractor.close()

    return {
        "written": written,
        "skipped": skipped,
        "errors": error_records,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", type=Path, required=True, help="Path to flattened WLASL metadata JSON.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing WLASL video files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/poses"), help="Where pose NPZ files are saved.")
    parser.add_argument("--include-face", action="store_true", help="Include face landmarks in extracted archives.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap per video.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract archives even if output already exists.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    """Run the extraction CLI."""

    args = parse_args()
    configure_logging(args.log_level)

    metadata = load_wlasl_metadata(args.metadata_path, video_dir=args.video_dir)
    summary = extract_pose_archives(
        metadata=metadata,
        output_dir=args.output_dir,
        include_face=args.include_face,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )

    error_path = args.output_dir / "extraction_errors.json"
    error_path.write_text(json.dumps(summary["errors"], indent=2), encoding="utf-8")
    LOGGER.info(
        "Pose extraction complete: written=%d skipped=%d errors=%d",
        summary["written"],
        summary["skipped"],
        len(summary["errors"]),
    )


if __name__ == "__main__":
    main()
