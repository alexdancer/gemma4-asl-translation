"""Prepare pose/text training manifests from raw WLASL assets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.extract_poses_batch import configure_logging, extract_pose_archives
from src.data.create_splits import (
    TOP50_CONTRACT_VERSION,
    create_top50_split_artifacts,
    save_splits,
    save_top50_contract,
    stratified_split,
)
from src.data.wlasl_loader import download_wlasl_metadata, load_wlasl_metadata

LOGGER = logging.getLogger(__name__)


def build_training_pairs(metadata: pd.DataFrame, pose_dir: Path) -> pd.DataFrame:
    """Create a training-pairs DataFrame containing pose archive paths and labels."""

    frame = metadata.copy()
    frame["pose_path"] = frame["sample_id"].map(lambda sample_id: str((pose_dir / f"{sample_id}.npz").as_posix()))
    frame = frame[frame["pose_path"].map(lambda value: Path(value).exists())].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No pose archives were found. Run extraction before building training pairs.")

    columns = ["sample_id", "gloss", "split", "signer_id", "video_id", "video_path", "pose_path"]
    existing_columns = [column for column in columns if column in frame.columns]
    return frame[existing_columns]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-url", type=str, default=None, help="Optional override for the WLASL metadata URL.")
    parser.add_argument("--metadata-path", type=Path, default=Path("data/raw/wlasl/WLASL_v0.3.json"), help="Local metadata JSON path.")
    parser.add_argument("--video-dir", type=Path, default=Path("data/raw/wlasl/videos"), help="Directory containing WLASL videos.")
    parser.add_argument("--pose-dir", type=Path, default=Path("data/processed/poses"), help="Directory containing extracted pose archives.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/training_pairs"), help="Output directory for CSV manifests.")
    parser.add_argument("--top50-contract", type=Path, default=Path("data/contracts/asl_top50_glosses_v1.json"), help="Versioned Top-50 gloss contract path.")
    parser.add_argument("--top50-output-dir", type=Path, default=Path("data/processed/splits/top50"), help="Output directory for Top-50 random and signer-independent split artifacts.")
    parser.add_argument("--top50-only", action="store_true", help="Only generate Top-50 split artifacts from metadata; do not require pose archives.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument("--extract-missing-poses", action="store_true", help="Extract pose archives before creating manifests.")
    parser.add_argument("--include-face", action="store_true", help="Include face landmarks during extraction when enabled.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap during extraction.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    """Run the training-data preparation CLI."""

    args = parse_args()
    configure_logging(args.log_level)

    metadata_path = download_wlasl_metadata(
        output_path=args.metadata_path,
        url=args.metadata_url or "https://raw.githubusercontent.com/dxli94/WLASL/main/start_kit/WLASL_v0.3.json",
        overwrite=False,
    )
    metadata = load_wlasl_metadata(metadata_path, video_dir=args.video_dir)

    save_top50_contract(args.top50_contract, overwrite=False)
    top50_outputs = create_top50_split_artifacts(
        metadata,
        output_dir=args.top50_output_dir,
        contract_path=args.top50_contract,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    LOGGER.info("Saved Top-50 %s split artifacts: %s", TOP50_CONTRACT_VERSION, top50_outputs)
    if args.top50_only:
        return

    if args.extract_missing_poses:
        summary = extract_pose_archives(
            metadata=metadata,
            output_dir=args.pose_dir,
            include_face=args.include_face,
            max_frames=args.max_frames,
            overwrite=False,
        )
        LOGGER.info(
            "Pose extraction summary during data prep: written=%d skipped=%d errors=%d",
            summary["written"],
            summary["skipped"],
            len(summary["errors"]),
        )

    pairs = build_training_pairs(metadata, pose_dir=args.pose_dir)
    train_df, val_df, test_df = stratified_split(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs_path = args.output_dir / "all_pairs.csv"
    pairs.to_csv(all_pairs_path, index=False)

    split_paths: Dict[str, Path] = save_splits(train_df, val_df, test_df, output_dir=args.output_dir)
    LOGGER.info("Saved all training pairs to %s", all_pairs_path)
    LOGGER.info("Saved split manifests: %s", split_paths)


if __name__ == "__main__":
    main()
