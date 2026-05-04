#!/usr/bin/env python3
"""Create compact Unsloth Dashboard training exports from raw pose JSONL.

The raw upload JSONL is mechanically valid but too long for practical dashboard
training: each sample is roughly 12k-15k Gemma tokens. This script creates
compact, quantized text encodings that preserve pose structure while fitting
smaller context lengths.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.q64_encoding import ALPHABET, encode_frames_q64, encode_summary_q64


def parse_pose_input(text: str) -> tuple[str, list[list[float]]]:
    sample_id = "unknown"
    pose_text = None
    for line in text.splitlines():
        if line.startswith("sample_id="):
            sample_id = line.split("=", 1)[1].strip()
        elif line.startswith("pose="):
            pose_text = line.split("=", 1)[1].strip()
    if pose_text is None:
        raise ValueError("input missing pose= line")

    frames: list[list[float]] = []
    for frame in pose_text.split(";"):
        frame = frame.strip()
        if not frame:
            continue
        frames.append([float(x) for x in frame.split(",") if x])
    return sample_id, frames


def build_record(obj: dict, mode: str) -> dict:
    sample_id, frames = parse_pose_input(obj["input"])
    features = min(len(f) for f in frames) if frames else 0

    if mode == "q64_full":
        pose = encode_frames_q64(frames, stride=1)
        input_text = (
            f"sample_id={sample_id}\n"
            f"encoding=q64_full clip=4 alphabet={ALPHABET}\n"
            f"frames={len(frames)} features_per_frame={features}\n"
            f"pose_q64={pose}"
        )
    elif mode == "q64_stride2":
        pose = encode_frames_q64(frames, stride=2)
        input_text = (
            f"sample_id={sample_id}\n"
            f"encoding=q64_stride2 clip=4 alphabet={ALPHABET}\n"
            f"frames_used={len(frames[::2])} original_frames={len(frames)} features_per_frame={features}\n"
            f"pose_q64={pose}"
        )
    elif mode == "q64_summary":
        pose = encode_summary_q64(frames)
        input_text = (
            f"sample_id={sample_id}\n"
            f"encoding=q64_summary clip=4 alphabet={ALPHABET}\n"
            f"original_frames={len(frames)} features_per_frame={features}\n"
            f"{pose}"
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    return {
        "instruction": "Classify this compact ASL pose encoding into its WLASL gloss. Reply with only the gloss word.",
        "input": input_text,
        "output": obj["output"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/exports/asl_unsloth_pose_train_upload.jsonl",
        help="Raw instruction/input/output JSONL export",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/exports",
        help="Output directory",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    modes = ["q64_full", "q64_stride2", "q64_summary"]
    manifest = {
        "source": str(input_path.resolve()),
        "samples": len(rows),
        "modes": {},
        "notes": [
            "q64_full keeps all 12 frames but replaces decimal floats with quantized characters.",
            "q64_stride2 keeps every other frame for lower context usage.",
            "q64_summary stores per-feature mean + last-first motion delta; smallest and most dashboard-friendly.",
            "All encodings quantize values clipped to [-4, 4] into 64 bins using the recorded alphabet.",
        ],
    }

    for mode in modes:
        records = [build_record(obj, mode) for obj in rows]
        jsonl_path = out_dir / f"asl_unsloth_pose_train_{mode}.jsonl"
        csv_path = out_dir / f"asl_unsloth_pose_train_{mode}.csv"
        with jsonl_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["instruction", "input", "output"])
            writer.writeheader()
            writer.writerows(records)
        manifest["modes"][mode] = {
            "jsonl": str(jsonl_path.resolve()),
            "csv": str(csv_path.resolve()),
            "avg_chars": round(mean(len(r["instruction"]) + len(r["input"]) + len(r["output"]) for r in records)),
        }

    manifest_path = out_dir / "asl_unsloth_pose_train_compact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
