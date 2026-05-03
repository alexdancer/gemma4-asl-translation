#!/usr/bin/env python3
"""Create Top-50 random-stratified Unsloth Dashboard JSONL splits.

Input records must use the instruction/input/output shape used by Unsloth
Dashboard. The gloss label is read from `output`. We select the 50 most common
labels, then split each label independently to preserve class coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "input", "output"])
        writer.writeheader()
        writer.writerows(records)


def sample_id(record: dict) -> str:
    match = re.search(r"^sample_id=([^\n]+)", record.get("input", ""), flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def split_group(records: list[dict], rng: random.Random) -> tuple[list[dict], list[dict], list[dict]]:
    items = list(records)
    rng.shuffle(items)
    n = len(items)
    if n < 3:
        raise ValueError(f"Need at least 3 examples per class for train/val/test; got {n}")

    # Preserve at least one validation and one test example per class.
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15)))
    if n_train + n_val > n - 1:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val
    if n_test < 1:
        raise AssertionError("split math produced empty test set")

    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/exports/asl_unsloth_pose_train_q64_full.jsonl",
        help="Compact q64_full JSONL source",
    )
    parser.add_argument("--out-dir", default="data/processed/exports")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="asl_unsloth_pose_train_q64_full_top50")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    counts = Counter(record["output"] for record in records)
    top_labels = [label for label, _count in counts.most_common(args.top_k)]
    top_set = set(top_labels)
    selected = [record for record in records if record["output"] in top_set]

    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in selected:
        by_label[record["output"]].append(record)

    rng = random.Random(args.seed)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    per_label: dict[str, dict] = {}

    for label in top_labels:
        label_train, label_val, label_test = split_group(by_label[label], rng)
        train.extend(label_train)
        val.extend(label_val)
        test.extend(label_test)
        per_label[label] = {
            "total": len(by_label[label]),
            "train": len(label_train),
            "val": len(label_val),
            "test": len(label_test),
        }

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    paths = {
        "train_jsonl": out_dir / f"{args.prefix}_train.jsonl",
        "val_jsonl": out_dir / f"{args.prefix}_val.jsonl",
        "test_jsonl": out_dir / f"{args.prefix}_test.jsonl",
        "train_csv": out_dir / f"{args.prefix}_train.csv",
        "val_csv": out_dir / f"{args.prefix}_val.csv",
        "test_csv": out_dir / f"{args.prefix}_test.csv",
        "manifest": out_dir / f"{args.prefix}_manifest.json",
    }

    write_jsonl(paths["train_jsonl"], train)
    write_jsonl(paths["val_jsonl"], val)
    write_jsonl(paths["test_jsonl"], test)
    write_csv(paths["train_csv"], train)
    write_csv(paths["val_csv"], val)
    write_csv(paths["test_csv"], test)

    split_ids = {
        "train": {sample_id(record) for record in train},
        "val": {sample_id(record) for record in val},
        "test": {sample_id(record) for record in test},
    }
    overlaps = {
        "train_val": sorted(split_ids["train"] & split_ids["val"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "val_test": sorted(split_ids["val"] & split_ids["test"]),
    }

    manifest = {
        "source": str(input_path.resolve()),
        "top_k": args.top_k,
        "seed": args.seed,
        "split_strategy": "random_stratified_70_15_15_with_min_1_val_1_test_per_class",
        "labels": top_labels,
        "counts": {
            "source_total": len(records),
            "selected_total": len(selected),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "per_label": per_label,
        "overlaps": overlaps,
        "files": {name: str(path.resolve()) for name, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    if any(overlaps.values()):
        raise SystemExit("ERROR: split leakage detected")


if __name__ == "__main__":
    main()
