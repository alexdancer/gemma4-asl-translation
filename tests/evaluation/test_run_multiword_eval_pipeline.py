from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.run_multiword_eval_pipeline import main


def test_run_multiword_eval_pipeline_end_to_end(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.jsonl"
    predicted_path = tmp_path / "predicted.jsonl"
    merged_out = tmp_path / "merged.jsonl"
    eval_out_dir = tmp_path / "eval"

    expected_path.write_text(
        json.dumps({"sample_id": "s1", "expected_words": ["hello", "world"]}) + "\n",
        encoding="utf-8",
    )
    predicted_path.write_text(
        json.dumps({"sample_id": "s1", "predicted_words": ["hello", "word"]}) + "\n",
        encoding="utf-8",
    )

    status = main(
        [
            "--expected-jsonl",
            str(expected_path),
            "--predicted-jsonl",
            str(predicted_path),
            "--merged-out-jsonl",
            str(merged_out),
            "--eval-out-dir",
            str(eval_out_dir),
        ]
    )

    assert status == 0
    assert merged_out.exists()
    assert (eval_out_dir / "predictions.csv").exists()
    metrics = json.loads((eval_out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["sample_count"] == 1
    assert metrics["word_error_rate"] == 0.5
