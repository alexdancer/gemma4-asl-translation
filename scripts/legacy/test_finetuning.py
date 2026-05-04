"""Fast smoke test for the Gemma ASL fine-tuning pipeline.

The default path loads the real Gemma 4 2B-E2B model through the project loader.
Use ``--mock-model`` when running on CPU-only CI or before Hugging Face/Unsloth
credentials are configured; the same data, forward, optimization, and checkpoint
steps still run.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pose_to_text_dataset import PoseToTextDataset, collate_pose_text_batch
from src.models.gemma_finetune import (
    FineTuneConfig,
    TrainingHistory,
    _prepare_model_inputs,
    _safe_model_forward,
    build_cosine_warmup_scheduler,
    save_training_checkpoint,
)
from src.models.gemma_loader import load_gemma_4_2b_e2b

LOGGER = logging.getLogger("asl_finetune_smoke")
SYNTHETIC_GLOSSES = ("hello", "thanks", "yes", "no", "please", "more", "help", "stop")


@dataclass
class SmokeResult:
    """Small result object used to print a readable final summary."""

    name: str
    passed: bool
    detail: str


class TinyTokenizer:
    """Character-level tokenizer for the ``--mock-model`` smoke path."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self) -> None:
        alphabet = list("abcdefghijklmnopqrstuvwxyz ")
        self._char_to_id = {char: index + 2 for index, char in enumerate(alphabet)}
        self._id_to_char = {index: char for char, index in self._char_to_id.items()}
        self._id_to_char[0] = ""
        self._id_to_char[1] = ""

    def __call__(self, texts: Iterable[str], padding: bool, truncation: bool, return_tensors: str) -> Dict[str, torch.Tensor]:
        encoded = [[self._char_to_id.get(char.lower(), 1) for char in text] + [1] for text in texts]
        max_length = max(len(row) for row in encoded)
        padded = [row + [self.pad_token_id] * (max_length - len(row)) for row in encoded]
        attention = [[1 if token != self.pad_token_id else 0 for token in row] for row in padded]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return "".join(self._id_to_char.get(int(token_id), "") for token_id in token_ids)

    def save_pretrained(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tiny_tokenizer.txt").write_text("mock tokenizer for fine-tuning smoke tests\n", encoding="utf-8")


class TinyCausalLM(torch.nn.Module):
    """A tiny language model that has the same loss/logits contract as Gemma."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 24) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> Any:
        logits = self.proj(self.embedding(input_ids))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=0,
            )
        return type("TinyOutput", (), {"loss": loss, "logits": logits})()

    def save_pretrained(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "tiny_model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast ASL fine-tuning smoke test.")
    parser.add_argument("--max-samples", type=int, default=8, help="Dataset subset size, capped at 50.")
    parser.add_argument("--batch-size", type=int, default=2, help="Small batch size for fast smoke runs.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/gemma_asl_smoke"))
    parser.add_argument("--manifest", type=Path, default=None, help="Optional CSV with pose_path and gloss columns.")
    parser.add_argument("--pose-root", type=Path, default=None, help="Optional root for relative pose paths.")
    parser.add_argument("--mock-model", action="store_true", help="Use a tiny local model instead of downloading Gemma.")
    parser.add_argument("--keep-output", action="store_true", help="Do not clear the smoke output directory first.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def find_manifest(explicit_manifest: Path | None) -> Path | None:
    """Find an existing training manifest, preferring explicit user input."""

    if explicit_manifest is not None:
        if not explicit_manifest.exists():
            raise FileNotFoundError(f"Requested manifest does not exist: {explicit_manifest}")
        return explicit_manifest

    candidates = (
        Path("data/processed/training_pairs/train.csv"),
        Path("data/processed/splits/train.csv"),
        Path("data/processed/splits/val.csv"),
    )
    return next((path for path in candidates if path.exists()), None)


def create_synthetic_dataset(root: Path, sample_count: int) -> Tuple[Path, Path]:
    """Create a tiny pose dataset when extracted WLASL features are not present."""

    pose_root = root / "poses"
    pose_root.mkdir(parents=True, exist_ok=True)
    rows: List[Mapping[str, str]] = []
    rng = np.random.default_rng(42)

    for index in range(sample_count):
        frames = int(rng.integers(6, 12))
        gloss = SYNTHETIC_GLOSSES[index % len(SYNTHETIC_GLOSSES)]
        pose_path = pose_root / f"sample_{index:03d}.npz"
        np.savez(
            pose_path,
            body=rng.normal(size=(frames, 17, 3)).astype(np.float32),
            left_hand=rng.normal(size=(frames, 21, 3)).astype(np.float32),
            right_hand=rng.normal(size=(frames, 21, 3)).astype(np.float32),
        )
        rows.append({"sample_id": f"synthetic_{index:03d}", "pose_path": pose_path.name, "gloss": gloss})

    manifest = root / "synthetic_train.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest, pose_root


def load_dataset(args: argparse.Namespace, run_dir: Path) -> Tuple[PoseToTextDataset, Path | None]:
    """Load up to 50 examples through the real PoseToTextDataset pipeline."""

    max_samples = max(1, min(args.max_samples, 50))
    manifest = find_manifest(args.manifest)
    pose_root = args.pose_root
    if manifest is None:
        LOGGER.warning("No processed manifest found; creating %d synthetic pose samples.", max_samples)
        manifest, pose_root = create_synthetic_dataset(run_dir / "synthetic_data", max_samples)

    dataset = PoseToTextDataset.from_csv(manifest, pose_root=pose_root, include_face=False, normalize=True)
    subset_size = min(max_samples, len(dataset))
    if subset_size < len(dataset):
        dataset = Subset(dataset, list(range(subset_size)))  # type: ignore[assignment]

    LOGGER.info("Loaded dataset subset: %d samples from %s", subset_size, manifest)
    return dataset, manifest


def load_model(use_mock_model: bool) -> Tuple[Any, Any]:
    """Load either real Gemma or a tiny contract-compatible model."""

    if use_mock_model:
        LOGGER.info("Loading mock model for local smoke validation.")
        return TinyCausalLM(), TinyTokenizer()

    LOGGER.info("Loading Gemma 4 2B-E2B through src.models.gemma_loader.load_gemma_4_2b_e2b.")
    try:
        return load_gemma_4_2b_e2b(lora_rank=4, load_in_4bit=torch.cuda.is_available(), max_seq_length=128)
    except Exception as exc:
        raise RuntimeError(
            "Gemma load failed. Verify Hugging Face access, Unsloth installation, and available VRAM. "
            "For a CPU-only pipeline check, rerun with --mock-model."
        ) from exc


def run_one_epoch(model: Any, tokenizer: Any, dataloader: DataLoader, output_dir: Path) -> Tuple[List[float], float, float, Path]:
    """Run one fast epoch, verify forward/backward, and save a checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    model.train()

    config = FineTuneConfig(
        output_dir=output_dir,
        learning_rate=1e-3,
        warmup_steps=0,
        num_epochs=1,
        max_grad_norm=1.0,
        log_every_steps=1,
        checkpoint_every_epochs=1,
        device=str(device),
    )
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate)
    scheduler = build_cosine_warmup_scheduler(optimizer, warmup_steps=0, total_steps=max(1, len(dataloader)))
    losses: List[float] = []
    probe_batch = dict(next(iter(dataloader)))

    with torch.no_grad():
        probe_inputs = _prepare_model_inputs(probe_batch, tokenizer, device)
        initial_probe_loss = float(_safe_model_forward(model, probe_inputs).loss.detach().cpu().item())

    optimizer.zero_grad(set_to_none=True)
    for step, raw_batch in enumerate(dataloader, start=1):
        batch = dict(raw_batch)
        model_inputs = _prepare_model_inputs(batch, tokenizer, device)
        outputs = _safe_model_forward(model, model_inputs)
        loss = getattr(outputs, "loss", None)
        logits = getattr(outputs, "logits", None)
        if loss is None or logits is None:
            raise RuntimeError("Forward pass failed: model output must include both loss and logits.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu().item()))
        LOGGER.info("step=%d/%d loss=%.4f lr=%.6g", step, len(dataloader), losses[-1], scheduler.get_last_lr()[0])

    model.eval()
    with torch.no_grad():
        probe_inputs = _prepare_model_inputs(probe_batch, tokenizer, device)
        final_probe_loss = float(_safe_model_forward(model, probe_inputs).loss.detach().cpu().item())

    checkpoint_dir = save_training_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        epoch=0,
        global_step=len(losses),
        history=TrainingHistory(train_loss=[float(sum(losses) / len(losses))], learning_rates=[scheduler.get_last_lr()[0]]),
        checkpoint_name="smoke-checkpoint",
    )
    return losses, initial_probe_loss, final_probe_loss, checkpoint_dir


def record_result(results: List[SmokeResult], name: str, passed: bool, detail: str) -> None:
    results.append(SmokeResult(name=name, passed=passed, detail=detail))
    LOGGER.info("%s %s - %s", "PASS" if passed else "FAIL", name, detail)


def main() -> int:
    configure_logging()
    args = parse_args()
    output_dir = args.output_dir.resolve()
    run_dir = output_dir / "_smoke_inputs"
    results: List[SmokeResult] = []
    start_time = time.perf_counter()

    if output_dir.exists() and not args.keep_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, tokenizer = load_model(args.mock_model)
        record_result(results, "model_loads", True, "model and tokenizer loaded")

        dataset, manifest = load_dataset(args, run_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pose_text_batch)
        first_batch = next(iter(dataloader))
        record_result(
            results,
            "data_pipeline",
            "pose_features" in first_batch and "texts" in first_batch,
            f"batch pose shape={tuple(first_batch['pose_features'].shape)} manifest={manifest}",
        )

        losses, initial_probe_loss, final_probe_loss, checkpoint_dir = run_one_epoch(model, tokenizer, dataloader, output_dir)
        record_result(results, "forward_and_train", len(losses) > 0, f"ran {len(losses)} optimization steps")

        loss_decreased = final_probe_loss <= initial_probe_loss
        record_result(
            results,
            "loss_decreases",
            loss_decreased,
            f"same-batch before={initial_probe_loss:.4f} after={final_probe_loss:.4f}",
        )

        checkpoint_ok = (checkpoint_dir / "trainer_state.pt").exists()
        record_result(results, "checkpoint_saved", checkpoint_ok, f"checkpoint={checkpoint_dir}")
    except Exception as exc:
        LOGGER.exception("Fine-tuning smoke test failed: %s", exc)
        record_result(results, "unexpected_error", False, str(exc))

    elapsed = time.perf_counter() - start_time
    print("\nFine-tuning smoke test summary")
    print("=" * 36)
    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.detail}")
    print(f"Runtime: {elapsed:.1f}s")

    return 0 if results and all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
