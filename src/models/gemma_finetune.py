"""Fine-tuning loop and checkpoint management for Gemma ASL transcription."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from src.models.utils import decode_text_batch, top_k_accuracy, word_accuracy

LOGGER = logging.getLogger(__name__)


class FineTuneError(RuntimeError):
    """Raised when model fine-tuning cannot proceed safely."""


@dataclass(frozen=True)
class FineTuneConfig:
    """Training configuration for Gemma LoRA fine-tuning."""

    output_dir: Path
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_delta: float = 0.0
    log_every_steps: int = 10
    checkpoint_every_epochs: int = 1
    metric_for_best_model: str = "val_accuracy"
    maximize_metric: bool = True
    device: Optional[str] = None


@dataclass
class EpochMetrics:
    """Aggregated metrics for one pass over a dataloader."""

    loss: float
    accuracy: float
    top5_accuracy: float
    steps: int
    examples: int
    elapsed_seconds: float


@dataclass
class TrainingHistory:
    """Serializable training history."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_top5_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_checkpoint: Optional[str] = None
    stopped_early: bool = False


def build_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create a cosine decay schedule with linear warmup."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    warmup_steps = max(0, warmup_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_device(config: FineTuneConfig) -> torch.device:
    """Pick a safe execution device."""

    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(batch: MutableMapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor fields in a batch mapping to the selected device."""

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _prepare_model_inputs(
    batch: Mapping[str, Any],
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Construct model inputs from text labels and optional pose tensors."""

    texts = list(batch["texts"])
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs: Dict[str, Any] = {key: value.to(device) for key, value in tokenized.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    if "pose_features" in batch:
        inputs["pose_features"] = batch["pose_features"].to(device)
    if "pose_attention_mask" in batch:
        inputs["pose_attention_mask"] = batch["pose_attention_mask"].to(device)

    return inputs


def _safe_model_forward(
    model: Any,
    model_inputs: Mapping[str, Any],
) -> Any:
    """Call model forward while tolerating models that ignore pose tensors."""

    try:
        return model(**model_inputs)
    except TypeError as exc:
        if "pose_features" not in model_inputs and "pose_attention_mask" not in model_inputs:
            raise
        LOGGER.debug("Model forward does not accept pose tensors directly: %s", exc)
        fallback_inputs = {
            key: value
            for key, value in model_inputs.items()
            if key not in {"pose_features", "pose_attention_mask"}
        }
        return model(**fallback_inputs)


def _extract_topk_predictions(logits: torch.Tensor, tokenizer: Any, k: int = 5) -> List[List[str]]:
    """Convert final-token logits into top-k candidate strings."""

    if logits.ndim != 3:
        raise ValueError("Expected logits with shape (batch, sequence, vocab).")

    topk_ids = torch.topk(logits[:, -1, :], k=min(k, logits.shape[-1]), dim=-1).indices
    candidates: List[List[str]] = []
    for row in topk_ids.detach().cpu().tolist():
        candidates.append([tokenizer.decode([token_id], skip_special_tokens=True).strip() for token_id in row])
    return candidates


def run_validation(
    model: Any,
    tokenizer: Any,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> EpochMetrics:
    """Evaluate a model on a validation split and compute word metrics."""

    model.eval()
    losses: List[float] = []
    predictions: List[str] = []
    references: List[str] = []
    candidate_predictions: List[List[str]] = []
    steps = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for raw_batch in tqdm(dataloader, desc="Validation", leave=False):
            batch = _move_batch_to_device(dict(raw_batch), device)
            model_inputs = _prepare_model_inputs(batch, tokenizer, device)
            outputs = _safe_model_forward(model, model_inputs)
            loss = getattr(outputs, "loss", None)
            logits = getattr(outputs, "logits", None)
            if loss is None or logits is None:
                raise FineTuneError("Model outputs must expose both 'loss' and 'logits'.")

            losses.append(float(loss.detach().cpu().item()))
            predicted_ids = torch.argmax(logits, dim=-1)
            predictions.extend(decode_text_batch(tokenizer, predicted_ids))
            references.extend(batch["texts"])
            candidate_predictions.extend(_extract_topk_predictions(logits, tokenizer, k=5))
            steps += 1

    elapsed = time.perf_counter() - start_time
    return EpochMetrics(
        loss=float(sum(losses) / max(1, len(losses))),
        accuracy=word_accuracy(predictions, references),
        top5_accuracy=top_k_accuracy(candidate_predictions, references, k=5),
        steps=steps,
        examples=len(references),
        elapsed_seconds=elapsed,
    )


def save_training_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    config: FineTuneConfig,
    epoch: int,
    global_step: int,
    history: TrainingHistory,
    checkpoint_name: str,
) -> Path:
    """Save adapter/tokenizer weights plus trainer state for resuming."""

    checkpoint_dir = Path(config.output_dir) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    state_path = checkpoint_dir / "trainer_state.pt"
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": asdict(config),
            "history": asdict(history),
        },
        state_path,
    )

    LOGGER.info("Saved checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


def load_training_checkpoint(
    checkpoint_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load trainer state from a checkpoint directory."""

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Trainer state file not found: {state_path}")

    if hasattr(model, "load_adapter"):
        try:
            model.load_adapter(str(checkpoint_dir))
        except Exception as exc:
            LOGGER.debug("Adapter-specific load failed, falling back to state-only resume: %s", exc)
    elif hasattr(model, "from_pretrained"):
        LOGGER.debug("Model does not expose adapter loading; assuming caller restored weights separately.")

    if hasattr(tokenizer, "from_pretrained"):
        tokenizer = tokenizer.__class__.from_pretrained(checkpoint_dir)

    try:
        trainer_state = torch.load(state_path, map_location=map_location, weights_only=False)
    except TypeError:
        trainer_state = torch.load(state_path, map_location=map_location)
    if optimizer is not None:
        optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(trainer_state["scheduler_state_dict"])
    return trainer_state


def train_gemma(
    model: Any,
    tokenizer: Any,
    train_dataloader: Iterable[Mapping[str, Any]],
    val_dataloader: Iterable[Mapping[str, Any]],
    config: FineTuneConfig,
    optimizer_factory: Optional[Callable[[Sequence[torch.nn.Parameter]], Optimizer]] = None,
) -> TrainingHistory:
    """Run end-to-end fine-tuning with validation, checkpoints, and early stopping."""

    device = _resolve_device(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        try:
            model.to(device)
        except RuntimeError as exc:
            raise FineTuneError(
                "Failed to move model to CUDA. Check available VRAM or disable 4-bit loading."
            ) from exc

    optimizer = optimizer_factory(model.parameters()) if optimizer_factory else AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_train_steps = max(1, len(train_dataloader) * config.num_epochs // max(1, config.gradient_accumulation_steps))
    scheduler = build_cosine_warmup_scheduler(optimizer, config.warmup_steps, total_train_steps)
    history = TrainingHistory()
    best_metric: Optional[float] = None
    epochs_without_improvement = 0
    global_step = 0

    LOGGER.info(
        "Starting fine-tuning for %d epochs on %s (%d optimization steps)",
        config.num_epochs,
        device,
        total_train_steps,
    )

    for epoch in range(config.num_epochs):
        model.train()
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", leave=False)
        running_losses: List[float] = []
        epoch_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(progress, start=1):
            try:
                batch = _move_batch_to_device(dict(raw_batch), device)
                model_inputs = _prepare_model_inputs(batch, tokenizer, device)
                outputs = _safe_model_forward(model, model_inputs)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    raise FineTuneError("Model outputs must expose a 'loss' field during training.")

                detached_loss = float(loss.detach().cpu().item())
                running_losses.append(detached_loss)
                loss = loss / max(1, config.gradient_accumulation_steps)
                loss.backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise FineTuneError(
                        "CUDA out of memory during training. Reduce batch size, sequence length, or LoRA rank."
                    ) from exc
                raise

            if step % max(1, config.gradient_accumulation_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                if global_step % max(1, config.log_every_steps) == 0:
                    average_loss = sum(running_losses[-config.log_every_steps :]) / min(
                        len(running_losses), config.log_every_steps
                    )
                    LOGGER.info(
                        "epoch=%d step=%d loss=%.4f lr=%.6f",
                        epoch + 1,
                        global_step,
                        average_loss,
                        current_lr,
                    )
                    progress.set_postfix(loss=f"{average_loss:.4f}", lr=f"{current_lr:.2e}")

        epoch_elapsed = time.perf_counter() - epoch_start
        average_train_loss = float(sum(running_losses) / max(1, len(running_losses)))
        history.train_loss.append(average_train_loss)
        history.learning_rates.append(float(scheduler.get_last_lr()[0]))

        val_metrics = run_validation(model, tokenizer, val_dataloader, device=device)
        history.val_loss.append(val_metrics.loss)
        history.val_accuracy.append(val_metrics.accuracy)
        history.val_top5_accuracy.append(val_metrics.top5_accuracy)

        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_accuracy=%.4f val_top5=%.4f epoch_seconds=%.2f val_seconds=%.2f",
            epoch + 1,
            average_train_loss,
            val_metrics.loss,
            val_metrics.accuracy,
            val_metrics.top5_accuracy,
            epoch_elapsed,
            val_metrics.elapsed_seconds,
        )

        tracked_metric = getattr(val_metrics, config.metric_for_best_model.removeprefix("val_"), None)
        if tracked_metric is None:
            tracked_metric = val_metrics.accuracy

        improved = (
            best_metric is None
            or (tracked_metric > best_metric + config.early_stopping_delta if config.maximize_metric else tracked_metric < best_metric - config.early_stopping_delta)
        )
        if improved:
            best_metric = float(tracked_metric)
            epochs_without_improvement = 0
            history.best_metric = best_metric
            best_checkpoint = save_training_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                history=history,
                checkpoint_name="best-checkpoint",
            )
            history.best_checkpoint = str(best_checkpoint)
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % max(1, config.checkpoint_every_epochs) == 0:
            save_training_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                history=history,
                checkpoint_name=f"epoch-{epoch + 1}",
            )

        history_path = Path(config.output_dir) / "training_history.json"
        history_path.write_text(json.dumps(asdict(history), indent=2), encoding="utf-8")

        if epochs_without_improvement >= config.early_stopping_patience:
            LOGGER.info(
                "Early stopping triggered after %d epochs without improvement.",
                epochs_without_improvement,
            )
            history.stopped_early = True
            break

    return history
