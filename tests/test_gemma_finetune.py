"""Unit tests for Gemma ASL fine-tuning utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from src.models.gemma_finetune import (
    FineTuneConfig,
    TrainingHistory,
    build_cosine_warmup_scheduler,
    load_training_checkpoint,
    save_training_checkpoint,
)


class DummyTokenizer:
    """Minimal tokenizer object with the save contract used by checkpoints."""

    def save_pretrained(self, checkpoint_dir: Path | str) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyModel(torch.nn.Module):
    """Small trainable model with the save contract used by checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def save_pretrained(self, checkpoint_dir: Path | str) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_dir / "model.pt")


def test_finetune_config_defaults_are_training_safe(tmp_path: Path) -> None:
    config = FineTuneConfig(output_dir=tmp_path / "checkpoints")

    assert config.output_dir == tmp_path / "checkpoints"
    assert config.learning_rate == pytest.approx(2e-4)
    assert config.gradient_accumulation_steps == 1
    assert config.max_grad_norm == pytest.approx(1.0)
    assert config.metric_for_best_model == "val_accuracy"
    assert config.maximize_metric is True


def test_gradient_clipping_caps_large_gradients() -> None:
    model = torch.nn.Linear(4, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)

    max_norm = 0.25
    original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    clipped_norm = torch.linalg.vector_norm(torch.stack([parameter.grad.norm() for parameter in model.parameters()]))

    assert original_norm > max_norm
    assert float(clipped_norm) <= max_norm + 1e-5


def test_cosine_warmup_scheduler_warms_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW([parameter], lr=1.0)
    scheduler = build_cosine_warmup_scheduler(optimizer, warmup_steps=2, total_steps=6)

    observed_lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        observed_lrs.append(scheduler.get_last_lr()[0])

    assert observed_lrs[0] > 0.0
    assert observed_lrs[1] >= observed_lrs[0]
    assert observed_lrs[-1] < observed_lrs[1]
    assert all(lr >= 0.0 for lr in observed_lrs)


def test_checkpoint_save_and_load_restores_trainer_state(tmp_path: Path) -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()
    optimizer = AdamW(model.parameters(), lr=0.01)
    scheduler = build_cosine_warmup_scheduler(optimizer, warmup_steps=0, total_steps=2)

    inputs = torch.ones(2, 2)
    loss = model.layer(inputs).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    history = TrainingHistory(train_loss=[1.0], val_loss=[1.2], best_metric=0.4)
    checkpoint_dir = save_training_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        config=FineTuneConfig(output_dir=tmp_path),
        epoch=0,
        global_step=1,
        history=history,
        checkpoint_name="unit-checkpoint",
    )

    new_model = DummyModel()
    new_optimizer = AdamW(new_model.parameters(), lr=0.01)
    new_scheduler = build_cosine_warmup_scheduler(new_optimizer, warmup_steps=0, total_steps=2)
    trainer_state = load_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=new_model,
        tokenizer=tokenizer,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        map_location="cpu",
    )

    assert (checkpoint_dir / "trainer_state.pt").exists()
    assert (checkpoint_dir / "model.pt").exists()
    assert (checkpoint_dir / "tokenizer.json").exists()
    assert trainer_state["epoch"] == 0
    assert trainer_state["global_step"] == 1
    assert trainer_state["history"]["best_metric"] == pytest.approx(0.4)
    assert new_optimizer.state_dict()["state"]
