"""Checkpoint loading seam for mobile export modules."""

from __future__ import annotations

from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_checkpoint(checkpoint_path: str) -> Tuple[object, AutoTokenizer]:
    """Load model and tokenizer from a Hugging Face-style checkpoint directory."""

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer
