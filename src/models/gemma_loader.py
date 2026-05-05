"""Gemma 4 E2B loading and fine-tuning utilities."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


def load_gemma_4_2b_e2b(
    lora_rank: int = 16,
    load_in_4bit: bool = True,
    max_seq_length: int = 2048,
) -> Tuple[object, AutoTokenizer]:
    """
    Load Gemma 4 E2B (instruction-tuned) with LoRA adapter for fine-tuning.

    This uses Unsloth for optimized loading and LoRA integration.
    Supports 4-bit quantization for memory efficiency on consumer GPUs.

    Args:
        lora_rank: LoRA rank (16 recommended, balance between speed and quality)
        load_in_4bit: Whether to quantize model to 4-bit (requires ~6-8GB VRAM)
        max_seq_length: Maximum sequence length for tokenization

    Returns:
        Tuple of (model, tokenizer) ready for training or inference

    Raises:
        ImportError: If unsloth is not installed
        RuntimeError: If HuggingFace authentication fails

    Example:
        >>> model, tokenizer = load_gemma_4_2b_e2b(lora_rank=16)
        >>> print(model)  # Shows LoRA config
    """

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "unsloth is required for optimized Gemma 4 loading. "
            "Install with: pip install unsloth"
        ) from exc

    LOGGER.info("Loading Gemma 4 E2B (instruction-tuned)")
    LOGGER.info(f"LoRA rank: {lora_rank}, 4-bit quantization: {load_in_4bit}")

    # Load base model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-4-E2B-it",
        max_seq_length=max_seq_length,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=load_in_4bit,
    )

    LOGGER.info(f"Loaded base model, parameters: {model.num_parameters():,}")

    # Attach LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"LoRA adapter added, trainable parameters: {trainable:,}")

    # Set tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_model_for_inference(model: object) -> object:
    """
    Prepare model for inference (generation mode).

    Must be called after training and before generating text.

    Args:
        model: Loaded model with LoRA adapter

    Returns:
        Model ready for inference

    Example:
        >>> model = prepare_model_for_inference(model)
        >>> outputs = model.generate(...)
    """

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError("unsloth is required") from exc

    LOGGER.info("Preparing model for inference")
    model = FastLanguageModel.for_inference(model)
    return model


def save_checkpoint(
    model: object,
    tokenizer: AutoTokenizer,
    checkpoint_path: str,
) -> None:
    """
    Save model and tokenizer checkpoint.

    Saves both the LoRA adapter weights and the tokenizer.

    Args:
        model: Model with LoRA adapter
        tokenizer: Tokenizer to save
        checkpoint_path: Directory path to save checkpoint

    Raises:
        OSError: If checkpoint directory cannot be created

    Example:
        >>> save_checkpoint(model, tokenizer, "models/checkpoints/epoch-1")
    """

    try:
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        LOGGER.info(f"Saved checkpoint to {checkpoint_path}")
    except OSError as exc:
        LOGGER.error(f"Failed to save checkpoint: {exc}")
        raise


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[object, AutoTokenizer]:
    """
    Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Directory path of saved checkpoint

    Returns:
        Tuple of (model, tokenizer) loaded from checkpoint

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist

    Example:
        >>> model, tokenizer = load_checkpoint("models/checkpoints/epoch-1")
    """

    from transformers import AutoModelForCausalLM

    try:
        LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        LOGGER.info("Checkpoint loaded successfully")
        return model, tokenizer
    except FileNotFoundError as exc:
        LOGGER.error(f"Checkpoint not found: {checkpoint_path}")
        raise


