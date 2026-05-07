#!/usr/bin/env python3
"""One-command LoRA -> merged HF -> Cactus convert -> validation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Cactus conversion pipeline from LoRA artifacts.")
    parser.add_argument("--base-model", default="google/gemma-4-E4B-it")
    parser.add_argument(
        "--base-model-local-dir",
        default="checkpoints/base_gemma4_e4b",
        help="Local cache directory for downloaded base model",
    )
    parser.add_argument(
        "--download-base-model",
        action="store_true",
        help="Download base model into --base-model-local-dir before merge",
    )
    parser.add_argument(
        "--prefer-local-base",
        action="store_true",
        help="Use --base-model-local-dir as merge base (expects downloaded model files there)",
    )
    parser.add_argument(
        "--lora-dir",
        default="checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline",
        help="Directory containing adapter_model.safetensors",
    )
    parser.add_argument(
        "--auto-base-from-adapter",
        action="store_true",
        help="Read lora_dir/adapter_config.json and use base_model_name_or_path for merge/download",
    )
    parser.add_argument(
        "--strict-base-match",
        action="store_true",
        help="Fail fast if selected base model does not match adapter_config.json base_model_name_or_path",
    )
    parser.add_argument(
        "--merged-out",
        default="checkpoints/gemma4_e4b_merged_hf",
        help="Output directory for merged HF model",
    )
    parser.add_argument(
        "--cactus-out",
        default="checkpoints/gemma4_e4b_cactus_int4",
        help="Output directory for converted Cactus weights",
    )
    parser.add_argument("--precision", default="INT4")
    parser.add_argument("--max-samples", type=int, default=1, help="Validation sample count")
    parser.add_argument("--skip-merge", action="store_true", help="Skip LoRA merge step")
    parser.add_argument("--skip-convert", action="store_true", help="Skip cactus convert step")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation step")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument(
        "--cactus-bin",
        default=os.path.expanduser("~/cactus/venv/bin/cactus"),
        help="Path to cactus binary",
    )
    return parser.parse_args()


def read_adapter_config(lora_dir: Path) -> dict[str, object]:
    config_path = lora_dir / "adapter_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def read_adapter_base_model(lora_dir: Path) -> str | None:
    payload = read_adapter_config(lora_dir)
    value = payload.get("base_model_name_or_path")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def download_base_model(repo_id: str, local_dir: Path) -> Path:
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading base model {repo_id} -> {local_dir}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
    return local_dir


def merge_lora(base_model: str, lora_dir: Path, merged_out: Path, python_bin: str, *, use_unsloth: bool) -> None:
    if use_unsloth:
        script = f'''
import importlib

base_id = {base_model!r}
lora_dir = {str(lora_dir)!r}
out_dir = {str(merged_out)!r}

print("Using Unsloth merge path...", flush=True)
try:
    unsloth = importlib.import_module("unsloth")
except Exception as exc:
    raise RuntimeError(
        "Adapter requires Unsloth merge path (unsloth_fixed=true), but `unsloth` is not importable. "
        "Install Unsloth in this environment and retry."
    ) from exc

FastModel = getattr(unsloth, "FastModel", None)
if FastModel is None:
    raise RuntimeError("Unsloth module does not expose FastModel; cannot run auto-merge.")

print("Loading tokenizer + model via Unsloth FastModel.from_pretrained...", flush=True)
model, tokenizer = FastModel.from_pretrained(
    model_name=base_id,
    load_in_4bit=True,
)

print("Loading LoRA adapter via PEFT-compatible load_adapter...", flush=True)
if not hasattr(model, "load_adapter"):
    raise RuntimeError("Loaded model has no load_adapter method; cannot attach LoRA adapter.")
model.load_adapter(lora_dir)

print("Exporting merged model with save_pretrained_merged(..., merged_16bit)...", flush=True)
if not hasattr(model, "save_pretrained_merged"):
    raise RuntimeError("Loaded model has no save_pretrained_merged method; cannot export merged weights.")
model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
print("DONE:", out_dir, flush=True)
'''
        run([python_bin, "-c", script])
        return

    script = f'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_id = {base_model!r}
lora_dir = {str(lora_dir)!r}
out_dir = {str(merged_out)!r}

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
print("Loading base model...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
print("Loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(base, lora_dir)
print("Merging...", flush=True)
merged = model.merge_and_unload()
print("Saving merged model...", flush=True)
merged.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print("DONE:", out_dir, flush=True)
'''
    run([python_bin, "-c", script])


def main() -> int:
    args = parse_args()
    lora_dir = Path(args.lora_dir)
    merged_out = Path(args.merged_out)
    cactus_out = Path(args.cactus_out)
    base_model_local_dir = Path(args.base_model_local_dir)

    adapter_base_model = read_adapter_base_model(lora_dir)
    adapter_config = read_adapter_config(lora_dir)
    adapter_unsloth_fixed = bool((adapter_config.get("auto_mapping") or {}).get("unsloth_fixed", False))

    selected_base_model = args.base_model
    if args.auto_base_from_adapter and adapter_base_model is not None:
        selected_base_model = adapter_base_model

    if args.strict_base_match and adapter_base_model is not None and selected_base_model != adapter_base_model:
        raise ValueError(
            "Base model mismatch: selected "
            f"{selected_base_model!r} but adapter expects {adapter_base_model!r}. "
            "Use --auto-base-from-adapter or set --base-model to match adapter_config.json."
        )

    if adapter_base_model is not None:
        print(f"Adapter expects base model: {adapter_base_model}", flush=True)
    print(f"Selected base model: {selected_base_model}", flush=True)
    if adapter_unsloth_fixed:
        print("Adapter indicates unsloth_fixed=true; using Unsloth merge path.", flush=True)

    if args.download_base_model:
        download_base_model(selected_base_model, base_model_local_dir)

    base_model_for_merge = selected_base_model
    if args.prefer_local_base:
        base_model_for_merge = str(base_model_local_dir.expanduser().resolve())

    if not args.skip_merge:
        adapter = lora_dir / "adapter_model.safetensors"
        if not adapter.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
        merge_lora(
            base_model_for_merge,
            lora_dir,
            merged_out,
            args.python,
            use_unsloth=adapter_unsloth_fixed,
        )

    if not args.skip_convert:
        run([
            args.cactus_bin,
            "convert",
            str(merged_out),
            str(cactus_out),
            "--precision",
            args.precision,
        ])

    if not args.skip_validate:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        run([
            args.python,
            "scripts/mobile/validate_cactus_model.py",
            "--cactus-weights",
            str(cactus_out),
            "--out-dir",
            "evaluation/results/cactus_model_validation",
            "--max-samples",
            str(args.max_samples),
        ], env=env)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
