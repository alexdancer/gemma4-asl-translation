#!/usr/bin/env python3
"""One-command LoRA -> merged HF -> Cactus convert -> validation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Cactus conversion pipeline from LoRA artifacts.")
    parser.add_argument("--base-model", default="google/gemma-4-E4B-it")
    parser.add_argument(
        "--lora-dir",
        default="checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline",
        help="Directory containing adapter_model.safetensors",
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


def merge_lora(base_model: str, lora_dir: Path, merged_out: Path, python_bin: str) -> None:
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
    torch_dtype=torch.bfloat16,
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

    if not args.skip_merge:
        adapter = lora_dir / "adapter_model.safetensors"
        if not adapter.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
        merge_lora(args.base_model, lora_dir, merged_out, args.python)

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
