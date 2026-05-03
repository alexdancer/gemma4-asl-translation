# Cactus Export Task A

Task A provides a deterministic export path for a Gemma ASL checkpoint into a
Cactus-style INT4 bundle for mobile integration work.

## Files

```text
src/mobile/cactus_export.py
tests/test_cactus_export.py
src/mobile/CACTUS_EXPORT_README.md
```

The exporter writes these artifacts to `cactus_export/` by default:

```text
model_int4.bin
config.json
special_tokens.json
export_manifest.json
quantization_report.json
```

`cactus_export/` is generated output and should not be committed.

## Usage

From the repo root:

```bash
venv/bin/python -m src.mobile.cactus_export \
  --checkpoint checkpoints/gemma_asl/best-checkpoint \
  --base-model google/gemma-4-E2B-it \
  --output-dir cactus_export
```

`--checkpoint` is required. The exporter does not create dummy artifacts or use
a mock fallback when model loading fails. Full Hugging Face checkpoints load
directly; adapter-only checkpoints with `adapter_config.json` load against
`--base-model`. Checkpoints must be directories saved with the Hugging Face
`save_pretrained` contract; single checkpoint files are rejected before export
artifacts are written.

If an adapter checkpoint cannot be loaded, the CLI exits non-zero and reports
the base model, checkpoint path, and adapter loader error. Common causes are
missing `peft`/`transformers` dependencies, an unavailable `--base-model`, or
missing tokenizer files in the adapter directory.

## Export Behavior

- Quantization is symmetric signed INT4 with optional per-channel scales.
- Floating point tensors with one or more dimensions are exported, including
  required bias and normalization vectors.
- Export validates tensor coverage against the loaded model state dict before
  writing the final weight bundle; missing required floating point tensors fail
  the export.
- INT4 values are packed two signed nibbles per byte in `model_int4.bin`.
- The binary starts with a small JSON tensor manifest, followed by scale data
  and packed tensor payloads.
- `export_manifest.json` includes the SHA-256 hash of `model_int4.bin`,
  projected latency and memory, pass/fail deployment criteria, and every
  generated artifact including the manifest itself.
- `quantization_report.json` includes per-layer ranges, scale shapes, and
  chunked reconstruction error statistics that avoid allocating a full
  reconstructed copy of giant tensors for metrics.
- Export failures propagate as a non-zero CLI exit with a JSON error on stderr.

The deterministic benchmark in Task A is a projection used to keep CI and local
tests stable. Real device latency must be measured in Task B with the mobile
runtime.

## Tests

```bash
venv/bin/python -m pytest -q tests/test_cactus_export.py
```

The tests cover quantization range and shape, JSON serialization, file output,
manifest integrity, size criteria, explicit checkpoint loading, persisted scales
and packed weights, tensor coverage in the serialized bundle, checkpoint and
adapter-path failures, Docker context hygiene, and the end-to-end export pipeline
with a patched lightweight checkpoint loader.

## Current Limits

- INT4 is the only supported quantization target.
- Cactus runtime loading is not implemented in this repo yet.
- Accuracy loss is reported as tensor reconstruction error only; task accuracy
  validation requires a trained checkpoint and evaluation slice.
- Adapter export depends on loading the requested base model at export time; pin
  `--base-model` to a locally available or authenticated Hugging Face model for
  offline/CI use.
