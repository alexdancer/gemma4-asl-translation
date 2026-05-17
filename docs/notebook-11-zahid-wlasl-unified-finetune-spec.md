# Notebook Spec: Unified Zahid→WLASL Top-50 Fine-tuning (Unsloth Gemma-4)

## Status
Approved planning artifact (pre-notebook). Do **not** generate the notebook from this file until explicit approval.

## Goal
Create one Colab notebook (N1) that performs:
1. **Phase 1 pretrain** on filtered `ZahidYasinMittha/American-Sign-Language-Dataset` mapped into canonical Top-50 labels.
2. **Phase 2 final-tune** on WLASL Top-50 using the **same adapter checkpoint** from phase 1 (R1/W1).
3. Export/push both stage adapters (O2/H1) and run final WLASL eval artifacts.

## Locked decisions
- External corpus role: pretraining source only; benchmark remains WLASL Top-50 (A, L2).
- Filtering policy: mapped-only rows; drop non-mapped rows (A2, F1).
- Task format: strict single-label classification output (T1).
- Scope: Top-50 canonical set only (S1).
- External split strategy: deterministic stratified split (P2) with explicit provenance (G1).
- Notebook scope: unified two-stage notebook (N1).
- Comparison baseline: out of scope for this pass (E1).
- Data volume: all filtered rows (D1).
- Modality: frame-based training using existing contract (V1).
- Mapping file location: `configs/label_mappings/zahid_to_top50.yaml` (C1).
- Phase-2 init: continue from phase-1 adapter/checkpoint (W1).
- Artifacts: publish both adapters with explicit stage names (O2/H1).
- Contract enforcement: hard fail-fast gates (Q1).

## Canonical mapping requirements
Mapping file path:
- `configs/label_mappings/zahid_to_top50.yaml`

Must include at minimum:
- `forget: forgot`
- `last year: year`

Rules:
- Case-insensitive normalization before lookup.
- Trim whitespace and collapse repeated spaces.
- Reject rows mapping to labels outside canonical Top-50 allowlist.

## Data contracts

### Frame contract (unchanged)
- Exactly `30` evenly sampled RGB frames per sample.
- Resolution `448x448`.
- Ordered frame paths.

### Zahid filtered manifest rows
Required fields:
- `sample_id`
- `gloss` (canonical mapped Top-50 label)
- `split` (`train|val|test`)
- `split_source` = `zahid_top50_stratified_seed42_v1`
- `frame_paths` (length = 30)
- `source_dataset` = `zahid`
- `source_label_raw` (original Zahid label)

### WLASL rows
Use existing Top-50 contract currently used in video notebook flow.

## Notebook structure (planned cells)
1. Runtime/install/preflight
   - Install unsloth stack.
   - Set `UNSLOTH_DISABLE_STATISTICS=1` before importing unsloth.
2. Config block
   - Base model: `google/gemma-4-26B-A4B-it`
   - Run names + HF repos (stage-specific)
   - Seeds, split params, frame params.
3. Load canonical Top-50 label allowlist
4. Load mapping YAML (`configs/label_mappings/zahid_to_top50.yaml`)
5. Ingest Zahid metadata and normalize labels
6. Apply mapping + strict filtering
7. Stratified split (seed 42) + write manifests
8. Hard gate checks (abort on failure)
9. Build Unsloth samples/messages for phase-1
10. Phase-1 training (Zahid mapped Top-50)
11. Save/push phase-1 adapter
12. Load WLASL Top-50 manifests
13. Build WLASL samples/messages
14. Continue training from phase-1 checkpoint (phase-2)
15. Save/push phase-2 final adapter
16. Run WLASL eval + write artifacts
17. Summary cell (paths, counts, metrics)

## Hard gate checks (must fail notebook if violated)
- Mapping file exists and parses.
- Canonical Top-50 allowlist length is exactly 50 and deduped.
- All mapped labels are in allowlist.
- Zahid filtered dataset non-empty and split non-empty.
- No overlap across train/val/test sample IDs.
- Each sample has exactly 30 frame paths; each frame exists and matches 448x448.
- Instruction/output contract sanity checks.

## Training defaults (initial)
- `per_device_train_batch_size=1`
- gradient accumulation enabled (effective batch via accum)
- LoRA rank/alpha baseline aligned to current notebook defaults unless memory issues
- conservative logging/eval cadence to reduce Colab overhead

## Output artifacts

### Stage 1 adapter repo
- `<namespace>/asl-gemma4-26b-a4b-zahid-pretrain-lora`

### Stage 2 adapter repo
- `<namespace>/asl-gemma4-26b-a4b-zahid-then-wlasl-top50-lora`

### Eval artifacts (local Colab workspace + optional hub upload)
- metrics JSON
- predictions JSONL/CSV
- invalid-output diagnostics
- confusion summaries

## Non-goals (this pass)
- No baseline comparison run (`base -> WLASL`) in this notebook.
- No pose-only branch.
- No expansion beyond Top-50 label space.

## Approval checklist before notebook generation
- [ ] Mapping YAML committed with approved mappings (`forget->forgot`, `last year->year`)
- [ ] Final stage repo names approved
- [ ] Colab runtime class confirmed (A100 preferred)
- [ ] User explicitly says "create notebook now"
