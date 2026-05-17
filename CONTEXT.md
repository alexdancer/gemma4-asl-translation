# ASL Project Context

## Glossary

### Video sample
A short ASL source video from WLASL or user upload that represents one labeled signing example. In the new video-finetune branch, a video sample is the domain object even when the implementation stores extracted images.

### Frame sequence
An ordered set of uniformly resized image frames extracted from one video sample for Gemma-4 multimodal fine-tuning or inference through Unsloth. Frame sequences are the canonical training input for the `video-finetune-main` branch.

### Top-50
A dedicated WLASL subset containing the 50 highest-support gloss labels. Top-50 remains its own first-class dataset category for initial fine-tuning and fast iteration.

### Full WLASL
The full available WLASL dataset is the broader target dataset category for the new video-finetuning branch. Extraction scripts should support full-WLASL frame generation while preserving Top-50 as a separate category/output.

### Video-finetune manifest
A JSONL file with one row per video sample. Each row records `sample_id`, `gloss`, `split`, and exactly 30 ordered `frame_paths`. The manifest is the notebook-facing source of truth; extracted frame files remain regular JPEG images on disk.

## Decisions captured in language

- The new video-finetuning path trains on ordered extracted image frames, not native video files.
- Code and documentation should preserve the domain term “video sample” so the project can later change the storage/input representation without renaming the product concept.
- Initial Gemma-4 E4B fine-tuning uses exactly 30 evenly sampled frames per video sample, resized to 448x448, with a fallback ladder to fewer/lower-resolution frames only if Colab memory limits require it.
- The dataset artifact is a simple folder tree of extracted frames plus one JSONL video-finetune manifest read by Colab.
- The extraction pipeline should run over the whole available WLASL dataset and also emit/keep Top-50 as its own first-class dataset category for initial fine-tuning.
- Top-50 should have its own copied frame folder and manifests, separate from the full-WLASL frame folder, to make upload/training handoff simpler even though it duplicates storage.
- Frame extraction should tolerate bad WLASL videos: try the normal extraction path first, retry with a decoded-frame-count fallback when metadata/FPS is unreliable, then skip and record failures in `failures.jsonl` if extraction still cannot produce a valid 20-frame sequence.
- The first training/evaluation target is exact WLASL gloss classification: the assistant should output exactly one canonical gloss label, with natural-English phrasing handled later as a demo/display layer.
- Dataset splitting should use official WLASL split metadata when available; otherwise use deterministic stratified-by-gloss splitting with a fixed seed and record the `split_source` in manifests.
- Top-50 prompts should include the full Top-50 allowed label list for output stability; full-WLASL prompts should not include the entire vocabulary and should instead instruct the model to reply with the exact WLASL gloss label.
- External ASL corpora (e.g., ZahidYasinMittha/American-Sign-Language-Dataset) are phase-1 pretraining sources, while WLASL Top-50 remains the final-tune/evaluation benchmark for continuity.
- Phase-1 external-corpus training uses only labels that map cleanly into the canonical Top-50 label space; non-mapped rows are dropped.
- Label normalization/mapping is versioned in-repo as a canonical mapping file, not ad-hoc notebook logic.
- Approved canonical mapping includes at least `forget -> forgot` and `last year -> year`.
- Phase-1 task format is strict single-label classification with exact one canonical label output.
- Phase-1 filtered external-corpus training uses deterministic stratified splits (P2) rather than train-only runs.
- Benchmark boundary discipline is required: external-corpus data is pretraining-only, while benchmark reporting remains strictly WLASL Top-50 evaluation artifacts.
- The new fine-tuning notebook scope is unified (N1): phase-1 external-corpus pretrain + phase-2 WLASL Top-50 final-tune + evaluation in one notebook.
- Phase transition mode is sequential (R1): continue training the same adapter from phase-1 external-corpus pretraining into phase-2 WLASL Top-50 final-tuning.
- Baseline comparison runs are out of scope for this pass (E1); focus is training with the new dataset.
- Phase-1 data volume policy is full filtered coverage (D1): use all filtered Zahid→Top-50 rows.
- Frame representation policy is V1: keep frame-based training with the existing canonical extraction contract (30 evenly sampled 448x448 RGB frames per video sample).
- Stratified split provenance for external-corpus manifests must be explicit (G1), e.g., `split_source: zahid_top50_stratified_seed42_v1`.
- Canonical external label mapping file location is C1: `configs/label_mappings/zahid_to_top50.yaml`.
- Video-finetune artifacts use self-contained dataset-category folders: `data/video_finetune/full/{frames,manifest.jsonl,train.jsonl,val.jsonl,test.jsonl}` and `data/video_finetune/top50/{frames,manifest.jsonl,train.jsonl,val.jsonl,test.jsonl,labels.txt}`.
- WLASL now lives locally on the Mac under `data/WLASL/`; scripts should accept configurable paths and default to the local repo-relative WLASL location when present.
- The WLASL start-kit preprocessing scripts are useful as metadata/reference code for locating sign clips, YouTube-ID mapping, `frame_start`/`frame_end`, and official splits, but the video-finetune branch should own a separate frame-extraction script that consumes already-preprocessed `start_kit/videos/{gloss}/{video_id}.mp4` by default.
- The first extractor implementation should use only existing preprocessed WLASL clips and log missing clips to `failures.jsonl`; raw-video regeneration is out of scope for the initial path.
- Local WLASL data and generated video-finetune artifacts are ignored by git; the repo tracks scripts/docs, not downloaded videos or extracted frames.
- Top-50 is computed from available local preprocessed clips per gloss, not from theoretical WLASL metadata counts, so the initial fine-tuning slice reflects videos that can actually be extracted locally.
