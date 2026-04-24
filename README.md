# ASL Transcription System

Phase 1 scaffold for a Kaggle hackathon project focused on real-time ASL transcription on mobile devices.

## Architecture

`MediaPipe -> Pose Encoder -> Gemma 4 2B-E2B -> Text`

## Project Layout

```text
src/
  data/
  models/
  mobile/
data/
  raw/
  processed/
notebooks/
tests/
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Phase 1 Scope

- Project scaffolding
- WLASL data loading
- Pose extraction with MediaPipe Holistic
- Train/validation/test split generation
- Basic exploratory notebooks and tests

## Notes

- WLASL metadata is downloaded from the public GitHub repository.
- Video files should be stored under `data/raw/wlasl/videos/`.
- Extracted pose sequences are written to `data/processed/`.
