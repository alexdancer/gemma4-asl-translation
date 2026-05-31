---
title: ASL Translation Prototype
emoji: 👐
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
startup_duration_timeout: 1h
---

# ASL Translation Prototype

This Space loads the ASL Gemma-4 LoRA adapter during app startup, before the Gradio UI is served, using the same Unsloth `FastVisionModel` path used by the working notebooks.

Current runtime behavior:

1. On container startup, `app.py` eagerly `snapshot_download`s `AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora`.
2. Startup then loads the local adapter snapshot with `FastVisionModel.from_pretrained(..., load_in_4bit=True, use_gradient_checkpointing="unsloth", max_seq_length=8192)` and caches it in a module-global model bundle.
3. The UI displays startup model status and diagnostics; there is no page-load `demo.load(...)` preload step.
4. After status shows ready, upload a short ASL video and click Translate.
5. The app samples 30 RGB frames at 448x448 by default and returns a strict prediction against the WLASL/model Top-50 label space used by the working Colab notebook.

If startup loading fails, the app fails closed with explicit diagnostics instead of returning a placeholder or fake translation.
