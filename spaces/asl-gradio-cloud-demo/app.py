from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
from PIL import Image

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

ADAPTER_MODEL_ID = os.environ.get(
    "ASL_ADAPTER_MODEL_ID",
    "AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora",
)
MODEL_BACKEND = "unsloth_fastvision"
FRAME_COUNT = int(os.environ.get("ASL_FRAME_COUNT", "30"))
FRAME_SIZE = int(os.environ.get("ASL_FRAME_SIZE", "448"))
MAX_NEW_TOKENS = int(os.environ.get("ASL_MAX_NEW_TOKENS", "12"))
MAX_SEQ_LENGTH = int(os.environ.get("ASL_MAX_SEQ_LENGTH", "8192"))
DEVICE_MAP_MODE = os.environ.get("ASL_DEVICE_MAP", "single").strip() or "single"
GPU_MEMORY_HEADROOM_GIB = float(os.environ.get("ASL_GPU_MEMORY_HEADROOM_GIB", "2"))
EAGER_LOAD_ON_STARTUP = os.environ.get("ASL_EAGER_LOAD_ON_STARTUP", "1").strip().lower() not in {"0", "false", "no", "off"}

TOP50_GLOSSES = [
    "drink",
    "like",
    "wrong",
    "forget",
    "computer",
    "finish",
    "hot",
    "mother",
    "now",
    "orange",
    "hearing",
    "color",
    "birthday",
    "need",
    "book",
    "before",
    "deaf",
    "fine",
    "no",
    "yes",
    "all",
    "black",
    "kiss",
    "study",
    "white",
    "dance",
    "but",
    "cook",
    "paper",
    "visit",
    "door",
    "college",
    "your",
    "use",
    "better",
    "last year",
    "will",
    "chair",
    "clothes",
    "candy",
    "year",
    "many",
    "woman",
    "blue",
    "fish",
    "hat",
    "bird",
    "cow",
    "enjoy",
    "meet",
]
TOP50_SET = set(TOP50_GLOSSES)
ALIASES = {
    "thank you": "thanks",
    "thank-you": "thanks",
    "dont like": "don't like",
    "do not like": "don't like",
    "dont understand": "don't understand",
    "do not understand": "don't understand",
    "dont know": "don't know",
    "do not know": "don't know",
}
SUPPORTED_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}

_MODEL_BUNDLE: "ModelBundle | None" = None
_LOAD_ERROR: dict[str, Any] | None = None


class ModelBundle:
    def __init__(self, *, processor: Any, model: Any, diagnostics: dict[str, Any], backend: str):
        self.processor = processor
        self.model = model
        self.backend = backend
        self.diagnostics = diagnostics


def get_runtime_config() -> dict[str, Any]:
    return {
        "mode": "model_preload",
        "model_backend": MODEL_BACKEND,
        "adapter_model_id": ADAPTER_MODEL_ID,
        "frame_count": FRAME_COUNT,
        "frame_size": FRAME_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "device_map": DEVICE_MAP_MODE,
        "gpu_memory_headroom_gib": GPU_MEMORY_HEADROOM_GIB,
        "loader": "snapshot_download(adapter_repo) -> unsloth FastVisionModel 4bit",
        "preload_trigger": "startup_import",
        "eager_load_on_startup": EAGER_LOAD_ON_STARTUP,
        "model_loaded": _MODEL_BUNDLE is not None,
    }


def validate_video_path(video_path: str | None) -> Path:
    if not video_path:
        raise gr.Error("Upload a short ASL video first.")
    path = Path(video_path)
    if not path.exists():
        raise gr.Error("Uploaded video file could not be found by the Space runtime.")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise gr.Error("Please upload an .mp4, .mov, .webm, or .m4v video.")
    return path


def _frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        return list(range(count))
    if count <= 1:
        return [0]
    last = max(total_frames - 1, 0)
    return [round(i * last / (count - 1)) for i in range(count)]


def sample_video_frames(video_path: Path, *, count: int = FRAME_COUNT, size: int = FRAME_SIZE) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise gr.Error("Could not open uploaded video. Try an .mp4/.mov clip.")

    frames: list[Image.Image] = []
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for index in _frame_indices(total_frames, count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
            frames.append(image)
    finally:
        capture.release()

    if not frames:
        raise gr.Error("Could not extract frames from uploaded video.")
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return frames[:count]


def build_prompt(_filename: str | None = None) -> str:
    return (
        "Identify the ASL sign shown across these frames. "
        "Return exactly one gloss label from the approved list.\n"
        f"Approved labels: {', '.join(TOP50_GLOSSES)}"
    )


def normalize_gloss(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = re.sub(r"^`+|`+$", "", normalized)
    normalized = re.sub(r"[^a-z0-9_ '\-]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_prediction(raw_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in str(raw_text).splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    normalized_candidate = normalize_gloss(first_line)
    candidate = ALIASES.get(normalized_candidate, normalized_candidate)
    if candidate in TOP50_SET:
        return {
            "status": "ok",
            "prediction": candidate,
            "candidate_gloss": candidate,
            "validation_reason": "candidate is in the approved Top-50 label set",
            "raw_output": raw_text,
            "low_confidence": False,
        }
    return {
        "status": "out_of_allowlist",
        "prediction": None,
        "candidate_gloss": candidate or None,
        "validation_reason": "model output is not in the approved Top-50 label set",
        "raw_output": raw_text,
        "low_confidence": True,
    }


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-4:]),
    }


def _snapshot_download_adapter() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(ADAPTER_MODEL_ID, token=os.environ.get("HF_TOKEN"))


def _load_unsloth_fastvision_model(adapter_local: str, device_map: Any, max_memory: dict[int, str] | None) -> tuple[Any, Any, str]:
    """Load the adapter with the same Unsloth path used by Notebooks 12/13."""

    import gc

    import torch
    from unsloth import FastVisionModel

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    load_kwargs: dict[str, Any] = {
        "model_name": adapter_local,
        "load_in_4bit": True,
        "use_gradient_checkpointing": "unsloth",
        "max_seq_length": MAX_SEQ_LENGTH,
        "device_map": device_map,
    }
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    try:
        model, tokenizer = FastVisionModel.from_pretrained(**load_kwargs)
    except TypeError as exc:
        if "max_memory" not in str(exc) or max_memory is None:
            raise
        load_kwargs.pop("max_memory", None)
        model, tokenizer = FastVisionModel.from_pretrained(**load_kwargs)
    FastVisionModel.for_inference(model)
    return model, tokenizer, "Unsloth FastVisionModel 4bit"

def _gpu_diagnostics() -> dict[str, Any]:
    import torch

    diagnostics: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        all_devices = []
        for device_index in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(device_index)
            all_devices.append(
                {
                    "index": device_index,
                    "name": torch.cuda.get_device_name(device_index),
                    "total_vram_gb": round(device_props.total_memory / (1024**3), 3),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                }
            )
        diagnostics.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_total_vram_gb": round(props.total_memory / (1024**3), 3),
                "cuda_compute_capability": f"{props.major}.{props.minor}",
                "cuda_devices": all_devices,
            }
        )
    return diagnostics


def _build_max_memory() -> dict[int, str] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    memory: dict[int, str] = {}
    for device_index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_index)
        total_gib = props.total_memory / (1024**3)
        usable_gib = max(1, int(total_gib - GPU_MEMORY_HEADROOM_GIB))
        memory[device_index] = f"{usable_gib}GiB"
    return memory


def _resolve_device_map_and_memory() -> tuple[Any, dict[int, str] | None]:
    mode = DEVICE_MAP_MODE.lower()
    if mode in {"single", "cuda:0", "0"}:
        return {"": 0}, None
    if mode in {"balanced", "balanced_low_0", "auto"}:
        return mode, _build_max_memory()
    return DEVICE_MAP_MODE, _build_max_memory()


def _format_status(*, loaded: bool, diagnostics: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> str:
    if loaded:
        gpu = diagnostics or {}
        return "\n".join(
            [
                "Model status: ready",
                f"Backend: {gpu.get('backend', MODEL_BACKEND)}",
                f"Adapter: {ADAPTER_MODEL_ID}",
                f"GPU: {gpu.get('cuda_device_name', 'unknown')}",
                f"VRAM: {gpu.get('cuda_total_vram_gb', 'unknown')} GB",
            ]
        )
    if error:
        return "\n".join(
            [
                "Model status: startup load failed",
                f"Error type: {error.get('type', 'unknown')}",
                f"Error: {error.get('message', '')}",
                "Submit is fail-closed until the model loads successfully.",
            ]
        )
    return "Model status: loading..."


def load_model_bundle() -> ModelBundle:
    """Load the ASL adapter once and cache it in module globals."""

    global _MODEL_BUNDLE, _LOAD_ERROR
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    diagnostics = _gpu_diagnostics()
    adapter_local = _snapshot_download_adapter()
    device_map, max_memory = _resolve_device_map_and_memory()
    model, processor, backend = _load_unsloth_fastvision_model(adapter_local, device_map, max_memory)
    if hasattr(model, "eval"):
        model.eval()
    diagnostics.update(
        {
            "status": "ready",
            "backend": backend,
            "adapter_model_id": ADAPTER_MODEL_ID,
            "frame_count": FRAME_COUNT,
            "frame_size": FRAME_SIZE,
            "device_map": device_map,
            "max_memory": max_memory,
        }
    )
    _MODEL_BUNDLE = ModelBundle(processor=processor, model=model, diagnostics=diagnostics, backend=backend)
    _LOAD_ERROR = None
    return _MODEL_BUNDLE


def model_status_payload() -> tuple[str, str]:
    if _MODEL_BUNDLE is not None:
        payload = {"status": "ready_cached", **_MODEL_BUNDLE.diagnostics}
        return _format_status(loaded=True, diagnostics=_MODEL_BUNDLE.diagnostics), json.dumps(payload, indent=2)
    if _LOAD_ERROR is not None:
        payload = {"status": "startup_load_failed", "error": _LOAD_ERROR}
        return _format_status(loaded=False, error=_LOAD_ERROR), json.dumps(payload, indent=2)
    payload = {"status": "not_loaded", "eager_load_on_startup": EAGER_LOAD_ON_STARTUP}
    return "Model status: not loaded", json.dumps(payload, indent=2)


def preload_model() -> tuple[str, str]:
    """Compatibility/manual retry hook; normal Space loading happens at startup import time."""

    try:
        bundle = load_model_bundle()
        return _format_status(loaded=True, diagnostics=bundle.diagnostics), json.dumps(bundle.diagnostics, indent=2)
    except Exception as exc:
        global _MODEL_BUNDLE, _LOAD_ERROR
        _MODEL_BUNDLE = None
        _LOAD_ERROR = _compact_error(exc)
        payload = {"status": "startup_load_failed", "error": _LOAD_ERROR}
        return _format_status(loaded=False, error=_LOAD_ERROR), json.dumps(payload, indent=2)


def require_loaded_model() -> ModelBundle:
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE
    if _LOAD_ERROR is not None:
        raise gr.Error(f"Model startup load failed. Diagnostics: {_LOAD_ERROR}")
    raise gr.Error("Model is not ready yet. The Space should load it at startup; check Model status diagnostics.")


def _extract_new_text(processor: Any, outputs: Any, inputs: Any) -> str:
    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if input_ids is not None:
        prompt_len = int(input_ids.shape[-1])
        try:
            generated = outputs[0][prompt_len:]
        except Exception:
            generated = outputs
    else:
        generated = outputs
    if hasattr(processor, "decode"):
        return str(processor.decode(generated, skip_special_tokens=True)).strip()
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    if isinstance(decoded, list):
        return str(decoded[0]).strip() if decoded else ""
    return str(decoded).strip()


def _get_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "parameters"):
        try:
            device = next(model.parameters()).device
        except Exception:
            device = None
    return device


def _move_inputs_to_model_device(inputs: Any, model: Any) -> Any:
    if not hasattr(inputs, "to"):
        return inputs
    device = _get_model_device(model)
    if device is None:
        return inputs
    return inputs.to(device)


def run_model_generation(frames: list[Image.Image], prompt: str) -> str:
    bundle = require_loaded_model()
    processor = bundle.processor
    model = bundle.model

    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image", "image": image} for image in frames)
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_model_device(inputs, model)

    import torch

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return _extract_new_text(processor, outputs, inputs)


def translate_video(video_path: str | None) -> tuple[str, str]:
    path = validate_video_path(video_path)
    bundle = require_loaded_model()
    try:
        frames = sample_video_frames(path)
        prompt = build_prompt(None)
        raw_text = run_model_generation(frames, prompt)
        normalized = normalize_prediction(raw_text)
        payload = {
            **normalized,
            "frame_count": len(frames),
            "frame_size": FRAME_SIZE,
            "model_backend": bundle.backend,
            "adapter_model_id": ADAPTER_MODEL_ID,
            "diagnostics": bundle.diagnostics,
        }
        if normalized["status"] == "ok":
            summary = "\n".join(
                [
                    f"Accepted prediction: {normalized['prediction']}",
                    f"Model candidate: {normalized['candidate_gloss']}",
                    "Status: ok",
                    f"Reason: {normalized['validation_reason']}",
                    f"Raw model output: {raw_text}",
                    f"Frames: {len(frames)} x {FRAME_SIZE}px",
                    f"Backend: {bundle.backend}",
                    f"Adapter: {ADAPTER_MODEL_ID}",
                ]
            )
        else:
            summary = "\n".join(
                [
                    "Accepted prediction: none",
                    f"Model candidate: {normalized['candidate_gloss'] or '(empty)'}",
                    f"Status: {normalized['status']}",
                    f"Reason: {normalized['validation_reason']}",
                    f"Raw model output: {raw_text}",
                    f"Frames: {len(frames)} x {FRAME_SIZE}px",
                    f"Backend: {bundle.backend}",
                    f"Adapter: {ADAPTER_MODEL_ID}",
                ]
            )
        return summary, json.dumps(payload, indent=2)
    except gr.Error:
        raise
    except Exception as exc:
        error = _compact_error(exc)
        payload = {
            "status": "inference_failed",
            "error": error,
            "model_backend": bundle.backend,
            "adapter_model_id": ADAPTER_MODEL_ID,
            "diagnostics": bundle.diagnostics,
        }
        summary = "\n".join(
            [
                "Inference failed; no translation was produced.",
                f"Error type: {error['type']}",
                f"Error: {error['message']}",
                "See Debug JSON for diagnostics.",
            ]
        )
        return summary, json.dumps(payload, indent=2)


def startup_load_model() -> None:
    if not EAGER_LOAD_ON_STARTUP:
        print("[boot] ASL eager model load disabled by ASL_EAGER_LOAD_ON_STARTUP", flush=True)
        return
    print(f"[boot] loading ASL adapter {ADAPTER_MODEL_ID} with {MODEL_BACKEND}...", flush=True)
    status, details = preload_model()
    print(status.replace("\n", " | "), flush=True)
    if _LOAD_ERROR is not None:
        print(details, flush=True)


startup_load_model()


def build_demo():
    with gr.Blocks(title="ASL Translation Prototype") as demo:
        gr.Markdown("# ASL Translation Prototype")
        gr.Markdown(
            f"This Space loads `{ADAPTER_MODEL_ID}` with `{MODEL_BACKEND}` during app startup, before the UI is served. "
            f"After status shows ready, upload a short ASL video. The app uses notebook-parity input: "
            f"{FRAME_COUNT} frames at {FRAME_SIZE}x{FRAME_SIZE} and the WLASL/model Top-50 label space."
        )
        initial_status, initial_debug = model_status_payload()
        status = gr.Textbox(label="Model status", value=initial_status, lines=6, interactive=False)
        preload_debug = gr.Textbox(label="Startup diagnostics", value=initial_debug, lines=12, interactive=False)
        video = gr.Video(label="ASL video", sources=["upload"], format=None)
        submit = gr.Button("Translate", variant="primary")
        result = gr.Textbox(label="Result", lines=8)
        debug = gr.Textbox(label="Debug JSON", lines=12)

        submit.click(fn=translate_video, inputs=video, outputs=[result, debug])
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
