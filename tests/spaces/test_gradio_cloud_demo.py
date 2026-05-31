from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_PATH = Path(__file__).resolve().parents[2] / "spaces" / "asl-gradio-cloud-demo" / "app.py"


def install_space_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Space app imports clean while local tests avoid installing Gradio/ZeroGPU."""

    cv2 = types.ModuleType("cv2")
    setattr(cv2, "CAP_PROP_FRAME_COUNT", 7)
    setattr(cv2, "CAP_PROP_POS_FRAMES", 1)
    setattr(cv2, "COLOR_BGR2RGB", 4)
    setattr(cv2, "cvtColor", lambda frame, _code: frame)
    setattr(cv2, "VideoCapture", lambda _path: None)
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    pil = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")

    class FakeImage:
        Resampling = SimpleNamespace(BICUBIC="bicubic")

        def __init__(self, size=(20, 16), mode="RGB"):
            self.size = size
            self.mode = mode

        @classmethod
        def fromarray(cls, _array):
            return cls()

        def convert(self, mode):
            self.mode = mode
            return self

        def resize(self, size, _resample):
            self.size = size
            return self

        def copy(self):
            return FakeImage(size=self.size, mode=self.mode)

    setattr(image_module, "Image", FakeImage)
    setattr(image_module, "Resampling", FakeImage.Resampling)
    setattr(image_module, "fromarray", FakeImage.fromarray)
    setattr(pil, "Image", image_module)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)

    gradio = types.ModuleType("gradio")

    class GradioError(Exception):
        pass

    class FakeComponent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeButton(FakeComponent):
        def click(self, **kwargs):
            self.click_kwargs = kwargs
            return self

    class FakeBlocks:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.load_kwargs = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def load(self, **kwargs):
            self.load_kwargs = kwargs
            return self

        def queue(self, **kwargs):
            self.queue_kwargs = kwargs
            return self

        def launch(self, **kwargs):
            self.launch_kwargs = kwargs
            return self

    setattr(gradio, "Error", GradioError)
    setattr(gradio, "Video", FakeComponent)
    setattr(gradio, "Textbox", FakeComponent)
    setattr(gradio, "Button", FakeButton)
    setattr(gradio, "Blocks", FakeBlocks)
    setattr(gradio, "Markdown", lambda *args, **kwargs: FakeComponent(*args, **kwargs))
    monkeypatch.setitem(sys.modules, "gradio", gradio)

    spaces = types.ModuleType("spaces")

    def gpu(fn=None, **kwargs):
        def decorate(wrapped):
            wrapped.gpu_kwargs = kwargs
            return wrapped

        if fn is None:
            return decorate
        return decorate(fn)

    setattr(spaces, "GPU", gpu)
    monkeypatch.setitem(sys.modules, "spaces", spaces)


def load_app_module(monkeypatch: pytest.MonkeyPatch):
    install_space_dependency_stubs(monkeypatch)
    monkeypatch.setenv("ASL_EAGER_LOAD_ON_STARTUP", "0")
    spec = importlib.util.spec_from_file_location("asl_gradio_cloud_demo", APP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_model_preload_space(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)

    config = app.get_runtime_config()

    assert config["mode"] == "model_preload"
    assert config["model_backend"] == "unsloth_fastvision"
    assert config["adapter_model_id"] == "AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora"
    assert config["frame_count"] == 30
    assert config["frame_size"] == 448
    assert config["device_map"] == "single"
    assert config["loader"] == "snapshot_download(adapter_repo) -> unsloth FastVisionModel 4bit"
    assert config["preload_trigger"] == "startup_import"
    assert config["eager_load_on_startup"] is False
    assert config["model_loaded"] is False
    assert "dance" in app.TOP50_SET
    assert "hello" not in app.TOP50_SET


def test_build_demo_uses_startup_status_without_page_load_preload(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    app._MODEL_BUNDLE = app.ModelBundle(
        processor="processor",
        model="model",
        diagnostics={"status": "ready", "backend": "test", "cuda_device_name": "Fake GPU", "cuda_total_vram_gb": 80},
        backend="test",
    )

    demo = app.build_demo()

    assert demo.load_kwargs is None
    assert app.get_runtime_config()["preload_trigger"] == "startup_import"


def test_load_model_bundle_loads_adapter_with_unsloth_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    calls: list[dict[str, object]] = []

    fake_model = SimpleNamespace(eval=lambda: calls.append({"method": "eval"}))

    def fake_load_unsloth(adapter_local, device_map, max_memory):
        calls.append({"adapter_local": adapter_local, "device_map": device_map, "max_memory": max_memory})
        return fake_model, "tokenizer", "Unsloth FastVisionModel 4bit"

    monkeypatch.setattr(
        app,
        "_gpu_diagnostics",
        lambda: {"cuda_available": True, "cuda_device_name": "Fake GPU", "cuda_total_vram_gb": 96.0},
    )
    monkeypatch.setattr(app, "_snapshot_download_adapter", lambda: "/tmp/adapter-snapshot")
    monkeypatch.setattr(app, "_resolve_device_map_and_memory", lambda: ("auto", {0: "20GiB", 1: "20GiB"}))
    monkeypatch.setattr(app, "_load_unsloth_fastvision_model", fake_load_unsloth)

    bundle = app.load_model_bundle()
    payload = bundle.diagnostics

    assert payload["status"] == "ready"
    assert payload["backend"] == "Unsloth FastVisionModel 4bit"
    assert payload["device_map"] == "auto"
    assert payload["max_memory"] == {0: "20GiB", 1: "20GiB"}
    assert app._MODEL_BUNDLE is not None
    assert app._MODEL_BUNDLE.backend == "Unsloth FastVisionModel 4bit"
    assert calls == [
        {"adapter_local": "/tmp/adapter-snapshot", "device_map": "auto", "max_memory": {0: "20GiB", 1: "20GiB"}},
        {"method": "eval"},
    ]


def test_unsloth_loader_uses_notebook_parity_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    calls: list[tuple[str, object]] = []

    torch_module = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            calls.append(("empty_cache", None))

    setattr(torch_module, "cuda", FakeCuda)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    unsloth_module = types.ModuleType("unsloth")

    class FakeFastVisionModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(("from_pretrained", kwargs))
            return SimpleNamespace(name="model"), "tokenizer"

        @staticmethod
        def for_inference(model):
            calls.append(("for_inference", model.name))

    setattr(unsloth_module, "FastVisionModel", FakeFastVisionModel)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_module)

    model, tokenizer, backend = app._load_unsloth_fastvision_model("/tmp/adapter", {"": 0}, None)

    assert model.name == "model"
    assert tokenizer == "tokenizer"
    assert backend == "Unsloth FastVisionModel 4bit"
    assert calls == [
        ("empty_cache", None),
        (
            "from_pretrained",
            {
                "model_name": "/tmp/adapter",
                "load_in_4bit": True,
                "use_gradient_checkpointing": "unsloth",
                "max_seq_length": 8192,
                "device_map": {"": 0},
            },
        ),
        ("for_inference", "model"),
    ]


def test_unsloth_loader_retries_without_max_memory_for_older_unsloth(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    calls: list[dict[str, object]] = []

    torch_module = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    setattr(torch_module, "cuda", FakeCuda)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    unsloth_module = types.ModuleType("unsloth")

    class FakeFastVisionModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(dict(kwargs))
            if "max_memory" in kwargs:
                raise TypeError("from_pretrained() got an unexpected keyword argument 'max_memory'")
            return SimpleNamespace(name="model"), "tokenizer"

        @staticmethod
        def for_inference(_model):
            pass

    setattr(unsloth_module, "FastVisionModel", FakeFastVisionModel)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_module)

    app._load_unsloth_fastvision_model("/tmp/adapter", "auto", {0: "20GiB"})

    assert calls[0]["max_memory"] == {0: "20GiB"}
    assert "max_memory" not in calls[1]
    assert calls[1]["device_map"] == "auto"

def test_preload_retry_fails_closed_with_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)

    def broken_load_unsloth(_adapter_local, _device_map, _max_memory):
        raise RuntimeError("cuda unavailable")

    monkeypatch.setattr(app, "_gpu_diagnostics", lambda: {"cuda_available": False})
    monkeypatch.setattr(app, "_snapshot_download_adapter", lambda: "/tmp/adapter-snapshot")
    monkeypatch.setattr(app, "_resolve_device_map_and_memory", lambda: ({"": 0}, None))
    monkeypatch.setattr(app, "_load_unsloth_fastvision_model", broken_load_unsloth)

    status, raw_json = app.preload_model()
    payload = json.loads(raw_json)

    assert "Model status: startup load failed" in status
    assert payload["status"] == "startup_load_failed"
    assert payload["error"]["type"] == "RuntimeError"
    assert "cuda unavailable" in payload["error"]["message"]
    with pytest.raises(app.gr.Error, match="Model startup load failed"):
        app.require_loaded_model()


def test_sample_video_frames_returns_30_rgb_448_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    class FakeCapture:
        def __init__(self, _path: str):
            self.positions: list[int] = []

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == app.cv2.CAP_PROP_FRAME_COUNT:
                return 90.0
            return 0.0

        def set(self, prop: int, value: float) -> None:
            assert prop == app.cv2.CAP_PROP_POS_FRAMES
            self.positions.append(int(value))

        def read(self):
            import numpy as np

            frame = np.zeros((16, 20, 3), dtype=np.uint8)
            frame[:, :, 0] = 255
            return True, frame

        def release(self) -> None:
            pass

    monkeypatch.setattr(app.cv2, "VideoCapture", FakeCapture)

    frames = app.sample_video_frames(video)

    assert len(frames) == 30
    assert all(frame.size == (448, 448) for frame in frames)
    assert all(frame.mode == "RGB" for frame in frames)



def test_run_model_generation_uses_notebook_style_text_then_images(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    captured: dict[str, object] = {}

    class FakeTensor:
        shape = (1, 3)

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["template_kwargs"] = kwargs
            return {"input_ids": FakeTensor()}

        def decode(self, generated, skip_special_tokens=True):
            captured["generated"] = generated
            captured["skip_special_tokens"] = skip_special_tokens
            return " thanks "

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return [[10, 11, 12, 99, 100]]

    torch_module = types.ModuleType("torch")

    class FakeInferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    setattr(torch_module, "inference_mode", lambda: FakeInferenceMode())
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    frames = [SimpleNamespace(name="frame-1"), SimpleNamespace(name="frame-2")]
    app._MODEL_BUNDLE = app.ModelBundle(
        processor=FakeTokenizer(),
        model=FakeModel(),
        diagnostics={"status": "ready"},
        backend="test",
    )

    raw = app.run_model_generation(frames, "Pick one label")

    assert raw == "thanks"
    message = captured["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0] == {"type": "text", "text": "Pick one label"}
    assert message["content"][1:] == [
        {"type": "image", "image": frames[0]},
        {"type": "image", "image": frames[1]},
    ]
    assert captured["template_kwargs"] == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert captured["generate_kwargs"]["max_new_tokens"] == app.MAX_NEW_TOKENS
    assert captured["generate_kwargs"]["do_sample"] is False
    assert captured["generated"] == [99, 100]

def test_translate_video_requires_preloaded_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    with pytest.raises(app.gr.Error, match="Model is not ready"):
        app.translate_video(str(video))


def test_translate_video_returns_structured_inference_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    frames = [SimpleNamespace(size=(448, 448), mode="RGB") for _ in range(30)]
    app._MODEL_BUNDLE = app.ModelBundle(processor="processor", model="model", diagnostics={"status": "ready"}, backend="test backend")

    monkeypatch.setattr(app, "sample_video_frames", lambda _path: frames)

    def broken_generation(_frames, _prompt):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(app, "run_model_generation", broken_generation)

    summary, raw_json = app.translate_video(str(video))
    raw = json.loads(raw_json)

    assert "Inference failed" in summary
    assert raw["status"] == "inference_failed"
    assert raw["error"]["type"] == "RuntimeError"
    assert "cuda out of memory" in raw["error"]["message"]


def test_translate_video_uses_preloaded_model_and_strict_guardrail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app_module(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    frames = [SimpleNamespace(size=(448, 448), mode="RGB") for _ in range(30)]
    app._MODEL_BUNDLE = app.ModelBundle(processor="processor", model="model", diagnostics={"status": "ready"}, backend="test backend")

    monkeypatch.setattr(app, "sample_video_frames", lambda _path: frames)
    monkeypatch.setattr(app, "run_model_generation", lambda _frames, _prompt: "dance")

    summary, raw_json = app.translate_video(str(video))
    raw = json.loads(raw_json)

    assert "Accepted prediction: dance" in summary
    assert "Status: ok" in summary
    assert raw["prediction"] == "dance"
    assert raw["candidate_gloss"] == "dance"
    assert raw["frame_count"] == 30
    assert raw["model_backend"] == "test backend"
