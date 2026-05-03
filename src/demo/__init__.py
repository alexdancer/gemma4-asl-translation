"""Demo-facing output contracts for ASL transcription."""

from src.demo.fallback_a import (
    DemoInferenceRunConfig,
    DemoInferenceRunResult,
    run_demo_inference_once,
)
from src.demo.output_contract import DemoOutput, DemoOutputConfig, format_demo_output, load_demo_output_config
from src.demo.prerecorded_q64 import (
    PrerecordedQ64DemoConfig,
    PrerecordedQ64DemoResult,
    run_prerecorded_q64_demo,
)

__all__ = [
    "DemoOutput",
    "DemoOutputConfig",
    "DemoInferenceRunConfig",
    "DemoInferenceRunResult",
    "PrerecordedQ64DemoConfig",
    "PrerecordedQ64DemoResult",
    "format_demo_output",
    "load_demo_output_config",
    "run_demo_inference_once",
    "run_prerecorded_q64_demo",
]
