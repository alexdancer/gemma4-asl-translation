"""Demo-facing output contracts for ASL transcription."""

from src.demo.fallback_a import (
    DemoInferenceRunConfig,
    DemoInferenceRunResult,
    run_demo_inference_once,
)
from src.demo.output_contract import DemoOutput, DemoOutputConfig, format_demo_output, load_demo_output_config
from src.demo.prerecorded_q64 import (
    DEMO_CLAIMS,
    DEMO_SCOPE,
    PrerecordedQ64DemoConfig,
    PrerecordedQ64DemoResult,
    run_prerecorded_q64_demo,
)
from src.demo.readiness_artifacts import write_prerecorded_q64_readiness_artifact

__all__ = [
    "DemoOutput",
    "DemoOutputConfig",
    "DemoInferenceRunConfig",
    "DemoInferenceRunResult",
    "DEMO_SCOPE",
    "DEMO_CLAIMS",
    "PrerecordedQ64DemoConfig",
    "PrerecordedQ64DemoResult",
    "format_demo_output",
    "load_demo_output_config",
    "run_demo_inference_once",
    "run_prerecorded_q64_demo",
    "write_prerecorded_q64_readiness_artifact",
]
