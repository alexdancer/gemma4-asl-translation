"""Demo-facing output contracts for ASL transcription."""

from src.demo.fallback_a import (
    DemoInferenceRunConfig,
    DemoInferenceRunResult,
    run_demo_inference_once,
)
from src.demo.output_contract import DemoOutput, DemoOutputConfig, format_demo_output
from src.demo.prerecorded_q64 import (
    DEMO_CLAIMS,
    DEMO_SCOPE,
    PrerecordedQ64DemoConfig,
    PrerecordedQ64DemoResult,
    run_prerecorded_q64_demo,
)
from src.demo.python_video_prompt_control import (
    PythonVideoPromptControlSmokeConfig,
    PythonVideoPromptControlSmokeResult,
    run_python_video_prompt_control_smoke,
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
    "PythonVideoPromptControlSmokeConfig",
    "PythonVideoPromptControlSmokeResult",
    "format_demo_output",
    "run_demo_inference_once",
    "run_prerecorded_q64_demo",
    "run_python_video_prompt_control_smoke",
    "write_prerecorded_q64_readiness_artifact",
]
