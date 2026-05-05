"""Issue #32 tracer slice: baseline freeze, conversion v1, local completion artifact."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_root = repo_root.expanduser().resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def resolve_git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class TracerSliceConfig:
    checkpoint_path: Path
    output_root: Path = Path("artifacts/cactus_tracer")
    conversion_output_version: str = "v1"
    git_sha: str | None = None
    prompt: str = (
        "You are an ASL gloss classifier. Return exactly one uppercase gloss label and no extra text."
    )
    allow_real_export: bool = True
    repo_root: Path = Path(".")


@dataclass(frozen=True)
class TracerSliceResult:
    freeze_metadata_path: str
    converted_weights_dir: str
    completion_artifact_path: str
    summary_path: str
    acceptance_proof_satisfied: bool


def write_frozen_baseline_metadata(config: TracerSliceConfig) -> Path:
    checkpoint = config.checkpoint_path.expanduser().resolve()
    output = config.output_root / "frozen_baseline_metadata.json"
    git_sha = config.git_sha or resolve_git_sha(config.repo_root)
    payload = {
        "scope": "cactus_tracer_slice",
        "checkpoint_id": checkpoint.name,
        "checkpoint_path": _to_repo_relative(checkpoint, config.repo_root),
        "git_sha": git_sha,
        "conversion_output_version": config.conversion_output_version,
        "captured_at": _utc_now(),
    }
    _write_json(output, payload)
    return output


def _write_conversion_fallback_manifest(
    weights_dir: Path,
    checkpoint_path: Path,
    conversion_output_version: str,
    error: str,
    repo_root: Path,
) -> None:
    payload = {
        "scope": "cactus_tracer_slice",
        "conversion_output_version": conversion_output_version,
        "checkpoint_path": _to_repo_relative(checkpoint_path, repo_root),
        "conversion_mode": "deterministic_fallback",
        "success": False,
        "error": error,
        "captured_at": _utc_now(),
    }
    _write_json(weights_dir / "conversion_manifest.json", payload)


def produce_cactus_weights_v1(config: TracerSliceConfig) -> tuple[Path, dict[str, Any]]:
    checkpoint = config.checkpoint_path.expanduser().resolve()
    version = config.conversion_output_version
    weights_dir = (config.output_root / "converted_weights" / version).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "VERSION").write_text(version + "\n", encoding="utf-8")

    if not config.allow_real_export:
        _write_conversion_fallback_manifest(
            weights_dir=weights_dir,
            checkpoint_path=checkpoint,
            conversion_output_version=version,
            error="Real export disabled via configuration.",
            repo_root=config.repo_root,
        )
        return weights_dir, {
            "mode": "deterministic_fallback",
            "success": False,
            "error": "Real export disabled via configuration.",
        }

    try:
        from src.mobile.cactus_export import CactusModelExporter

        exporter = CactusModelExporter(output_dir=weights_dir, model_version=version)
        export_result = exporter.export(checkpoint_path=checkpoint)
        manifest_payload = {
            "scope": "cactus_tracer_slice",
            "conversion_output_version": version,
            "checkpoint_path": _to_repo_relative(checkpoint, config.repo_root),
            "conversion_mode": "real_export",
            "success": bool(export_result.get("success", False)),
            "error": None,
            "export_result": export_result,
            "captured_at": _utc_now(),
        }
        _write_json(weights_dir / "conversion_manifest.json", manifest_payload)
        return weights_dir, {
            "mode": "real_export",
            "success": manifest_payload["success"],
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - exercised in integration environments
        _write_conversion_fallback_manifest(
            weights_dir=weights_dir,
            checkpoint_path=checkpoint,
            conversion_output_version=version,
            error=f"{type(exc).__name__}: {exc}",
            repo_root=config.repo_root,
        )
        return weights_dir, {
            "mode": "deterministic_fallback",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _deterministic_fallback_response(prompt: str) -> str:
    _ = prompt
    return "HELLO"


def run_local_completion(
    *,
    weights_dir: Path,
    prompt: str,
    artifact_path: Path,
    repo_root: Path,
    prefer_cactus_engine: bool = True,
) -> dict[str, Any]:
    start = time.perf_counter()
    runtime_mode = "deterministic_fallback"
    response = _deterministic_fallback_response(prompt)
    success = False
    error: str | None = "Cactus engine not attempted."
    runtime_warning: str | None = None

    if prefer_cactus_engine:
        try:
            from src.cactus import cactus_complete, cactus_destroy, cactus_init

            weights = weights_dir.expanduser().resolve()
            if not weights.exists():
                raise FileNotFoundError(f"Cactus weights directory not found: {weights}")

            model = cactus_init(str(weights), None, False)
            try:
                messages = json.dumps([{"role": "user", "content": prompt}])
                options = json.dumps({"temperature": 0.0, "max_tokens": 8})
                raw = cactus_complete(model, messages, options, None, None)
                payload = json.loads(raw)
                runtime_mode = "cactus_engine"
                success = bool(payload.get("success", False))
                response = str(payload.get("response", ""))
                error = payload.get("error")
            finally:
                cactus_destroy(model)
        except Exception as exc:
            runtime_mode = "deterministic_fallback"
            response = _deterministic_fallback_response(prompt)
            success = False
            error = f"{type(exc).__name__}: {exc}"
            runtime_warning = f"Fell back to deterministic tracer completion: {type(exc).__name__}: {exc}"

    timing_ms = round((time.perf_counter() - start) * 1000.0, 3)
    artifact = {
        "scope": "cactus_tracer_slice",
        "runtime_mode": runtime_mode,
        "success": success,
        "error": error,
        "response": response,
        "timing_ms": timing_ms,
        "prompt": prompt,
        "weights_dir": _to_repo_relative(weights_dir, repo_root),
        "captured_at": _utc_now(),
    }
    if runtime_warning is not None:
        artifact["runtime_warning"] = runtime_warning

    _write_json(artifact_path, artifact)
    return artifact


def _acceptance_proof_satisfied(conversion_status: dict[str, Any], completion_payload: dict[str, Any]) -> bool:
    return (
        conversion_status.get("mode") == "real_export"
        and bool(conversion_status.get("success", False))
        and completion_payload.get("runtime_mode") == "cactus_engine"
        and bool(completion_payload.get("success", False))
        and not completion_payload.get("error")
    )


def run_cactus_tracer_slice(config: TracerSliceConfig) -> TracerSliceResult:
    freeze_path = write_frozen_baseline_metadata(config)
    weights_dir, conversion_status = produce_cactus_weights_v1(config)

    completion_path = (config.output_root / f"local_completion_{config.conversion_output_version}.json").resolve()
    completion_payload = run_local_completion(
        weights_dir=weights_dir,
        prompt=config.prompt,
        artifact_path=completion_path,
        repo_root=config.repo_root,
        prefer_cactus_engine=True,
    )

    proof_satisfied = _acceptance_proof_satisfied(conversion_status, completion_payload)
    summary_path = (config.output_root / "run_summary.json").resolve()
    summary_payload = {
        "scope": "cactus_tracer_slice",
        "freeze_metadata_path": _to_repo_relative(freeze_path, config.repo_root),
        "converted_weights_dir": _to_repo_relative(weights_dir, config.repo_root),
        "completion_artifact_path": _to_repo_relative(completion_path, config.repo_root),
        "conversion_status": conversion_status,
        "completion_status": {
            "runtime_mode": completion_payload["runtime_mode"],
            "success": completion_payload["success"],
            "error": completion_payload["error"],
        },
        "acceptance_proof_satisfied": proof_satisfied,
        "captured_at": _utc_now(),
    }
    _write_json(summary_path, summary_payload)

    return TracerSliceResult(
        freeze_metadata_path=str(freeze_path.resolve()),
        converted_weights_dir=str(weights_dir.resolve()),
        completion_artifact_path=str(completion_path.resolve()),
        summary_path=str(summary_path.resolve()),
        acceptance_proof_satisfied=proof_satisfied,
    )


def result_to_dict(result: TracerSliceResult) -> dict[str, str]:
    return asdict(result)
