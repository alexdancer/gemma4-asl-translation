"""ASL v2 cloud translation API skeleton.

This module provides a minimal WSGI-compatible app with a single endpoint:
POST /v1/translate-sign

Behavior for slice #51:
- Accept multipart/form-data with a `video` file part
- Return mock success payload with PRD fields
- Return standardized error payload on invalid requests
- Guard with a 12-second timeout scaffold
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest

from src.telemetry_slo import TelemetryEvent, append_event
from src.frame_extraction import (
    FrameExtractionError,
    MAX_FRAMES,
    OVERFLOW_POLICY,
    extract_frames_from_canonical_video,
)
from src.video_ingest import VideoIngestError, process_uploaded_video
from src.pose_handoff import PosePipelineError, run_pose_extraction_pipeline

Response = tuple[str, list[tuple[str, str]], bytes]
CloudInferCallable = Callable[..., dict[str, Any]]
MAX_UPLOAD_BYTES = 60_000_000
DEFAULT_RATE_LIMIT_REQUESTS = 60
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60

_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_HITS_BY_KEY: dict[str, list[float]] = {}


class CloudInferError(RuntimeError):
    """Structured inference-stage error for deterministic API error mapping.

    Lower layers raise this with explicit code/retryable/status so the top-level
    request handler can return a stable error contract without ad-hoc parsing.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        status: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.details = details or {}


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    started: float
    method: str
    path: str
    content_type: str
    body: bytes


def _safe_append_event(event: TelemetryEvent) -> None:
    try:
        append_event(event)
    except Exception as exc:
        print(json.dumps({"event": "telemetry_write_failed", "error": str(exc)}))


def _redacted_filename_hint(filename: str) -> str:
    digest = hashlib.sha256(filename.encode("utf-8", errors="ignore")).hexdigest()[:10]
    ext = os.path.splitext(filename)[1].lower() or ".bin"
    return f"file_{digest}{ext}"


def _json_response(status: str, payload: dict[str, Any]) -> Response:
    return status, [("Content-Type", "application/json")], json.dumps(payload).encode("utf-8")


def _error(
    error_code: str,
    message: str,
    request_id: str,
    retryable: bool,
    status: str = "400 Bad Request",
    details: dict[str, Any] | None = None,
) -> Response:
    payload = {
        "error_code": error_code,
        "message": message,
        "request_id": request_id,
        "retryable": retryable,
    }
    if details:
        payload["details"] = details
    return _json_response(status, payload)


def _build_request_context(environ: dict[str, Any]) -> RequestContext:
    return RequestContext(
        request_id=environ.get("HTTP_X_REQUEST_ID") or str(uuid.uuid4()),
        started=time.monotonic(),
        method=environ.get("REQUEST_METHOD", ""),
        path=environ.get("PATH_INFO", ""),
        content_type=environ.get("CONTENT_TYPE", ""),
        body=environ.get("wsgi.input_body", b""),
    )


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _load_allowed_api_keys() -> set[str]:
    """Load active API keys from env (current + next rotation set)."""

    key_sets = [os.environ.get("ASL_V1_API_KEYS", ""), os.environ.get("ASL_V1_API_KEYS_NEXT", "")]
    allowed: set[str] = set()
    for key_set in key_sets:
        for key in key_set.split(","):
            normalized = key.strip()
            if normalized:
                allowed.add(normalized)
    return allowed


def _extract_api_key(environ: dict[str, Any]) -> str | None:
    """Extract API key from Authorization Bearer token or X-API-Key header."""

    auth_header = str(environ.get("HTTP_AUTHORIZATION") or "").strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    x_api_key = str(environ.get("HTTP_X_API_KEY") or "").strip()
    return x_api_key or None


def _check_rate_limit(*, api_key: str, now: float) -> tuple[bool, dict[str, Any] | None]:
    """Enforce lightweight per-key rate limiting with an in-memory sliding window."""

    limit = _parse_positive_int_env("ASL_V1_RATE_LIMIT_REQUESTS", DEFAULT_RATE_LIMIT_REQUESTS)
    window_seconds = _parse_positive_int_env("ASL_V1_RATE_LIMIT_WINDOW_SECONDS", DEFAULT_RATE_LIMIT_WINDOW_SECONDS)
    window_floor = now - float(window_seconds)

    with _RATE_LIMIT_LOCK:
        hits = _RATE_LIMIT_HITS_BY_KEY.get(api_key, [])
        hits = [ts for ts in hits if ts >= window_floor]
        if len(hits) >= limit:
            retry_after_seconds = max(1, int(window_seconds - (now - min(hits))))
            _RATE_LIMIT_HITS_BY_KEY[api_key] = hits
            return False, {
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after_seconds": retry_after_seconds,
            }

        hits.append(now)
        _RATE_LIMIT_HITS_BY_KEY[api_key] = hits

    return True, None


def _build_video_ingest_payload(
    *,
    original_probe: Any,
    normalized_probe: Any,
    normalization_applied: bool,
    profile: Any,
) -> dict[str, Any]:
    return {
        "normalization_applied": normalization_applied,
        "target": {
            "max_duration_seconds": profile.max_duration_seconds,
            "fps": profile.fps,
            "width": profile.width,
            "height": profile.height,
            "video_codec": profile.video_codec,
            "pixel_format": profile.pixel_format,
            "audio": "stripped",
        },
        "original": {
            "duration_seconds": original_probe.duration_seconds,
            "fps": original_probe.fps,
            "width": original_probe.width,
            "height": original_probe.height,
            "codec": original_probe.codec,
            "pixel_format": original_probe.pixel_format,
            "has_audio": original_probe.has_audio,
        },
        "normalized": {
            "duration_seconds": normalized_probe.duration_seconds,
            "fps": normalized_probe.fps,
            "width": normalized_probe.width,
            "height": normalized_probe.height,
            "codec": normalized_probe.codec,
            "pixel_format": normalized_probe.pixel_format,
            "has_audio": normalized_probe.has_audio,
        },
    }


def _log_translate_sign_event(
    *,
    request_id: str,
    filename: str,
    bytes_received: int,
    canonical_video_bytes: bytes,
    normalization_applied: bool,
    original_probe: Any,
    normalized_probe: Any,
    frame_extraction: Any,
    latency_ms: int,
) -> None:
    print(
        json.dumps(
            {
                "event": "translate_sign",
                "request_id": request_id,
                "filename_hint": _redacted_filename_hint(filename),
                "bytes_received": bytes_received,
                "normalized_bytes": len(canonical_video_bytes),
                "normalization_applied": normalization_applied,
                "original_duration_seconds": original_probe.duration_seconds,
                "normalized_duration_seconds": normalized_probe.duration_seconds,
                "original_fps": original_probe.fps,
                "normalized_fps": normalized_probe.fps,
                "original_width": original_probe.width,
                "original_height": original_probe.height,
                "normalized_width": normalized_probe.width,
                "normalized_height": normalized_probe.height,
                "frame_count": frame_extraction.frame_count,
                "first_ts_ms": frame_extraction.first_ts_ms,
                "last_ts_ms": frame_extraction.last_ts_ms,
                "effective_fps": frame_extraction.effective_fps,
                "cadence": frame_extraction.cadence,
                "latency_ms": latency_ms,
                "status": "ok",
            }
        )
    )


def _extract_video_part(content_type: str, body: bytes) -> tuple[bytes, str] | None:
    if "multipart/form-data" not in content_type:
        return None

    envelope = b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    message = BytesParser(policy=default).parsebytes(envelope)

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        if name != "video":
            continue

        filename = part.get_param("filename", header="content-disposition") or "upload.bin"
        payload = part.get_payload(decode=True) or b""
        return payload, filename

    return None


def _build_inference_input_payload(*, filename: str, pose_handoff: Any, include_pose_sequence: bool) -> dict[str, Any]:
    """Build provider input from pose extraction output.

    Default path keeps payload compact (`pose_summary`). Full per-frame
    `pose_sequence` is opt-in for debug/experiments to control request size.
    """

    payload = {
        "filename": filename,
        "pose_summary": {
            "frame_count": int(pose_handoff.frame_count),
            "first_ts_ms": int(pose_handoff.first_ts_ms),
            "last_ts_ms": int(pose_handoff.last_ts_ms),
        },
    }
    if include_pose_sequence:
        payload["pose_sequence"] = [
            {
                "index": int(frame.index),
                "timestamp_ms": int(frame.timestamp_ms),
                "landmarks": frame.landmarks,
            }
            for frame in getattr(pose_handoff, "frames", [])
        ]
    return payload


def _normalize_inference_result(*, result: dict[str, Any], include_provider_debug: bool) -> dict[str, Any]:
    """Normalize provider response into the app-facing response contract.

    Providers may return different shapes (`prediction`, `translation`, nested
    `output.translation`), but the API always emits prediction/confidence with
    predictable validation and error typing.
    """

    def _as_int(value: Any, *, field_name: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise CloudInferError(
                "INFERENCE_INVALID_RESPONSE",
                f"Model provider returned invalid {field_name}",
                retryable=False,
                status="422 Unprocessable Entity",
            ) from exc

    def _as_float(value: Any, *, field_name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise CloudInferError(
                "INFERENCE_INVALID_RESPONSE",
                f"Model provider returned invalid {field_name}",
                retryable=False,
                status="422 Unprocessable Entity",
            ) from exc

    def _pick_value(primary: Any, secondary: Any, default: Any) -> Any:
        if primary is not None:
            return primary
        if secondary is not None:
            return secondary
        return default

    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y"}:
                return True
            if lowered in {"0", "false", "no", "n", ""}:
                return False
        return False

    prediction = result.get("prediction") or result.get("translation") or result.get("output", {}).get("translation")
    output_obj = result.get("output") if isinstance(result.get("output"), dict) else {}
    transcript_words = result.get("transcript_words") or output_obj.get("transcript_words") or result.get("words") or []
    if transcript_words and not isinstance(transcript_words, list):
        transcript_words = []

    if transcript_words:
        normalized_words: list[dict[str, Any]] = []
        for item in transcript_words:
            if not isinstance(item, dict):
                continue
            word_value = item.get("word")
            if not word_value:
                continue
            normalized_words.append(
                {
                    "word": str(word_value),
                    "start_ms": _as_int(
                        _pick_value(item.get("start_ms"), item.get("start"), 0),
                        field_name="transcript_words.start_ms",
                    ),
                    "end_ms": _as_int(
                        _pick_value(item.get("end_ms"), item.get("end"), 0),
                        field_name="transcript_words.end_ms",
                    ),
                    "confidence": _as_float(
                        item.get("confidence") if item.get("confidence") is not None else 0.0,
                        field_name="transcript_words.confidence",
                    ),
                }
            )
        transcript_words = normalized_words

    if not prediction and transcript_words:
        prediction = " ".join(word["word"] for word in transcript_words)

    if "confidence" in result and result.get("confidence") is not None:
        confidence = result.get("confidence")
    else:
        confidence = output_obj.get("confidence")
    alternatives = result.get("alternatives") or output_obj.get("alternatives") or []

    if not prediction:
        raise CloudInferError(
            "INFERENCE_INVALID_RESPONSE",
            "Model provider returned empty prediction",
            retryable=False,
            status="422 Unprocessable Entity",
        )
    if confidence is None:
        raise CloudInferError(
            "INFERENCE_INVALID_RESPONSE",
            "Model provider response missing confidence",
            retryable=False,
            status="422 Unprocessable Entity",
        )

    normalized = {
        "prediction": str(prediction),
        "confidence": _as_float(confidence, field_name="confidence"),
        "alternatives": alternatives,
        "transcript_words": transcript_words
        if transcript_words
        else [
            {
                "word": str(prediction),
                "start_ms": 0,
                "end_ms": 0,
                "confidence": _as_float(confidence, field_name="confidence"),
            }
        ],
        "sequence_confidence": _as_float(
            result.get("sequence_confidence")
            if result.get("sequence_confidence") is not None
            else output_obj.get("sequence_confidence")
            if output_obj.get("sequence_confidence") is not None
            else confidence,
            field_name="sequence_confidence",
        ),
        "low_confidence": _as_bool(
            result.get("low_confidence")
            if result.get("low_confidence") is not None
            else output_obj.get("low_confidence")
            if output_obj.get("low_confidence") is not None
            else False
        ),
    }
    if include_provider_debug:
        normalized["provider_debug"] = result
    return normalized


def _default_cloud_infer(
    *,
    video_bytes: bytes,
    filename: str,
    request_id: str,
    timeout_seconds: float,
    pose_handoff: Any | None = None,
) -> dict[str, Any]:
    """Invoke upstream model provider and return normalized internal fields.

    This function is the network boundary: it validates required env config,
    sends request payload (optionally enriched with pose data), and converts
    upstream transport/schema failures into typed inference errors.
    """

    endpoint = os.environ.get("ASL_CLOUD_INFER_URL")
    api_key = os.environ.get("ASL_CLOUD_API_KEY")
    model_name = os.environ.get("ASL_CLOUD_MODEL", "cactus-asl-v2")

    if not endpoint:
        raise RuntimeError("ASL_CLOUD_INFER_URL is not configured")
    if not api_key:
        raise RuntimeError("ASL_CLOUD_API_KEY is not configured")

    include_pose_sequence = os.environ.get("ASL_INFER_INCLUDE_POSE_SEQUENCE", "0") == "1"
    input_payload = {
        "filename": filename,
        "video_base64": base64.b64encode(video_bytes).decode("ascii"),
        "encoding": "base64",
    }
    if pose_handoff is not None:
        input_payload.update(
            _build_inference_input_payload(
                filename=filename,
                pose_handoff=pose_handoff,
                include_pose_sequence=include_pose_sequence,
            )
        )

    payload = {
        "request_id": request_id,
        "model": model_name,
        "input": input_payload,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Request-ID": request_id,
        },
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except TimeoutError as exc:
        raise TimeoutError("upstream timeout") from exc
    except urlerror.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError("upstream timeout") from exc
        raise RuntimeError(f"upstream request failed: {exc.reason}") from exc

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CloudInferError(
            "INFERENCE_UPSTREAM_MALFORMED",
            "Model provider returned malformed JSON",
            retryable=True,
            status="502 Bad Gateway",
        ) from exc

    prediction = decoded.get("prediction") or decoded.get("translation") or decoded.get("output", {}).get("translation")
    output_obj = decoded.get("output") if isinstance(decoded.get("output"), dict) else {}
    if "confidence" in decoded and decoded.get("confidence") is not None:
        confidence = decoded.get("confidence")
    else:
        confidence = output_obj.get("confidence")
    if not prediction or confidence is None:
        raise CloudInferError(
            "INFERENCE_INVALID_RESPONSE",
            "Model provider response missing required prediction/confidence fields",
            retryable=False,
            status="422 Unprocessable Entity",
        )

    latency_ms = int(decoded.get("latency_ms") or decoded.get("timing", {}).get("latency_ms") or 0)
    return {
        "request_id": decoded.get("request_id", request_id),
        "prediction": str(prediction),
        "confidence": float(confidence),
        "alternatives": decoded.get("alternatives") or decoded.get("output", {}).get("alternatives") or [],
        "transcript_words": decoded.get("transcript_words")
        or decoded.get("output", {}).get("transcript_words")
        or decoded.get("words")
        or [],
        "sequence_confidence": decoded.get("sequence_confidence")
        if decoded.get("sequence_confidence") is not None
        else decoded.get("output", {}).get("sequence_confidence"),
        "low_confidence": decoded.get("low_confidence")
        if decoded.get("low_confidence") is not None
        else decoded.get("output", {}).get("low_confidence"),
        "provider_raw": decoded,
        "latency_ms": latency_ms,
    }


def _cloud_infer_supports_pose_handoff(cloud_infer: CloudInferCallable) -> bool:
    """Return True if injected inference callable accepts `pose_handoff`.

    Keeps backward compatibility with older callables in tests/runtime while
    enabling richer inference input for newer implementations.
    """

    try:
        signature = inspect.signature(cloud_infer)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "pose_handoff" in signature.parameters


def _record_infer_failure_telemetry(*, request_id: str, started: float, outcome: str) -> None:
    """Emit a standardized telemetry event for inference failure paths."""

    latency_ms = max(int((time.monotonic() - started) * 1000), 1)
    _safe_append_event(
        TelemetryEvent(
            request_id=request_id,
            latency_ms=latency_ms,
            outcome=outcome,
            confidence=0.0,
            model_tag=os.environ.get("ASL_CLOUD_MODEL", "cactus-asl-v2"),
        )
    )


def _build_infer_kwargs(
    *,
    cloud_infer: CloudInferCallable,
    canonical_video_bytes: bytes,
    filename: str,
    request_id: str,
    timeout_seconds: float,
    pose_extraction: Any,
) -> dict[str, Any]:
    """Build cloud inference kwargs with backward-compatible pose injection."""

    infer_kwargs: dict[str, Any] = {
        "video_bytes": canonical_video_bytes,
        "filename": filename,
        "request_id": request_id,
        "timeout_seconds": timeout_seconds,
    }
    if _cloud_infer_supports_pose_handoff(cloud_infer):
        infer_kwargs["pose_handoff"] = pose_extraction
    return infer_kwargs


def translate_sign_wsgi_app(
    environ: dict[str, Any],
    _start_response: Callable[[str, list[tuple[str, str]]], None] | None = None,
    timeout_seconds: float = 12.0,
) -> Response:
    """Handle POST /v1/translate-sign and return JSON response."""

    context = _build_request_context(environ)

    if context.method != "POST" or context.path != "/v1/translate-sign":
        return _error("NOT_FOUND", "Endpoint not found", context.request_id, retryable=False, status="404 Not Found")

    # 12-second timeout guard scaffold
    elapsed = time.monotonic() - context.started
    if elapsed > timeout_seconds:
        return _error("TIMEOUT", "Request exceeded timeout", context.request_id, retryable=True, status="504 Gateway Timeout")

    allowed_keys = _load_allowed_api_keys()
    presented_key = _extract_api_key(environ)
    if not allowed_keys or not presented_key or presented_key not in allowed_keys:
        return _error(
            "UNAUTHORIZED",
            "Missing or invalid API key",
            context.request_id,
            retryable=False,
            status="401 Unauthorized",
        )

    allowed_request, rate_limit_details = _check_rate_limit(api_key=presented_key, now=time.monotonic())
    if not allowed_request:
        return _error(
            "RATE_LIMITED",
            "Rate limit exceeded for API key",
            context.request_id,
            retryable=True,
            status="429 Too Many Requests",
            details=rate_limit_details,
        )

    video_part = _extract_video_part(context.content_type, context.body)
    if video_part is None:
        return _error(
            "INVALID_REQUEST",
            "Expected multipart/form-data with a video part named 'video'",
            context.request_id,
            retryable=False,
        )

    video_bytes, filename = video_part
    if not video_bytes:
        return _error("INVALID_VIDEO", "Uploaded video is empty", context.request_id, retryable=False)
    if len(video_bytes) > MAX_UPLOAD_BYTES:
        return _error(
            "PAYLOAD_TOO_LARGE",
            "Uploaded video exceeds 60MB limit",
            context.request_id,
            retryable=False,
            status="413 Payload Too Large",
        )

    video_ingest = environ.get("video_ingest_callable") or process_uploaded_video
    try:
        original_probe, normalized_probe, canonical_video_bytes, normalization_applied, profile = video_ingest(
            video_bytes,
            filename,
        )
    except (VideoIngestError, RuntimeError, ValueError) as exc:
        return _error("INVALID_VIDEO", f"Uploaded video could not be processed: {exc}", context.request_id, retryable=False)

    if original_probe.duration_seconds > profile.max_duration_seconds:
        return _error(
            "VIDEO_DURATION_EXCEEDED",
            f"Uploaded video exceeds maximum duration of {profile.max_duration_seconds:.0f} seconds",
            context.request_id,
            retryable=False,
        )

    frame_extractor = environ.get("frame_extractor_callable") or extract_frames_from_canonical_video
    try:
        frame_extraction = frame_extractor(canonical_video_bytes, probe=normalized_probe)
    except FrameExtractionError as exc:
        return _error(
            exc.code,
            str(exc),
            context.request_id,
            retryable=exc.retryable,
            status=exc.status,
            details=exc.details,
        )
    except Exception as exc:
        return _error(
            "FRAME_EXTRACTION_FAILED",
            f"Frame extraction failed: {exc}",
            context.request_id,
            retryable=False,
            status="422 Unprocessable Entity",
        )

    pose_pipeline = environ.get("pose_pipeline_callable") or run_pose_extraction_pipeline
    try:
        pose_extraction = pose_pipeline(request_id=context.request_id, frame_extraction=frame_extraction)
    except PosePipelineError as exc:
        return _error(
            exc.code,
            str(exc),
            context.request_id,
            retryable=exc.retryable,
            status=exc.status,
            details=exc.details,
        )
    except Exception as exc:
        return _error(
            "POSE_EXTRACTION_FAILED",
            f"Pose extraction failed: {exc}",
            context.request_id,
            retryable=False,
            status="422 Unprocessable Entity",
        )

    cloud_infer = environ.get("cloud_infer_callable") or _default_cloud_infer

    try:
        infer_kwargs = _build_infer_kwargs(
            cloud_infer=cloud_infer,
            canonical_video_bytes=canonical_video_bytes,
            filename=filename,
            request_id=context.request_id,
            timeout_seconds=timeout_seconds,
            pose_extraction=pose_extraction,
        )
        result = cloud_infer(**infer_kwargs)
    except TimeoutError:
        _record_infer_failure_telemetry(request_id=context.request_id, started=context.started, outcome="timeout")
        return _error(
            "TIMEOUT",
            "Cloud inference timed out",
            context.request_id,
            retryable=True,
            status="504 Gateway Timeout",
        )
    except CloudInferError as exc:
        print(json.dumps({"event": "cloud_infer_error", "request_id": context.request_id, "error": str(exc), "code": exc.code}))
        _record_infer_failure_telemetry(request_id=context.request_id, started=context.started, outcome="upstream_failure")
        return _error(
            exc.code,
            str(exc),
            context.request_id,
            retryable=exc.retryable,
            status=exc.status,
            details=exc.details,
        )
    except Exception as exc:
        print(json.dumps({"event": "cloud_infer_error", "request_id": context.request_id, "error": str(exc)}))
        _record_infer_failure_telemetry(request_id=context.request_id, started=context.started, outcome="upstream_failure")
        return _error(
            "UPSTREAM_FAILURE",
            "Cloud inference request failed",
            context.request_id,
            retryable=True,
            status="503 Service Unavailable",
        )

    include_provider_debug = environ.get("inference_debug") is True
    try:
        # Normalize once at the API boundary to keep client-visible schema stable
        # even when upstream provider response shape varies.
        normalized_inference = _normalize_inference_result(result=result, include_provider_debug=include_provider_debug)
    except CloudInferError as exc:
        return _error(
            exc.code,
            str(exc),
            context.request_id,
            retryable=exc.retryable,
            status=exc.status,
            details=exc.details,
        )

    payload = {
        "request_id": result.get("request_id", context.request_id),
        "prediction": normalized_inference["prediction"],
        "confidence": normalized_inference["confidence"],
        "alternatives": normalized_inference["alternatives"],
        "translation": normalized_inference["prediction"],
        "transcript_words": normalized_inference["transcript_words"],
        "sequence_confidence": normalized_inference["sequence_confidence"],
        "low_confidence": normalized_inference["low_confidence"],
        "status": "completed",
        "latency_ms": int(result.get("latency_ms", (time.monotonic() - context.started) * 1000)),
        "video_ingest": _build_video_ingest_payload(
            original_probe=original_probe,
            normalized_probe=normalized_probe,
            normalization_applied=normalization_applied,
            profile=profile,
        ),
        "frame_extraction": {
            "frame_count": frame_extraction.frame_count,
            "first_ts_ms": frame_extraction.first_ts_ms,
            "last_ts_ms": frame_extraction.last_ts_ms,
            "effective_fps": frame_extraction.effective_fps,
            "cadence": frame_extraction.cadence,
            "max_frames": MAX_FRAMES,
            "overflow_policy": OVERFLOW_POLICY,
        },
        "pose_extraction": {
            "frame_count": pose_extraction.frame_count,
            "first_ts_ms": pose_extraction.first_ts_ms,
            "last_ts_ms": pose_extraction.last_ts_ms,
            "aligned_with_frame_timestamps": True,
        },
    }
    if include_provider_debug and "provider_debug" in normalized_inference:
        payload["provider_debug"] = normalized_inference["provider_debug"]

    _safe_append_event(
        TelemetryEvent(
            request_id=payload["request_id"],
            latency_ms=payload["latency_ms"],
            outcome="success",
            confidence=payload["confidence"],
            model_tag=os.environ.get("ASL_CLOUD_MODEL", "cactus-asl-v2"),
        )
    )

    _log_translate_sign_event(
        request_id=payload["request_id"],
        filename=filename,
        bytes_received=len(video_bytes),
        canonical_video_bytes=canonical_video_bytes,
        normalization_applied=normalization_applied,
        original_probe=original_probe,
        normalized_probe=normalized_probe,
        frame_extraction=frame_extraction,
        latency_ms=payload["latency_ms"],
    )

    return _json_response("200 OK", payload)


__all__ = ["translate_sign_wsgi_app"]
