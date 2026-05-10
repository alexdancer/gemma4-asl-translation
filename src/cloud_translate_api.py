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


def _default_cloud_infer(*, video_bytes: bytes, filename: str, request_id: str, timeout_seconds: float) -> dict[str, Any]:
    endpoint = os.environ.get("ASL_CLOUD_INFER_URL")
    api_key = os.environ.get("ASL_CLOUD_API_KEY")
    model_name = os.environ.get("ASL_CLOUD_MODEL", "cactus-asl-v2")

    if not endpoint:
        raise RuntimeError("ASL_CLOUD_INFER_URL is not configured")
    if not api_key:
        raise RuntimeError("ASL_CLOUD_API_KEY is not configured")

    payload = {
        "request_id": request_id,
        "model": model_name,
        "input": {
            "filename": filename,
            "video_base64": base64.b64encode(video_bytes).decode("ascii"),
            "encoding": "base64",
        },
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

    decoded = json.loads(raw)
    gloss = decoded.get("gloss") or decoded.get("output", {}).get("gloss")
    translation = decoded.get("translation") or decoded.get("output", {}).get("translation")
    confidence = decoded.get("confidence") or decoded.get("output", {}).get("confidence")

    if gloss is None or translation is None or confidence is None:
        raise RuntimeError("upstream response missing required fields")

    latency_ms = int(decoded.get("latency_ms") or decoded.get("timing", {}).get("latency_ms") or 0)
    return {
        "request_id": decoded.get("request_id", request_id),
        "gloss": str(gloss),
        "translation": str(translation),
        "confidence": float(confidence),
        "latency_ms": latency_ms,
    }


def _cloud_infer_supports_pose_handoff(cloud_infer: CloudInferCallable) -> bool:
    try:
        signature = inspect.signature(cloud_infer)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "pose_handoff" in signature.parameters


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
        infer_kwargs = {
            "video_bytes": canonical_video_bytes,
            "filename": filename,
            "request_id": context.request_id,
            "timeout_seconds": timeout_seconds,
        }
        if _cloud_infer_supports_pose_handoff(cloud_infer):
            infer_kwargs["pose_handoff"] = pose_extraction

        result = cloud_infer(**infer_kwargs)
    except TimeoutError:
        latency_ms = max(int((time.monotonic() - context.started) * 1000), 1)
        _safe_append_event(
            TelemetryEvent(
                request_id=context.request_id,
                latency_ms=latency_ms,
                outcome="timeout",
                confidence=0.0,
                model_tag=os.environ.get("ASL_CLOUD_MODEL", "cactus-asl-v2"),
            )
        )
        return _error(
            "TIMEOUT",
            "Cloud inference timed out",
            context.request_id,
            retryable=True,
            status="504 Gateway Timeout",
        )
    except Exception as exc:
        print(json.dumps({"event": "cloud_infer_error", "request_id": context.request_id, "error": str(exc)}))
        latency_ms = max(int((time.monotonic() - context.started) * 1000), 1)
        _safe_append_event(
            TelemetryEvent(
                request_id=context.request_id,
                latency_ms=latency_ms,
                outcome="upstream_failure",
                confidence=0.0,
                model_tag=os.environ.get("ASL_CLOUD_MODEL", "cactus-asl-v2"),
            )
        )
        return _error(
            "UPSTREAM_FAILURE",
            "Cloud inference request failed",
            context.request_id,
            retryable=True,
            status="503 Service Unavailable",
        )

    payload = {
        "request_id": result.get("request_id", context.request_id),
        "gloss": result["gloss"],
        "translation": result["translation"],
        "confidence": float(result["confidence"]),
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
