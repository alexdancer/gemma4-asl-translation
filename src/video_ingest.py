from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True)
class CanonicalVideoProfile:
    max_duration_seconds: float = 10.0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    video_codec: str = "libx264"
    pixel_format: str = "yuv420p"


@dataclass(frozen=True)
class VideoProbeResult:
    duration_seconds: float
    fps: float
    width: int
    height: int
    codec: str = "unknown"
    pixel_format: str = "unknown"
    has_audio: bool = False


IngestResult: TypeAlias = tuple[VideoProbeResult, VideoProbeResult, bytes, bool, CanonicalVideoProfile]


class VideoIngestError(RuntimeError):
    pass


def process_uploaded_video(video_bytes: bytes, filename: str, profile: CanonicalVideoProfile | None = None) -> IngestResult:
    profile = profile or CanonicalVideoProfile()

    suffix = Path(filename).suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="asl_ingest_") as tmp_dir:
        in_path = Path(tmp_dir) / f"input{suffix}"
        out_path = Path(tmp_dir) / "normalized.mp4"
        in_path.write_bytes(video_bytes)

        original = probe_video(in_path)
        if original.duration_seconds > profile.max_duration_seconds:
            # duration guardrail is a hard reject policy
            return original, original, video_bytes, False, profile

        if _normalization_required(original, profile):
            normalize_video(in_path, out_path, profile)
            normalized = probe_video(out_path)
            canonical_bytes = out_path.read_bytes()
            return original, normalized, canonical_bytes, True, profile

        return original, original, video_bytes, False, profile


def _build_probe_cmd(video_path: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]


def probe_video(video_path: Path) -> VideoProbeResult:
    try:
        completed = subprocess.run(_build_probe_cmd(video_path), capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoIngestError(f"ffprobe failed: {exc.stderr.strip() or exc}") from exc
    except FileNotFoundError as exc:
        raise VideoIngestError("ffprobe is not installed") from exc

    payload = json.loads(completed.stdout)
    return _probe_result_from_payload(payload)


def _probe_result_from_payload(payload: dict) -> VideoProbeResult:
    format_payload = payload.get("format") or {}
    streams = payload.get("streams") or []
    video_stream = _first_video_stream(streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)

    duration_seconds = _extract_duration_seconds(format_payload, video_stream)
    fps = _parse_fps(str(video_stream.get("avg_frame_rate") or "0/1"))
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoIngestError("video resolution invalid")

    codec = str(video_stream.get("codec_name") or "unknown")
    pixel_format = str(video_stream.get("pix_fmt") or "unknown")

    return VideoProbeResult(
        duration_seconds=duration_seconds,
        fps=fps,
        width=width,
        height=height,
        codec=codec,
        pixel_format=pixel_format,
        has_audio=has_audio,
    )


def _first_video_stream(streams: list[dict]) -> dict:
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video_stream:
        raise VideoIngestError("video stream not found")
    return video_stream


def _extract_duration_seconds(format_payload: dict, video_stream: dict) -> float:
    duration_raw = format_payload.get("duration") or video_stream.get("duration")
    if duration_raw is None:
        raise VideoIngestError("video duration missing")
    return float(duration_raw)


def _build_normalize_cmd(input_path: Path, output_path: Path, profile: CanonicalVideoProfile) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        f"scale={profile.width}:{profile.height}",
        "-r",
        str(profile.fps),
        "-c:v",
        profile.video_codec,
        "-pix_fmt",
        profile.pixel_format,
        str(output_path),
    ]


def normalize_video(input_path: Path, output_path: Path, profile: CanonicalVideoProfile) -> None:
    try:
        subprocess.run(_build_normalize_cmd(input_path, output_path, profile), capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoIngestError(f"ffmpeg normalize failed: {exc.stderr.strip() or exc}") from exc
    except FileNotFoundError as exc:
        raise VideoIngestError("ffmpeg is not installed") from exc


def _normalization_required(probe: VideoProbeResult, profile: CanonicalVideoProfile) -> bool:
    if probe.width != profile.width or probe.height != profile.height:
        return True
    if abs(probe.fps - profile.fps) > 0.01:
        return True
    if probe.codec.lower() not in {"h264", "avc1"}:
        return True
    if probe.pixel_format.lower() != profile.pixel_format.lower():
        return True
    if probe.has_audio:
        return True
    return False


def _parse_fps(value: str) -> float:
    if "/" not in value:
        return float(value)
    num_s, den_s = value.split("/", 1)
    num = float(num_s)
    den = float(den_s)
    if den == 0:
        return 0.0
    return num / den
