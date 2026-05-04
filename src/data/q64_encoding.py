"""Shared compact q64 pose encoding helpers."""

from __future__ import annotations

from statistics import mean

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def q64(value: float, clip: float = 4.0) -> str:
    """Quantize a normalized float into one URL-safe base64-ish character."""

    if value < -clip:
        value = -clip
    elif value > clip:
        value = clip
    idx = round(((value + clip) / (2 * clip)) * 63)
    return ALPHABET[max(0, min(63, idx))]


def encode_frames_q64(frames: list[list[float]], stride: int = 1) -> str:
    """Encode selected frames as q64 character rows separated by |."""

    selected = frames[::stride]
    return "|".join("".join(q64(v) for v in frame) for frame in selected)


def encode_summary_q64(frames: list[list[float]]) -> str:
    """Encode per-feature mean and motion delta as compact q64 strings."""

    if not frames:
        return ""
    width = min(len(frame) for frame in frames)
    trimmed = [frame[:width] for frame in frames]
    means = [mean(frame[index] for frame in trimmed) for index in range(width)]
    if len(trimmed) > 1:
        deltas = [trimmed[-1][index] - trimmed[0][index] for index in range(width)]
    else:
        deltas = [0.0] * width
    return "m=" + "".join(q64(value) for value in means) + "\nd=" + "".join(q64(value) for value in deltas)
