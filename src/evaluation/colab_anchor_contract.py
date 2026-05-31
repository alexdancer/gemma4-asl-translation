"""Build deterministic Colab anchor + label contract artifacts for Top-50 evaluation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.unsloth_asl import load_manifest_labels, load_q64_jsonl, normalize_gloss


@dataclass(frozen=True)
class AliasEntry:
    alias: str
    canonical: str


def load_alias_map(path: Path | str | None) -> dict[str, list[str]]:
    """Load canonical->aliases mapping JSON.

    Expected shape:
    {
      "thank you": ["thankyou", "thanks"]
    }
    """

    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("alias map must be a JSON object of canonical -> [aliases].")
    decoded: dict[str, list[str]] = {}
    for canonical, aliases in payload.items():
        if not isinstance(aliases, list):
            raise ValueError(f"aliases for {canonical!r} must be a list.")
        decoded[str(canonical)] = [str(alias) for alias in aliases]
    return decoded


def flatten_alias_map(
    labels: Sequence[str],
    canonical_to_aliases: Mapping[str, Sequence[str]],
) -> list[AliasEntry]:
    """Normalize and validate alias mapping against canonical labels."""

    canonical_set = {normalize_gloss(label) for label in labels}
    flattened: dict[str, str] = {}

    for canonical, aliases in canonical_to_aliases.items():
        canonical_norm = normalize_gloss(canonical)
        if canonical_norm not in canonical_set:
            raise ValueError(f"Alias map canonical label not in allowlist: {canonical}")
        # canonical maps to itself
        existing = flattened.get(canonical_norm)
        if existing is not None and existing != canonical_norm:
            raise ValueError(f"Alias collision for canonical label {canonical_norm}")
        flattened[canonical_norm] = canonical_norm

        for alias in aliases:
            alias_norm = normalize_gloss(alias)
            if not alias_norm:
                continue
            prior = flattened.get(alias_norm)
            if prior is not None and prior != canonical_norm:
                raise ValueError(
                    f"Alias collision: {alias!r} maps to both {prior!r} and {canonical_norm!r}"
                )
            flattened[alias_norm] = canonical_norm

    return [
        AliasEntry(alias=alias, canonical=canonical)
        for alias, canonical in sorted(flattened.items(), key=lambda item: (item[1], item[0]))
    ]


def top_k_anchors(records: Sequence[Mapping[str, Any]], labels: Sequence[str], top_k: int = 10) -> list[dict[str, int | str]]:
    """Select deterministic top-k anchors by frequency then lexical tie-break."""

    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    canonical = {normalize_gloss(label) for label in labels}
    counts: Counter[str] = Counter()
    for record in records:
        output = record.get("output")
        if output is None:
            continue
        normalized = normalize_gloss(str(output))
        if normalized not in canonical:
            raise ValueError(f"Record output not in manifest labels: {output!r}")
        counts[normalized] += 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [{"gloss": gloss, "count": count} for gloss, count in ranked]


def build_colab_anchor_contract(
    *,
    manifest_path: Path | str,
    records_path: Path | str,
    top_k: int = 10,
    canonical_to_aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    labels = list(load_manifest_labels(manifest_path))
    records = load_q64_jsonl(records_path)
    alias_entries = flatten_alias_map(labels, canonical_to_aliases or {})
    anchors = top_k_anchors(records, labels, top_k=top_k)

    return {
        "contract_version": "colab_top50_anchor_contract_v1",
        "top_k": top_k,
        "source": {
            "manifest_path": str(Path(manifest_path)),
            "records_path": str(Path(records_path)),
            "record_count": len(records),
        },
        "allowlist": labels,
        "anchors": anchors,
        "normalization_rules": {
            "lowercase": True,
            "strip_outer_backticks": True,
            "strip_non_alnum_underscore_space_hyphen": True,
            "collapse_whitespace": True,
            "comparison": "normalized_exact_with_alias_map",
        },
        "alias_map": [
            {"alias": item.alias, "canonical": item.canonical}
            for item in alias_entries
        ],
    }


def write_colab_anchor_contract(path: Path | str, contract: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return output_path
