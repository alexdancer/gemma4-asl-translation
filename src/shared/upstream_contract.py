from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProofFields:
    runtime_mode: str
    cloud_handoff: bool
    model_id: str
    model_version: str


@dataclass(frozen=True)
class UpstreamFailureEvidence:
    upstream_status: int | None
    upstream_code: str
    upstream_message: str
    upstream_request_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "upstream_status": self.upstream_status,
            "upstream_code": self.upstream_code,
            "upstream_message": self.upstream_message,
            "upstream_request_id": self.upstream_request_id,
        }


class ProofValidationError(ValueError):
    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def validate_proof_fields(*, result: dict[str, Any], request_id: str) -> ProofFields:
    """Validate required upstream proof fields.

    Correlation rule: when upstream includes request_id, it must match the
    caller request_id. Missing upstream request_id is allowed.
    """

    required_fields = ("runtime_mode", "cloud_handoff", "model_id", "model_version")
    missing = [field for field in required_fields if field not in result]
    if missing:
        raise ProofValidationError(
            f"Missing upstream proof fields: {', '.join(missing)}",
            kind="missing",
        )

    runtime_mode = result.get("runtime_mode")
    model_id = result.get("model_id")
    model_version = result.get("model_version")
    cloud_handoff = result.get("cloud_handoff")

    if not isinstance(runtime_mode, str) or not runtime_mode.strip():
        raise ProofValidationError("Invalid upstream proof field: runtime_mode", kind="invalid")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ProofValidationError("Invalid upstream proof field: model_id", kind="invalid")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ProofValidationError("Invalid upstream proof field: model_version", kind="invalid")
    if not isinstance(cloud_handoff, bool):
        raise ProofValidationError("Invalid upstream proof field: cloud_handoff", kind="invalid")

    upstream_request_id = str(result.get("request_id") or "").strip()
    if upstream_request_id and upstream_request_id != request_id:
        raise ProofValidationError("Invalid upstream proof field: request_id mismatch", kind="invalid")

    return ProofFields(
        runtime_mode=runtime_mode.strip(),
        cloud_handoff=cloud_handoff,
        model_id=model_id.strip(),
        model_version=model_version.strip(),
    )


def normalize_upstream_failure_evidence(
    *,
    upstream_status: int | None,
    upstream_code: str,
    upstream_message: str,
    upstream_request_id: str | None,
) -> UpstreamFailureEvidence:
    code = str(upstream_code or "CLOUD_HANDOFF_FAILED").strip() or "CLOUD_HANDOFF_FAILED"
    message = str(upstream_message or "Cloud handoff request failed").strip() or "Cloud handoff request failed"
    request_id = str(upstream_request_id or "").strip() or None
    return UpstreamFailureEvidence(
        upstream_status=upstream_status,
        upstream_code=code,
        upstream_message=message,
        upstream_request_id=request_id,
    )
