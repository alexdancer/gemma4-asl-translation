from __future__ import annotations

from src.shared.upstream_contract import (
    ProofValidationError,
    normalize_upstream_failure_evidence,
    validate_proof_fields,
)


def test_validate_proof_fields_accepts_matching_or_missing_upstream_request_id():
    result = {
        "runtime_mode": "cactus_engine",
        "cloud_handoff": True,
        "model_id": "m1",
        "model_version": "v1",
    }
    proof = validate_proof_fields(result=result, request_id="req-1")
    assert proof.runtime_mode == "cactus_engine"
    assert proof.cloud_handoff is True
    assert proof.model_id == "m1"
    assert proof.model_version == "v1"

    with_request = dict(result)
    with_request["request_id"] = "req-1"
    proof2 = validate_proof_fields(result=with_request, request_id="req-1")
    assert proof2.model_id == "m1"


def test_validate_proof_fields_rejects_request_id_mismatch():
    try:
        validate_proof_fields(
            result={
                "runtime_mode": "cactus_engine",
                "cloud_handoff": True,
                "model_id": "m1",
                "model_version": "v1",
                "request_id": "other",
            },
            request_id="req-1",
        )
        raise AssertionError("expected ProofValidationError")
    except ProofValidationError as exc:
        assert exc.kind == "invalid"
        assert "request_id mismatch" in str(exc)


def test_normalize_upstream_failure_evidence_defaults():
    evidence = normalize_upstream_failure_evidence(
        upstream_status=None,
        upstream_code="",
        upstream_message="",
        upstream_request_id="",
    ).to_dict()
    assert evidence["upstream_status"] is None
    assert evidence["upstream_code"] == "CLOUD_HANDOFF_FAILED"
    assert evidence["upstream_message"] == "Cloud handoff request failed"
    assert evidence["upstream_request_id"] is None
