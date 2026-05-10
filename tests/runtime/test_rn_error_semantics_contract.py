from pathlib import Path


def test_rn_app_contains_structured_error_mapping_and_async_polling_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    app = repo_root / "apps" / "mobile-rn" / "App.tsx"
    source = app.read_text(encoding="utf-8")

    assert "type TranslateFailure" in source
    assert "retryable: boolean" in source
    assert "const mapErrorMessage" in source
    assert "UNAUTHORIZED" in source
    assert "RATE_LIMITED" in source
    assert "PAYLOAD_TOO_LARGE" in source
    assert "TIMEOUT" in source

    # Async poll path contract (submission returns queued/processing and app polls)
    assert "submitResponse.status === 202" in source
    assert "status === 'queued'" in source
    assert "status === 'processing'" in source
    assert "const pollUrl" in source
    assert "const maxAttempts = 20" in source

    # Terminal failed state should be surfaced with no UI dead-end
    assert "setFailure(timeoutFailure)" in source
    assert "Retryable:" in source
