from pathlib import Path


def test_rn_app_contains_structured_error_mapping_and_async_polling_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    app = repo_root / "apps" / "mobile-rn" / "App.tsx"
    ux = repo_root / "apps" / "mobile-rn" / "src" / "inferenceUx.ts"
    app_source = app.read_text(encoding="utf-8")
    ux_source = ux.read_text(encoding="utf-8")

    # Error contract now lives in src/inferenceUx.ts and is imported by App.tsx.
    assert "type TranslateFailure" in ux_source
    assert "retryable: boolean" in ux_source
    assert "const mapErrorMessage" in ux_source
    assert "UNAUTHORIZED" in ux_source
    assert "RATE_LIMITED" in ux_source
    assert "PAYLOAD_TOO_LARGE" in ux_source
    assert "TIMEOUT" in ux_source

    # App must use structured error mapping contract.
    assert "mapErrorMessage" in app_source
    assert "TranslateFailure" in app_source

    # Async poll path contract (submission returns queued/processing and app polls)
    assert "submitResponse.status === 202" in app_source
    assert "status === 'queued'" in app_source
    assert "status === 'processing'" in app_source
    assert "const pollUrl" in app_source
    assert "const maxAttempts = 20" in app_source

    # Terminal failed state should be surfaced with no UI dead-end
    assert "setFailure(timeoutFailure)" in app_source
    assert "Retryable:" in app_source
