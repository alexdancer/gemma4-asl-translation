from pathlib import Path


def test_rn_app_is_minimal_upload_status_result_flow() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    app = repo_root / "apps" / "mobile-rn" / "App.tsx"
    source = app.read_text(encoding="utf-8")

    # Keep the essential user path.
    assert "Select Video" in source
    assert "Run Cloud Translation" in source
    assert "Status" in source
    assert "Result" in source
    assert "Error" in source

    # Keep async transport behavior from #64/#72.
    assert "submitResponse.status === 202" in source
    assert "status === 'queued'" in source
    assert "status === 'processing'" in source

    # Remove non-essential diagnostics from main UX path.
    assert "Check Cloud Connection" not in source
    assert "Endpoint check" not in source
    assert "URI: {selectedFile?.uri" not in source
    assert "bytes</Text>" not in source
    assert "Request ID:" not in source
