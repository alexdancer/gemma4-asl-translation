# Cloud SLO Telemetry Reporting (ASL v2)

This repo now records privacy-safe cloud inference telemetry for MVP SLO checks.

## Privacy guardrail
- **Raw uploaded video bytes are not persisted.**
- Persisted telemetry fields only:
  - `request_id`
  - `latency_ms`
  - `outcome`
  - `confidence`
  - `model_tag`

## Telemetry storage
- Default path: `evaluation/results/runtime/cloud_telemetry.jsonl`
- Override with env var: `ASL_TELEMETRY_PATH`

## Cloud config
- `ASL_CLOUD_INFER_URL` (required for real backend)
- `ASL_CLOUD_API_KEY` (server-side key; never client-supplied)
- `ASL_CLOUD_MODEL` (optional tag, defaults to `cactus-asl-v2`)

## Generate SLO snapshot
```bash
python3 scripts/runtime/report_cloud_slo.py
```

Example output:
```json
{
  "count": 42,
  "success_count": 39,
  "success_rate": 0.9286,
  "p95_latency_ms": 781,
  "telemetry_path": "evaluation/results/runtime/cloud_telemetry.jsonl"
}
```
