# ASL v2 PRD — Cloud-First Video Translation

## Summary
ASL v2 pivots from local-first inference to a **cloud-first architecture** using Cactus cloud path. The product goal is simple and shippable: **record a short signing video on iPhone → upload → return gloss + natural-language translation** with acceptable reliability and latency.

## Why we are changing course
- Current Gemma4 E4B local path is too slow for practical in-app UX.
- E2B may help, but not enough to justify blocking product delivery on local optimization.
- We need a reliable, demoable end-to-end path now.

## Locked product/runtime decisions
1. Cloud-first; local disabled for v2 MVP.
2. Raw video upload first (no on-device feature extraction in MVP).
3. Use Cactus cloud-first path directly.
4. Cloud-only runtime in UI/runtime (no local toggle).
5. Return both **gloss** and **natural-language translation**.
6. Synchronous request/response for MVP.
7. Max clip length: **5 seconds**.
8. Hard-block clips >5s before upload.
9. Backend manages cloud key (server-side only).
10. Auto-retry once on failure, then clear user-visible error.
11. MVP SLOs: success rate >=90%; p95 latency <=8s for <=5s clips.
12. Ship with current best available cloud model first; iterate model in parallel.
13. Privacy: retain metadata + outputs only; no long-term raw video retention.
14. App->backend auth: single rotatable app API key.

## API contract (MVP)

### Endpoint
`POST /v1/translate-sign`

### Request
- Content-Type: `multipart/form-data`
- Fields:
  - `video` (required): recorded clip file
  - `request_id` (optional): client correlation id
  - `source_lang` (optional, default `asl`)
  - `target_lang` (optional, default `en`)

### Success response (200)
```json
{
  "request_id": "req_123",
  "gloss": "THANK-YOU",
  "translation": "Thank you",
  "confidence": 0.93,
  "latency_ms": 4210
}
```

### Error response (4xx/5xx)
```json
{
  "error_code": "TIMEOUT",
  "message": "Inference timed out",
  "request_id": "req_123",
  "retryable": true
}
```

### Timeout
- Server hard timeout: **12 seconds**.

## iOS MVP behavior
- User records or selects clip.
- App enforces 5-second maximum locally.
- App sends multipart upload to backend.
- App waits synchronously for result.
- On failure, app retries once automatically.
- If second attempt fails, app shows clear error.

## Data/privacy
- Do not persist raw video long-term.
- Persist minimal telemetry:
  - request_id
  - timing
  - success/failure + error_code
  - gloss/translation/confidence
  - model/version tag

## Model/training plan (parallel track)
- Continue Gemma4 tuning work in cloud GPU environments (Colab/other).
- Evaluate available ASL datasets and preprocessing compatibility.
- Keep this track independent from MVP app integration so product flow is not blocked.

## Out of scope (MVP)
- User accounts and per-user auth.
- Async jobs/polling/callback workflow.
- On-device pose extraction upload path.
- Local inference fallback mode in app.

## Acceptance criteria
1. iPhone app can send <=5s clip and receive gloss + translation.
2. End-to-end cloud path works with Cactus-backed inference.
3. Error handling uses standardized schema and single auto-retry.
4. Logs avoid long-term raw-video retention.
5. MVP SLOs tracked and reported (success >=90%, p95 <=8s).
