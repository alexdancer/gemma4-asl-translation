# ASL v1 Multi-word Translation Plan (Top-50)

## Purpose
This document captures locked product/architecture decisions and the execution checklist for the ASL v1 multi-word milestone.

## Locked Decisions
1. Server-side preprocessing/inference for v1.
2. Output is structured multi-word with timestamps/confidence.
3. Timestamp granularity is word-level.
4. Bootstrap with synthetic multi-word phrases from top-50 single-word clips.
5. Phrase length target: 2–3 words.
6. Synthetic inter-word gaps are randomized.
7. Repeated words are allowed with low probability.
8. Primary success metrics: WER + timestamp MAE.
9. App uploads raw video only.
10. Video ingest guardrails: max 10s, max 720p, max 30 FPS (downsample if higher).
11. Video optimization/normalization is backend-only.
12. Training target uses explicit `<sep>` and `<eos>` tokens.
13. Confidence includes per-word + overall sequence confidence.
14. Low-confidence behavior: return best guess plus low-confidence flag.
15. Runtime mode for v1 is clip-based only (non-streaming).
16. API mode is sync fast path with async fallback.
17. Async completion mechanism starts with polling (`GET /jobs/{id}`).
18. Primary inference target is Cactus cloud endpoint.
19. v1 auth is app API key only.
20. Decoding is constrained to top-50 vocabulary.
21. Timestamp method: synthetic boundaries + lightweight inference heuristic.
22. Client processing UX uses stage states.
23. Error contract is structured: error code + user-safe message + retryable flag.
24. Immediate deliverable preference: checklist/issues first, then implementation.

## Implementation Checklist (No-code Planning Baseline)

### Phase 0 — Contracts
- [ ] Define translate response schema
  - [ ] `transcript_words[]`: `{word, start_ms, end_ms, confidence}`
  - [ ] `sequence_confidence`
  - [ ] `low_confidence`
  - [ ] `status`, `request_id`
- [ ] Define error schema
  - [ ] `error_code`, `message`, `retryable`, `request_id`
- [ ] Define stage/status schema
  - [ ] `uploading`, `processing`, `decoding`, `completed`, `failed`

### Phase 1 — Synthetic Data Pipeline
- [ ] Build phrase synthesizer from top-50 single-word clips
- [ ] Enforce phrase length 2–3 words
- [ ] Add randomized inter-word gap generation
- [ ] Allow repeated words at low probability
- [ ] Emit word boundary ground truth from synthesis
- [ ] Preserve `<sep>/<eos>` target formatting

### Phase 2 — Training / Evaluation
- [ ] Update sequence target generation for multi-word outputs
- [ ] Add constrained top-50 decoding logic
- [ ] Add WER metric
- [ ] Add timestamp MAE metric

### Phase 3 — Backend API
- [ ] Implement `POST /translate` sync fast path
- [ ] Implement async fallback job creation when over threshold
- [ ] Implement `GET /jobs/{id}` polling endpoint
- [ ] Emit stage/status updates for client UX
- [ ] Apply low-confidence flag policy in response

### Phase 4 — Video Ingest Optimization
- [ ] Enforce upload guardrails (10s/720p/30fps)
- [ ] Backend normalization/transcoding to canonical ingest profile
- [ ] Reuse existing server-side frame extraction + pose extraction pipeline
- [ ] Log normalized video attributes (duration/fps/resolution)

### Phase 5 — Auth/Safety (v1)
- [ ] API key middleware
- [ ] Key rotation support
- [ ] Basic per-key rate limiting

### Phase 6 — Cactus Integration
- [ ] Integrate Cactus cloud inference call as primary backend target
- [ ] Map provider output into v1 response schema
- [ ] Normalize provider errors into structured contract

### Phase 7 — Client Contract Readiness
- [ ] Implement raw video upload flow
- [ ] Implement stage-state rendering
- [ ] Implement polling for async fallback jobs
- [ ] Handle low-confidence and structured error responses

### Phase 8 — Acceptance Gates
- [ ] Synthetic multi-word WER reported and acceptable
- [ ] Timestamp MAE reported and acceptable
- [ ] End-to-end flow validated:
  - [ ] Sync path success
  - [ ] Async fallback + polling success
  - [ ] Error path behavior validated

## Suggested Sub-Issue Breakdown
1. ASL-MW-01 — Synthetic phrase generator (2–3 words, randomized gaps, repeat policy).
2. ASL-MW-02 — Multi-word target formatting + constrained decoding (`<sep>/<eos>`, top-50).
3. ASL-MW-03 — Evaluation harness (WER + timestamp MAE).
4. ASL-API-01 — Translate API contract + structured errors/status.
5. ASL-API-02 — Sync fast path + async fallback + polling endpoint.
6. ASL-VID-01 — Backend video normalization + ingest guardrails.
7. ASL-AUTH-01 — API key auth + rotation + rate limiting.
8. ASL-CACTUS-01 — Cactus cloud integration + response mapping.
9. ASL-RN-01 — RN upload + stage UI + polling integration.

## API Key Rotation Runbook (ASL-AUTH-01)

- Auth headers accepted by backend:
  - `Authorization: Bearer <key>` (primary)
  - `X-API-Key: <key>` (compatibility)
- Env keys:
  - `ASL_V1_API_KEYS` — current active keys (comma-separated)
  - `ASL_V1_API_KEYS_NEXT` — next keys during rotation window (comma-separated, optional)
  - `ASL_V1_RATE_LIMIT_REQUESTS` — per-key request cap in window (default `60`)
  - `ASL_V1_RATE_LIMIT_WINDOW_SECONDS` — window size in seconds (default `60`)

Rotation steps:
1. Generate new key(s), set them in `ASL_V1_API_KEYS_NEXT`, keep old keys in `ASL_V1_API_KEYS`.
2. Roll out app/backend clients to start using new key(s).
3. Promote new key(s) into `ASL_V1_API_KEYS`.
4. Remove old keys and clear `ASL_V1_API_KEYS_NEXT`.

Error semantics:
- Invalid/missing key -> `401 UNAUTHORIZED` (non-retryable)
- Per-key limit exceeded -> `429 RATE_LIMITED` (retryable, includes `details.retry_after_seconds`)
