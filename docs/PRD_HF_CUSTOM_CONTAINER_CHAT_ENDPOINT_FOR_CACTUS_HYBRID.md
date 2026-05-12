## Problem Statement

The current ASL v2 translation flow is blocked at the upstream inference layer. The mobile app and local FastAPI path are functioning, but cloud handoff fails because the selected Gemma4 merged model is not reliably served through the current Hugging Face router/provider path. This causes Cactus hybrid to return upstream failures instead of verified translation results, preventing a shippable end-to-end flow.

From the user perspective: video upload and local backend wiring can work, but translation still fails due to endpoint/runtime incompatibility. The product needs a dependable, proof-bearing cloud inference path that preserves the locked architecture and fail-closed behavior.

## Solution

Stand up a Hugging Face Inference Endpoint using a custom container that exposes a minimal OpenAI-compatible `/v1/chat/completions` interface for the selected ASL model family. Keep Cactus hybrid as the routing authority and keep ASL prompt shaping in Cactus, while the container remains a model-serving adapter.

The solution is to:
- keep the request path as RN app -> local FastAPI `/v1/translate-sign` -> Cactus hybrid service -> HF custom endpoint,
- enforce fail-closed upstream handling with explicit evidence fields,
- return a minimal but valid chat response contract so Cactus can normalize and pass through proof data,
- validate first on a dev endpoint before production promotion.

## User Stories

1. As an ASL app user, I want my recorded signing clip to return a translation reliably, so that the feature works consistently in normal use.
2. As an ASL app user, I want clear errors when the cloud model is unavailable, so that I do not receive silent or unverifiable results.
3. As an ASL app user, I want no manual endpoint/API-key configuration in the app, so that setup remains simple and zero-config.
4. As an iOS tester, I want the app to keep using `/v1/translate-sign` locally, so that device testing does not require workflow changes.
5. As a backend operator, I want Cactus to remain the hybrid routing authority, so that routing policy is centralized and auditable.
6. As a backend operator, I want Cactus to fail closed on handoff/proof errors, so that only verifiable outputs are returned.
7. As a backend operator, I want upstream timeout behavior to be explicit and bounded, so that failures resolve predictably.
8. As a backend operator, I want request IDs to propagate through the stack, so that I can correlate failures across services.
9. As a model operator, I want a custom container runtime for Gemma4 compatibility, so that model architecture support is controlled.
10. As a model operator, I want a health/readiness signal at startup, so that bad deploys are caught quickly.
11. As a model operator, I want the container API surface to be minimal initially, so that first deployment risk is reduced.
12. As a platform engineer, I want a dev-only endpoint rollout first, so that E2E validation can occur before production cutover.
13. As a platform engineer, I want static bearer auth for initial bring-up, so that integration can be unblocked quickly.
14. As a platform engineer, I want token rotation and hardening staged after first green E2E, so that security improvements are deliberate.
15. As a product owner, I want proof fields to remain mandatory in successful outputs, so that runtime provenance is preserved.
16. As a product owner, I want explicit upstream failure evidence in errors, so that operations can diagnose outages quickly.
17. As a QA engineer, I want deterministic contract tests at the Cactus boundary, so that regressions are caught before device tests.
18. As a QA engineer, I want end-to-end validation from RN to cloud inference, so that real user flows are verified.
19. As a maintainer, I want prompt shaping to stay in Cactus, so that model-serving containers remain replaceable.
20. As a maintainer, I want a clear hardening phase after first success, so that usage metrics/streaming/retries can be added without blocking MVP.
21. As an on-call engineer, I want logs to separate client-safe errors from provider internals, so that debugging is effective without leaking sensitive data.
22. As a release manager, I want a documented promotion gate from dev endpoint to prod, so that rollout risk is controlled.

## Implementation Decisions

- Preserve architecture shape: mobile app calls local translation API, local API delegates to Cactus hybrid, Cactus performs cloud handoff.
- Keep Cactus as the single routing authority for hybrid/local-cloud behavior.
- Use a Hugging Face Inference Endpoint with a custom container for initial Gemma4-compatible deployment.
- Endpoint contract for first cut is OpenAI-chat compatible at `/v1/chat/completions`.
- Container returns minimal fields only (`id`, `object`, `created`, `model`, first choice message content) for first green path.
- Keep ASL translation prompt shaping and domain framing in Cactus; container performs model inference only.
- Auth path uses static bearer token from service configuration in first cut.
- Upstream timeout budget is 20 seconds at the Cactus boundary for first cut.
- Failure policy is fail-closed; no silent fallback to unverifiable translation paths.
- Error responses must surface upstream status/message/request correlation evidence in a sanitized form.
- Proof/provenance fields remain required at the backend boundary for successful responses.
- Deployment progression is dev endpoint first, then promotion after successful end-to-end verification.
- Post-green hardening track includes usage accounting, richer metadata, retry policy refinement, and operational runbook improvements.

## Testing Decisions

- Good tests assert externally observable behavior and contracts, not internal implementation details.
- Test the Cactus boundary contract for:
  - successful normalization of minimal chat response,
  - mandatory proof field enforcement,
  - request ID correlation,
  - fail-closed mapping on upstream errors/timeouts.
- Test endpoint adapter behavior for:
  - startup model load/readiness signals,
  - minimal OpenAI-chat response shape,
  - deterministic error responses for invalid requests.
- Test integration behavior for:
  - local FastAPI to Cactus to dev endpoint connectivity,
  - explicit `UPSTREAM_FAILURE` evidence propagation,
  - no silent fallback behavior.
- Test mobile-visible behavior for:
  - successful translation rendering,
  - explicit error UX when upstream fails.
- Prior art should follow existing runtime and API contract tests already used for cloud translation and Cactus hybrid behavior in this codebase.

## Out of Scope

- Production blue/green or multi-endpoint traffic management in first implementation.
- Streaming responses and token usage reporting in first implementation.
- Dynamic token broker, mTLS, or advanced auth hardening in first implementation.
- Reworking mobile UX beyond required error/proof behavior and existing zero-config direction.
- On-device feature extraction/local inference optimization experiments.
- Model-training improvements and quality-tuning beyond serving compatibility unblock.

## Further Notes

- This PRD reflects locked decisions from the current planning sequence: custom container, chat contract, model-only container responsibilities, static bearer auth, fail-closed policy, 20s timeout, dev-first rollout, and checklist-first execution mode.
- The work should prioritize unblock-to-green in the shortest safe path, then proceed to hardening as a separate follow-up.
- Operational evidence should include endpoint health, boundary contract validation, and one successful RN->FastAPI->Cactus->HF translation flow with proof fields preserved.