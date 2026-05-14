# HF Endpoint Debug Handoff — 2026-05-13

## Goal
Get real ASL translation output (semantic text), not stub filename output (`ASL_TRANSLATION_FOR_<filename>`), through:
- Mobile RN app -> local backend (`/v1/translate-sign`) -> Cactus hybrid service -> HF custom endpoint.

## Current blocker (primary)
HF custom endpoint real backend path still fails inference for target model with provider/task routing issues.

Latest observed error:
- `HTTP 502`
- `{"error":{"code":"INFERENCE_FAILED","message":"backend inference failed: StopIteration()"}}`

Interpretation:
- Request reaches endpoint.
- Runtime backend attempts Hugging Face `InferenceClient` call.
- Provider/task auto-resolution path fails internally (iterator exhausted), not a simple caller-auth problem.

## What is already fixed
1. Local API proof passthrough bug fixed (`src/cloud_translate_api.py`) so `INFERENCE_PROOF_MISSING` is resolved.
2. RN UI mapping patched (`apps/mobile-rn/App.tsx`) to normalize result fields (`gloss ?? prediction ?? translation`) so app can display model output schema variants.
3. HF custom endpoint now supports `ASL_HF_ENDPOINT_BACKEND=real` (code patch in `src/hf_custom_endpoint_service.py`).
4. Added fallback from chat path to `text_generation` for non-chat model errors (`model_not_supported`).
5. Added better inference error surfacing helper so blank exception strings now show representation (e.g., `StopIteration()`).

## What was verified
- Endpoint health is good:
  - `ready: true`
  - `backend: real`
  - expected model id/version visible in `/healthz`.
- Caller auth to endpoint works (`/healthz` returns 200 with bearer token).
- Model repo exists and is reachable from valid token locally.
- Private-model auth confusion occurred multiple times due to token placement/refresh and secret visibility behavior in CLI (`secrets` appear as null/redacted).
- Model was temporarily set public for debugging; still ended at runtime `StopIteration()` in real backend path.

## Important operational notes
- HF CLI command is `hf endpoints ...` (not `hf endpoint ...`).
- This CLI version does **not** support `hf endpoints logs`; logs must be viewed in HF web UI.
- Local shell had `cat` alias/toolchain issue (`bat` crash due to `llhttp/libgit2` mismatch). Use `/bin/cat` explicitly for debugging files.

## Endpoint details used
- Endpoint name: `asl-gemma4-e2b-q64-top50-mer-ohp`
- Endpoint URL: `https://ycd984e4ou9u6p8s.us-east-1.aws.endpoints.huggingface.cloud`
- Container image currently used during latest tests: `alexd05/asl-hf-custom-endpoint:real-v3`
- Env:
  - `ASL_HF_ENDPOINT_BACKEND=real`
  - `ASL_HF_ENDPOINT_MODEL_ID=AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit`
  - `ASL_HF_ENDPOINT_MODEL_VERSION=gemma4-e2b-q64-top50`

## Next step (recommended code change)
Patch `RealBackend.generate()` in `src/hf_custom_endpoint_service.py` to avoid brittle provider auto-routing assumptions:
1. Try `client.chat.completions.create(...)`.
2. If unsupported/empty/StopIteration, try alternate HF chat API surface (`chat_completion` compatible path if available).
3. If still failing, return detailed structured failure including attempted method + exception class/message (already partially improved).
4. Add/extend tests for:
   - fallback branch selection,
   - non-empty content extraction,
   - deterministic error payload when all methods fail.

## Security note
A live HF token was exposed during debugging in chat. Treat as compromised:
- Revoke/rotate exposed token(s).
- Replace endpoint secret with fresh token.

## Session stop state
Paused intentionally by user. No further deployment or code changes should be assumed complete beyond items listed above.
