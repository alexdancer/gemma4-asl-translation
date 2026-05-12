# Handoff: Cactus Hybrid Inference Direction (for next chat)

## Current Decision Lock (final)
1. Keep current architecture shape: RN app -> local backend (`/v1/translate-sign`) -> upstream inference service.
2. Upstream must be Cactus hybrid inference service.
3. Hybrid routing authority lives inside Cactus service (not RN, not custom backend routing).
4. Cloud handoff provider for phase 1: Hugging Face OpenAI-compatible endpoint.
5. Required proof fields from upstream on every response:
   - `runtime_mode`
   - `cloud_handoff`
   - `model_id`
   - `model_version`
6. Backend must enforce proof contract at boundary.
7. RN UX must be zero-config for normal users:
   - remove manual endpoint input
   - remove manual API-key input
   - use build-time config
8. On proof failure/misconfiguration, show explicit user-facing error.
9. If handoff path is down: fail closed (do not silently return unverifiable fallback).
10. Hosting target: user’s PC (Arch GPU machine), not cloud VM.

## Relevant GitHub issues (active)
- #75 `ASL-CACTUS-01: Deploy Cactus hybrid inference service on Arch with HF handoff` (AFK)
- #76 `ASL-CACTUS-02: Enforce upstream Cactus proof contract in cloud_translate_api` (AFK)
  - Blocked by #75
- #77 `ASL-CACTUS-03: RN zero-config inference wiring and proof-failure UX` (AFK)
  - Blocked by #76
- #78 `ASL-CACTUS-04: Operator runbook for Arch-hosted Cactus hybrid + HF handoff` (AFK)
  - Blocked by #75
- #73 `ASL-E2E-01: HITL real-device proof run + evidence bundle` (HITL)
  - Updated to this new direction
  - Blocked by #75, #76, #77

## Dependency order for execution
1. #75
2. #76 and #78 (can proceed after #75)
3. #77 (after #76)
4. #73 (HITL proof run)

## Cactus docs pages previously identified as relevant
- Home: `https://docs.cactuscompute.com/v1.14/`
- Choose SDK: `https://docs.cactuscompute.com/v1.14/docs/choose-sdk/`
- React Native SDK: `https://docs.cactuscompute.com/v1.14/react-native/`
- Fine-tuning/deployment guide: `https://docs.cactuscompute.com/v1.14/docs/finetuning/`
- Cactus engine API: `https://docs.cactuscompute.com/v1.14/docs/cactus_engine/`

## Existing local repo doc added earlier
- `docs/RN_HF_FASTAPI_PROVIDER_SETUP.md`

## Notes for next chat
- User explicitly wants Cactus hybrid engine to be genuinely used, not just named.
- User prefers practical implementation over extra debate once decisions are locked.
- Start from issue #75 in AFK sequence unless user requests a different entry point.
