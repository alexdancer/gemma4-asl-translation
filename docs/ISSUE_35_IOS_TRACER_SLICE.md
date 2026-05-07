# Issue #35 — iOS tracer slice #1

## Scope delivered

- Added deterministic tracer module: `src/mobile/ios_tracer_slice.py`
- Added runnable tracer CLI: `scripts/mobile/run_ios_tracer_slice.py`
- Added SwiftUI app scaffold files under `ios/ASL-App/ASL-App/`
  - `ASL_AppApp.swift`
  - `ContentView.swift` (runtime selector, input-path selector, clip selector, strict-proof toggle, full result contract rendering)
  - `LocalCactusInferenceClient.swift` (fixture + RealLocal seam, retry-once behavior, strict-proof no-fallback behavior)
  - `Resources/local_cactus_response.json`
  - `Resources/local_cactus_clips.json` (3 locked clips for Tensor + Video paths)
- Added artifact logging contract in app Documents directory:
  - Per-run JSON files (`ios_inference_v1` schema)
  - Rolling session index (`session_index.json`)
- Added behavior tests: `tests/runtime/test_ios_tracer_slice.py`

## What this proves now

- The tracer flow records a user button tap event.
- The tap event triggers a local-inference seam.
- The UI contract payload includes `predicted_gloss` and `confidence`.
- SwiftUI scaffold contains a visible button and result rendering text fields.

## Follow-up required outside this Linux environment

This environment cannot run Xcode/iPhone runtime checks. Device-only validation remains:

1. Build and run app target on the target iPhone.
2. Tap **Run Local Cactus Inference** in app runtime.
3. Capture screenshot/video proof that gloss + confidence render in-app.
4. Replace fixture-backed local client with real Cactus runtime call at the TODO seam in `LocalCactusInferenceClient.swift`.

## New blocker tracking (sub-issue)

- Sub-issue created: **#50**
  - https://github.com/alexdancer/sign-language-asl/issues/50
  - Title: `Issue 35 blocker: produce Cactus-compatible model artifacts for iOS real proof run`

### Why #50 exists

Issue #35 now has SDK wiring, strict proof route handling, and proof-run UI flow, but still needs a runtime-loadable Cactus model artifact bundle so `cactusInit` succeeds and `cactusComplete` returns real output in app runtime.

### Completion gate from #50 back into #35

- Successful strict proof run with:
  - `runtimeMode=local_cactus`
  - `routeReason=local_cactus_runtime_success`
  - non-empty gloss + confidence
  - no fallback route
- Artifact evidence recorded in `ios_tracer_artifacts/session_index.json`.
