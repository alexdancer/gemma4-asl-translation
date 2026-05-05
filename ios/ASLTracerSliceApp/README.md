# ASL Tracer Slice iOS App (Issue #35)

This folder contains a minimal SwiftUI tracer scaffold for the issue #35 vertical path:

- user taps **Run Local Cactus Inference**
- app triggers a local inference client seam
- app renders predicted gloss and confidence in UI

The current implementation uses a bundled JSON fixture (`local_cactus_response.json`) to provide deterministic local behavior in environments where the Cactus iOS runtime cannot be exercised.

## Device follow-up validation

On macOS with Xcode installed:

1. Create/open an iOS app target and include the three Swift files under `ASLTracerSliceApp/`.
2. Ensure `Resources/local_cactus_response.json` is bundled in the app target.
3. Run on target iPhone.
4. Tap **Run Local Cactus Inference** and confirm gloss + confidence render.

This repository's Linux environment cannot run Xcode/iPhone runtime checks.
