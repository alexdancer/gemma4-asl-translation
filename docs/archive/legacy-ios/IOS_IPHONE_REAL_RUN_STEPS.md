# iPhone Run Guide (ASL-App + local Cactus runtime)

> **Archive notice (2026-05):** `ios/ASL-App` has been removed from the repo. Primary development is React Native at `apps/mobile-rn`. This guide is retained only as historical reference for past device-evidence workflows.

This is the exact checklist to run the app on a **real iPhone** and validate local runtime behavior.

---

## 0) Prerequisites

- Xcode installed
- `cactus-ios.xcframework` already built and added to the app
- Model folder available at:
  - `/Users/alex/Documents/ASL-project/sign-language-asl/checkpoints/gemma4_e4b_cactus_int4`
- iPhone connected via USB (or trusted for wireless debugging)

---

## 1) Verify Xcode target/device

1. Open:
   - `ios/ASL-App/ASL-App.xcodeproj`
2. In the top device selector, choose your **iPhone** (not “My Mac” and not simulator).
3. In **Signing & Capabilities** for target `ASL-App`:
   - Select your Team
   - Ensure a valid Bundle Identifier
   - Let Xcode manage signing

---

## 2) Confirm framework linkage (one-time check)

In target `ASL-App`:

1. **General → Frameworks, Libraries, and Embedded Content**
   - Ensure `cactus-ios.xcframework` exists
   - Embed setting should be **Embed & Sign**
2. **Build Phases**
   - Framework should appear in both:
     - **Link Binary With Libraries**
     - **Embed Frameworks**

---

## 3) Add model as Folder Reference (critical)

Use this to bundle the full model directory into the iOS app.

1. In Finder, locate:
   - `/Users/alex/Documents/ASL-project/sign-language-asl/checkpoints/gemma4_e4b_cactus_int4`
2. Rename a copy (or the folder itself) to:
   - `cactus-model`
3. In Xcode Project Navigator, right-click `ASL-App` folder → **Add Files to "ASL-App"...**
4. Select the `cactus-model` folder
5. In the add dialog:
   - ✅ **Copy items if needed** (recommended)
   - ✅ Target membership: `ASL-App`
   - ✅ Choose **Create folder references** (blue folder)
     - Do **not** choose “Create groups”
6. Click **Add**

Result: you should see a **blue folder** named `cactus-model` in Xcode.

---

## 4) Scheme configuration for iPhone strict-proof run

1. Product → Scheme → **Edit Scheme...**
2. Select **Run** action
3. Use **Build Configuration: Release** (preferred for latency)
4. Under **Arguments → Environment Variables**:
   - Preferred for bundled model test: **remove** `CACTUS_MODEL_PATH` or leave it unset
   - (Optional explicit path test) set `CACTUS_MODEL_PATH` only if you intentionally test env-path behavior

For strict in-app bundle proof, rely on bundled `cactus-model` first.

---

## 5) Clean install on device

1. Product → **Clean Build Folder** (`Shift+Cmd+K`)
2. Run (`Cmd+R`) to install on iPhone
3. If prompted on iPhone:
   - Trust developer profile in Settings
   - Re-run app

---

## 6) Execute validation run in app

1. In app UI:
   - `runtimeMode` should report `local_cactus` after inference
2. Run one of your test prompts/clips
3. Capture these output fields from the UI:
   - `runtimeMode`
   - `routeReason`
   - `statusMessage`
   - `latencyMs`
   - gloss + confidence (non-empty)

Target success values:
- `runtimeMode=local_cactus`
- `routeReason=local_cactus_runtime_success`
- non-empty gloss/confidence

---

## 7) Tracer artifact check (Issue #50 evidence)

After successful run, verify tracer output exists:

- `ios_tracer_artifacts/session_index.json`

If present, keep this file plus screenshots/log output for issue evidence.

---

## 8) If you still see strict proof latency failure

If route reason is `strict_proof_local_runtime_failed` with latency-related status:

1. Confirm you are on **physical iPhone** (not simulator)
2. Confirm **Release** build
3. Re-run with a short clip/input and minimal background load
4. Capture exact `statusMessage` text (includes detailed runtime diagnostics)

---

## 9) Quick verification checklist

- [ ] Running on iPhone device target
- [ ] `cactus-model` added as **Folder Reference** (blue folder)
- [ ] `cactus-ios.xcframework` linked + Embed & Sign
- [ ] Release run configuration
- [ ] `runtimeMode=local_cactus`
- [ ] `routeReason=local_cactus_runtime_success`
- [ ] non-empty gloss/confidence
- [ ] `ios_tracer_artifacts/session_index.json` generated

---

If any one of these fails, capture screenshot + exact status text and iterate from that single failing checkpoint.