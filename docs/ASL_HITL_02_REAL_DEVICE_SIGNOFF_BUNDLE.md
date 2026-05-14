# ASL-HITL-02 Real-Device Sign-off Bundle + Promotion Recommendation

Issue: #85  
Parent: #80

## Purpose

Human-validated real-device sign-off for locked flow:

RN app -> `/v1/translate-sign` -> FastAPI -> Cactus hybrid -> HF dev custom endpoint.

This bundle captures:
1) happy-path user-visible success,  
2) fail-closed user-visible failure,  
3) provenance proof fields,  
4) explicit promotion go/no-go recommendation.

## Locked constraints

- No manual endpoint input in app UI.
- No manual API-key entry in app UI.
- External API contract remains `/v1/translate-sign`.

## Operator preflight (before human run)

Run these first and save outputs in evidence folder.

```bash
mkdir -p evidence/hitl-02

bash -n scripts/runtime/verify_rn_fastapi_cactus_hf_e2e.sh \
  | tee evidence/hitl-02/00-bashn-verify-script.txt

bash -n scripts/runtime/prove_cactus_hybrid.sh \
  | tee evidence/hitl-02/01-bashn-proof-script.txt

. .venv/bin/activate
python -m pytest tests/runtime/test_e2e_rn_fastapi_cactus_hf_chain.py -q \
  | tee evidence/hitl-02/02-runtime-e2e-test.txt
python -m pytest tests/runtime/test_cactus_hybrid_service.py -q \
  | tee evidence/hitl-02/03-runtime-cactus-test.txt
```

Expected pass signals:
- both `bash -n` commands exit `0`
- both pytest commands pass

## Human run protocol (device)

### A) Happy path (required)

1. Start backend stack with dev endpoint config.
2. On physical iPhone, open app with default cloud-first flow.
3. Upload known-good short ASL video clip.
4. Wait for result.
5. Capture:
   - app screen showing translated output
   - backend terminal snippet around request_id
   - response JSON showing proof fields.

Save artifacts:
- `evidence/hitl-02/A1-app-happy-path.png`
- `evidence/hitl-02/A2-backend-happy-path.log`
- `evidence/hitl-02/A3-response-happy-path.json`

### B) Fail-closed scenario (required)

Use one controlled misconfiguration: invalid upstream token.

1. Set bad `ASL_HF_TOKEN` for Cactus service.
2. Repeat same device upload flow.
3. Confirm user-visible explicit error (not silent success/fallback).
4. Capture:
   - app error screen/text
   - backend log snippet with request_id + upstream failure
   - response JSON with fail-closed error code.
5. Restore valid token and re-check healthy state.

Save artifacts:
- `evidence/hitl-02/B1-app-fail-closed.png`
- `evidence/hitl-02/B2-backend-fail-closed.log`
- `evidence/hitl-02/B3-response-fail-closed.json`

## Required evidence assertions

For happy-path JSON (`A3-response-happy-path.json`), verify:

```bash
jq -e '.prediction != null and .prediction != ""' evidence/hitl-02/A3-response-happy-path.json
jq -e '.runtime_mode != null and .cloud_handoff == true and .model_id != null and .model_version != null' evidence/hitl-02/A3-response-happy-path.json
```

For fail-closed JSON (`B3-response-fail-closed.json`), verify:

```bash
jq -e '.ok == false and .error_code != null and .error_message != null and .error_message != ""' evidence/hitl-02/B3-response-fail-closed.json
```

## Promotion recommendation template (go/no-go)

Fill this section after evidence capture.

### Decision
- [ ] GO
- [ ] NO-GO

### Summary (3-5 bullets)
- <bullet>
- <bullet>
- <bullet>

### Gate checklist
- [ ] Device happy-path translation observed end-to-end.
- [ ] No manual endpoint/key entry used in UI.
- [ ] Proof fields present in success response: `runtime_mode`, `cloud_handoff`, `model_id`, `model_version`.
- [ ] At least one fail-closed user-visible error validated.
- [ ] Evidence artifacts saved under `evidence/hitl-02/`.

### Risks / follow-ups
- <risk or follow-up owner>

## Notes

- This issue is HITL by definition. Human/device evidence cannot be fully automated inside CI.
- Downstream issue #73 consumes this bundle as proof source.
