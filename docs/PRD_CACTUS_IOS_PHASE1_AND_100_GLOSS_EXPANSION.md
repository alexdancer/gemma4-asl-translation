# PRD — Cactus iOS Phase 1 + Top-50 Proof + 100-Gloss Expansion

## Problem Statement

We need to turn the current ASL Top-50 proof pipeline into a judge-ready, local-first Cactus deployment that runs on iOS, while preserving measurable quality and setting up a safe path to vocabulary expansion.

From Alex’s perspective, there are three immediate outcomes required before broader scope:

1. Load the current validated model into Cactus.
2. Use it end-to-end in a simple iOS app.
3. Measure how well it performs on the Top-50 gloss task versus the validated Python baseline.

At the same time, the project must avoid destabilizing scope before the hackathon deadline and create a clear next step toward more words and later multi-word transcription.

## Solution

Deliver in phased, gated order:

1. **Phase 1 (Cactus-first proof):**
   - Freeze current validated checkpoint as the parity baseline.
   - Convert and load into Cactus.
   - Validate with staged Top-50 evaluation (25-sample smoke, then full split).
   - Ship a minimal iOS Swift app that runs local Cactus inference on prerecorded clips.

2. **Phase 1.5 (demo reliability + trust):**
   - Enforce local-first routing with explicit fallback reasons.
   - Add low-confidence abstain behavior.
   - Produce failover mode + metrics dashboard + reproducibility bundle.

3. **Phase 2 (retraining after Cactus proof):**
   - Expand from 50 to 100 glosses.
   - Require minimum 40 samples per new gloss.
   - Promote only if quality, validity, and latency gates pass.

4. **Phase 3 (future):**
   - Multi-word/sentence transcription roadmap after single-gloss 100-class stability.

## User Stories

1. As a hackathon judge, I want to see ASL inference running locally on iOS, so that I can trust this is a real edge deployment.
2. As Alex, I want the same frozen checkpoint used across Python and Cactus, so that parity conclusions are credible.
3. As Alex, I want a deterministic staged evaluation process, so that I can catch failures quickly before full runs.
4. As Alex, I want Cactus to run first locally, so that we align with the Cactus special track intent.
5. As Alex, I want route decisions recorded, so that fallback usage is explainable during debugging and judging.
6. As a demo viewer, I want the app to show gloss output and confidence, so that model behavior is understandable.
7. As a judge, I want low-confidence abstains instead of forced wrong labels, so that the system appears trustworthy.
8. As Alex, I want a prerecorded clip mode first, so that I can de-risk integration and prove end-to-end behavior fast.
9. As Alex, I want no crashes in a repeated demo loop, so that live presentation risk is controlled.
10. As Alex, I want Python-vs-Cactus side-by-side metrics, so that runtime drift is visible.
11. As Alex, I want Cactus quality to stay within a defined delta of Python baseline, so that deployment quality is acceptable.
12. As Alex, I want invalid-label rate bounded, so that outputs remain contract-safe.
13. As Alex, I want fallback behavior separated by engineering mode and demo mode, so that debugging and presentation needs are both met.
14. As Alex, I want a one-command reproducibility bundle, so that judges/reviewers can verify claims quickly.
15. As Alex, I want explicit submission artifacts (video, links, write-up inputs), so that non-code deliverables are not missed.
16. As Alex, I want a scope freeze date, so that last-week quality does not regress from feature churn.
17. As Alex, I want retraining blocked until Cactus runtime issues are ruled out, so that we do not confuse deployment drift with model quality.
18. As Alex, I want vocabulary expansion to 100 glosses, so that capability increases while remaining measurable.
19. As Alex, I want a minimum sample floor per new gloss, so that expansion quality does not collapse from sparse classes.
20. As Alex, I want a strict promotion gate for retrained models, so that regressions are rejected automatically.
21. As Alex, I want future multi-word work explicitly staged after 100-gloss stability, so that roadmap sequencing stays realistic.
22. As a judge, I want clear evidence of local-first intelligent routing, so that the project fits track expectations.
23. As Alex, I want iOS-first implementation, so that I can test directly on my available device.
24. As Alex, I want a minimal Swift app before a broader wrapper strategy, so that initial delivery complexity stays manageable.

## Implementation Decisions

- **Execution order is sequential:** Cactus deployment and evaluation gates are completed before retraining starts.
- **Baseline freeze:** Current validated checkpoint is the parity anchor for all Cactus proof steps.
- **Evaluation protocol:** Two-stage quality gate (stratified smoke subset first, full Top-50 split second).
- **Quality thresholds:** Cactus Top-1 normalized gloss accuracy must remain within 5% of Python baseline; invalid-label rate must be <=2%.
- **Routing policy:** Local Cactus inference first, with explicit fallback triggers for low confidence, invalid parse, or latency cap breach.
- **Fallback strategy:** Engineering mode uses Python fallback for parity debugging; demo mode uses deterministic prerecorded failover path.
- **Abstain policy:** Slightly conservative low-confidence abstain threshold calibrated on held-out validation.
- **iOS scope:** Native Swift app first with prerecorded-clip inference path before live-camera expansion.
- **iOS Phase 1 acceptance gate:** App installs on target iPhone, loads local Cactus model, completes 3 locked demo clips, reports gloss/confidence/route reason, and survives a 5-minute loop without crashes.
- **Vocabulary expansion gate:** Move from 50 to 100 glosses in retraining phase, with at least 40 samples per newly added gloss.
- **Model promotion gate:** Retrained model only promotes if it improves Top-50 accuracy, keeps invalid-label <=2%, and limits end-to-end latency regression to <=20%.
- **Submission-readiness requirements:** Demo failover mode, metrics dashboard artifact, and judge-ready reproducibility bundle are mandatory.
- **Timeline control:** Feature freeze and demo/write-up freeze are treated as contractual milestones before final submission.

## Testing Decisions

- A good test verifies **external behavior and contractual outputs**, not internal implementation details.
- Prioritize tests that prove:
  - shared q64 prompt/input contract consistency,
  - Python-vs-Cactus normalized output parity behavior,
  - routing decision correctness,
  - abstain/fallback behavior under edge conditions,
  - iOS app stability under repeated demo use.
- Module families to test:
  - q64 contract and record-validation modules,
  - prompt-control evaluation and normalization modules,
  - Cactus runtime adapter/parity report modules,
  - routing-policy and demo-output contract modules,
  - iOS integration seam for model lifecycle + inference result rendering.
- Prior art for test style:
  - existing evaluator contract tests,
  - existing demo contract tests,
  - existing runtime/parity harness tests,
  - existing smoke-style artifact tests across data/demo/runtime flows.

## Out of Scope

- Full production-grade live ASL recognition across unconstrained vocabulary.
- Immediate multi-word/sentence transcription in the same phase as Cactus proof.
- Cross-platform Flutter migration before iOS Phase 1 proof is complete.
- Broad cloud-first architecture; this phase is local-first with bounded fallback behavior.
- Expansion beyond 100 glosses in the first retraining milestone.

## Further Notes

- Track alignment includes Main + Cactus + Unsloth objectives through one coherent delivery path.
- Required submission artifacts (video + links/write-up inputs) should be treated as first-class engineering deliverables.
- Locked freeze schedule:
  - **Feature freeze:** May 12, 2026 (America/Chicago)
  - **Demo/write-up freeze:** May 15, 2026 (America/Chicago)
  - **Hackathon deadline:** May 18, 2026 at 6:59 PM CDT (23:59 UTC)
