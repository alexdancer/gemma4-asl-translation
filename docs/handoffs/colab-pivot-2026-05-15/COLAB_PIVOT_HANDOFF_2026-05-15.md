# ASL Project Handoff — Pivot to Colab-First (Deadline Mode)

Date: 2026-05-15
Purpose: start fresh Hermes session with clear direction.

## Context recap

We spent multiple iterations stabilizing mobile + hosted Hugging Face custom endpoint path:
- Legacy RN/local-API endpoint bridge was removed in the Colab-first pivot; current critical path is Colab-only evaluation/inference.
- Container deploys succeeded through multiple tags (latest validated tag: `real-v20-amd64`)
- Transport/auth/startup issues were mostly resolved
- Output contract bugs were addressed (JSON parser strictness, then plain-text fallback)

Current status of serving path:
- Infra path works (requests reach endpoint, endpoint responds)
- But semantic quality still unstable for real phone clips
- User observed repeated incorrect output (`hello`) for different signs (`thank you`, `yes`)

So bottleneck is no longer basic infrastructure. Bottleneck is model behavior quality under current serving setup.

## Why we are pivoting

Deadline priority changed execution strategy.

Hosted endpoint + app loop introduces too many moving pieces:
- network/auth/headers
- container/runtime differences
- prompt/contract parsing layers
- app integration noise

These layers slow diagnosis of true model quality.

Decision: **pivot to Colab-first direct model workflow** for faster iteration and clearer evidence.

## Decision

We are moving from endpoint-first debugging to **direct model evaluation in Google Colab**.

Primary goal now:
1. Validate and improve model quality directly (pose/frame-derived input -> model output)
2. Prove behavior on target signs and Top-50 set
3. Only after quality is stable, re-wrap into endpoint/app integration

## What this means operationally

Short-term (now):
- Pause endpoint-heavy debugging as primary track
- Use Colab notebook for direct inference/eval loops
- Focus on measurable outputs and confusion patterns

Later (after quality proof):
- Bring proven config back into hosted endpoint
- Re-test RN app path with minimal integration changes

## Known reference facts

- Active repo path: `/Users/alex/Documents/ASL-project/sign-language-asl`
- Project north-star remains: app uploads video -> backend extracts signal -> model returns translation
- Model family direction remains E2B
- Top-50 label universe is defined in repo manifests
- Latest endpoint smoke can return success responses, but semantic collapse still observed in user clips

## Explicit handoff objective for next Hermes session

Start a **Colab-first execution session** that:
1. Sets up direct E2B inference notebook workflow
2. Runs focused checks for problematic signs (`yes`, `thank you`) plus a small representative Top-50 slice
3. Produces clear artifact outputs (predictions + simple metrics)
4. Recommends exact model-side adjustments before any endpoint/app rework

## Suggested kickoff prompt for next session

"Use this handoff doc. Execute Colab-first model validation for the ASL E2B Top-50 project in deadline mode. Prioritize direct inference quality evidence over endpoint integration. Produce runnable steps, required files, and measurable outputs for yes/thank-you + small Top-50 regression set."

---

End of handoff.
