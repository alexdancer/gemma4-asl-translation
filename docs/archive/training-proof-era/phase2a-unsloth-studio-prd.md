# PRD: Phase2A Unsloth Studio Fine-Tuning Workflow

**Created:** 2026-04-30 CDT  
**Owner:** Alex / Gem  
**Project:** Kaggle Gemma 4 Good Hackathon  
**Status:** Draft  
**Primary decision:** Use Unsloth Studio as the Phase2A fine-tuning interface instead of building a custom Unsloth training script first.

## Source References

- Unsloth Gemma 4 fine-tuning docs: https://unsloth.ai/docs/models/gemma-4/train
- Unsloth Studio docs: https://unsloth.ai/docs/new/studio

## Problem Statement

Phase2A needs a reliable fine-tuning path for Gemma 4, but writing and maintaining custom Unsloth training code creates avoidable implementation overhead. The higher-leverage work is ensuring that the Phase2A dataset is clean, correctly formatted, representative of the target behavior, and paired with sane fine-tuning parameters.

Alex needs a workflow that makes fine-tuning fast to run, easy to monitor, easy to reproduce, and easy to compare across candidate runs without spending the bulk of Phase2A on training infrastructure.

## Solution

Use **Unsloth Studio** as the primary fine-tuning workflow for Phase2A. Treat Studio as the training orchestration layer and focus engineering effort on:

1. Producing a validated Phase2A training dataset.
2. Creating a repeatable dataset QA checklist.
3. Selecting model, context length, LoRA/QLoRA, and training hyperparameters intentionally.
4. Running small pilot jobs before full candidate jobs.
5. Capturing run configuration, training metrics, evaluation outputs, and exported artifacts.
6. Keeping a fallback path open for custom Unsloth code only if Studio blocks a required capability.

## Goals

- Reduce Phase2A training setup time.
- Avoid unnecessary custom training code unless needed.
- Improve dataset quality before spending GPU time.
- Make training runs reproducible enough to compare.
- Produce at least one usable Phase2A fine-tuned Gemma 4 candidate.
- Establish an evaluation loop that can identify regressions and improvements quickly.

## Non-Goals

- Building a full custom Unsloth training pipeline in Phase2A.
- Automating every Studio action immediately.
- Full production MLOps infrastructure.
- Multi-model benchmark infrastructure beyond what is needed to choose a Phase2A candidate.
- Final competition submission packaging unless required by the Phase2A milestone.

## User Stories

1. As Alex, I want to use Unsloth Studio for fine-tuning, so that I can spend less time writing training code.
2. As Alex, I want a dataset QA checklist, so that bad examples do not silently poison training.
3. As Alex, I want malformed records caught before training, so that Studio runs do not fail late.
4. As Alex, I want duplicate examples identified, so that the model does not overfit repeated patterns.
5. As Alex, I want train/validation splits defined, so that improvements can be measured honestly.
6. As Alex, I want token length stats, so that context length is chosen from evidence instead of guesswork.
7. As Alex, I want outlier examples inspected, so that unusually long or malformed samples are handled intentionally.
8. As Alex, I want the dataset format aligned with Gemma 4 chat expectations, so that training teaches the intended interaction pattern.
9. As Alex, I want a small fixed eval set, so that every fine-tuned candidate is tested against the same prompts.
10. As Alex, I want a recommended first-run configuration, so that I can start training without over-optimizing prematurely.
11. As Alex, I want a pilot run before a full run, so that obvious data/config issues are caught cheaply.
12. As Alex, I want clear guidance on E2B vs E4B, so that I can trade off speed, quality, and available VRAM.
13. As Alex, I want guidance on QLoRA vs LoRA, so that memory constraints do not derail training.
14. As Alex, I want context length guidance, so that training is efficient while preserving needed examples.
15. As Alex, I want loss interpretation guidance for Gemma 4, so that normal high loss values do not trigger false alarms.
16. As Alex, I want training metrics recorded, so that runs can be compared later.
17. As Alex, I want exported model artifacts tracked, so that the best run is not lost.
18. As Alex, I want qualitative eval outputs saved, so that model behavior can be reviewed directly.
19. As Alex, I want a fallback plan, so that custom code can be introduced only if Studio is insufficient.
20. As Alex, I want the workflow documented in markdown, so that Phase2A work remains reproducible and shareable.
21. As a collaborator, I want to understand the accepted dataset schema, so that I can contribute examples safely.
22. As a collaborator, I want run naming conventions, so that experiment artifacts are not ambiguous.
23. As a reviewer, I want success criteria, so that the Phase2A candidate can be accepted or rejected objectively.
24. As a reviewer, I want out-of-scope boundaries, so that the team does not expand Phase2A into infrastructure work.
25. As the project, I want each training run to produce a short run report, so that later decisions are evidence-based.

## Requirements

### Dataset Requirements

- Dataset must be in a Studio-compatible format before training.
- Each example must have a clear prompt/input and target response/output.
- Chat-style data should preserve role boundaries.
- Examples must avoid accidental leakage of private data, secrets, or irrelevant local context.
- Duplicates and near-duplicates should be detected and reviewed.
- Empty, malformed, or extremely low-signal examples must be removed.
- Train/validation split must be stable across runs.
- Token length distribution must be measured before selecting context length.
- A small fixed evaluation set must be separated from training data.

### Training Workflow Requirements

- Use Unsloth Studio as the primary training interface.
- Start with a small pilot run to verify dataset formatting, memory fit, and loss behavior.
- Use a consistent run naming convention.
- Capture the full Studio configuration for each run.
- Monitor loss trend, gradient norms if available, GPU utilization, and failure messages.
- Save exported artifacts after successful training.
- Save qualitative outputs from the fixed eval set after each candidate run.

### Reproducibility Requirements

Each run should record:

- Run name.
- Date/time.
- Base model.
- Dataset version/hash if available.
- Train/validation split details.
- Context length.
- Fine-tuning method.
- LoRA rank/alpha/dropout if configured.
- Learning rate.
- Epoch count or max steps.
- Batch size and gradient accumulation if visible/configurable.
- Warmup setting.
- Save/export format.
- Validation results.
- Notes on qualitative behavior.

## Recommended First-Run Configuration

This is the initial baseline, not a final claim of optimality.

### Model Choice

Preferred:

- **Gemma 4 E4B QLoRA**, if the available hardware supports it.

Fallback:

- **Gemma 4 E2B** for faster iteration or lower VRAM environments.

Rationale:

- Unsloth’s Gemma 4 docs recommend E4B QLoRA over E2B LoRA when feasible because E4B is larger and the quantization tradeoff is expected to be small.
- The docs note E2B can train on lower VRAM, making it useful for quick validation loops.

### Training Type

- Use **QLoRA** for the first serious Phase2A run unless Studio/hardware constraints suggest otherwise.
- Use LoRA only if memory is sufficient and the expected quality gain justifies the extra cost.

### Context Length

- Choose context length based on token statistics.
- Baseline rule: pick the smallest context length that covers roughly 95–98% of useful training examples.
- Do not increase context length just to cover a few outliers unless those outliers are strategically important.

### Hyperparameter Starting Point

- Epochs: **1 pilot epoch**, then **2–3 candidate epochs** if the pilot looks healthy.
- Learning rate: start around **2e-4**.
- Warmup: around **5%** of steps if configurable.
- LoRA rank: **16** for pilot; consider **32** for candidate if quality is promising.
- LoRA alpha: match rank or use Studio default if it provides a recommended value.
- LoRA dropout: **0.0–0.05**.
- Batch size: largest stable value that fits memory.
- Gradient accumulation: use Studio defaults or enough accumulation to reach a stable effective batch size.
- Validation: enable if Studio supports it directly; otherwise evaluate separately with the held-out set.

### Loss Interpretation

- For Gemma 4 E2B/E4B, Unsloth documents that loss around **13–15 can be normal** for these multimodal-style models.
- Do not judge runs by absolute loss alone.
- Prefer: healthy downward trend, no instability, validation behavior, and qualitative eval outputs.

## Operational Checklist

### Before Training

- [ ] Confirm target behavior for Phase2A.
- [ ] Freeze a dataset version.
- [ ] Validate schema/format.
- [ ] Remove malformed examples.
- [ ] Remove or flag duplicates.
- [ ] Check for private data or accidental leakage.
- [ ] Compute token length distribution.
- [ ] Inspect longest examples manually.
- [ ] Create stable train/validation/eval splits.
- [ ] Prepare fixed Phase2A eval prompts.
- [ ] Select base model and context length.
- [ ] Record intended Studio configuration.

### During Training

- [ ] Start with a pilot run.
- [ ] Confirm training starts without data errors.
- [ ] Monitor loss trend rather than only absolute loss.
- [ ] Watch for memory failures or unstable gradients.
- [ ] Stop early if the dataset/config is clearly wrong.
- [ ] Save screenshots or exported run settings where useful.

### After Training

- [ ] Export/save model artifacts.
- [ ] Run fixed eval prompts.
- [ ] Save qualitative outputs.
- [ ] Compare against base model behavior if possible.
- [ ] Write a short run report.
- [ ] Decide: reject, rerun with changes, or promote candidate.

## Implementation Decisions

- Treat Unsloth Studio as the Phase2A training orchestration layer.
- Treat dataset preparation and evaluation as the main engineering deliverables.
- Keep custom Unsloth code as a fallback, not the first implementation path.
- Use a staged training approach: data validation → pilot run → candidate run → eval → iteration.
- Prefer evidence-based context length selection using token distribution.
- Prefer stable, named dataset versions over ad hoc file edits.
- Keep a fixed eval set separate from training and validation.
- Record every run’s configuration and result summary.
- Use qualitative review alongside metrics because the competition outcome depends on behavior, not loss alone.

## Proposed Workflow Modules

These are conceptual workflow modules, not necessarily code modules.

### Dataset Quality Gate

Responsible for deciding whether the dataset is safe and ready to train.

Inputs:

- Raw or prepared Phase2A examples.
- Expected schema.
- Privacy constraints.

Outputs:

- Clean training file.
- Validation file.
- Eval set.
- QA report.

### Training Configuration Sheet

Responsible for recording Studio choices before and after each run.

Inputs:

- Base model choice.
- Dataset version.
- Hardware constraints.
- Token stats.

Outputs:

- Run configuration.
- Rationale for parameter choices.
- Notes for reproducibility.

### Evaluation Harness

Responsible for comparing candidate behavior against the fixed Phase2A eval prompts.

Inputs:

- Base model outputs if available.
- Fine-tuned model outputs.
- Fixed eval prompt set.

Outputs:

- Qualitative comparison.
- Pass/fail notes.
- Recommended next action.

### Run Report

Responsible for making each experiment understandable later.

Inputs:

- Studio config.
- Training metrics.
- Eval outputs.
- Observations.

Outputs:

- Short markdown run report.
- Candidate promotion/rejection decision.

## Testing Decisions

Testing should focus on external behavior and artifact quality, not implementation details.

### Dataset Tests

- Schema validation: every record has the expected fields and non-empty content.
- Role validation: chat examples preserve supported roles and ordering.
- Token validation: examples fit within the selected context length or are intentionally excluded.
- Duplicate validation: exact duplicates are removed; near-duplicates are reviewed.
- Split validation: train, validation, and eval sets do not overlap.
- Privacy validation: obvious secrets, credentials, and irrelevant personal data are absent.

### Training Workflow Tests

- Pilot run successfully starts and consumes the dataset.
- Pilot run produces a saved adapter/model artifact.
- Training metrics are visible enough to judge whether the run is healthy.
- Export flow produces the expected artifact format.

### Evaluation Tests

- Fixed eval prompts run against each candidate.
- Outputs are saved with the run report.
- Regressions are called out explicitly.
- Candidate is accepted only if it improves target Phase2A behavior without obvious degradation.

## Success Criteria

Phase2A Unsloth Studio workflow is successful when:

- A clean dataset version exists.
- Dataset QA has been completed and recorded.
- At least one pilot training run completes.
- At least one candidate run completes with saved artifacts.
- Candidate outputs are evaluated against a fixed eval set.
- A run report identifies whether the candidate should be promoted, rerun, or rejected.
- No custom training code was needed unless Studio proved insufficient.

## Risks and Mitigations

### Risk: Studio Beta Limitations

Unsloth Studio is documented as Beta, so some workflows may change or be incomplete.

Mitigation:

- Keep the custom Unsloth code path as a fallback.
- Record Studio version/details in run reports when available.

### Risk: Bad Dataset Produces Misleading Training Success

Training may complete successfully while teaching the wrong behavior.

Mitigation:

- Require dataset QA and fixed qualitative evals before promoting any candidate.

### Risk: Misreading Gemma 4 Loss

Gemma 4 E2B/E4B may show high absolute loss values that are normal according to Unsloth docs.

Mitigation:

- Judge by trend, stability, validation behavior, and qualitative outputs.

### Risk: Context Length Too Large

Overly large context length may waste memory and reduce throughput.

Mitigation:

- Choose context length from token distribution.
- Trim or exclude rare outliers unless strategically valuable.

### Risk: Context Length Too Small

Important examples may be truncated or excluded.

Mitigation:

- Inspect long examples manually.
- Preserve examples needed for target behavior.

## Out of Scope

- Building a custom dashboard.
- Building a full training automation platform.
- Hyperparameter sweeps beyond a small number of intentional runs.
- Large-scale model comparison unrelated to Phase2A.
- Advanced RL training unless Phase2A explicitly requires it.
- Final deployment/inference serving beyond exported artifacts and local evaluation.

## Further Notes

- The main strategic shift is that Phase2A should be dataset-first, not training-code-first.
- The first run should optimize for fast feedback, not final quality.
- Every run should leave behind enough evidence that future Alex/Gem can understand what changed and why.
- If Studio cannot support a required dataset format, parameter, export, or evaluation path, then custom Unsloth scripting becomes justified.
