# PRD: Unsloth Top-50 Proof-of-Learning and Demo Integration

## Problem Statement

Alex needs a credible, repeatable way to prove that the ASL pose-to-gloss pipeline is learning before investing more time in larger training runs or live-camera demo work. The current full-dataset dashboard run proved that Unsloth Studio can train a Gemma 4 adapter on compact q64 pose records, but it did not provide clean proof metrics because the model saw examples that would otherwise be used for testing.

The project needs a clean Top-50 proof-of-learning workflow that separates training, validation, and final testing; records all dashboard settings; freezes checkpoints before evaluation; and uses one shared prediction contract for both evaluation and later demo integration. This prevents the team from confusing training loss with real accuracy and prevents evaluation/demo drift caused by separate prompt formatting or output parsing logic.

## Solution

Create and operate a Top-50 ASL proof-of-learning workflow around Unsloth Studio training and local project evaluation.

The solution uses Unsloth Studio for the manual fine-tuning run, because the dashboard is already working for q64_full records and reduces custom training code overhead. The project owns dataset preparation, experiment records, checkpoint freezing, and evaluation.

The workflow is:

1. Produce compact q64_full pose records that fit the selected Gemma 4 context length.
2. Create a random stratified Top-50 train/validation/test split with no test leakage.
3. Run a 3-epoch Gemma 4 E4B IT QLoRA baseline in Unsloth Studio using train and validation files only.
4. Freeze the resulting checkpoint into the project before evaluation.
5. Evaluate free-generation predictions on the untouched Top-50 test split using strict normalized exact-match accuracy.
6. Use the same q64 prediction contract for later demo integration.
7. If Top-50 passes the agreed success gate, integrate the model into a prerecorded/known-good demo path before scaling to the full 250-gloss set.

## User Stories

1. As Alex, I want a clean Top-50 proof-of-learning workflow, so that I can know whether the model actually learns ASL pose-to-gloss mapping.
2. As Alex, I want training, validation, and test data separated, so that evaluation is not contaminated by examples the model already saw.
3. As Alex, I want the Top-50 labels selected consistently, so that repeated experiments are comparable.
4. As Alex, I want q64_full records that fit the dashboard context limit, so that Unsloth Studio training does not truncate important input.
5. As Alex, I want a dashboard run checklist, so that manual Unsloth Studio settings are entered correctly.
6. As Alex, I want an experiment record for the dashboard run, so that the run can be reproduced later.
7. As Alex, I want to upload a training file and separate eval file, so that the dashboard can report validation behavior while preserving a final test set.
8. As Alex, I want to keep the held-out test file untouched, so that final metrics are trustworthy.
9. As Alex, I want the checkpoint copied into the project before evaluation, so that dashboard outputs are not lost or overwritten.
10. As Alex, I want a run summary captured with model, dataset, training settings, and losses, so that we can compare future runs.
11. As Alex, I want strict normalized exact-match accuracy, so that the primary metric reflects whether the model outputs the correct gloss.
12. As Alex, I want invalid-output rate measured, so that I can tell whether the model is failing formatting rather than classification.
13. As Alex, I want per-class accuracy, so that weak signs are visible instead of hidden by aggregate accuracy.
14. As Alex, I want a confusion matrix, so that I can see which signs the model mixes up.
15. As Alex, I want predictions saved to a reviewable table, so that I can inspect raw model outputs and normalized predictions.
16. As Alex, I want evaluation to use free generation first, so that the metric reflects realistic demo behavior.
17. As Alex, I want constrained label scoring deferred until needed, so that the first metric remains simple and honest.
18. As Alex, I want mock evaluation support, so that evaluator plumbing can be tested without loading a GPU model.
19. As Alex, I want real checkpoint evaluation support, so that once training finishes I can measure the adapter on the held-out test split.
20. As Alex, I want dependency errors to be clear, so that missing Unsloth or PEFT packages do not create confusing failures.
21. As Alex, I want one shared prediction contract, so that evaluation and demo use the same prompt formatting and output normalization.
22. As Alex, I want q64 JSONL inference implemented before video inference, so that model behavior can be tested without MediaPipe or video bugs.
23. As Alex, I want prerecorded/known-good demo integration before live camera, so that we isolate model inference before adding real-time capture noise.
24. As Alex, I want demo integration before full-250 scaling if Top-50 succeeds, so that the Kaggle prototype has a working story quickly.
25. As Alex, I want the current full-dataset checkpoint treated as a smoke artifact, so that it does not accidentally become proof evidence.
26. As Alex, I want Top-50 success thresholds, so that the team can make a go/no-go decision instead of debating vague results.
27. As Alex, I want a yellow-zone tuning path, so that middling accuracy leads to specific changes instead of abandoning the approach.
28. As Alex, I want a no-go debug threshold, so that weak results trigger investigation before scaling.
29. As a collaborator, I want documented dashboard settings, so that I can reproduce Alex's training run.
30. As a collaborator, I want generated split manifests, so that I can verify label counts and leakage checks.
31. As a collaborator, I want a stable evaluation command, so that I can run the same measurement on any frozen checkpoint.
32. As a reviewer, I want a clear distinction between training loss and test accuracy, so that the model is judged by behavior.
33. As a reviewer, I want a compact explanation of q64_full encoding, so that the training inputs are understandable.
34. As a reviewer, I want evidence from held-out examples, so that claims about ASL recognition are credible.
35. As the project, I want future experiments to vary learning rate, epochs, encoding, or LoRA settings intentionally, so that tuning remains controlled.
36. As the project, I want top-50 proof metrics before larger data work, so that scarce time is spent on a validated path.
37. As the project, I want the demo scoped honestly around supported signs, so that the prototype does not overclaim coverage.
38. As the project, I want a clean handoff from training to evaluation to demo, so that progress is not lost between tools.
39. As the project, I want all artifacts recorded in project-controlled locations, so that scratch workspace outputs do not become hidden dependencies.
40. As the project, I want this workflow to remain compatible with a future coded Unsloth pipeline, so that dashboard work is not wasted if automation becomes necessary.

## Implementation Decisions

- Use Unsloth Studio as the training interface for the Top-50 baseline instead of building custom training orchestration first.
- Use Gemma 4 E4B IT with QLoRA 4-bit as the primary model/method for the baseline.
- Use q64_full as the first compact pose encoding because it preserves all selected frames while fitting within the target context length.
- Use a random stratified Top-50 split as the first proof gate, with signer-independent evaluation deferred until the basic signal is proven.
- Use a 70/15/15 train/validation/test split with a fixed seed and no sample overlap across splits.
- Keep the held-out test split out of all dashboard training and validation inputs.
- Use three training epochs as the baseline run, while monitoring validation loss for overfitting.
- Use AdamW 8-bit as the optimizer for the dashboard run, with conservative batch size and gradient accumulation where available.
- Treat the current full-dataset dashboard checkpoint as a smoke-test artifact only, not proof-of-learning evidence.
- Freeze/copy the dashboard output checkpoint into the project before evaluation.
- Record run settings in an experiment record and dashboard checklist before executing the run.
- Use strict normalized exact-match top-1 accuracy as the primary evaluation metric.
- Normalize model outputs by trimming whitespace, punctuation, and case, but do not give semantic credit for different words.
- Use invalid-output rate, per-class accuracy, confusion counts, and predictions output as secondary evaluation artifacts.
- Evaluate free-generation output first because it matches demo behavior.
- Add constrained Top-50 scoring only if free-generation accuracy is poor and the team needs to determine whether the model learned signal but failed output formatting.
- Build one shared q64 prediction contract used by both evaluation and future demo code.
- Implement q64 JSONL prediction/evaluation before video input support.
- Integrate prerecorded/known-good video before live camera to isolate video pipeline failures from model failures.
- If Top-50 reaches the go threshold, prioritize demo integration before scaling to full 250-gloss training.
- If Top-50 is in the yellow zone, tune training settings or encoding before scaling.
- If Top-50 is below the debug threshold, investigate encoding, prompt, checkpoint loading, split distribution, or evaluator bugs before further training.

## Testing Decisions

- Tests should verify external behavior of the evaluation contract rather than implementation details.
- The q64 inference contract should be tested with mock predictors so that CI or local smoke checks do not require GPU model loading.
- Output normalization should be tested for case-insensitivity, whitespace handling, punctuation handling, and invalid label detection.
- Metrics should be tested for exact-match accuracy, invalid-output rate, per-class accuracy, and confusion matrix counts.
- Artifact writing should be tested by verifying that predictions and metrics files are created with expected fields.
- Split generation should be tested or verified for row counts, label coverage, and zero sample overlap across train, validation, and test splits.
- The dashboard checklist should be treated as an operational test aid: the train and eval files must be uploaded correctly, and the test file must remain untouched.
- Real model evaluation should be run after checkpoint freezing, not in lightweight tests.
- The first real evaluation should run against the full Top-50 test split and write stable metrics artifacts.
- If real evaluation fails due to dependencies, the error should guide the user to install project dependencies or run mock mode.
- Prior art exists in the project around demo output contracts, readiness reporting, and Phase2A reporting; new evaluator tests should follow the same pattern of testing behavior and output artifacts.

## Out of Scope

- Training the full 250-gloss model as part of this PRD.
- Signer-independent evaluation as the first proof gate.
- Live camera integration as the first demo path.
- Replacing Unsloth Studio with a fully coded training loop unless the dashboard blocks the required workflow.
- Building a full model registry or experiment tracking platform.
- Optimizing mobile deployment or Cactus export from this checkpoint.
- Claiming production-grade ASL recognition from the Top-50 proof run.
- Collecting new ASL data or expanding the dataset.
- Implementing constrained label scoring unless free generation needs diagnosis.

## Further Notes

The Top-50 proof run is intentionally small and may have limited generalization because each class has few examples. Its purpose is to prove that the encoding, training flow, checkpoint handling, and evaluation loop work end-to-end. If it succeeds, the next high-leverage move is a scoped prerecorded demo using supported signs and known-good clips. If it fails, the result should guide whether to tune q64 encoding, prompt format, hyperparameters, or move from dashboard training to coded Unsloth training.

The agreed go/no-go ladder is:

- Strong go: at least 80% strict normalized exact-match test accuracy.
- Go: at least 70% strict normalized exact-match test accuracy.
- Yellow: 40% to 70%, tune before deciding.
- No-go/debug: below 40%, investigate before scaling or demo integration.
