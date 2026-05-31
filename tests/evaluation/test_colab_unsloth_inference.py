from __future__ import annotations

from src.evaluation.colab_unsloth_inference import (
    RetryConfig,
    build_retry_prompt,
    evaluate_anchor_samples,
    infer_with_single_retry,
)


def test_infer_with_single_retry_keeps_first_pass_when_valid() -> None:
    calls: list[dict[str, object]] = []

    def _predict(prompt: str, decode: dict[str, object]) -> str:
        calls.append({"prompt": prompt, "decode": decode})
        return "yes"

    result = infer_with_single_retry(
        sample_id="s1",
        expected_gloss="yes",
        record_input="pose: ...",
        allowlist=["yes", "no"],
        alias_entries=None,
        predict_fn=_predict,
    )

    assert result.used_retry is False
    assert result.final_valid is True
    assert result.final_gloss == "yes"
    assert len(calls) == 1


def test_infer_with_single_retry_uses_deterministic_retry_on_oov() -> None:
    outputs = iter(["nonsense token", "no"])
    seen_decode: list[dict[str, object]] = []

    def _predict(prompt: str, decode: dict[str, object]) -> str:
        seen_decode.append(decode)
        return next(outputs)

    result = infer_with_single_retry(
        sample_id="s2",
        expected_gloss="no",
        record_input="pose: ...",
        allowlist=["yes", "no"],
        alias_entries=None,
        predict_fn=_predict,
        retry_config=RetryConfig(max_new_tokens=2, temperature=0.0, top_p=1.0),
    )

    assert result.used_retry is True
    assert result.first_pass.valid is False
    assert result.retry_pass is not None
    assert result.retry_pass.valid is True
    assert result.final_gloss == "no"
    assert seen_decode[1] == {"max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0}


def test_retry_prompt_contains_previous_output_and_allowlist() -> None:
    prompt = build_retry_prompt(
        record_input="signal",
        allowlist=["thank you", "yes"],
        first_output="foobar",
    )
    assert "previous output" in prompt.lower()
    assert "foobar" in prompt
    assert "thank you" in prompt
    assert "yes" in prompt


def test_evaluate_anchor_samples_logs_first_and_retry_fields() -> None:
    outputs = iter(["bad", "yes", "no"])

    def _predict(prompt: str, decode: dict[str, object]) -> str:
        return next(outputs)

    rows = evaluate_anchor_samples(
        samples=[
            {"sample_id": "a", "expected_gloss": "yes", "input": "x"},
            {"sample_id": "b", "expected_gloss": "no", "input": "y"},
        ],
        allowlist=["yes", "no"],
        alias_entries=None,
        predict_fn=_predict,
    )

    assert rows[0]["retry_used"] is True
    assert rows[0]["first_pass_raw"] == "bad"
    assert rows[0]["retry_raw"] == "yes"
    assert rows[1]["retry_used"] is False
    assert rows[1]["first_pass_raw"] == "no"


def test_alias_entries_allow_retry_recovery_to_canonical_gloss() -> None:
    outputs = iter(["oops", "thankyou"])

    def _predict(prompt: str, decode: dict[str, object]) -> str:
        return next(outputs)

    result = infer_with_single_retry(
        sample_id="s3",
        expected_gloss="thank you",
        record_input="pose",
        allowlist=["thank you", "yes"],
        alias_entries=[{"alias": "thankyou", "canonical": "thank you"}],
        predict_fn=_predict,
    )

    assert result.used_retry is True
    assert result.final_valid is True
    assert result.final_gloss == "thank you"
    assert result.correct is True


def test_retry_runs_once_and_can_still_fail_closed() -> None:
    calls = 0

    def _predict(prompt: str, decode: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        return "still_oov"

    result = infer_with_single_retry(
        sample_id="s4",
        expected_gloss="yes",
        record_input="pose",
        allowlist=["yes", "no"],
        alias_entries=None,
        predict_fn=_predict,
    )

    assert calls == 2
    assert result.used_retry is True
    assert result.final_valid is False
    assert result.final_gloss is None
    assert result.correct is False
