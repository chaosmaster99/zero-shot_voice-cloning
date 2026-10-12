from __future__ import annotations

import pytest

from backend.evaluation.wer import compute_wer_from_asr, normalize_text, word_error_rate


def test_word_error_rate_zero_for_identical_text():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_word_error_rate_nonzero_for_mismatch():
    assert word_error_rate("hello world", "hello there") == pytest.approx(0.5)


def test_wer_ignores_case_and_punctuation():
    assert word_error_rate("Hello, world!", "hello world") == 0.0


def test_wer_counts_insertions_and_deletions():
    assert word_error_rate("a b c d", "a b") == pytest.approx(0.5)
    assert word_error_rate("a b", "a b c d") == pytest.approx(1.0)


def test_empty_transcript_is_total_error_not_a_crash():
    assert compute_wer_from_asr("hello world", "") == 1.0


def test_missing_reference_text_is_an_error():
    with pytest.raises(ValueError):
        compute_wer_from_asr("   ", "hello")


def test_normalize_text():
    assert normalize_text("  Don't   STOP!! ") == "don t stop"
