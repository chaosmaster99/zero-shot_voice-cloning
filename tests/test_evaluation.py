from __future__ import annotations

from backend.evaluation.wer import word_error_rate


def test_word_error_rate_zero_for_identical_text():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_word_error_rate_nonzero_for_mismatch():
    score = word_error_rate("hello world", "hello there")
    assert score > 0.0
