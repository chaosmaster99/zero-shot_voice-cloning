from __future__ import annotations

import pytest

from backend.core.text import split_sentences


def test_splits_on_sentence_boundaries():
    text = "Welcome to our speech processing project. Today we explore how prosody shapes meaning."
    assert split_sentences(text) == [
        "Welcome to our speech processing project.",
        "Today we explore how prosody shapes meaning.",
    ]


def test_single_sentence_is_unchanged():
    assert split_sentences("Hello there, how are you today?") == ["Hello there, how are you today?"]


def test_question_and_exclamation_boundaries():
    parts = split_sentences("Are you ready for the demo? Yes we are! Let us begin the presentation now.")
    assert len(parts) == 3


def test_abbreviations_and_decimals_do_not_split():
    assert len(split_sentences("Dr. Smith measured 3.5 percent growth in the first quarter.")) == 1


def test_short_fragments_are_merged():
    parts = split_sentences("Stop. We will begin the meeting at nine o'clock tomorrow morning.")
    assert len(parts) == 1


def test_whitespace_is_normalized_and_empty_input_is_empty():
    assert split_sentences("   ") == []
    assert split_sentences("A  very   spaced   sentence here.") == ["A very spaced sentence here."]
