from __future__ import annotations

import re
from typing import List

_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "no.", "jr.", "sr."}


def split_sentences(text: str, min_words: int = 3) -> List[str]:
    """Split text into sentences for sentence-level synthesis.

    Splits after . ! ? when the next sentence starts with a capital letter or digit, keeps
    abbreviations (Dr., e.g.) and decimals (3.5) together, and merges fragments shorter than
    `min_words` into their neighbour so the TTS never gets a one-word utterance.
    """
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    raw = [part.strip() for part in _BOUNDARY.split(normalized) if part and part.strip()]
    merged: List[str] = []
    for part in raw:
        if merged and merged[-1].split()[-1].lower() in _ABBREVIATIONS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)

    result: List[str] = []
    for part in merged:
        if result and len(part.split()) < min_words:
            result[-1] = f"{result[-1]} {part}"
        else:
            result.append(part)
    if len(result) > 1 and len(result[0].split()) < min_words:
        result[1] = f"{result[0]} {result[1]}"
        result.pop(0)
    return result
