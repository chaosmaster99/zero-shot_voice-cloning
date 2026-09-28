from __future__ import annotations

import re
from typing import Tuple


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    if not hyp:
        return 1.0

    dp = [[0 for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    edits = dp[len(ref)][len(hyp)]
    return float(edits / max(len(ref), 1))


def compute_wer_from_asr(reference_text: str, asr_transcript: str) -> float:
    if not reference_text or not asr_transcript:
        raise ValueError("Reference text and ASR transcript are required to compute WER.")
    return word_error_rate(reference_text, asr_transcript)
