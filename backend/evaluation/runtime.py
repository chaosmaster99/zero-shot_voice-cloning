from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from backend.evaluation.prosody_metrics import analyze_prosody
from backend.evaluation.wer import compute_wer_from_asr, normalize_text

Clip = Tuple[np.ndarray, int]


def count_words(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    words = normalize_text(text).split()
    return len(words) or None


def evaluate_clips(
    reference: Clip,
    clips: Dict[str, Clip],
    target_text: Optional[str],
    *,
    include_contours: bool = True,
    with_speaker: bool = True,
    with_asr: bool = True,
) -> Dict[str, Any]:
    """Speaker similarity (ECAPA), WER (Whisper) and prosody statistics for each clip.

    A missing optional model never aborts the request: the affected metric is omitted and
    a human-readable warning is returned instead.
    """
    from backend.evaluation.speaker_similarity import embedding_similarity, speaker_embedding

    words = count_words(target_text)
    warnings: list[str] = []
    result: Dict[str, Any] = {
        "reference": {"prosody": analyze_prosody(*reference, include_contours=include_contours)},
        "clips": {},
        "warnings": warnings,
    }

    reference_embedding = None
    if with_speaker:
        try:
            reference_embedding = speaker_embedding(*reference)
        except Exception as exc:
            warnings.append(f"Speaker similarity unavailable: {exc}")

    asr_failed = False
    for name, (audio, sample_rate) in clips.items():
        entry: Dict[str, Any] = {"prosody": analyze_prosody(audio, sample_rate, num_words=words, include_contours=include_contours)}

        if reference_embedding is not None:
            try:
                entry["speaker_similarity"] = embedding_similarity(reference_embedding, speaker_embedding(audio, sample_rate))
            except Exception as exc:
                warnings.append(f"Speaker similarity failed for {name}: {exc}")
                reference_embedding = None

        if with_asr and target_text and not asr_failed:
            try:
                from backend.evaluation.asr import transcribe

                transcript = transcribe(audio, sample_rate)
                entry["transcript"] = transcript
                entry["wer"] = compute_wer_from_asr(target_text, transcript)
            except Exception as exc:
                warnings.append(f"WER unavailable: {exc}")
                asr_failed = True

        result["clips"][name] = entry
    return result
