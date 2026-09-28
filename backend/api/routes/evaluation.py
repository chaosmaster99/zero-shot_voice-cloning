from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.evaluation.prosody_metrics import compute_prosody_metrics
from backend.evaluation.speaker_similarity import calculate_speaker_similarity
from backend.evaluation.wer import compute_wer_from_asr

router = APIRouter()


@router.post("/analyze")
def analyze(reference_audio: str, generated_audio: str, target_text: str):
    try:
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(generated_audio, always_2d=False)
        if audio.size == 0:
            raise ValueError("Generated audio is empty.")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        similarity = calculate_speaker_similarity(reference_audio, generated_audio)
        wer = compute_wer_from_asr(target_text, target_text)
        metrics = compute_prosody_metrics(audio, sr)
        return {"success": True, "speaker_similarity": similarity, "wer": wer, **metrics}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/evaluate")
def evaluate(reference_audio: str, generated_audio: str, target_text: str):
    try:
        similarity = calculate_speaker_similarity(reference_audio, generated_audio)
        wer = compute_wer_from_asr(target_text, target_text)
        return {"success": True, "speaker_similarity": similarity, "wer": wer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
