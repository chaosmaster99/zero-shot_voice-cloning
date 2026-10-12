from __future__ import annotations

import librosa
import numpy as np

from backend.config import get_device
from backend.evaluation.models import get_whisper

ASR_SR = 16000
WINDOW_SECONDS = 30


def transcribe(audio: np.ndarray, sample_rate: int) -> str:
    """Whisper transcription (windowed so clips longer than 30 s are not truncated)."""
    import torch

    processor, model = get_whisper()
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != ASR_SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=ASR_SR)

    window = WINDOW_SECONDS * ASR_SR
    pieces = []
    for start in range(0, max(len(audio), 1), window):
        chunk = audio[start : start + window]
        if chunk.size < ASR_SR // 10:
            continue
        features = processor(chunk, sampling_rate=ASR_SR, return_tensors="pt").input_features.to(get_device())
        with torch.inference_mode():
            tokens = model.generate(features)
        pieces.append(processor.batch_decode(tokens, skip_special_tokens=True)[0].strip())
    return " ".join(p for p in pieces if p).strip()
