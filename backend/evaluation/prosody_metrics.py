from __future__ import annotations

import numpy as np
import librosa


def compute_prosody_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    if audio.size == 0:
        return {"f0_mean": 0.0, "energy_mean": 0.0, "duration": 0.0, "rate": 0.0, "pause_count": 0}

    f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sample_rate)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=256)[0]
    duration = len(audio) / sample_rate
    rate = 1.0 / max(duration, 1e-6)
    pause_count = int(np.sum(rms < 0.01))

    return {
        "f0_mean": float(np.nanmean(f0)) if f0.size else 0.0,
        "energy_mean": float(np.mean(rms)) if rms.size else 0.0,
        "duration": float(duration),
        "rate": float(rate),
        "pause_count": pause_count,
    }
