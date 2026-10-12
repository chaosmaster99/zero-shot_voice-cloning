from __future__ import annotations

from typing import Any, Dict

import librosa
import numpy as np

WAVEFORM_BINS = 500
MEL_BANDS = 64
MEL_MAX_FRAMES = 240


def waveform_envelope(audio: np.ndarray, bins: int = WAVEFORM_BINS) -> Dict[str, Any]:
    """Min/max per bin so the browser can draw the waveform without the raw samples."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return {"min": [], "max": []}
    bins = min(bins, audio.size)
    edges = np.linspace(0, audio.size, bins + 1).astype(int)
    mins = [float(audio[a:b].min()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    maxs = [float(audio[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    return {"min": [round(v, 4) for v in mins], "max": [round(v, 4) for v in maxs]}


def mel_image(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Mel-spectrogram in dB, scaled to 0..255 and shaped [mel][frame] (low frequencies first)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < 2048:
        return {"data": [], "bands": MEL_BANDS, "frames": 0}
    hop = max(256, int(len(audio) / MEL_MAX_FRAMES))
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=hop, n_mels=MEL_BANDS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=70.0)
    scaled = np.clip((mel_db + 70.0) / 70.0, 0.0, 1.0) * 255.0
    return {"data": scaled.astype(np.uint8).tolist(), "bands": int(mel.shape[0]), "frames": int(mel.shape[1])}
