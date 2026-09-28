from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity


def _load_audio_embedding(path: str | Path):
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.size == 0:
        raise ValueError(f"Audio is empty: {path}")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def calculate_speaker_similarity(reference_audio: str | Path, generated_audio: str | Path) -> float:
    ref_audio, _ = _load_audio_embedding(reference_audio)
    gen_audio, _ = _load_audio_embedding(generated_audio)

    ref_feature = np.asarray(ref_audio[: min(len(ref_audio), 16000)], dtype=np.float32)
    gen_feature = np.asarray(gen_audio[: min(len(gen_audio), 16000)], dtype=np.float32)

    ref_vec = ref_feature.reshape(1, -1)
    gen_vec = gen_feature.reshape(1, -1)
    sim = cosine_similarity(ref_vec, gen_vec)[0][0]
    return float(np.clip(sim, -1.0, 1.0))
