from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from backend.evaluation.models import get_speaker_encoder

EMBED_SR = 16000


def _load(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"Audio is empty: {path}")
    return audio, int(sample_rate)


def speaker_embedding(audio: np.ndarray, sample_rate: int):
    """ECAPA-TDNN (VoxCeleb) speaker embedding as a 1-D torch tensor."""
    import torch

    encoder = get_speaker_encoder()
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != EMBED_SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=EMBED_SR)
    with torch.inference_mode():
        embedding = encoder.encode_batch(torch.from_numpy(audio).unsqueeze(0).to(encoder.device))
    return embedding.squeeze()


def embedding_similarity(first, second) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item())


def audio_similarity(reference: np.ndarray, reference_sr: int, generated: np.ndarray, generated_sr: int) -> float:
    return embedding_similarity(speaker_embedding(reference, reference_sr), speaker_embedding(generated, generated_sr))


def calculate_speaker_similarity(reference_audio: str | Path, generated_audio: str | Path) -> float:
    """Cosine similarity between ECAPA speaker embeddings of two audio files."""
    ref, ref_sr = _load(reference_audio)
    gen, gen_sr = _load(generated_audio)
    return audio_similarity(ref, ref_sr, gen, gen_sr)
