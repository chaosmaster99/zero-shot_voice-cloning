from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

from backend.config import MAX_REFERENCE_SECONDS, MIN_REFERENCE_SECONDS, SAMPLE_RATE


def validate_reference_audio(path: str | Path, *, min_seconds: float = MIN_REFERENCE_SECONDS, max_seconds: float = MAX_REFERENCE_SECONDS) -> Tuple[float, float]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")

    audio, sr = sf.read(str(path), always_2d=False)
    if audio.size == 0:
        raise ValueError("Reference audio is empty.")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    duration = audio.shape[0] / float(sr)
    if duration < min_seconds:
        raise ValueError(f"Reference audio must be at least {min_seconds:.1f} seconds long.")
    if duration > max_seconds:
        raise ValueError(f"Reference audio must be no longer than {max_seconds:.1f} seconds.")

    return audio, float(sr)


def preprocess_reference_audio(path: str | Path, *, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    audio, sr = validate_reference_audio(path)
    audio = librosa.to_mono(audio.astype(np.float32)) if audio.ndim > 1 else audio.astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio = librosa.util.normalize(audio)
    audio = remove_silence(audio, sample_rate=sr)
    if audio.size == 0:
        raise ValueError("No speech content found after silence cleanup.")
    return audio.astype(np.float32), sr


def remove_silence(audio: np.ndarray, sample_rate: int, top_db: float = 30.0) -> np.ndarray:
    non_silent = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=256)
    if len(non_silent) == 0:
        return audio
    segments = [audio[start:end] for start, end in non_silent]
    return np.concatenate(segments) if segments else audio


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio.astype(np.float32), sample_rate)
    return str(out_path)


def calculate_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def calculate_duration(audio: np.ndarray, sample_rate: int) -> float:
    return float(len(audio) / sample_rate)


def compute_stft(audio: np.ndarray, sample_rate: int, n_fft: int = 2048, hop_length: int = 512):
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)


def compute_mel_spectrogram(audio: np.ndarray, sample_rate: int, n_mels: int = 80):
    return librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
