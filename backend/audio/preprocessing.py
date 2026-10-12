from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

from backend.config import MAX_REFERENCE_SECONDS, MIN_REFERENCE_SECONDS, SAMPLE_RATE

MIN_SPEECH_RMS = 1e-3
MIN_SNR_DB = 10.0


@dataclass
class ReferenceInfo:
    """Facts about a preprocessed reference recording."""

    raw_duration: float
    duration: float
    sample_rate: int
    snr_db: float | None = None
    warnings: List[str] = field(default_factory=list)


def read_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """Read any soundfile-supported file as mono float32."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")
    try:
        audio, sr = sf.read(str(path), always_2d=False)
    except Exception as exc:  # soundfile raises LibsndfileError / RuntimeError
        raise ValueError("Unsupported or corrupt audio file. Upload a WAV, FLAC or OGG recording.") from exc
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def validate_reference_audio(
    path: str | Path,
    *,
    min_seconds: float = MIN_REFERENCE_SECONDS,
    max_seconds: float = MAX_REFERENCE_SECONDS,
) -> Tuple[np.ndarray, float]:
    audio, sr = read_audio(path)
    if audio.size == 0:
        raise ValueError("Reference audio is empty.")

    duration = audio.shape[0] / float(sr)
    if duration < min_seconds:
        raise ValueError(f"Reference audio must be at least {min_seconds:.1f} seconds long (got {duration:.1f}s).")
    if duration > max_seconds:
        raise ValueError(f"Reference audio must be no longer than {max_seconds:.1f} seconds (got {duration:.1f}s).")
    return audio, float(sr)


def estimate_snr_db(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> float:
    """Crude SNR: loud-frame level vs quiet-frame level."""
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size < 4:
        return float("inf")
    speech = float(np.percentile(rms, 90))
    noise = max(float(np.percentile(rms, 10)), 1e-6)
    return float(20.0 * np.log10(max(speech, 1e-6) / noise))


def preprocess_reference_audio(
    path: str | Path,
    *,
    target_sr: int = SAMPLE_RATE,
    min_seconds: float = MIN_REFERENCE_SECONDS,
    max_seconds: float = MAX_REFERENCE_SECONDS,
) -> Tuple[np.ndarray, int, ReferenceInfo]:
    """Mono -> resample -> peak-normalize -> trim leading/trailing silence."""
    audio, sr = validate_reference_audio(path, min_seconds=min_seconds, max_seconds=max_seconds)
    raw_duration = len(audio) / sr
    audio = audio.astype(np.float32)
    if int(sr) != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        raise ValueError("No speech detected: the recording is silent.")
    audio = audio / peak * 0.95

    trimmed, _ = librosa.effects.trim(audio, top_db=35, frame_length=1024, hop_length=256)
    if trimmed.size == 0 or float(np.sqrt(np.mean(np.square(trimmed)))) < MIN_SPEECH_RMS:
        raise ValueError("No speech detected after silence cleanup.")
    audio = trimmed.astype(np.float32)

    info = ReferenceInfo(raw_duration=raw_duration, duration=len(audio) / sr, sample_rate=sr)
    if info.duration < min_seconds * 0.6:
        raise ValueError("Not enough speech in the reference recording; provide 5-10 seconds of continuous speech.")
    info.snr_db = estimate_snr_db(audio)
    if info.snr_db < MIN_SNR_DB:
        info.warnings.append(
            f"Reference looks noisy (estimated SNR {info.snr_db:.0f} dB); cloning quality may suffer."
        )
    return audio, sr, info


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.asarray(audio, dtype=np.float32), sample_rate)
    return str(out_path)


def calculate_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def calculate_duration(audio: np.ndarray, sample_rate: int) -> float:
    return float(len(audio) / sample_rate)
