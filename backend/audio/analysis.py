from __future__ import annotations

import numpy as np
import librosa


def waveform(audio: np.ndarray):
    return np.asarray(audio, dtype=np.float32)


def stft(audio: np.ndarray, sample_rate: int, n_fft: int = 2048, hop_length: int = 512):
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length), sample_rate


def spectrogram(audio: np.ndarray, sample_rate: int, n_fft: int = 2048, hop_length: int = 512):
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    return magnitude, sample_rate


def mel_spectrogram(audio: np.ndarray, sample_rate: int, n_mels: int = 80):
    return librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels), sample_rate


def compute_f0(audio: np.ndarray, sample_rate: int):
    f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sample_rate)
    return f0


def compute_rms_energy(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512):
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def estimate_duration(audio: np.ndarray, sample_rate: int) -> float:
    return float(len(audio) / sample_rate)


def detect_pauses(audio: np.ndarray, sample_rate: int, energy_threshold: float = 0.01):
    rms = compute_rms_energy(audio, frame_length=2048, hop_length=256)
    mask = rms < energy_threshold
    frames = np.where(mask)[0]
    if len(frames) == 0:
        return []
    starts = []
    ends = []
    current_start = frames[0]
    prev = frames[0]
    for idx in frames[1:]:
        if idx - prev > 1:
            starts.append(current_start)
            ends.append(prev)
            current_start = idx
        prev = idx
    starts.append(current_start)
    ends.append(prev)
    return [(start, end) for start, end in zip(starts, ends)]
