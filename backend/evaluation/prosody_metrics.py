from __future__ import annotations

from typing import Any, Dict, List, Optional

import librosa
import numpy as np

from backend.audio.analysis import (  # noqa: F401  (extract_f0 / alpha_ratio_db re-exported)
    F0_MAX_HZ,
    F0_MIN_HZ,
    MIN_PAUSE_SECONDS,
    alpha_ratio_db,
    detect_pauses,
    extract_f0,
    intonation_features,
    phrase_intervals,
    remove_dc,
    speech_intervals,
)

MAX_CONTOUR_POINTS = 300

_EMPTY = {
    "f0_mean": 0.0,
    "f0_median": 0.0,
    "f0_std_semitones": 0.0,
    "f0_slope": 0.0,
    "f0_tremor": 0.0,
    "energy_mean": 0.0,
    "alpha_ratio_db": 0.0,
    "duration": 0.0,
    "speech_duration": 0.0,
    "rate": None,
    "pause_count": 0,
    "pause_total": 0.0,
}


def _as_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return remove_dc(audio, sample_rate)


def _downsample(values: np.ndarray, max_points: int = MAX_CONTOUR_POINTS) -> np.ndarray:
    if len(values) <= max_points:
        return values
    index = np.linspace(0, len(values) - 1, max_points).astype(int)
    return values[index]


def analyze_prosody(
    audio: np.ndarray,
    sample_rate: int,
    num_words: Optional[int] = None,
    include_contours: bool = False,
) -> Dict[str, Any]:
    """Prosody and voice-quality statistics measured on speech regions only.

    f0_mean / f0_median   -- Hz over voiced frames (unvoiced and silent frames excluded).
    f0_std_semitones      -- intonation range: robust F0 spread in semitones (slopes and tremor removed).
    f0_slope              -- mean per-phrase F0 slope, semitones/second.
    f0_tremor             -- peak depth (semitones) of the 5.5 Hz F0 modulation.
    energy_mean           -- RMS over speech regions (silence excluded).
    alpha_ratio_db        -- spectral tilt: 1-5 kHz vs 50 Hz-1 kHz energy, dB.
    speech_duration       -- total non-silent time in seconds.
    pause_count/total     -- internal pauses of at least 0.1 s, and their summed length.
    rate                  -- words per second of *speech* time when num_words is given.
    """
    audio = _as_mono(audio, sample_rate)
    if audio.size == 0:
        metrics = dict(_EMPTY)
        if include_contours:
            metrics["contours"] = {"times": [], "f0": [], "energy_db": []}
        return metrics

    duration = len(audio) / sample_rate
    intervals = speech_intervals(audio)
    speech_samples = int(sum(end - start for start, end in intervals))
    speech_duration = speech_samples / sample_rate
    energy = 0.0
    if speech_samples:
        speech = np.concatenate([audio[start:end] for start, end in intervals])
        energy = float(np.sqrt(np.mean(np.square(speech))))

    f0_frames, f0_times = extract_f0(audio, sample_rate)
    voiced = f0_frames[~np.isnan(f0_frames)]
    phrases = [(start / sample_rate, end / sample_rate) for start, end in phrase_intervals(audio, sample_rate)]
    pauses = detect_pauses(audio, sample_rate, min_pause_seconds=MIN_PAUSE_SECONDS)

    metrics: Dict[str, Any] = {
        "f0_mean": float(np.mean(voiced)) if voiced.size else 0.0,
        "f0_median": float(np.median(voiced)) if voiced.size else 0.0,
        **intonation_features(f0_frames, f0_times, phrases),
        "energy_mean": energy,
        "alpha_ratio_db": alpha_ratio_db(audio, sample_rate),
        "duration": float(duration),
        "speech_duration": float(speech_duration),
        "rate": float(num_words / speech_duration) if num_words and speech_duration > 0 else None,
        "pause_count": len(pauses),
        "pause_total": float(sum(end - start for start, end in pauses)),
    }

    if include_contours:
        hop = max(1, sample_rate // 100)
        rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop)[0]
        energy_db = 20.0 * np.log10(np.maximum(rms, 1e-5))
        times = f0_times
        energy_times = np.arange(len(energy_db)) * hop / sample_rate

        def _clean(values: np.ndarray) -> List[Optional[float]]:
            return [None if np.isnan(v) else round(float(v), 2) for v in values]

        metrics["contours"] = {
            "times": [round(float(t), 3) for t in _downsample(times)],
            "f0": _clean(_downsample(f0_frames)),
            "energy_times": [round(float(t), 3) for t in _downsample(energy_times)],
            "energy_db": _clean(_downsample(energy_db)),
        }
    return metrics


def compute_prosody_metrics(audio: np.ndarray, sample_rate: int, num_words: Optional[int] = None) -> Dict[str, Any]:
    return analyze_prosody(audio, sample_rate, num_words=num_words, include_contours=False)
