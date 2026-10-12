from __future__ import annotations

import librosa
import numpy as np
import parselmouth
from scipy.signal import butter, sosfiltfilt
from typing import Dict, List

DC_CUTOFF_HZ = 30.0
# Silent gaps at least this long are pauses (shorter ones are articulatory closures). Shared by the
# controller (what it scales) and the metrics (what is measured) so the two always agree.
MIN_PAUSE_SECONDS = 0.1
F0_MIN_HZ = 60.0
F0_MAX_HZ = 600.0
# Vocal tremor rate. Physiological tremor in fearful / sad voices sits around 4-7 Hz.
TREMOR_HZ = 5.5
OCTAVE_JUMP_ST = 7.0  # F0 jumps larger than this vs. the local median are tracker octave errors
MIN_PHRASE_VOICED_FRAMES = 8
SLOW_KNOT_SECONDS = 0.3  # knot spacing of the slow-intonation spline used to isolate tremor
ALPHA_SPLIT_HZ = 1000.0  # eGeMAPS alpha ratio: energy 1-5 kHz over 50 Hz-1 kHz


def remove_dc(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Zero-phase 30 Hz high-pass.

    A DC offset disturbs F0 tracking and, because Praat's resynthesis subtracts the global mean,
    turns digital-zero silences into a constant noise floor that corrupts pause detection.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate < 4 * DC_CUTOFF_HZ or audio.size < 64:
        return audio
    sos = butter(2, DC_CUTOFF_HZ, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


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


def speech_intervals(audio: np.ndarray, top_db: float = 35.0, frame_length: int = 1024, hop_length: int = 256):
    """Return non-silent (start, end) sample indices.

    The threshold is relative to the signal peak, so applying a gain does not change
    which regions count as speech.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-6:
        return np.zeros((0, 2), dtype=int)
    return librosa.effects.split(audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)


def phrase_intervals(audio: np.ndarray, sample_rate: int, min_pause_seconds: float = MIN_PAUSE_SECONDS) -> np.ndarray:
    """Intonation phrases: speech intervals merged across gaps shorter than a pause.

    Shared by the controller (where contour slopes are applied) and the metrics (where they are
    measured), so both see the same phrases.
    """
    intervals = speech_intervals(audio)
    if len(intervals) == 0:
        return intervals
    min_gap = int(round(min_pause_seconds * sample_rate))
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] < min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=int)


def detect_pauses(audio: np.ndarray, sample_rate: int, min_pause_seconds: float = MIN_PAUSE_SECONDS, top_db: float = 35.0):
    """Return internal pauses as (start_s, end_s). Leading/trailing silence is not a pause."""
    intervals = speech_intervals(audio, top_db=top_db)
    pauses = []
    for (_, prev_end), (next_start, _) in zip(intervals[:-1], intervals[1:]):
        length = (next_start - prev_end) / sample_rate
        if length >= min_pause_seconds:
            pauses.append((prev_end / sample_rate, next_start / sample_rate))
    return pauses


def extract_f0(audio: np.ndarray, sample_rate: int, time_step: float = 0.01):
    """Frame-wise F0 in Hz (NaN where unvoiced) and the frame times.

    Praat autocorrelation tracking with a two-pass, per-speaker search range
    (0.75*Q1 .. 2.5*Q3 of a first wide pass) so octave errors do not leak into the statistics.
    """
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=float(sample_rate))
    first = sound.to_pitch(time_step=time_step, pitch_floor=F0_MIN_HZ, pitch_ceiling=600.0)
    wide = first.selected_array["frequency"]
    wide = wide[wide > 0]
    floor, ceiling = F0_MIN_HZ, 600.0
    if wide.size >= 10:
        q1, q3 = np.percentile(wide, [25, 75])
        floor, ceiling = max(F0_MIN_HZ, 0.75 * q1), min(F0_MAX_HZ, 2.5 * q3)
    pitch = sound.to_pitch_ac(
        time_step=time_step, pitch_floor=floor, pitch_ceiling=ceiling, octave_jump_cost=0.6, voicing_threshold=0.5
    )
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 <= 0] = np.nan
    return correct_octave_errors(f0), np.asarray(pitch.xs(), dtype=float)


def correct_octave_errors(f0: np.ndarray, window_frames: int = 51, jump_st: float = OCTAVE_JUMP_ST) -> np.ndarray:
    """Fold or drop octave errors: frames more than `jump_st` semitones from the running median
    of the surrounding voiced frames (+-0.25 s) are moved by whole octaves when that brings them
    within 3 st of it, and discarded otherwise. Real intonation does not jump that far between
    neighbouring frames; tracker halving/doubling (creak, low-energy edges) does.
    """
    voiced = np.flatnonzero(~np.isnan(f0))
    if voiced.size < 5:
        return f0
    st = 12.0 * np.log2(f0[voiced])
    half = window_frames // 2
    reference = np.array([np.median(st[max(0, k - half) : k + half + 1]) for k in range(len(st))])
    deviation = st - reference
    corrected = f0.copy()
    for k in np.flatnonzero(np.abs(deviation) > jump_st):
        folded = deviation[k] - 12.0 * np.round(deviation[k] / 12.0)
        corrected[voiced[k]] = 2.0 ** ((reference[k] + folded) / 12.0) if abs(folded) < 3.0 else np.nan
    return corrected


def alpha_ratio_db(audio: np.ndarray, sample_rate: int) -> float:
    """eGeMAPS alpha ratio of the long-term speech spectrum: 1-5 kHz over 50 Hz-1 kHz energy, in dB.

    Rises with vocal effort / a flatter, "brighter" spectrum; falls for soft, dark voices.
    """
    intervals = speech_intervals(audio)
    if len(intervals) == 0:
        return 0.0
    speech = np.concatenate([audio[start:end] for start, end in intervals])
    power = np.mean(np.abs(librosa.stft(speech, n_fft=1024, hop_length=256)) ** 2, axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
    low = power[(freqs >= 50.0) & (freqs < ALPHA_SPLIT_HZ)].sum()
    high = power[(freqs >= ALPHA_SPLIT_HZ) & (freqs < min(5000.0, sample_rate / 2))].sum()
    return float(10.0 * np.log10(max(high, 1e-12) / max(low, 1e-12)))


def intonation_features(f0: np.ndarray, times: np.ndarray, phrases: List[tuple]) -> Dict[str, float]:
    """Decompose the voiced F0 contour (semitones re. its median) into tremor, phrase slope and range.

    f0_tremor        -- amplitude (peak, semitones) of the TREMOR_HZ component, estimated by least
                        squares together with a slow per-phrase spline, so intonation does not
                        leak into it (lock-in detection). Natural speech has
                        almost no coherent component at that frequency, so baselines are near 0.
    f0_slope         -- mean per-phrase regression slope (semitones/second, weighted by voiced
                        frames) of the contour with tremor removed. Declination gives < 0.
    f0_std_semitones -- robust spread (IQR/1.349) of the contour with tremor and per-phrase slopes
                        removed: the intonation *range*, independent of the slope/tremor controls,
                        and insensitive to isolated octave-tracking errors.
    """
    voiced = ~np.isnan(f0)
    out = {"f0_std_semitones": 0.0, "f0_slope": 0.0, "f0_tremor": 0.0}
    if voiced.sum() < 2:
        return out
    st = np.full_like(f0, np.nan)
    st[voiced] = 12.0 * np.log2(f0[voiced] / float(np.median(f0[voiced])))

    # Slow intonation: a linear spline per phrase (knots every SLOW_KNOT_SECONDS, bridging unvoiced
    # gaps), which cannot follow the 5.5 Hz tremor. Voiced runs are often shorter than one tremor
    # cycle, so per-run lines would absorb the tremor; a phrase-wide spline does not.
    blocks, rows = [], []
    for start, end in phrases:
        inside = np.flatnonzero(voiced & (times >= start) & (times < end))
        if inside.size < MIN_PHRASE_VOICED_FRAMES:
            continue
        knots = np.linspace(start, end, max(2, int(np.ceil((end - start) / SLOW_KNOT_SECONDS)) + 1))
        hats = np.stack([np.interp(times[inside], knots, np.eye(len(knots))[j]) for j in range(len(knots))], axis=1)
        blocks.append(hats)
        rows.append(inside)
    tremor_fit = np.zeros_like(st)
    if rows:
        rows_all = np.concatenate(rows)
        design = np.zeros((len(rows_all), sum(b.shape[1] for b in blocks) + 2))
        r = c = 0
        for block in blocks:
            design[r : r + block.shape[0], c : c + block.shape[1]] = block
            r += block.shape[0]
            c += block.shape[1]
        omega = 2.0 * np.pi * TREMOR_HZ * times[rows_all]
        design[:, -2], design[:, -1] = np.sin(omega), np.cos(omega)
        coef, *_ = np.linalg.lstsq(design, st[rows_all], rcond=None)
        out["f0_tremor"] = float(np.hypot(coef[-2], coef[-1]))
        all_omega = 2.0 * np.pi * TREMOR_HZ * times
        tremor_fit = coef[-2] * np.sin(all_omega) + coef[-1] * np.cos(all_omega)
    clean = st - tremor_fit

    detrended = clean.copy()
    slopes, weights = [], []
    for start, end in phrases:
        inside = voiced & (times >= start) & (times < end)
        if inside.sum() < MIN_PHRASE_VOICED_FRAMES or np.ptp(times[inside]) < 0.25:
            continue
        slope = float(np.polyfit(times[inside], clean[inside], 1)[0])
        slopes.append(slope)
        weights.append(int(inside.sum()))
        detrended[inside] -= slope * (times[inside] - float(np.median(times[inside])))
    if slopes:
        out["f0_slope"] = float(np.average(slopes, weights=weights))

    values = detrended[voiced]
    q25, q75 = np.percentile(values, [25, 75])
    out["f0_std_semitones"] = float(0.7413 * (q75 - q25))
    return out


def contour_statistics(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Median F0 (Hz) plus the intonation features, exactly as the evaluation measures them."""
    audio = remove_dc(audio, sample_rate)
    f0, times = extract_f0(audio, sample_rate)
    voiced = ~np.isnan(f0)
    phrases = [(start / sample_rate, end / sample_rate) for start, end in phrase_intervals(audio, sample_rate)]
    return {"f0_median": float(np.median(f0[voiced])) if voiced.any() else 0.0, **intonation_features(f0, times, phrases)}
