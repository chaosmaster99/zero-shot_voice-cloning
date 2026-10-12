from __future__ import annotations

import math
from typing import Dict

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from backend.audio.analysis import (
    MIN_PAUSE_SECONDS,
    TREMOR_HZ,
    alpha_ratio_db,
    contour_statistics,
    extract_f0,
    phrase_intervals,
    remove_dc,
    speech_intervals,
)

PEAK_CEILING = 0.99
LIMITER_KNEE = 0.85
F0_ABS_MIN = 40.0
F0_ABS_MAX = 1000.0

# Spectral-tilt (brightness) control: gain ramps from 0 dB below to the full gain above this band,
# which is centred on the 1 kHz split of the alpha ratio used to measure it.
TILT_LOW_HZ = 900.0
TILT_HIGH_HZ = 1100.0
TILT_ITERATIONS = 3  # closed-loop refinements of the tilt gain
CONTOUR_ITERATIONS = 4  # closed-loop refinements of the F0 register and range
RANGE_GAIN_BOUNDS = (0.6, 1.6)  # total correction allowed on the range (the soft limit can cap it)
PITCH_TOLERANCE_ST = 0.05  # ~0.3 % F0
# Soft limit on the reshaped contour (semitones from the median): beyond the knee, deviations are
# compressed smoothly to at most knee + span. Keeps wide-range voices out of the creak / falsetto
# regions where PSOLA sounds rough and pitch trackers make octave errors.
CONTOUR_KNEE_ST = 8.0
CONTOUR_SPAN_ST = 4.0
RANGE_TOLERANCE = 0.02

NEUTRAL: Dict[str, float] = {
    "pitch": 0.0,         # % change of median F0
    "pitch_range": 1.0,   # multiplier on F0 variation around the median
    "pitch_slope": 0.0,   # semitones per second added to every phrase (declination / incline)
    "tremor": 0.0,        # peak depth (semitones) of a 5.5 Hz F0 modulation
    "energy": 0.0,        # % change of speech RMS
    "brightness": 0.0,    # dB change of high-frequency (>1 kHz) energy: spectral tilt / vocal effort
    "rate": 1.0,          # speaking-rate multiplier
    "pause": 1.0,         # multiplier on internal pause length
}


def active_rms(audio: np.ndarray) -> float:
    """RMS over speech regions only, so silence does not dilute the level."""
    intervals = speech_intervals(audio)
    if len(intervals) == 0:
        return 0.0
    speech = np.concatenate([audio[start:end] for start, end in intervals])
    return float(np.sqrt(np.mean(np.square(speech))))


def soft_limit(audio: np.ndarray, ceiling: float = PEAK_CEILING, knee: float = LIMITER_KNEE) -> np.ndarray:
    """Smoothly compress only the few samples above `knee`, leaving the RMS (and thus energy control) intact."""
    magnitude = np.abs(audio)
    over = magnitude > knee
    if not over.any():
        return audio
    span = ceiling - knee
    limited = audio.copy()
    limited[over] = np.sign(audio[over]) * (knee + span * np.tanh((magnitude[over] - knee) / span))
    return limited


class ProsodyController:
    """Post-synthesis prosody and voice-quality control (DSP only, no neural model).

    Intonation (one PSOLA pass, Praat Manipulation)
      pitch        -- relative change of median F0 in percent (+10 -> F0 x1.10).
      pitch_range  -- multiplier on F0 variation around the median, applied in the log-F0 domain.
      pitch_slope  -- semitones per second added as a linear ramp inside every phrase (centred, so
                      the phrase median is kept). Negative = stronger declination (sad, final,
                      assertive); positive = rising / lifted contours (happy, excited, surprised).
      tremor       -- peak depth in semitones of a 5.5 Hz F0 modulation (shaky, fearful voice).
    Timing
      rate         -- speaking-rate multiplier (PSOLA duration tier; speech *and* pauses scale).
      pause        -- multiplier on internal pause length relative to the speech rate.
    Loudness and voice quality
      energy       -- relative change of speech RMS in percent.
      brightness   -- dB gain of the spectrum above ~1 kHz (spectral tilt): vocal effort.
                      Positive = tense / bright (anger, joy), negative = soft / dark (sadness, calm).

    Every control is isolated: after each stage that changes the level as a side effect, the
    speech RMS is restored, and only the final `energy` stage changes it on purpose. Slopes and
    tremor are generated in output time, so a rate change does not alter their requested size.
    """

    def __init__(self, **params: float):
        unknown = set(params) - set(NEUTRAL)
        if unknown:
            raise TypeError(f"Unknown prosody control(s): {', '.join(sorted(unknown))}.")
        values = {**NEUTRAL, **{k: float(v) for k, v in params.items() if v is not None}}
        for name, value in values.items():
            setattr(self, name, value)
        if self.pitch <= -100.0:
            raise ValueError("pitch must be greater than -100 (percent).")
        if self.energy <= -100.0:
            raise ValueError("energy must be greater than -100 (percent).")
        if self.rate <= 0.0:
            raise ValueError("rate must be positive.")
        if self.pause < 0.0:
            raise ValueError("pause must not be negative.")
        if self.pitch_range <= 0.0:
            raise ValueError("pitch_range must be positive.")
        if self.tremor < 0.0:
            raise ValueError("tremor must not be negative.")

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> "ProsodyController":
        return cls(**{name: params[name] for name in NEUTRAL if name in params})

    @property
    def params(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in NEUTRAL}

    def _changed(self, name: str) -> bool:
        return abs(getattr(self, name) - NEUTRAL[name]) > 1e-8

    @property
    def is_identity(self) -> bool:
        return not any(self._changed(name) for name in NEUTRAL)

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        processed = np.asarray(audio, dtype=np.float32)
        if processed.ndim > 1:
            processed = processed.mean(axis=1)
        if processed.size == 0 or self.is_identity:
            return processed

        changes_f0 = any(self._changed(name) for name in ("pitch", "pitch_range", "pitch_slope", "tremor"))
        if changes_f0 or self._changed("rate") or self._changed("brightness"):
            processed = remove_dc(processed, sample_rate)
        if not changes_f0:
            return self._render(processed, sample_rate)

        # Closed loop on the *final* output: median F0 and intonation range are measured exactly as
        # the evaluation measures them, and the pitch tier is corrected from the error. Every
        # iteration re-renders from the original signal (one PSOLA generation, never cascaded).
        # Needed because the open-loop mapping misses on real voices: range expansion and spectral
        # changes alter which frames a pitch tracker treats as voiced, by up to several percent.
        source = contour_statistics(processed, sample_rate)
        if source["f0_median"] <= 0.0:
            return self._render(processed, sample_rate)
        target_median = source["f0_median"] * (1.0 + self.pitch / 100.0)
        target_spread = source["f0_std_semitones"] * self.pitch_range
        offset, range_gain = 0.0, 1.0
        best, best_error = None, np.inf
        for iteration in range(CONTOUR_ITERATIONS + 1):
            result = self._render(processed, sample_rate, offset, range_gain)
            achieved = contour_statistics(result, sample_rate)
            if achieved["f0_median"] <= 0.0:
                return result if best is None else best
            pitch_error = 12.0 * np.log2(target_median / achieved["f0_median"])
            range_error = target_spread / achieved["f0_std_semitones"] if achieved["f0_std_semitones"] > 1e-6 else 1.0
            # Register first: a range the contour limit cannot reach must not drag the median.
            error = abs(pitch_error) + 1.5 * abs(np.log(range_error))
            if error < best_error:
                best, best_error = result, error
            if abs(pitch_error) < PITCH_TOLERANCE_ST and abs(range_error - 1.0) < RANGE_TOLERANCE:
                break
            if iteration < CONTOUR_ITERATIONS:
                offset += float(np.clip(pitch_error, -2.0, 2.0))
                if source["f0_std_semitones"] > 1e-6:  # also holds the range when only pitch/slope change
                    range_gain = float(np.clip(range_gain * np.clip(range_error, 0.8, 1.25), *RANGE_GAIN_BOUNDS))
        return best

    def _render(self, audio: np.ndarray, sample_rate: int, offset: float = 0.0, range_gain: float = 1.0) -> np.ndarray:
        """The full chain: pauses -> PSOLA (F0 + rate) -> spectral tilt -> energy -> limiter."""
        reference_rms = active_rms(audio)

        def restore_level(signal: np.ndarray) -> np.ndarray:
            if reference_rms <= 0.0:
                return signal
            current = active_rms(signal)
            return signal * (reference_rms / current) if current > 0.0 else signal

        needs_resynthesis = any(self._changed(name) for name in ("pitch", "pitch_range", "pitch_slope", "tremor", "rate"))
        # Spectral balance of the input: the tilt stage restores it (+ the requested change), so
        # pitch shifting cannot alter the measured brightness as a side effect.
        target_alpha = alpha_ratio_db(audio, sample_rate) + self.brightness

        processed = audio
        if self._changed("pause"):
            processed = self._scale_pauses(processed, sample_rate)
        if needs_resynthesis:
            processed = restore_level(self._resynthesize(processed, sample_rate, offset, range_gain))
        if needs_resynthesis or self._changed("brightness"):
            processed = restore_level(self._tilt_to(processed, sample_rate, target_alpha))
        if self._changed("energy"):
            processed = processed * (1.0 + self.energy / 100.0)

        processed = soft_limit(processed.astype(np.float32))
        return np.clip(processed, -PEAK_CEILING, PEAK_CEILING).astype(np.float32)

    # ------------------------------------------------------------------ intonation / rate
    def _resynthesize(self, audio: np.ndarray, sample_rate: int, offset: float = 0.0, range_gain: float = 1.0) -> np.ndarray:
        """One PSOLA pass (Praat overlap-add) for all F0 and timing changes."""
        sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sample_rate))
        floor, ceiling = _speaker_f0_bounds(sound)
        manipulation = call(sound, "To Manipulation", 0.01, floor, ceiling)

        if any(self._changed(name) for name in ("pitch", "pitch_range", "pitch_slope", "tremor")):
            tier = call(manipulation, "Extract pitch tier")
            count = int(call(tier, "Get number of points"))
            if count >= 2:
                times = np.array([call(tier, "Get time from index", i) for i in range(1, count + 1)])
                freqs = np.array([call(tier, "Get value at index", i) for i in range(1, count + 1)])
                semitones = self._target_contour(times, freqs, audio, sample_rate, range_gain) + offset
                shifted = np.clip(float(np.median(freqs)) * np.power(2.0, semitones / 12.0), F0_ABS_MIN, F0_ABS_MAX)
                new_tier = call("Create PitchTier", "prosody", sound.xmin, sound.xmax)
                for time, freq in zip(times, shifted):
                    call(new_tier, "Add point", float(time), float(freq))
                call([new_tier, manipulation], "Replace pitch tier")

        if self._changed("rate"):
            duration_tier = call("Create DurationTier", "rate", sound.xmin, sound.xmax)
            call(duration_tier, "Add point", sound.xmin, 1.0 / self.rate)
            call([duration_tier, manipulation], "Replace duration tier")

        return np.asarray(call(manipulation, "Get resynthesis (overlap-add)").values[0], dtype=np.float32)

    def _target_contour(
        self, times: np.ndarray, freqs: np.ndarray, audio: np.ndarray, sample_rate: int, range_gain: float = 1.0
    ) -> np.ndarray:
        """New F0 contour in semitones relative to the pulse median (input time axis).

        Range, slope and tremor are a shape transform of the contour. The register offset is then
        *solved* rather than assumed: the same transform is applied to the input's F0 analysis
        frames (the frames the median-F0 measurement uses), and the offset is chosen so that the
        predicted frame median moves by exactly the requested percentage. Without this, range
        and slope changes around the pulse median shift the frame median by several percent.
        """
        centre_hz = float(np.median(freqs))
        phrases = []
        if self._changed("pitch_slope"):
            # Phrases as they will exist in the *output*: the duration change divides every gap by
            # the rate, so a closure that becomes >= MIN_PAUSE_SECONDS long after slowing down is a
            # phrase boundary there (and in the measurement), and vice versa.
            for start, end in phrase_intervals(audio, sample_rate, min_pause_seconds=MIN_PAUSE_SECONDS * self.rate):
                inside = (times >= start / sample_rate) & (times < end / sample_rate)
                if inside.sum() >= 2:
                    phrases.append((start / sample_rate, end / sample_rate, float(np.median(times[inside]))))

        def shape(at: np.ndarray, semitones: np.ndarray) -> np.ndarray:
            shaped = self.pitch_range * range_gain * semitones
            # Input-time slope = requested / rate and tremor frequency = TREMOR_HZ / rate, so the
            # *output* (after the duration change) has exactly the requested values.
            for start, end, centre in phrases:
                inside = (at >= start) & (at < end)
                shaped[inside] += (self.pitch_slope / self.rate) * (at[inside] - centre)
            magnitude = np.abs(shaped)
            over = magnitude > CONTOUR_KNEE_ST
            shaped[over] = np.sign(shaped[over]) * (
                CONTOUR_KNEE_ST + CONTOUR_SPAN_ST * np.tanh((magnitude[over] - CONTOUR_KNEE_ST) / CONTOUR_SPAN_ST)
            )
            if self._changed("tremor"):
                shaped = shaped + self.tremor * np.sin(2.0 * np.pi * (TREMOR_HZ / self.rate) * at)
            return shaped

        offset = 12.0 * math.log2(1.0 + self.pitch / 100.0)
        frame_f0, frame_times = extract_f0(audio, sample_rate)
        voiced = ~np.isnan(frame_f0)
        if voiced.sum() >= 5:
            frame_st = 12.0 * np.log2(frame_f0[voiced] / centre_hz)
            current = float(np.median(frame_st))
            predicted = float(np.median(shape(frame_times[voiced], frame_st.copy())))
            offset += current - predicted  # keep the frame median where the pitch control puts it
        return offset + shape(times, 12.0 * np.log2(freqs / centre_hz))

    # ------------------------------------------------------------------ voice quality
    def _tilt_to(self, audio: np.ndarray, sample_rate: int, target_alpha: float) -> np.ndarray:
        """Zero-phase spectral tilt (0 dB below TILT_LOW_HZ, g dB above TILT_HIGH_HZ), with the gain g
        found in closed loop so the alpha ratio of the result equals `target_alpha`.

        A shelf of g dB changes the alpha ratio by slightly less than g (the 1 kHz transition band
        and the voice's own spectrum), so g is refined from the measured error a few times.
        """
        n_fft, hop = 1024, 256
        spectrum = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        ramp = np.clip((freqs - TILT_LOW_HZ) / (TILT_HIGH_HZ - TILT_LOW_HZ), 0.0, 1.0)[:, None]

        def render(gain_db: float) -> np.ndarray:
            shaped = spectrum * np.power(10.0, gain_db * ramp / 20.0)
            return librosa.istft(shaped, hop_length=hop, length=len(audio)).astype(np.float32)

        gain = target_alpha - alpha_ratio_db(audio, sample_rate)
        result = render(gain)
        for _ in range(TILT_ITERATIONS):
            error = target_alpha - alpha_ratio_db(result, sample_rate)
            if abs(error) < 0.05:
                break
            gain += error
            result = render(gain)
        return result

    # ------------------------------------------------------------------ pauses
    def _scale_pauses(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Stretch or shrink silent gaps between speech; leading/trailing silence is kept."""
        intervals = speech_intervals(audio)
        if len(intervals) < 2:
            return audio

        pieces = [audio[: intervals[0][0]]]
        for index, (start, end) in enumerate(intervals):
            pieces.append(audio[start:end])
            if index + 1 < len(intervals):
                pieces.append(self._resize_gap(audio[end : intervals[index + 1][0]], sample_rate))
        pieces.append(audio[intervals[-1][1] :])
        return np.concatenate(pieces)

    def _resize_gap(self, gap: np.ndarray, sample_rate: int) -> np.ndarray:
        if len(gap) / sample_rate < MIN_PAUSE_SECONDS:
            return gap  # articulatory closure, not a pause
        target = int(round(len(gap) * self.pause))
        if target == len(gap):
            return gap
        if target < len(gap):
            head = target // 2
            tail = target - head
            return np.concatenate([gap[:head], gap[len(gap) - tail :]])
        middle = len(gap) // 2
        return np.concatenate([gap[:middle], np.zeros(target - len(gap), dtype=gap.dtype), gap[middle:]])


def _speaker_f0_bounds(sound: "parselmouth.Sound") -> tuple[float, float]:
    """Per-speaker F0 search range (Hirst & De Looze: 0.75*Q1 .. 2.5*Q3) to avoid octave errors."""
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=60.0, pitch_ceiling=600.0)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if f0.size < 10:
        return 75.0, 500.0
    q1, q3 = np.percentile(f0, [25, 75])
    return float(max(F0_ABS_MIN, 0.75 * q1)), float(min(F0_ABS_MAX, 2.5 * q3))


def pitch_semitones(percent: float) -> float:
    """Helper for reports: convert a percent F0 change into semitones."""
    return 12.0 * math.log2(1.0 + percent / 100.0)
