from __future__ import annotations

import math
from typing import Any, Dict, Optional

MIN_BASELINE_PAUSE_SECONDS = 0.05


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator is None or denominator <= 1e-9 or numerator is None:
        return None
    return float(numerator) / float(denominator)


def compute_control_accuracy(baseline: Dict[str, Any], controlled: Dict[str, Any], params: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Requested vs measured value per control, both measured with `analyze_prosody`.

    Definitions (matching ProsodyController):
      pitch        % change of median F0.
      pitch_range  ratio of the intonation range (F0 spread in semitones, slopes and tremor removed).
      pitch_slope  added per-phrase F0 slope (semitones/second). The baseline's own declination is
                   also scaled by the range and rate changes, so the added part is
                   slope_after - range_ratio * rate_ratio * slope_before.
      tremor       added 5.5 Hz F0 modulation depth (semitones). The baseline's incoherent
                   micro-modulation (also scaled by the range) is removed in quadrature.
      energy       % change of speech RMS.
      brightness   change of the alpha ratio (dB): 1-5 kHz vs 50 Hz-1 kHz speech energy.
      rate         ratio of speech durations (baseline / controlled).
      pause        ratio of total internal pause time, divided by the speech-rate change so
                   that the pause control is reported independently of rate.
    Controls whose baseline statistic is missing are skipped (older metric dicts).
    """
    report: Dict[str, Dict[str, Any]] = {}

    f0_ratio = _ratio(controlled.get("f0_median"), baseline.get("f0_median"))
    measured_pitch = None if f0_ratio is None else 100.0 * (f0_ratio - 1.0)
    report["pitch"] = _entry(params.get("pitch", 0.0), measured_pitch, "%")

    range_ratio = _ratio(controlled.get("f0_std_semitones"), baseline.get("f0_std_semitones"))
    report["pitch_range"] = _entry(params.get("pitch_range", 1.0), range_ratio, "x")

    measured_rate = _ratio(baseline.get("speech_duration"), controlled.get("speech_duration"))

    if "f0_slope" in baseline and "f0_slope" in controlled:
        added_slope = None
        if range_ratio is not None and measured_rate is not None:
            added_slope = controlled["f0_slope"] - range_ratio * measured_rate * baseline["f0_slope"]
        report["pitch_slope"] = _entry(params.get("pitch_slope", 0.0), added_slope, "st/s")

    if "f0_tremor" in baseline and "f0_tremor" in controlled:
        scaled_base = (range_ratio or 1.0) * baseline["f0_tremor"]
        added_tremor = math.sqrt(max(controlled["f0_tremor"] ** 2 - scaled_base**2, 0.0))
        report["tremor"] = _entry(params.get("tremor", 0.0), added_tremor, "st")

    energy_ratio = _ratio(controlled.get("energy_mean"), baseline.get("energy_mean"))
    measured_energy = None if energy_ratio is None else 100.0 * (energy_ratio - 1.0)
    report["energy"] = _entry(params.get("energy", 0.0), measured_energy, "%")

    if "alpha_ratio_db" in baseline and "alpha_ratio_db" in controlled:
        report["brightness"] = _entry(
            params.get("brightness", 0.0), controlled["alpha_ratio_db"] - baseline["alpha_ratio_db"], "dB"
        )

    report["rate"] = _entry(params.get("rate", 1.0), measured_rate, "x")

    measured_pause = None
    if (baseline.get("pause_total") or 0.0) > MIN_BASELINE_PAUSE_SECONDS:
        pause_ratio = _ratio(controlled.get("pause_total", 0.0), baseline.get("pause_total"))
        if pause_ratio is not None and measured_rate:
            measured_pause = pause_ratio * measured_rate
    report["pause"] = _entry(params.get("pause", 1.0), measured_pause, "x")
    return report


def _entry(requested: float, measured: Optional[float], unit: str) -> Dict[str, Any]:
    error = None if measured is None else float(measured) - float(requested)
    return {
        "requested": float(requested),
        "measured": None if measured is None else float(measured),
        "abs_error": None if error is None else abs(error),
        "unit": unit,
    }
