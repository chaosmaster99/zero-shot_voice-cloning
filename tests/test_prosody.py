from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation.prosody_metrics import compute_prosody_metrics
from backend.prosody.controller import ProsodyController

SR = 24000


@pytest.fixture(scope="module")
def baseline(speechlike) -> dict:
    return compute_prosody_metrics(speechlike, SR)


def _relative(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def test_baseline_measurements_are_sane(speechlike, baseline):
    assert 120 < baseline["f0_median"] < 140
    assert baseline["pause_count"] == 2
    assert 0.9 < baseline["pause_total"] < 1.1
    assert baseline["f0_std_semitones"] > 0.3


def test_identity_is_noop(speechlike):
    out = ProsodyController().apply(speechlike, SR)
    assert np.array_equal(out, speechlike)


@pytest.mark.parametrize("percent", [-20.0, -10.0, 10.0, 20.0])
def test_pitch_percent_matches_measured_f0(speechlike, baseline, percent):
    out = ProsodyController(pitch=percent).apply(speechlike, SR)
    measured = _relative(compute_prosody_metrics(out, SR)["f0_median"], baseline["f0_median"])
    assert abs(measured - percent) < 3.0
    assert abs(len(out) - len(speechlike)) < SR * 0.05


@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_pitch_range_scales_f0_variation_and_keeps_median(speechlike, baseline, factor):
    out = ProsodyController(pitch_range=factor).apply(speechlike, SR)
    measured = compute_prosody_metrics(out, SR)
    assert abs(measured["f0_std_semitones"] / baseline["f0_std_semitones"] - factor) < 0.2
    assert abs(_relative(measured["f0_median"], baseline["f0_median"])) < 3.0


@pytest.mark.parametrize("percent", [-20.0, 20.0])
def test_energy_percent_matches_measured_rms(speechlike, baseline, percent):
    out = ProsodyController(energy=percent).apply(speechlike, SR)
    measured = _relative(compute_prosody_metrics(out, SR)["energy_mean"], baseline["energy_mean"])
    assert abs(measured - percent) < 3.0
    assert np.max(np.abs(out)) <= 1.0


@pytest.mark.parametrize("rate", [0.8, 1.2])
def test_rate_scales_speech_duration(speechlike, baseline, rate):
    out = ProsodyController(rate=rate).apply(speechlike, SR)
    measured = compute_prosody_metrics(out, SR)
    assert abs(baseline["speech_duration"] / measured["speech_duration"] - rate) < 0.05


def test_controls_do_not_leak_into_each_other(speechlike, baseline):
    pitched = compute_prosody_metrics(ProsodyController(pitch=10).apply(speechlike, SR), SR)
    assert abs(_relative(pitched["energy_mean"], baseline["energy_mean"])) < 5.0
    assert abs(pitched["speech_duration"] / baseline["speech_duration"] - 1.0) < 0.03

    stretched = compute_prosody_metrics(ProsodyController(rate=1.2).apply(speechlike, SR), SR)
    assert abs(_relative(stretched["f0_median"], baseline["f0_median"])) < 3.0
    assert abs(_relative(stretched["energy_mean"], baseline["energy_mean"])) < 5.0

    louder = compute_prosody_metrics(ProsodyController(energy=20).apply(speechlike, SR), SR)
    assert abs(_relative(louder["f0_median"], baseline["f0_median"])) < 1.0


@pytest.mark.parametrize("multiplier", [0.5, 1.5, 2.0])
def test_pause_multiplier_scales_internal_pauses(speechlike, baseline, multiplier):
    out = ProsodyController(pause=multiplier).apply(speechlike, SR)
    measured = compute_prosody_metrics(out, SR)
    assert measured["pause_count"] == baseline["pause_count"]
    assert abs(measured["pause_total"] / baseline["pause_total"] - multiplier) < 0.1
    assert abs(measured["speech_duration"] - baseline["speech_duration"]) < 0.05


def test_pause_leaves_leading_and_trailing_silence(speechlike):
    out = ProsodyController(pause=2.0).apply(speechlike, SR)
    lead = int(0.2 * SR)
    assert np.array_equal(out[:lead], speechlike[:lead])
    assert np.array_equal(out[-lead:], speechlike[-lead:])


def test_combined_controls_are_each_close(speechlike, baseline):
    controller = ProsodyController(pitch=15, energy=10, rate=1.1, pause=1.5, pitch_range=1.2)
    measured = compute_prosody_metrics(controller.apply(speechlike, SR), SR)
    assert abs(_relative(measured["f0_median"], baseline["f0_median"]) - 15) < 4.0
    assert abs(_relative(measured["energy_mean"], baseline["energy_mean"]) - 10) < 4.0
    assert abs(baseline["speech_duration"] / measured["speech_duration"] - 1.1) < 0.06


def test_energy_never_clips(speechlike):
    out = ProsodyController(energy=80).apply(speechlike * 5.0, SR)
    assert np.max(np.abs(out)) <= 1.0


def test_stereo_input_is_downmixed(speechlike):
    stereo = np.stack([speechlike, speechlike], axis=1)
    out = ProsodyController(energy=10).apply(stereo, SR)
    assert out.ndim == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"pitch": -100}, {"energy": -100}, {"rate": 0}, {"pause": -1}, {"pitch_range": 0}],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        ProsodyController(**kwargs)


def test_dc_offset_with_digital_silence_does_not_break_pause_structure(speechlike):
    """Regression: recordings with a DC offset and exact-zero silences (common with TTS/SAPI output).

    Praat subtracts the mean during resynthesis, which used to turn the zero silences into a constant
    floor, destroying pause detection and the energy/rate measurements.
    """
    from backend.audio.analysis import speech_intervals

    signal = speechlike.copy()
    mask = np.zeros(len(signal), dtype=bool)
    for start, end in speech_intervals(signal):
        mask[start:end] = True
    signal[~mask] = 0.0
    signal[mask] += 0.02

    base = compute_prosody_metrics(signal, SR)
    out = compute_prosody_metrics(ProsodyController(pitch=10).apply(signal, SR), SR)
    assert out["pause_count"] == base["pause_count"] == 2
    assert abs(out["speech_duration"] / base["speech_duration"] - 1.0) < 0.03
    assert abs(_relative(out["energy_mean"], base["energy_mean"])) < 5.0
    assert abs(_relative(out["f0_median"], base["f0_median"]) - 10.0) < 3.0


# ----------------------------------------------------------------------------- contour, tremor, tilt
@pytest.mark.parametrize("slope", [-3.0, 2.0])
def test_pitch_slope_adds_requested_semitones_per_second(speechlike, baseline, slope):
    measured = compute_prosody_metrics(ProsodyController(pitch_slope=slope).apply(speechlike, SR), SR)
    assert abs((measured["f0_slope"] - baseline["f0_slope"]) - slope) < 0.3
    assert abs(_relative(measured["f0_median"], baseline["f0_median"])) < 4.0  # ramp is centred per phrase
    assert abs(measured["f0_std_semitones"] / baseline["f0_std_semitones"] - 1.0) < 0.1  # range is not affected


def test_slope_is_specified_in_output_time_under_rate_change(speechlike, baseline):
    from backend.evaluation.control_accuracy import compute_control_accuracy

    params = {"pitch_slope": -2.0, "rate": 1.25}
    measured = compute_prosody_metrics(ProsodyController(**params).apply(speechlike, SR), SR)
    report = compute_control_accuracy(baseline, measured, params)
    assert report["pitch_slope"]["abs_error"] < 0.4


def test_tremor_depth_is_measured_by_lock_in(speechlike, baseline):
    assert baseline["f0_tremor"] < 0.1  # no coherent 5.5 Hz component in the source
    measured = compute_prosody_metrics(ProsodyController(tremor=0.6).apply(speechlike, SR), SR)
    assert abs(measured["f0_tremor"] - 0.6) < 0.1
    assert abs(_relative(measured["f0_median"], baseline["f0_median"])) < 2.0
    assert abs(measured["f0_std_semitones"] / baseline["f0_std_semitones"] - 1.0) < 0.1


@pytest.mark.parametrize("db", [-6.0, 6.0])
def test_brightness_shifts_alpha_ratio_and_keeps_level(speechlike, baseline, db):
    measured = compute_prosody_metrics(ProsodyController(brightness=db).apply(speechlike, SR), SR)
    assert abs((measured["alpha_ratio_db"] - baseline["alpha_ratio_db"]) - db) < 1.0
    assert abs(_relative(measured["energy_mean"], baseline["energy_mean"])) < 2.0
    assert abs(_relative(measured["f0_median"], baseline["f0_median"])) < 1.0


@pytest.mark.parametrize("preset", ["happy", "sad", "angry", "calm", "excited", "afraid"])
def test_every_preset_is_reproduced_within_tolerance(speechlike, baseline, preset):
    from backend.evaluation.control_accuracy import compute_control_accuracy
    from backend.prosody.presets import PRESETS

    params = PRESETS[preset]["params"]
    out = ProsodyController.from_params(params).apply(speechlike, SR)
    report = compute_control_accuracy(baseline, compute_prosody_metrics(out, SR), params)
    tolerance = {"pitch": 4.0, "pitch_range": 0.15, "pitch_slope": 0.6, "tremor": 0.15, "energy": 4.0,
                 "brightness": 1.0, "rate": 0.06, "pause": 0.2}
    for control, row in report.items():
        assert row["abs_error"] is not None and row["abs_error"] < tolerance[control], (preset, control, row)
    assert np.max(np.abs(out)) <= 1.0
