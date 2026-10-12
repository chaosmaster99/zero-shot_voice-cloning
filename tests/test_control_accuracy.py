from __future__ import annotations

import pytest

from backend.evaluation.control_accuracy import compute_control_accuracy
from backend.evaluation.prosody_metrics import compute_prosody_metrics
from backend.prosody.controller import ProsodyController

SR = 24000


def test_identity_reports_zero_error(speechlike):
    metrics = compute_prosody_metrics(speechlike, SR)
    report = compute_control_accuracy(metrics, metrics, {"pitch": 0, "pitch_range": 1, "energy": 0, "rate": 1, "pause": 1})
    for control, row in report.items():
        assert row["measured"] is not None, control
        assert row["abs_error"] < 1e-9, control


def test_report_matches_requested_values_for_combined_controls(speechlike):
    params = {"pitch": 10.0, "pitch_range": 1.0, "energy": 20.0, "rate": 1.2, "pause": 1.5}
    base = compute_prosody_metrics(speechlike, SR)
    out = compute_prosody_metrics(ProsodyController.from_params(params).apply(speechlike, SR), SR)
    report = compute_control_accuracy(base, out, params)
    assert report["pitch"]["abs_error"] < 4.0  # percentage points
    assert report["energy"]["abs_error"] < 4.0
    assert report["rate"]["abs_error"] < 0.07
    assert report["pause"]["abs_error"] < 0.2


def test_missing_baseline_pauses_are_reported_as_unmeasurable():
    base = {"f0_median": 120.0, "f0_std_semitones": 1.0, "energy_mean": 0.1, "speech_duration": 2.0, "pause_total": 0.0}
    out = dict(base)
    report = compute_control_accuracy(base, out, {"pause": 2.0})
    assert report["pause"]["measured"] is None
    assert report["pause"]["abs_error"] is None


def test_error_is_signed_difference_magnitude():
    base = {"f0_median": 100.0, "f0_std_semitones": 1.0, "energy_mean": 0.1, "speech_duration": 2.0, "pause_total": 1.0}
    out = {"f0_median": 118.0, "f0_std_semitones": 1.0, "energy_mean": 0.1, "speech_duration": 2.0, "pause_total": 1.0}
    row = compute_control_accuracy(base, out, {"pitch": 20.0})["pitch"]
    assert row["measured"] == pytest.approx(18.0)
    assert row["abs_error"] == pytest.approx(2.0)
