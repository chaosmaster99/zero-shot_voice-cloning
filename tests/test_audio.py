from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from backend.audio.preprocessing import preprocess_reference_audio, validate_reference_audio
from tests.conftest import synth_speechlike


def test_short_reference_is_rejected(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(8000, dtype=np.float32), 22050)
    with pytest.raises(ValueError, match="at least"):
        validate_reference_audio(path, min_seconds=5.0, max_seconds=10.0)


def test_long_reference_is_rejected(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(22050 * 12, dtype=np.float32), 22050)
    with pytest.raises(ValueError, match="no longer than"):
        validate_reference_audio(path, min_seconds=5.0, max_seconds=10.0)


def test_empty_audio_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype=np.float32), 22050)
    with pytest.raises(ValueError):
        validate_reference_audio(path)


def test_corrupt_file_is_a_clean_value_error(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"this is not audio")
    with pytest.raises(ValueError, match="corrupt"):
        validate_reference_audio(path)


def test_silent_reference_is_rejected(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(24000 * 6, dtype=np.float32), 24000)
    with pytest.raises(ValueError, match="No speech"):
        preprocess_reference_audio(path, target_sr=24000)


def test_preprocess_resamples_normalizes_and_trims(tmp_path):
    speech = synth_speechlike(phrases=2, phrase_seconds=2.5, pause_seconds=0.3, lead=0.8, sample_rate=16000)
    stereo = np.stack([speech, speech], axis=1)
    path = tmp_path / "ref.wav"
    sf.write(path, stereo, 16000)

    audio, sr, info = preprocess_reference_audio(path, target_sr=24000, min_seconds=5.0, max_seconds=10.0)

    assert sr == 24000
    assert audio.ndim == 1
    assert abs(float(np.max(np.abs(audio))) - 0.95) < 0.01
    assert info.duration < info.raw_duration  # leading/trailing silence removed
    assert info.duration > 4.5
