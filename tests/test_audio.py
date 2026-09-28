from __future__ import annotations

import numpy as np

from backend.audio.preprocessing import validate_reference_audio


def test_valid_reference_audio_raises_for_short_file(tmp_path):
    wave_path = tmp_path / "short.wav"
    audio = np.zeros(8000, dtype=np.float32)
    import soundfile as sf
    sf.write(wave_path, audio, 22050)

    try:
        validate_reference_audio(wave_path, min_seconds=5.0, max_seconds=10.0)
        assert False, "Expected validation to fail for too-short audio"
    except ValueError:
        pass


def test_empty_audio_raises_error(tmp_path):
    wave_path = tmp_path / "empty.wav"
    import soundfile as sf
    sf.write(wave_path, np.zeros(0, dtype=np.float32), 22050)

    try:
        validate_reference_audio(wave_path)
        assert False, "Expected empty audio to fail validation"
    except ValueError:
        pass
