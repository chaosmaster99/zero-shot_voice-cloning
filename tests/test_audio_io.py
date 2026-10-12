from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from backend.utils.audio_io import install_torchaudio_shim, soundfile_load  # noqa: E402


def test_soundfile_load_matches_torchaudio_contract(tmp_path):
    path = tmp_path / "tone.wav"
    samples = (0.3 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)).astype(np.float32)
    sf.write(path, samples, 16000)

    waveform, sample_rate = soundfile_load(path)
    assert sample_rate == 16000
    assert waveform.shape == (1, 8000) and waveform.dtype == torch.float32
    assert np.allclose(waveform[0].numpy(), samples, atol=1e-3)

    stereo = np.stack([samples, samples], axis=1)
    sf.write(path, stereo, 16000)
    assert soundfile_load(path)[0].shape == (2, 8000)
    assert soundfile_load(path, channels_first=False)[0].shape == (8000, 2)


def test_install_is_idempotent():
    import torchaudio

    install_torchaudio_shim()
    first = torchaudio.load
    install_torchaudio_shim()
    assert torchaudio.load is first
