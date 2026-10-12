from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Keep API-test output out of data/generated (must be set before backend.config is imported).
os.environ.setdefault("OUTPUT_DIRECTORY", tempfile.mkdtemp(prefix="vc_test_out_"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SR = 24000


def synth_speechlike(
    phrases: int = 3,
    phrase_seconds: float = 1.0,
    pause_seconds: float = 0.5,
    f0: float = 130.0,
    vibrato: float = 0.12,
    amplitude: float = 0.15,
    sample_rate: int = SR,
    lead: float = 0.2,
) -> np.ndarray:
    """Harmonic, amplitude-modulated phrases separated by silences, with a slowly varying F0.

    A deterministic stand-in for speech used to test signal-level behaviour with known ground truth.
    """
    rng = np.random.default_rng(0)

    def silence(seconds: float) -> np.ndarray:
        return rng.normal(0.0, 0.0005, int(seconds * sample_rate)).astype(np.float32)

    def phrase(seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * sample_rate)) / sample_rate
        inst_f0 = f0 * (1.0 + vibrato * np.sin(2 * np.pi * 0.8 * t))
        phase = 2 * np.pi * np.cumsum(inst_f0) / sample_rate
        wave = sum(np.sin(k * phase) / k for k in range(1, 12))
        wave = wave / np.max(np.abs(wave)) * amplitude
        envelope = np.sqrt(0.5 * (1 + np.sin(2 * np.pi * 4 * t)))
        fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.03)
        return (wave * envelope * fade).astype(np.float32)

    parts = [silence(lead)]
    for index in range(phrases):
        parts.append(phrase(phrase_seconds))
        parts.append(silence(pause_seconds if index < phrases - 1 else lead))
    return np.concatenate(parts)


@pytest.fixture(scope="session")
def sample_rate() -> int:
    return SR


@pytest.fixture(scope="session")
def speechlike() -> np.ndarray:
    return synth_speechlike()
