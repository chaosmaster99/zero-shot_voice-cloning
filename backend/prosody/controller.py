from __future__ import annotations

import numpy as np
import librosa


class ProsodyController:
    """Apply deterministic, signal-based prosody changes to a generated waveform."""

    def __init__(self, pitch: float = 0.0, energy: float = 0.0, rate: float = 1.0, pause: float = 1.0):
        self.pitch = float(pitch)
        self.energy = float(energy)
        self.rate = float(rate)
        self.pause = float(pause)

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.size == 0:
            return audio

        processed = np.asarray(audio, dtype=np.float32)

        if abs(self.pitch) > 1e-8:
            try:
                processed = librosa.effects.pitch_shift(processed, sr=sample_rate, n_steps=self.pitch)
            except Exception:
                processed = processed

        if abs(self.energy) > 1e-8:
            gain = 10 ** (self.energy / 20.0)
            processed = processed * gain

        if abs(self.rate - 1.0) > 1e-8 and self.rate > 0:
            processed = librosa.effects.time_stretch(processed, rate=self.rate)

        if abs(self.pause - 1.0) > 1e-8 and self.pause > 0:
            pause_adjust = max(0.0, self.pause - 1.0)
            if pause_adjust > 0:
                silence_length = int(len(processed) * min(0.2, pause_adjust * 0.08))
                if silence_length > 0:
                    silence = np.zeros(silence_length, dtype=np.float32)
                    processed = np.concatenate([processed, silence])

        processed = np.clip(processed, -1.0, 1.0)
        return processed.astype(np.float32)
