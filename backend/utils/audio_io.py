"""torchaudio >= 2.9 delegates audio I/O to torchcodec, which needs FFmpeg shared libraries that
are not available on a default Windows install. F5-TTS only needs `torchaudio.load` to read the
(already preprocessed, WAV) reference, so it is served by libsndfile instead."""

from __future__ import annotations

from typing import Any, Tuple


def soundfile_load(
    uri: Any,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    **_: Any,
) -> Tuple[Any, int]:
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(str(uri), dtype="float32", always_2d=True, start=frame_offset, frames=num_frames)
    tensor = torch.from_numpy(data)
    if channels_first:
        tensor = tensor.T
    return tensor.contiguous(), int(sample_rate)


def install_torchaudio_shim() -> None:
    """Idempotently route torchaudio.load through soundfile (process-local)."""
    import torchaudio

    if getattr(torchaudio.load, "_is_soundfile_shim", False):
        return
    soundfile_load._is_soundfile_shim = True  # type: ignore[attr-defined]
    torchaudio.load = soundfile_load  # type: ignore[assignment]
