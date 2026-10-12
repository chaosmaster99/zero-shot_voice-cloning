"""Lazy loaders for the pretrained *evaluation* models (speaker encoder and ASR).

They are imported on first use so the API can start, and report a clear error, even when
the optional heavy dependencies are not installed.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from backend.config import get_device

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-base.en")
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

_lock = threading.Lock()
_speaker_encoder: Any = None
_whisper: Any = None


class ModelUnavailable(RuntimeError):
    """Raised when an evaluation model or its dependency cannot be loaded."""


def get_speaker_encoder() -> Any:
    global _speaker_encoder
    with _lock:
        if _speaker_encoder is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ModelUnavailable("speechbrain is not installed (pip install speechbrain).") from exc
            _speaker_encoder = EncoderClassifier.from_hparams(
                source=SPEAKER_MODEL,
                savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"),
                local_strategy=LocalStrategy.COPY,
                run_opts={"device": "cuda:0" if get_device() == "cuda" else "cpu"},
            )
        return _speaker_encoder


def get_whisper() -> Any:
    global _whisper
    with _lock:
        if _whisper is None:
            try:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ModelUnavailable("transformers is not installed (pip install transformers).") from exc
            processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
            model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(get_device())
            model.eval()
            _whisper = (processor, model)
        return _whisper
