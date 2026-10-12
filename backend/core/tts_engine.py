from __future__ import annotations

import importlib.util
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.config import F5_MODEL, F5_NFE_STEPS, MODEL_NAME, SEED, get_device


class TTSUnavailable(RuntimeError):
    """The selected zero-shot TTS backbone cannot be used in this environment."""


class BaseTTSEngine(ABC):
    name: str = "base"

    @abstractmethod
    def availability(self) -> Tuple[bool, str]:
        """(usable, human-readable reason)."""

    @abstractmethod
    def synthesize(
        self,
        reference_audio: str | Path,
        text: str,
        output_path: str | Path,
        *,
        reference_text: str = "",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Write a WAV for `text` in the voice of `reference_audio` and return metadata."""


class F5TTSEngine(BaseTTSEngine):
    """Pretrained F5-TTS (flow-matching, zero-shot) used through its Python API.

    The model is loaded once per process. Nothing is trained or fine-tuned here: the speaker
    is specified only by the reference recording and its transcript at inference time.
    """

    name = "f5-tts"

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()

    def availability(self) -> Tuple[bool, str]:
        if importlib.util.find_spec("f5_tts") is None:
            return False, "f5-tts is not installed (pip install f5-tts)."
        return True, f"{F5_MODEL} on {get_device()}"

    def _load(self) -> Any:
        if self._model is None:
            ok, reason = self.availability()
            if not ok:
                raise TTSUnavailable(reason)
            from backend.utils.audio_io import install_torchaudio_shim

            install_torchaudio_shim()
            from f5_tts.api import F5TTS

            self._model = F5TTS(model=F5_MODEL, device=get_device())
        return self._model

    def synthesize(
        self,
        reference_audio: str | Path,
        text: str,
        output_path: str | Path,
        *,
        reference_text: str = "",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not text.strip():
            raise ValueError("Text cannot be empty.")

        reference_text = (reference_text or "").strip()
        reference_text_source = "user"
        if not reference_text:
            from backend.audio.preprocessing import read_audio
            from backend.evaluation.asr import transcribe

            audio, sample_rate = read_audio(reference_audio)
            reference_text = transcribe(audio, sample_rate)
            reference_text_source = "whisper"
            if not reference_text:
                raise ValueError("Could not transcribe the reference recording; please provide its transcript.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            model = self._load()
            used_seed = SEED if seed is None else int(seed)
            model.infer(
                ref_file=str(reference_audio),
                ref_text=reference_text,
                gen_text=text,
                file_wave=str(output),
                seed=used_seed,
                nfe_step=F5_NFE_STEPS,
                remove_silence=False,
                show_info=lambda *_: None,
            )
        if not output.exists():
            raise RuntimeError("F5-TTS completed without producing the output WAV.")
        return {
            "output_path": str(output),
            "engine": self.name,
            "model": F5_MODEL,
            "seed": used_seed,
            "nfe_steps": F5_NFE_STEPS,
            "reference_text": reference_text,
            "reference_text_source": reference_text_source,
        }


class NullTTSEngine(BaseTTSEngine):
    name = "null"

    def availability(self) -> Tuple[bool, str]:
        return False, "No TTS engine is configured (check MODEL_NAME)."

    def synthesize(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise TTSUnavailable("No TTS engine is configured for this project. Set MODEL_NAME=f5-tts.")


_engines: Dict[str, BaseTTSEngine] = {}


def get_tts_engine(model_name: str = MODEL_NAME) -> BaseTTSEngine:
    key = (model_name or "").lower()
    if key not in _engines:
        _engines[key] = F5TTSEngine() if key in {"f5", "f5-tts", "f5tts"} else NullTTSEngine()
    return _engines[key]
