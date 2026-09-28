from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess
import sys
import os

from backend.config import MODEL_NAME, MODEL_PATH


class BaseTTSEngine(ABC):
    name: str = "base"

    @abstractmethod
    def synthesize(self, reference_audio: str | Path, text: str, output_path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class F5TTSEngine(BaseTTSEngine):
    name = "f5-tts"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or MODEL_PATH or ""

    def synthesize(self, reference_audio: str | Path, text: str, output_path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        ref_text = str(kwargs.get("reference_text", "")).strip()
        if not ref_text:
            raise ValueError("F5-TTS requires the reference transcript. Provide reference_text or configure an ASR transcription step.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "f5_tts.infer.infer_cli",
            "-r", str(reference_audio),
            "-s", ref_text,
            "-t", text,
            "-w", output.name,
            "-o", str(output.parent),
            "--device", kwargs.get("device", "cpu"),
            "--nfe_step", os.getenv("F5_NFE_STEPS", "8"),
            "--remove_silence",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"F5-TTS inference failed: {details}")
        if not output.exists():
            raise RuntimeError("F5-TTS completed without producing the requested output WAV.")
        return {"output_path": str(output), "engine": self.name}


class NullTTSEngine(BaseTTSEngine):
    name = "null"

    def synthesize(self, reference_audio: str | Path, text: str, output_path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("No TTS engine is configured for the active project. Set a valid backend or install the selected model package.")


def get_tts_engine(model_name: str = MODEL_NAME) -> BaseTTSEngine:
    model_name = (model_name or "").lower()
    if model_name in {"f5", "f5-tts", "f5tts"}:
        return F5TTSEngine()
    return NullTTSEngine()
