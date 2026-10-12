from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _configure_ffmpeg() -> None:
    """Make FFmpeg shared DLLs available to audio libraries on Windows."""
    ffmpeg_bin = os.getenv("FFMPEG_BIN")
    candidates = [Path(ffmpeg_bin)] if ffmpeg_bin else []
    ffmpeg_executable = shutil.which("ffmpeg")
    if ffmpeg_executable:
        candidates.append(Path(ffmpeg_executable).parent)
    candidates.extend(Path(path) for path in os.getenv("PATH", "").split(os.pathsep) if path)
    candidates.extend(Path(path) for path in Path(os.getenv("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages").glob("**/bin"))

    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "ffmpeg.exe").exists():
            continue
        os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(candidate))
        return


_configure_ffmpeg()

DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
EVALUATION_DIR = DATA_DIR / "evaluation"
REFERENCES_DIR = DATA_DIR / "references"
RESULTS_DIR = PROJECT_ROOT / "results"

DEVICE = os.getenv("DEVICE", "auto")  # auto | cuda | cpu
MODEL_NAME = os.getenv("MODEL_NAME", "f5-tts")
F5_MODEL = os.getenv("F5_MODEL", "F5TTS_v1_Base")
F5_NFE_STEPS = int(os.getenv("F5_NFE_STEPS", "32"))
SEED = int(os.getenv("SEED", "42"))
# Sentence-level synthesis: each sentence is generated separately and joined with an explicit gap,
# which gives the pause control a well-defined handle (and a neutral baseline pause of this length).
SPLIT_SENTENCES = os.getenv("SPLIT_SENTENCES", "true").lower() == "true"
SENTENCE_PAUSE_SECONDS = float(os.getenv("SENTENCE_PAUSE_SECONDS", "0.35"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))  # F5-TTS / Vocos native rate
MIN_REFERENCE_SECONDS = float(os.getenv("MIN_REFERENCE_SECONDS", "5.0"))
MAX_REFERENCE_SECONDS = float(os.getenv("MAX_REFERENCE_SECONDS", "10.0"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "600"))
OUTPUT_DIRECTORY = Path(os.getenv("OUTPUT_DIRECTORY", str(GENERATED_DIR)))
ENABLE_EVALUATION = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]

for _directory in (OUTPUT_DIRECTORY, REFERENCES_DIR, EVALUATION_DIR, RESULTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

DEFAULT_PROSODY = {
    "pitch": 0.0,
    "pitch_range": 1.0,
    "energy": 0.0,
    "rate": 1.0,
    "pause": 1.0,
}


def get_device() -> Literal["cpu", "cuda"]:
    preference = DEVICE.lower()
    if preference in {"cpu", "cuda"}:
        return preference  # type: ignore[return-value]
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
