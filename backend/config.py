from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
EVALUATION_DIR = DATA_DIR / "evaluation"
REFERENCES_DIR = DATA_DIR / "references"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

DEVICE = os.getenv("DEVICE", "cpu")
MODEL_NAME = os.getenv("MODEL_NAME", "f5-tts")
MODEL_PATH = os.getenv("MODEL_PATH", "")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "22050"))
MIN_REFERENCE_SECONDS = float(os.getenv("MIN_REFERENCE_SECONDS", "5.0"))
MAX_REFERENCE_SECONDS = float(os.getenv("MAX_REFERENCE_SECONDS", "10.0"))
OUTPUT_DIRECTORY = Path(os.getenv("OUTPUT_DIRECTORY", str(GENERATED_DIR)))
ENABLE_GPU = os.getenv("ENABLE_GPU", "false").lower() == "true"
ENABLE_EVALUATION = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROSODY = {
    "pitch": 0.0,
    "energy": 0.0,
    "rate": 1.0,
    "pause": 1.0,
}


def get_device() -> Literal["cpu", "cuda"]:
    return "cuda" if DEVICE.lower() == "cuda" else "cpu"
