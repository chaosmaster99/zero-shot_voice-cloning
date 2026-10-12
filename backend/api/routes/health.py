from __future__ import annotations

from fastapi import APIRouter

from backend.config import F5_MODEL, MODEL_NAME, SAMPLE_RATE, get_device
from backend.core.tts_engine import get_tts_engine
from backend.evaluation.models import SPEAKER_MODEL, WHISPER_MODEL
from backend.prosody.presets import LIMITS, list_presets
from backend.utils.system_info import system_info

router = APIRouter()


@router.get("/health")
def health_check():
    usable, reason = get_tts_engine().availability()
    return {
        "status": "ok",
        "service": "controllable-voice-cloning",
        "device": get_device(),
        "tts_available": usable,
        "tts_detail": reason,
    }


@router.get("/models")
def list_models():
    usable, reason = get_tts_engine().availability()
    return {
        "selected": MODEL_NAME,
        "tts": {"name": MODEL_NAME, "checkpoint": F5_MODEL, "available": usable, "detail": reason, "sample_rate": SAMPLE_RATE},
        "evaluation": {"speaker_encoder": SPEAKER_MODEL, "asr": WHISPER_MODEL},
        "pretrained_only": True,
        "system": system_info(),
    }


@router.get("/presets")
def presets():
    return {"presets": list_presets(), "limits": {name: {"min": low, "max": high} for name, (low, high) in LIMITS.items()}}
