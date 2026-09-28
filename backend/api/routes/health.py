from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "controllable-voice-cloning"}


@router.get("/models")
def list_models():
    return {"models": ["f5-tts"], "selected": "f5-tts"}
