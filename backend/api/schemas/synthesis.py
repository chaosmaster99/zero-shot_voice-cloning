from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SynthesisRequest(BaseModel):
    reference_audio: str = Field(..., description="Path to the 5–10 second reference audio file.")
    reference_text: str = Field(..., min_length=1, description="Transcript of the reference audio for F5-TTS conditioning.")
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    pitch: float = Field(default=0.0, description="Pitch adjustment percentage.")
    energy: float = Field(default=0.0, description="Energy adjustment percentage.")
    rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Speaking rate multiplier.")
    pause: float = Field(default=1.0, ge=0.5, le=2.0, description="Pause multiplier.")
    emotion: str = Field(default="neutral", description="Optional emotion preset.")


class SynthesisResponse(BaseModel):
    success: bool
    request_id: str
    baseline_audio: Optional[str] = None
    controlled_audio: Optional[str] = None
    metrics: dict
    error: Optional[str] = None
