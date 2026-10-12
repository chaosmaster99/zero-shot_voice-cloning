from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProsodyParams(BaseModel):
    pitch: float = Field(0.0, description="Median F0 change in percent.")
    pitch_range: float = Field(1.0, description="Intonation range multiplier.")
    pitch_slope: float = Field(0.0, description="Added per-phrase F0 slope, semitones/second.")
    tremor: float = Field(0.0, description="5.5 Hz F0 tremor depth, semitones.")
    energy: float = Field(0.0, description="Speech RMS change in percent.")
    brightness: float = Field(0.0, description="Spectral tilt above 1 kHz, dB.")
    rate: float = Field(1.0, description="Speaking-rate multiplier.")
    pause: float = Field(1.0, description="Pause-length multiplier.")


class SynthesisResponse(BaseModel):
    success: bool = True
    request_id: str
    emotion: str
    prosody: ProsodyParams
    text: Optional[str] = None
    urls: Dict[str, str]
    reference: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    control_accuracy: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    analysis: Dict[str, Any] = Field(default_factory=dict)
    timings: Dict[str, float] = Field(default_factory=dict)
    run: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    notice: str = "Generated speech is synthetic and must not be used for unauthorized impersonation or deception."
