from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from backend.api.deps import guarded, new_request_id, save_upload
from backend.api.routes.synthesis import build_urls
from backend.api.schemas.synthesis import SynthesisResponse
from backend.config import OUTPUT_DIRECTORY
from backend.core.pipeline import apply_prosody_only

router = APIRouter(prefix="/prosody")


@router.post("/apply", response_model=SynthesisResponse)
def apply_prosody(
    audio: UploadFile = File(..., description="Any speech recording; prosody is applied directly, without TTS."),
    emotion: str = Form("neutral"),
    pitch: Optional[float] = Form(None),
    pitch_range: Optional[float] = Form(None),
    pitch_slope: Optional[float] = Form(None, description="Semitones/second added per phrase."),
    tremor: Optional[float] = Form(None, description="5.5 Hz F0 tremor depth in semitones."),
    energy: Optional[float] = Form(None),
    brightness: Optional[float] = Form(None, description="Spectral tilt above 1 kHz in dB."),
    rate: Optional[float] = Form(None),
    pause: Optional[float] = Form(None),
    target_text: str = Form("", description="Optional transcript for WER; leave empty to skip ASR."),
):
    request_id = new_request_id()
    source_path = save_upload(audio, request_id, "lab")
    try:
        result = guarded(
            request_id,
            lambda: apply_prosody_only(
                request_id=request_id,
                audio_path=source_path,
                output_dir=OUTPUT_DIRECTORY,
                emotion=emotion,
                overrides={
                "pitch": pitch, "pitch_range": pitch_range, "pitch_slope": pitch_slope, "tremor": tremor,
                "energy": energy, "brightness": brightness, "rate": rate, "pause": pause,
            },
                target_text=target_text.strip() or None,
            ),
        )
    finally:
        source_path.unlink(missing_ok=True)
    result["urls"] = build_urls(result["files"])
    return result
