from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.api.deps import guarded, new_request_id, save_upload
from backend.api.schemas.synthesis import SynthesisResponse
from backend.config import MAX_TEXT_CHARS, OUTPUT_DIRECTORY
from backend.core.pipeline import synthesize_with_prosody

router = APIRouter()
logger = logging.getLogger("voice_cloning")


def build_urls(files: dict) -> dict:
    return {name: f"/audio/{filename}" for name, filename in files.items()}


@router.post("/synthesize", response_model=SynthesisResponse)
def synthesize(
    reference_audio: UploadFile = File(..., description="5-10 second recording of the target speaker."),
    text: str = Form(..., description="Text to synthesize."),
    reference_text: str = Form("", description="Transcript of the reference; transcribed with Whisper when empty."),
    emotion: str = Form("neutral", description="Preset key or 'custom'."),
    pitch: Optional[float] = Form(None),
    pitch_range: Optional[float] = Form(None),
    pitch_slope: Optional[float] = Form(None, description="Semitones/second added per phrase."),
    tremor: Optional[float] = Form(None, description="5.5 Hz F0 tremor depth in semitones."),
    energy: Optional[float] = Form(None),
    brightness: Optional[float] = Form(None, description="Spectral tilt above 1 kHz in dB."),
    rate: Optional[float] = Form(None),
    pause: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"Text is limited to {MAX_TEXT_CHARS} characters.")

    request_id = new_request_id()
    source_path = save_upload(reference_audio, request_id, "reference")

    def work():
        result = synthesize_with_prosody(
            request_id=request_id,
            reference_path=source_path,
            reference_text=reference_text,
            text=text,
            output_dir=OUTPUT_DIRECTORY,
            emotion=emotion,
            overrides={
                "pitch": pitch, "pitch_range": pitch_range, "pitch_slope": pitch_slope, "tremor": tremor,
                "energy": energy, "brightness": brightness, "rate": rate, "pause": pause,
            },
            seed=seed,
        )
        logger.info(
            "synthesis request=%s model=%s emotion=%s prosody=%s ref=%.1fs chars=%d tts=%.2fs prosody_t=%.2fs eval=%.2fs total=%.2fs",
            request_id,
            result["run"]["model"],
            result["emotion"],
            result["prosody"],
            result["reference"]["duration"],
            len(text),
            result["timings"]["tts"],
            result["timings"]["prosody"],
            result["timings"].get("evaluation", 0.0),
            result["timings"]["total"],
        )
        return result

    try:
        result = guarded(request_id, work)
    finally:
        source_path.unlink(missing_ok=True)
    result["urls"] = build_urls(result["files"])
    return result
