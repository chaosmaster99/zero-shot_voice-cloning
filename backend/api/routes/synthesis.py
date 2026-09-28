from __future__ import annotations

import uuid
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.api.schemas.synthesis import SynthesisRequest, SynthesisResponse
from backend.audio.preprocessing import preprocess_reference_audio, save_audio
from backend.config import OUTPUT_DIRECTORY
from backend.core.tts_engine import get_tts_engine
from backend.evaluation.speaker_similarity import calculate_speaker_similarity
from backend.evaluation.wer import compute_wer_from_asr
from backend.evaluation.runtime import evaluate_generated_audio

router = APIRouter()


@router.post("/synthesize", response_model=SynthesisResponse)
async def synthesize(
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(...),
    text: str = Form(...),
    pitch: float = Form(0.0),
    energy: float = Form(0.0),
    rate: float = Form(1.0),
    pause: float = Form(1.0),
    emotion: str = Form("neutral"),
):
    request_id = str(uuid.uuid4())
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        started_at = time.perf_counter()
        source_path = Path(OUTPUT_DIRECTORY) / f"{request_id}_reference{Path(reference_audio.filename or '.wav').suffix}"
        source_path.write_bytes(await reference_audio.read())
        ref_audio, sr = preprocess_reference_audio(source_path)
        baseline_path = Path(OUTPUT_DIRECTORY) / f"{request_id}_baseline.wav"
        controlled_path = Path(OUTPUT_DIRECTORY) / f"{request_id}_controlled.wav"

        engine = get_tts_engine()
        engine.synthesize(source_path, text, baseline_path, reference_text=reference_text)
        from backend.prosody.controller import ProsodyController
        import soundfile as sf

        generated, output_sr = sf.read(str(baseline_path), always_2d=False)
        controlled = ProsodyController(pitch=pitch, energy=energy, rate=rate, pause=pause).apply(generated, output_sr)
        save_audio(controlled_path, controlled, output_sr)

        metrics = evaluate_generated_audio(source_path, controlled_path, text)
        metrics["rtf"] = float((time.perf_counter() - started_at) / max(metrics["duration"], 1e-6))
        return SynthesisResponse(success=True, request_id=request_id, baseline_audio=str(baseline_path), controlled_audio=str(controlled_path), metrics=metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


