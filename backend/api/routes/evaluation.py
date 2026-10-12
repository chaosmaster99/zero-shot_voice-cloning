from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from backend.api.deps import guarded, new_request_id, save_upload
from backend.audio.preprocessing import read_audio
from backend.evaluation.prosody_metrics import analyze_prosody
from backend.evaluation.runtime import evaluate_clips

router = APIRouter()


@router.post("/evaluate")
def evaluate(
    reference_audio: UploadFile = File(...),
    generated_audio: UploadFile = File(...),
    target_text: str = Form(""),
):
    """Speaker similarity, WER (when target_text is given) and prosody of an arbitrary generated clip."""
    request_id = new_request_id()
    reference_path = save_upload(reference_audio, request_id, "ref")
    generated_path = save_upload(generated_audio, request_id, "gen")

    def work():
        reference = read_audio(reference_path)
        generated = read_audio(generated_path)
        text = target_text.strip() or None
        evaluation = evaluate_clips(reference, {"generated": generated}, text, include_contours=False, with_asr=bool(text))
        clip = evaluation["clips"]["generated"]
        return {
            "success": True,
            "speaker_similarity": clip.get("speaker_similarity"),
            "wer": clip.get("wer"),
            "transcript": clip.get("transcript"),
            "prosody": clip["prosody"],
            "reference_prosody": evaluation["reference"]["prosody"],
            "warnings": evaluation["warnings"],
        }

    try:
        return guarded(request_id, work)
    finally:
        reference_path.unlink(missing_ok=True)
        generated_path.unlink(missing_ok=True)


@router.post("/analyze")
def analyze(audio: UploadFile = File(...)):
    """Prosody statistics and contours of a single recording."""
    request_id = new_request_id()
    path = save_upload(audio, request_id, "analyze")

    def work():
        samples, sample_rate = read_audio(path)
        if samples.size == 0:
            raise ValueError("Audio is empty.")
        return {"success": True, "prosody": analyze_prosody(samples, sample_rate, include_contours=True)}

    try:
        return guarded(request_id, work)
    finally:
        path.unlink(missing_ok=True)
