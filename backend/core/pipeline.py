"""End-to-end flows shared by the API and the experiment scripts."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np

from backend.audio.preprocessing import preprocess_reference_audio, read_audio, save_audio
from backend.audio.visualization import mel_image, waveform_envelope
from backend.config import ENABLE_EVALUATION, F5_MODEL, MODEL_NAME, SAMPLE_RATE, SENTENCE_PAUSE_SECONDS, SPLIT_SENTENCES
from backend.core.text import split_sentences
from backend.core.tts_engine import get_tts_engine
from backend.evaluation.control_accuracy import compute_control_accuracy
from backend.evaluation.runtime import evaluate_clips
from backend.prosody.controller import ProsodyController
from backend.prosody.presets import resolve_prosody
from backend.utils.system_info import system_info


EDGE_PAD_SECONDS = 0.05
# Baselines peak at -6 dBFS so the prosody controller has headroom for louder emotions
# (+40 % energy, brighter spectra) without hitting the limiter.
BASELINE_PEAK = 0.5


def synthesize_baseline(
    engine: Any,
    reference_path: Path,
    text: str,
    output_path: Path,
    *,
    reference_text: str = "",
    seed: Optional[int] = None,
    sentence_pause: float = SENTENCE_PAUSE_SECONDS,
    split: bool = SPLIT_SENTENCES,
) -> Dict[str, Any]:
    """Zero-shot baseline: one TTS call per sentence, joined with an explicit neutral gap.

    The joined file is the *baseline* of the study (same voice, text, backbone and seed as the
    controlled output). Its inter-sentence pauses are what the pause control later scales.
    """
    sentences = split_sentences(text) if split else [text.strip()]
    if not sentences:
        raise ValueError("Text must not be empty.")

    pieces = []
    sample_rate = None
    meta: Dict[str, Any] = {}
    reference_text_source = "user" if reference_text.strip() else None
    with tempfile.TemporaryDirectory() as workdir:
        for index, sentence in enumerate(sentences):
            part_path = Path(workdir) / f"sentence_{index}.wav"
            meta = engine.synthesize(reference_path, sentence, part_path, reference_text=reference_text, seed=seed)
            reference_text = meta.get("reference_text", reference_text)  # transcribe the reference only once
            reference_text_source = reference_text_source or meta.get("reference_text_source")
            audio, sample_rate = read_audio(part_path)
            trimmed, _ = librosa.effects.trim(audio, top_db=40, frame_length=1024, hop_length=256)
            pieces.append(trimmed if trimmed.size else audio)

    assert sample_rate is not None
    gap = np.zeros(int(round(sentence_pause * sample_rate)), dtype=np.float32)
    edge = np.zeros(int(round(EDGE_PAD_SECONDS * sample_rate)), dtype=np.float32)
    joined = [edge]
    for index, piece in enumerate(pieces):
        if index:
            joined.append(gap)
        joined.append(piece.astype(np.float32))
    joined.append(edge)
    audio = np.concatenate(joined)

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * (BASELINE_PEAK / peak)
    save_audio(output_path, audio, sample_rate)
    meta["reference_text_source"] = reference_text_source or meta.get("reference_text_source", "user")
    return {**meta, "output_path": str(output_path), "sentences": sentences, "sentence_pause": sentence_pause if len(sentences) > 1 else 0.0}


def _reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _vram_peak_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    return None


def _visual(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    return {"waveform": waveform_envelope(audio), "mel": mel_image(audio, sample_rate), "duration": len(audio) / sample_rate}


def _summary(evaluation: Dict[str, Any], accuracy: Dict[str, Any], params: Dict[str, float]) -> Dict[str, Any]:
    baseline = evaluation["clips"].get("baseline", {})
    controlled = evaluation["clips"].get("controlled", {})
    reference_prosody = evaluation["reference"]["prosody"]
    base_prosody = baseline.get("prosody", {})
    ctrl_prosody = controlled.get("prosody", {})
    return {
        "speaker_similarity": controlled.get("speaker_similarity"),
        "baseline_speaker_similarity": baseline.get("speaker_similarity"),
        "wer": controlled.get("wer"),
        "baseline_wer": baseline.get("wer"),
        "transcript": controlled.get("transcript"),
        "f0_median": ctrl_prosody.get("f0_median"),
        "baseline_f0_median": base_prosody.get("f0_median"),
        "reference_f0_median": reference_prosody.get("f0_median"),
        "energy": ctrl_prosody.get("energy_mean"),
        "baseline_energy": base_prosody.get("energy_mean"),
        "duration": ctrl_prosody.get("duration"),
        "baseline_duration": base_prosody.get("duration"),
        "pitch_change_pct": accuracy["pitch"]["measured"],
        "energy_change_pct": accuracy["energy"]["measured"],
        "rate_ratio": accuracy["rate"]["measured"],
        "identity_drop": (
            None
            if baseline.get("speaker_similarity") is None or controlled.get("speaker_similarity") is None
            else baseline["speaker_similarity"] - controlled["speaker_similarity"]
        ),
    }


def synthesize_with_prosody(
    *,
    request_id: str,
    reference_path: Path,
    reference_text: str,
    text: str,
    output_dir: Path,
    emotion: Optional[str] = None,
    overrides: Optional[Dict[str, Optional[float]]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Reference -> zero-shot TTS baseline -> prosody controller -> controlled audio + evaluation."""
    timings: Dict[str, float] = {}
    overall = time.perf_counter()

    emotion_key, params = resolve_prosody(emotion, **(overrides or {}))

    started = time.perf_counter()
    ref_audio, ref_sr, ref_info = preprocess_reference_audio(reference_path)
    timings["reference_processing"] = time.perf_counter() - started

    ref_clean_path = output_dir / f"{request_id}_reference.wav"
    baseline_path = output_dir / f"{request_id}_baseline.wav"
    controlled_path = output_dir / f"{request_id}_controlled.wav"
    save_audio(ref_clean_path, ref_audio, ref_sr)

    engine = get_tts_engine()
    _reset_vram_peak()
    started = time.perf_counter()
    tts_meta = synthesize_baseline(engine, ref_clean_path, text, baseline_path, reference_text=reference_text, seed=seed)
    timings["tts"] = time.perf_counter() - started

    baseline_audio, baseline_sr = read_audio(baseline_path)

    started = time.perf_counter()
    controller = ProsodyController.from_params(params)
    controlled_audio = controller.apply(baseline_audio, baseline_sr)
    save_audio(controlled_path, controlled_audio, baseline_sr)
    timings["prosody"] = time.perf_counter() - started

    baseline_duration = len(baseline_audio) / baseline_sr
    timings["rtf"] = timings["tts"] / max(baseline_duration, 1e-6)

    result: Dict[str, Any] = {
        "request_id": request_id,
        "emotion": emotion_key,
        "prosody": params,
        "text": text,
        "reference": {
            "duration": ref_info.duration,
            "raw_duration": ref_info.raw_duration,
            "snr_db": ref_info.snr_db,
            "transcript": tts_meta.get("reference_text", reference_text),
            "transcript_source": tts_meta.get("reference_text_source", "user"),
        },
        "warnings": list(ref_info.warnings),
        "files": {
            "reference": ref_clean_path.name,
            "baseline": baseline_path.name,
            "controlled": controlled_path.name,
        },
        "analysis": {
            "reference": _visual(ref_audio, ref_sr),
            "baseline": _visual(baseline_audio, baseline_sr),
            "controlled": _visual(controlled_audio, baseline_sr),
        },
    }

    if ENABLE_EVALUATION:
        started = time.perf_counter()
        evaluation = evaluate_clips(
            (ref_audio, ref_sr),
            {"baseline": (baseline_audio, baseline_sr), "controlled": (controlled_audio, baseline_sr)},
            text,
            include_contours=True,
        )
        timings["evaluation"] = time.perf_counter() - started
        accuracy = compute_control_accuracy(
            evaluation["clips"]["baseline"]["prosody"], evaluation["clips"]["controlled"]["prosody"], params
        )
        result["evaluation"] = evaluation
        result["control_accuracy"] = accuracy
        result["metrics"] = _summary(evaluation, accuracy, params)
        result["warnings"].extend(evaluation.get("warnings", []))
    else:
        result["warnings"].append("Evaluation is disabled (ENABLE_EVALUATION=false).")

    timings["total"] = time.perf_counter() - overall
    result["timings"] = timings
    result["run"] = {
        "backbone": MODEL_NAME,
        "model": tts_meta.get("model", F5_MODEL),
        "seed": tts_meta.get("seed"),
        "nfe_steps": tts_meta.get("nfe_steps"),
        "vram_peak_mb": _vram_peak_mb(),
        "sentences": tts_meta.get("sentences"),
        "sentence_pause": tts_meta.get("sentence_pause"),
        "sample_rate": baseline_sr,
        "reference_sample_rate": SAMPLE_RATE,
        "system": system_info(),
    }
    (output_dir / f"{request_id}_meta.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "analysis"}, indent=2, default=float), encoding="utf-8"
    )
    return result


def apply_prosody_only(
    *,
    request_id: str,
    audio_path: Path,
    output_dir: Path,
    emotion: Optional[str] = None,
    overrides: Optional[Dict[str, Optional[float]]] = None,
    target_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Prosody Lab: apply the controller to any speech recording (no TTS involved)."""
    emotion_key, params = resolve_prosody(emotion, **(overrides or {}))
    audio, sample_rate = read_audio(audio_path)
    if audio.size < sample_rate * 0.5:
        raise ValueError("Audio is too short; provide at least half a second of speech.")
    if audio.size > sample_rate * 60:
        raise ValueError("Audio is too long for the lab; provide at most 60 seconds.")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        raise ValueError("No speech detected: the recording is silent.")
    audio = (audio * (BASELINE_PEAK / peak)).astype(np.float32)  # same headroom as a TTS baseline

    started = time.perf_counter()
    controlled = ProsodyController.from_params(params).apply(audio, sample_rate)
    prosody_seconds = time.perf_counter() - started

    original_path = output_dir / f"{request_id}_original.wav"
    controlled_path = output_dir / f"{request_id}_controlled.wav"
    save_audio(original_path, audio, sample_rate)
    save_audio(controlled_path, controlled, sample_rate)

    evaluation = evaluate_clips(
        (audio, sample_rate),
        {"baseline": (audio, sample_rate), "controlled": (controlled, sample_rate)},
        target_text,
        include_contours=True,
        with_speaker=True,
        with_asr=bool(target_text),
    )
    accuracy = compute_control_accuracy(
        evaluation["clips"]["baseline"]["prosody"], evaluation["clips"]["controlled"]["prosody"], params
    )
    return {
        "request_id": request_id,
        "emotion": emotion_key,
        "prosody": params,
        "files": {"baseline": original_path.name, "controlled": controlled_path.name},
        "analysis": {"baseline": _visual(audio, sample_rate), "controlled": _visual(controlled, sample_rate)},
        "evaluation": evaluation,
        "control_accuracy": accuracy,
        "metrics": _summary(evaluation, accuracy, params),
        "timings": {"prosody": prosody_seconds},
        "warnings": list(evaluation.get("warnings", [])),
    }
