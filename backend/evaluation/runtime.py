from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from torch.nn.functional import cosine_similarity
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from backend.evaluation.prosody_metrics import compute_prosody_metrics
from backend.evaluation.wer import compute_wer_from_asr

_speaker_encoder: EncoderClassifier | None = None
_whisper_processor: WhisperProcessor | None = None
_whisper_model: WhisperForConditionalGeneration | None = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def _speaker_similarity(reference_path: str | Path, generated_path: str | Path) -> float:
    global _speaker_encoder
    if _speaker_encoder is None:
        from speechbrain.utils.fetching import LocalStrategy

        _speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"),
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": _device()},
        )

    def embedding(path: str | Path) -> torch.Tensor:
        audio, sample_rate = _load_audio(path)
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        return _speaker_encoder.encode_batch(torch.from_numpy(audio).unsqueeze(0).to(_device())).squeeze()

    return float(cosine_similarity(embedding(reference_path).unsqueeze(0), embedding(generated_path).unsqueeze(0)).item())


def _transcribe(path: str | Path) -> str:
    global _whisper_processor, _whisper_model
    if _whisper_processor is None or _whisper_model is None:
        _whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        _whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(_device())
        _whisper_model.eval()

    audio, sample_rate = _load_audio(path)
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    inputs = _whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        tokens = _whisper_model.generate(inputs.input_features.to(_device()))
    return _whisper_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def evaluate_generated_audio(reference_path: str | Path, generated_path: str | Path, target_text: str) -> dict[str, Any]:
    generated, generated_sr = _load_audio(generated_path)
    reference, reference_sr = _load_audio(reference_path)
    generated_metrics = compute_prosody_metrics(generated, generated_sr)
    reference_metrics = compute_prosody_metrics(reference, reference_sr)
    transcript = _transcribe(generated_path)

    return {
        "speaker_similarity": _speaker_similarity(reference_path, generated_path),
        "wer": compute_wer_from_asr(target_text, transcript),
        "transcript": transcript,
        "f0_change": float(generated_metrics["f0_mean"] - reference_metrics["f0_mean"]),
        "energy_change": float(generated_metrics["energy_mean"] - reference_metrics["energy_mean"]),
        "duration": generated_metrics["duration"],
        "rtf": 0.0,
    }