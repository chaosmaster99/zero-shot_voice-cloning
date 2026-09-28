from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.wer import compute_wer_from_asr


def transcribe(model, processor, audio_path: Path, device: str) -> str:
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    audio = audio.mean(axis=1) if getattr(audio, "ndim", 1) > 1 else audio
    audio = librosa.resample(audio.astype("float32"), orig_sr=sample_rate, target_sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(input_features)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe generated audio and compute WER.")
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    report = {"model": args.model, "device": device, "target_text": args.target_text, "results": {}}
    for audio_path in sorted(args.audio_dir.glob("*.wav")):
        transcript = transcribe(model, processor, audio_path, device)
        report["results"][audio_path.name] = {
            "transcript": transcript,
            "wer": compute_wer_from_asr(args.target_text, transcript),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()