from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from torch.nn.functional import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = librosa.resample(audio.astype("float32"), orig_sr=sample_rate, target_sr=16000)
    return torch.from_numpy(audio).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure speaker identity similarity with ECAPA embeddings.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"),
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": device},
    )
    reference_embedding = encoder.encode_batch(load_audio(args.reference).to(device)).squeeze()

    report = {"model": "speechbrain/spkrec-ecapa-voxceleb", "device": device, "results": {}}
    for audio_path in sorted(args.audio_dir.glob("*.wav")):
        generated_embedding = encoder.encode_batch(load_audio(audio_path).to(device)).squeeze()
        score = cosine_similarity(reference_embedding.unsqueeze(0), generated_embedding.unsqueeze(0)).item()
        report["results"][audio_path.name] = {"speaker_similarity": float(score)}

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()