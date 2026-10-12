"""Validate the emotion presets on real zero-shot clones.

For every reference speaker x sentence: synthesize the F5-TTS baseline once, apply every emotion
preset with the DSP ProsodyController, and report
  * control accuracy (requested vs measured) for each of the eight controls,
  * speaker similarity (ECAPA) and WER (Whisper) cost,
  * optionally (--ser), the output distribution of an independent pretrained speech-emotion
    recogniser (wav2vec2 fine-tuned on IEMOCAP: neutral / happy / angry / sad) as an objective
    check that the presets move the perceived emotion in the intended direction.

Usage:
    python experiments/run_preset_validation.py --max-sentences 2 --ser --save-audio
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.audio.preprocessing import preprocess_reference_audio, read_audio, save_audio  # noqa: E402
from backend.config import SEED  # noqa: E402
from backend.core.pipeline import synthesize_baseline  # noqa: E402
from backend.core.tts_engine import get_tts_engine  # noqa: E402
from backend.evaluation.asr import transcribe  # noqa: E402
from backend.evaluation.control_accuracy import compute_control_accuracy  # noqa: E402
from backend.evaluation.prosody_metrics import analyze_prosody  # noqa: E402
from backend.evaluation.runtime import count_words  # noqa: E402
from backend.evaluation.speaker_similarity import embedding_similarity, speaker_embedding  # noqa: E402
from backend.evaluation.wer import compute_wer_from_asr  # noqa: E402
from backend.prosody.controller import ProsodyController  # noqa: E402
from backend.prosody.presets import PRESETS  # noqa: E402
from experiments.run_study import load_speakers  # noqa: E402

SER_MODEL = "superb/wav2vec2-base-superb-er"
SER_LABELS = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}


def load_ser():
    from transformers import pipeline

    return pipeline("audio-classification", model=SER_MODEL, top_k=None)


def ser_scores(classifier, audio: np.ndarray, sr: int) -> Dict[str, float]:
    import librosa

    wave = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
    return {f"ser_{SER_LABELS.get(item['label'], item['label'])}": float(item["score"]) for item in classifier(wave.astype(np.float32))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references-dir", type=Path, default=PROJECT_ROOT / "data" / "references")
    parser.add_argument("--sentences", type=Path, default=PROJECT_ROOT / "experiments" / "sentences.txt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "preset_validation")
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--ser", action="store_true", help=f"Also score every clip with {SER_MODEL}.")
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()

    engine = get_tts_engine()
    usable, reason = engine.availability()
    if not usable:
        raise SystemExit(f"TTS backbone unavailable: {reason}")
    classifier = load_ser() if args.ser else None

    out = args.output_dir
    (out / "audio").mkdir(parents=True, exist_ok=True)
    speakers = load_speakers(args.references_dir, args.max_speakers)
    sentences = [s.strip() for s in args.sentences.read_text(encoding="utf-8").splitlines() if s.strip()]
    sentences = sentences[: args.max_sentences] if args.max_sentences else sentences

    rows: List[Dict[str, Any]] = []
    for speaker in speakers:
        ref_audio, ref_sr, _ = preprocess_reference_audio(speaker["path"])
        ref_clean = out / "audio" / f"{speaker['id']}_reference.wav"
        save_audio(ref_clean, ref_audio, ref_sr)
        ref_embedding = speaker_embedding(ref_audio, ref_sr)

        for sentence_id, text in enumerate(sentences):
            base_path = out / "audio" / f"{speaker['id']}_s{sentence_id}_neutral.wav"
            if not base_path.exists():
                synthesize_baseline(engine, ref_clean, text, base_path, reference_text=speaker["transcript"], seed=SEED)
            baseline, sr = read_audio(base_path)
            words = count_words(text)
            base_metrics = analyze_prosody(baseline, sr, num_words=words)

            for key, spec in PRESETS.items():
                if key == "custom":
                    continue
                params = spec["params"]
                audio = baseline if key == "neutral" else ProsodyController.from_params(params).apply(baseline, sr)
                if args.save_audio and key != "neutral":
                    save_audio(out / "audio" / f"{speaker['id']}_s{sentence_id}_{key}.wav", audio, sr)
                metrics = analyze_prosody(audio, sr, num_words=words)
                accuracy = compute_control_accuracy(base_metrics, metrics, params)
                row: Dict[str, Any] = {
                    "speaker": speaker["id"],
                    "sentence_id": sentence_id,
                    "preset": key,
                    "speaker_similarity": embedding_similarity(ref_embedding, speaker_embedding(audio, sr)),
                    "wer": compute_wer_from_asr(text, transcribe(audio, sr)),
                    "peak": float(np.max(np.abs(audio))),
                }
                for control, entry in accuracy.items():
                    row[f"req_{control}"] = entry["requested"]
                    row[f"meas_{control}"] = entry["measured"]
                    row[f"err_{control}"] = entry["abs_error"]
                if classifier is not None:
                    row.update(ser_scores(classifier, audio, sr))
                rows.append(row)
                print(f"[{speaker['id']} s{sentence_id} {key:8s}] sim={row['speaker_similarity']:.3f} wer={row['wer']:.2f} "
                      + " ".join(f"{c}={accuracy[c]['measured']:.2f}" for c in accuracy if accuracy[c]["measured"] is not None), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "results.csv", index=False)
    controls = [c[4:] for c in frame.columns if c.startswith("err_")]
    accuracy_table = pd.DataFrame({
        control: {
            "requested (mean)": frame[f"req_{control}"].mean(),
            "mean abs error": frame[f"err_{control}"].mean(),
            "max abs error": frame[f"err_{control}"].max(),
        }
        for control in controls
    }).T.round(3)
    by_preset = frame.groupby("preset", sort=False)[
        ["speaker_similarity", "wer"] + [c for c in frame.columns if c.startswith("ser_")]
    ].mean().round(3)
    errors = frame.groupby("preset", sort=False)[[f"err_{c}" for c in controls]].mean().round(3)
    report = (
        "## Control accuracy over all presets\n\n" + accuracy_table.to_markdown()
        + "\n\n## Mean absolute error per preset\n\n" + errors.to_markdown()
        + "\n\n## Identity, intelligibility" + (" and emotion-recogniser scores" if args.ser else "") + " per preset\n\n"
        + by_preset.to_markdown() + "\n"
    )
    (out / "summary.md").write_text(report, encoding="utf-8")
    (out / "presets.json").write_text(json.dumps({k: v["params"] for k, v in PRESETS.items()}, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
