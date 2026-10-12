"""Full research study: zero-shot cloning + prosody control + identity-prosody trade-off.

For every reference speaker x fixed target sentence the script
  1. generates the zero-shot baseline with the pretrained TTS backbone,
  2. applies each prosody condition with the ProsodyController,
  3. measures speaker similarity (ECAPA), WER (Whisper), F0 / energy / rate / pauses,
  4. computes control error against the *baseline*,
and writes CSV tables, a metadata JSON (reproducibility, spec section 43) and figures.

Reference layout (one pair per speaker, 5-10 s each):
    data/references/<speaker_id>.wav
    data/references/<speaker_id>.txt     <- exact transcript of the recording

Usage:
    python experiments/run_study.py --references-dir data/references --output-dir results/study
    python experiments/run_study.py --no-tts ...   # smoke-test mode: the reference recording itself is the "baseline"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.audio.preprocessing import preprocess_reference_audio, read_audio, save_audio  # noqa: E402
from backend.config import F5_MODEL, F5_NFE_STEPS, MODEL_NAME, SEED  # noqa: E402
from backend.core.pipeline import synthesize_baseline  # noqa: E402
from backend.core.tts_engine import get_tts_engine  # noqa: E402
from backend.evaluation.asr import transcribe  # noqa: E402
from backend.evaluation.control_accuracy import compute_control_accuracy  # noqa: E402
from backend.evaluation.prosody_metrics import analyze_prosody  # noqa: E402
from backend.evaluation.runtime import count_words  # noqa: E402
from backend.evaluation.speaker_similarity import embedding_similarity, speaker_embedding  # noqa: E402
from backend.evaluation.wer import compute_wer_from_asr  # noqa: E402
from backend.prosody.controller import ProsodyController  # noqa: E402
from backend.prosody.presets import NEUTRAL_PARAMS, PRESETS  # noqa: E402
from backend.utils.system_info import system_info  # noqa: E402


def with_params(**overrides: float) -> Dict[str, float]:
    return {**NEUTRAL_PARAMS, **overrides}


STRONG_UP = with_params(pitch=20.0, pitch_range=1.5, energy=20.0, rate=1.2, pause=1.5)
STRONG_DOWN = with_params(pitch=-20.0, pitch_range=0.6, energy=-20.0, rate=0.8, pause=1.5)
LEVELS = {0: 0.0, 1: 1 / 3, 2: 2 / 3, 3: 1.0}


def interpolate(target: Dict[str, float], strength: float) -> Dict[str, float]:
    return {k: NEUTRAL_PARAMS[k] + strength * (target[k] - NEUTRAL_PARAMS[k]) for k in NEUTRAL_PARAMS}


def build_conditions() -> List[Dict[str, Any]]:
    """Experiments 2-5, presets, ablation (spec section 29) and the trade-off sweep (section 28)."""
    rows: List[Dict[str, Any]] = [{"experiment": "baseline", "condition": "baseline", "level": 0, "params": with_params()}]

    for value in (-20, -10, 10, 20):
        rows.append({"experiment": "pitch", "condition": f"pitch {value:+d}%", "level": value, "params": with_params(pitch=value)})
    for value in (-20, 20):
        rows.append({"experiment": "energy", "condition": f"energy {value:+d}%", "level": value, "params": with_params(energy=value)})
    for value in (0.8, 1.2):
        rows.append({"experiment": "rate", "condition": f"rate {value}x", "level": value, "params": with_params(rate=value)})
    for value in (0.5, 2.0):
        rows.append({"experiment": "pause", "condition": f"pause {value}x", "level": value, "params": with_params(pause=value)})
    for value in (0.6, 1.5):
        rows.append({"experiment": "pitch_range", "condition": f"range {value}x", "level": value, "params": with_params(pitch_range=value)})
    for value in (-3.0, 2.0):
        rows.append({"experiment": "pitch_slope", "condition": f"slope {value:+.1f} st/s", "level": value, "params": with_params(pitch_slope=value)})
    for value in (0.3, 0.6):
        rows.append({"experiment": "tremor", "condition": f"tremor {value} st", "level": value, "params": with_params(tremor=value)})
    for value in (-6.0, 6.0):
        rows.append({"experiment": "brightness", "condition": f"brightness {value:+.0f} dB", "level": value, "params": with_params(brightness=value)})

    rows.append(
        {"experiment": "combined", "condition": "pitch+15 energy+10 rate1.1", "level": 1,
         "params": with_params(pitch=15.0, energy=10.0, rate=1.1)}
    )

    for key, spec in PRESETS.items():
        if key in ("neutral", "custom"):
            continue
        rows.append({"experiment": "preset", "condition": key, "level": 1, "params": with_params(**spec["params"])})

    ablation = [
        ("A baseline", with_params()),
        ("B +pitch", with_params(pitch=15.0)),
        ("C +energy", with_params(pitch=15.0, energy=10.0)),
        ("D +rate", with_params(pitch=15.0, energy=10.0, rate=1.1)),
        ("E +pause +range (full)", with_params(pitch=15.0, energy=10.0, rate=1.1, pause=1.5, pitch_range=1.2)),
    ]
    for order, (name, params) in enumerate(ablation):
        rows.append({"experiment": "ablation", "condition": name, "level": order, "params": params})

    for direction, target in (("up", STRONG_UP), ("down", STRONG_DOWN)):
        for level, strength in LEVELS.items():
            rows.append(
                {"experiment": f"tradeoff_{direction}", "condition": f"{direction} level {level}", "level": level,
                 "strength": strength, "params": interpolate(target, strength)}
            )
    return rows


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def load_speakers(references_dir: Path, max_speakers: int | None) -> List[Dict[str, Any]]:
    speakers = []
    for wav in sorted(references_dir.glob("*.wav")) + sorted(references_dir.glob("*.flac")):
        transcript_file = wav.with_suffix(".txt")
        if not transcript_file.exists():
            print(f"skip {wav.name}: no transcript ({transcript_file.name})")
            continue
        speakers.append({"id": wav.stem, "path": wav, "transcript": transcript_file.read_text(encoding="utf-8").strip()})
    if not speakers:
        raise SystemExit(f"No <speaker>.wav + <speaker>.txt pairs found in {references_dir}.")
    return speakers[:max_speakers] if max_speakers else speakers


def generate_rows(args, out: Path, speakers: List[Dict[str, Any]], sentences: List[str], conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    engine = get_tts_engine()
    if not args.no_tts:
        usable, reason = engine.availability()
        if not usable:
            raise SystemExit(f"TTS backbone unavailable: {reason}")

    rows: List[Dict[str, Any]] = []
    for speaker in speakers:
        ref_audio, ref_sr, ref_info = preprocess_reference_audio(
            speaker["path"], min_seconds=args.min_reference_seconds, max_seconds=args.max_reference_seconds
        )
        ref_clean = out / "baselines" / f"{speaker['id']}_reference.wav"
        save_audio(ref_clean, ref_audio, ref_sr)
        ref_embedding = speaker_embedding(ref_audio, ref_sr)

        items = [(0, speaker["transcript"])] if args.no_tts else list(enumerate(sentences))
        for sentence_id, text in items:
            baseline_path = out / "baselines" / f"{speaker['id']}_s{sentence_id}.wav"
            tts_seconds = None
            if args.no_tts:
                baseline_path = ref_clean
            elif not baseline_path.exists():
                tic = time.perf_counter()
                synthesize_baseline(engine, ref_clean, text, baseline_path, reference_text=speaker["transcript"], seed=SEED)
                tts_seconds = time.perf_counter() - tic

            baseline, sr = read_audio(baseline_path)
            words = count_words(text)
            baseline_metrics = analyze_prosody(baseline, sr, num_words=words)
            baseline_wer = compute_wer_from_asr(text, transcribe(baseline, sr))
            baseline_similarity = embedding_similarity(ref_embedding, speaker_embedding(baseline, sr))
            print(f"[{speaker['id']} s{sentence_id}] baseline sim={baseline_similarity:.3f} wer={baseline_wer:.3f}", flush=True)

            for condition in conditions:
                params = condition["params"]
                is_baseline = condition["experiment"] == "baseline" or all(
                    abs(params[k] - NEUTRAL_PARAMS[k]) < 1e-9 for k in params
                )
                if is_baseline:
                    controlled, metrics = baseline, baseline_metrics
                    similarity, wer = baseline_similarity, baseline_wer
                else:
                    controlled = ProsodyController.from_params(params).apply(baseline, sr)
                    metrics = analyze_prosody(controlled, sr, num_words=words)
                    similarity = embedding_similarity(ref_embedding, speaker_embedding(controlled, sr))
                    wer = compute_wer_from_asr(text, transcribe(controlled, sr))
                    if args.save_audio:
                        name = f"{speaker['id']}_s{sentence_id}_{condition['experiment']}_{condition['condition']}".replace(" ", "_")
                        save_audio(out / "audio" / f"{name}.wav", controlled, sr)

                accuracy = compute_control_accuracy(baseline_metrics, metrics, params)
                row: Dict[str, Any] = {
                    "speaker": speaker["id"],
                    "sentence_id": sentence_id,
                    "experiment": condition["experiment"],
                    "condition": condition["condition"],
                    "level": condition["level"],
                    "strength": condition.get("strength"),
                    **{f"req_{k}": v for k, v in params.items()},
                    "speaker_similarity": similarity,
                    "identity_drop": baseline_similarity - similarity,
                    "wer": wer,
                    "wer_delta": wer - baseline_wer,
                    "f0_median": metrics["f0_median"],
                    "f0_std_semitones": metrics["f0_std_semitones"],
                    "f0_slope": metrics["f0_slope"],
                    "f0_tremor": metrics["f0_tremor"],
                    "alpha_ratio_db": metrics["alpha_ratio_db"],
                    "energy": metrics["energy_mean"],
                    "duration": metrics["duration"],
                    "speech_duration": metrics["speech_duration"],
                    "pause_total": metrics["pause_total"],
                    "pause_count": metrics["pause_count"],
                    "words_per_second": metrics["rate"],
                    "baseline_source": "reference_recording" if args.no_tts else "zero_shot_tts",
                    "tts_seconds": tts_seconds,
                }
                for control, entry in accuracy.items():
                    row[f"meas_{control}"] = entry["measured"]
                    row[f"err_{control}"] = entry["abs_error"]
                rows.append(row)

    return rows


def finalize(frame: pd.DataFrame, out: Path, args, speakers: List[Dict[str, Any]], sentences: List[str], started: float) -> None:
    (out / "figures").mkdir(exist_ok=True)
    metric_cols = ["speaker_similarity", "identity_drop", "wer", "wer_delta"] + [c for c in frame.columns if c.startswith("err_")]
    summary = frame.groupby(["experiment", "condition"], sort=False)[metric_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(out / "summary.csv")
    (out / "summary.md").write_text(frame.groupby(["experiment", "condition"], sort=False)[metric_cols].mean().round(3).to_markdown(), encoding="utf-8")

    metadata = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "backbone": MODEL_NAME,
        "tts_checkpoint": F5_MODEL,
        "nfe_steps": F5_NFE_STEPS,
        "seed": SEED,
        "baseline_source": str(frame["baseline_source"].iloc[0]),
        "speakers": [s["id"] for s in speakers],
        "sentences": sentences,
        "n_rows": len(frame),
        "finalized_from_csv": bool(args.from_csv),
        "elapsed_seconds": None if args.from_csv else time.perf_counter() - started,
        "speaker_encoder": "speechbrain/spkrec-ecapa-voxceleb",
        "asr": "openai/whisper-base.en",
        "system": system_info(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    from experiments.plot_study import make_figures

    make_figures(frame, out / "figures")
    print(f"Done: {len(frame)} rows -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references-dir", type=Path, default=PROJECT_ROOT / "data" / "references")
    parser.add_argument("--sentences", type=Path, default=PROJECT_ROOT / "experiments" / "sentences.txt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "study")
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--no-tts", action="store_true", help="Use the reference recording as baseline (smoke test; NOT zero-shot).")
    parser.add_argument("--save-audio", action="store_true", help="Keep every controlled WAV.")
    parser.add_argument("--from-csv", type=Path, default=None, help="Skip generation; rebuild summary, metadata and figures from an existing results.csv.")
    parser.add_argument("--min-reference-seconds", type=float, default=5.0)
    parser.add_argument("--max-reference-seconds", type=float, default=10.0)
    args = parser.parse_args()

    out = args.output_dir
    (out / "baselines").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    if args.save_audio:
        (out / "audio").mkdir(exist_ok=True)

    speakers = load_speakers(args.references_dir, args.max_speakers)
    sentences = [s.strip() for s in args.sentences.read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.max_sentences:
        sentences = sentences[: args.max_sentences]
    conditions = build_conditions()

    started = time.perf_counter()
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        frame = pd.DataFrame(generate_rows(args, out, speakers, sentences, conditions))
        frame.to_csv(out / "results.csv", index=False)
    finalize(frame, out, args, speakers, sentences, started)


if __name__ == "__main__":
    main()
