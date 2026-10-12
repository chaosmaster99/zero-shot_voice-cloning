"""Evaluate a folder of generated WAVs against one reference: speaker similarity, WER and prosody.

    python experiments/evaluation/evaluate_folder.py --reference ref.wav --audio-dir out/ \
        --target-text "Welcome to our speech processing project." --report out/eval.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.audio.preprocessing import read_audio  # noqa: E402
from backend.evaluation.runtime import evaluate_clips  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--target-text", default="", help="Text the clips should say; enables WER.")
    parser.add_argument("--report", required=True, type=Path, help="CSV output path.")
    args = parser.parse_args()

    reference = read_audio(args.reference)
    clips = {path.name: read_audio(path) for path in sorted(args.audio_dir.glob("*.wav"))}
    if not clips:
        raise SystemExit(f"No WAV files in {args.audio_dir}")

    text = args.target_text.strip() or None
    evaluation = evaluate_clips(reference, clips, text, include_contours=False, with_asr=bool(text))
    rows = []
    for name, entry in evaluation["clips"].items():
        rows.append(
            {
                "file": name,
                "speaker_similarity": entry.get("speaker_similarity"),
                "wer": entry.get("wer"),
                "transcript": entry.get("transcript"),
                **{k: v for k, v in entry["prosody"].items()},
            }
        )
    frame = pd.DataFrame(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.report, index=False)
    print(frame.drop(columns=["transcript"], errors="ignore").round(3).to_string(index=False))
    for warning in evaluation["warnings"]:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
