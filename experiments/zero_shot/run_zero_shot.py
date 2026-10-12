"""Experiment 1 building block: one zero-shot generation pass (reference + text -> WAV)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.audio.preprocessing import preprocess_reference_audio, save_audio  # noqa: E402
from backend.config import OUTPUT_DIRECTORY  # noqa: E402
from backend.core.tts_engine import get_tts_engine  # noqa: E402


def run_zero_shot(reference_audio: str, text: str, reference_text: str = "", output_path: Optional[str] = None) -> str:
    if not text.strip():
        raise ValueError("Text must not be empty.")

    audio, sample_rate, info = preprocess_reference_audio(reference_audio)
    for warning in info.warnings:
        print(f"warning: {warning}")

    output = Path(output_path) if output_path else Path(OUTPUT_DIRECTORY) / "zero_shot_output.wav"
    cleaned_reference = output.with_name(output.stem + "_reference.wav")
    save_audio(cleaned_reference, audio, sample_rate)

    meta = get_tts_engine().synthesize(cleaned_reference, text, output, reference_text=reference_text)
    print(f"reference transcript ({meta['reference_text_source']}): {meta['reference_text']}")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a zero-shot speech generation pass.")
    parser.add_argument("--reference", required=True, help="Path to the 5-10 second reference recording.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--reference-text", default="", help="Exact transcript of the reference (auto-transcribed with Whisper if omitted).")
    parser.add_argument("--output", default=None, help="Optional output WAV path.")
    args = parser.parse_args()

    print(f"Generated audio: {run_zero_shot(args.reference, args.text, args.reference_text, args.output)}")


if __name__ == "__main__":
    main()
