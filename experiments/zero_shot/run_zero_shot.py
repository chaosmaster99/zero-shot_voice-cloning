from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.audio.preprocessing import preprocess_reference_audio, save_audio
from backend.config import OUTPUT_DIRECTORY
from backend.core.tts_engine import get_tts_engine


def run_zero_shot(reference_audio: str, reference_text: str, text: str, output_path: str | None = None):
    if not text.strip():
        raise ValueError("Text must not be empty.")
    if not reference_text.strip():
        raise ValueError("Reference text must not be empty.")

    processed_audio, sr = preprocess_reference_audio(reference_audio)
    output = Path(output_path) if output_path else Path(OUTPUT_DIRECTORY) / "zero_shot_output.wav"
    output.parent.mkdir(parents=True, exist_ok=True)

    engine = get_tts_engine()
    try:
        engine.synthesize(reference_audio, text, output, reference_text=reference_text)
        if not output.exists():
            raise RuntimeError("TTS completed without producing an output WAV.")
        return str(output)
    except Exception as exc:
        raise RuntimeError(f"Zero-shot synthesis failed for this environment: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Run a zero-shot speech generation pass.")
    parser.add_argument("--reference", required=True, help="Path to the 5–10 second reference WAV file.")
    parser.add_argument("--reference-text", required=True, help="Exact transcript of the reference recording.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--output", default=None, help="Optional output WAV path.")
    args = parser.parse_args()

    output = run_zero_shot(args.reference, args.reference_text, args.text, args.output)
    print(f"Generated audio: {output}")


if __name__ == "__main__":
    main()
