"""Create a small demo reference set in data/references from the clips bundled with F5-TTS.

Four short English recordings (three different voices) are enough to smoke-test the whole
pipeline and the study script. For real experiments use a proper corpus (VCTK / LibriTTS):
one `<speaker>.wav` (5-10 s) + `<speaker>.txt` (exact transcript) per speaker.

    python scripts/prepare_demo_references.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import REFERENCES_DIR  # noqa: E402

KNOWN_TRANSCRIPTS = {"basic_ref_en": "Some call me nature, others call me mother nature."}
CLIPS = {
    "basic_ref_en": "basic/basic_ref_en.wav",
    "voice_main": "multi/main.flac",
    "voice_town": "multi/town.flac",
    "voice_country": "multi/country.flac",
}


def main() -> None:
    import f5_tts

    from backend.audio.preprocessing import read_audio
    from backend.evaluation.asr import transcribe

    examples = Path(next(iter(f5_tts.__path__))) / "infer" / "examples"
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    for name, relative in CLIPS.items():
        source = examples / relative
        audio, sample_rate = read_audio(source)
        target = REFERENCES_DIR / f"{name}.wav"
        sf.write(str(target), audio, sample_rate)
        transcript = KNOWN_TRANSCRIPTS.get(name) or transcribe(audio, sample_rate)
        (REFERENCES_DIR / f"{name}.txt").write_text(transcript, encoding="utf-8")
        print(f"{name}: {len(audio) / sample_rate:.1f}s  \"{transcript}\"")


if __name__ == "__main__":
    main()
