"""Download / warm up every pretrained model once so the first request is not slow.

    python scripts/download_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def step(name: str, fn) -> None:
    started = time.perf_counter()
    print(f"[{name}] loading ...", flush=True)
    fn()
    print(f"[{name}] ready in {time.perf_counter() - started:.0f}s", flush=True)


def main() -> None:
    from backend.core.tts_engine import get_tts_engine
    from backend.evaluation.models import get_speaker_encoder, get_whisper

    step("whisper (ASR)", get_whisper)
    step("ecapa (speaker encoder)", get_speaker_encoder)
    engine = get_tts_engine()
    usable, reason = engine.availability()
    if not usable:
        raise SystemExit(reason)
    step("f5-tts (zero-shot backbone)", engine._load)
    print("All models are cached.")


if __name__ == "__main__":
    main()
