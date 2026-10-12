from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Callable, TypeVar

from fastapi import HTTPException, UploadFile

from backend.config import MAX_UPLOAD_BYTES, OUTPUT_DIRECTORY
from backend.core.tts_engine import TTSUnavailable
from backend.evaluation.models import ModelUnavailable

logger = logging.getLogger("voice_cloning")

ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
T = TypeVar("T")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def save_upload(upload: UploadFile, request_id: str, label: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type '{suffix}'. Use WAV, FLAC, OGG or MP3.")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    destination = Path(OUTPUT_DIRECTORY) / f"{request_id}_{label}_upload{suffix}"
    destination.write_bytes(data)
    return destination


def guarded(request_id: str, work: Callable[[], T]) -> T:
    """Run `work`, mapping domain errors to clean HTTP errors (never a raw traceback)."""
    try:
        return work()
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (TTSUnavailable, ModelUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        if type(exc).__name__ == "OutOfMemoryError":
            logger.error("Request %s ran out of GPU memory", request_id)
            raise HTTPException(
                status_code=503,
                detail="The GPU ran out of memory. Try a shorter text, or set DEVICE=cpu in .env.",
            ) from exc
        logger.exception("Request %s failed", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while processing request {request_id}. See the server log for details.",
        ) from exc


def cleanup_old_outputs(max_age_hours: float = 24.0) -> int:
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for path in Path(OUTPUT_DIRECTORY).glob("*"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed
