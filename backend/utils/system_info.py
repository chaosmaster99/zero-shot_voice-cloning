from __future__ import annotations

import importlib.metadata
import platform
from functools import lru_cache
from typing import Any, Dict

PACKAGES = ("torch", "f5-tts", "speechbrain", "transformers", "librosa", "praat-parselmouth", "numpy", "fastapi")


@lru_cache(maxsize=1)
def system_info() -> Dict[str, Any]:
    """Software/hardware facts recorded with every run for reproducibility (spec section 43)."""
    info: Dict[str, Any] = {"python": platform.python_version(), "os": platform.platform(), "packages": {}}
    for package in PACKAGES:
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda_available"] = False
        info["gpu"] = None
    return info
