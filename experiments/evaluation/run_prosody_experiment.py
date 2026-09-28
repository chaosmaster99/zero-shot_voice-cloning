from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.prosody_metrics import compute_prosody_metrics
from backend.prosody.controller import ProsodyController


VARIANTS = {
    "baseline": {},
    "pitch_up": {"pitch": 3.0},
    "pitch_down": {"pitch": -3.0},
    "energy_up": {"energy": 3.0},
    "energy_down": {"energy": -3.0},
    "rate_fast": {"rate": 1.2},
    "rate_slow": {"rate": 0.8},
    "pauses": {"pause": 2.0},
}


def run_experiment(input_path: Path, output_dir: Path) -> dict[str, dict]:
    audio, sample_rate = sf.read(str(input_path), always_2d=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for name, settings in VARIANTS.items():
        processed = ProsodyController(**settings).apply(audio, sample_rate)
        output_path = output_dir / f"{input_path.stem}_{name}.wav"
        sf.write(str(output_path), processed, sample_rate)
        results[name] = {
            "settings": settings,
            "output_path": str(output_path),
            "metrics": compute_prosody_metrics(processed, sample_rate),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and measure controlled prosody variants.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    report = run_experiment(args.input, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()