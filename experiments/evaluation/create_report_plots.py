from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def short_name(filename: str) -> str:
    return filename.removeprefix("test1_zero_shot_gpu_").removesuffix(".wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create research plots from evaluation reports.")
    parser.add_argument("--prosody-report", required=True, type=Path)
    parser.add_argument("--asr-report", required=True, type=Path)
    parser.add_argument("--speaker-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    prosody = json.loads(args.prosody_report.read_text(encoding="utf-8"))
    asr = json.loads(args.asr_report.read_text(encoding="utf-8"))["results"]
    speaker = json.loads(args.speaker_report.read_text(encoding="utf-8"))["results"]
    names = list(prosody)
    labels = [name.replace("_", " ").title() for name in names]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "speaker_similarity": [speaker[f"test1_zero_shot_gpu_{name}.wav"]["speaker_similarity"] for name in names],
        "wer": [asr[f"test1_zero_shot_gpu_{name}.wav"]["wer"] for name in names],
        "f0_mean_hz": [prosody[name]["metrics"]["f0_mean"] for name in names],
        "energy_rms": [prosody[name]["metrics"]["energy_mean"] for name in names],
        "duration_seconds": [prosody[name]["metrics"]["duration"] for name in names],
        "pause_count": [prosody[name]["metrics"]["pause_count"] for name in names],
    }

    for metric, values in figures.items():
        figure, axis = plt.subplots(figsize=(11, 5.5))
        axis.bar(labels, values, color="#2b6cb0")
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{metric}.png", dpi=180)
        plt.close(figure)

    print(f"Created {len(figures)} plots in {args.output_dir}")


if __name__ == "__main__":
    main()