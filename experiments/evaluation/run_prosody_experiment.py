"""Apply a fixed set of prosody variants to one recording and report measured prosody.

Signal-level only (no TTS, no neural models), so it runs anywhere:
    python experiments/evaluation/run_prosody_experiment.py --input clip.wav --output-dir out --report out/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.audio.preprocessing import read_audio, save_audio  # noqa: E402
from backend.evaluation.control_accuracy import compute_control_accuracy  # noqa: E402
from backend.evaluation.prosody_metrics import analyze_prosody  # noqa: E402
from backend.prosody.controller import ProsodyController  # noqa: E402
from backend.prosody.presets import NEUTRAL_PARAMS, PRESETS  # noqa: E402

VARIANTS = {
    "baseline": {},
    "pitch_up": {"pitch": 10.0},
    "pitch_down": {"pitch": -10.0},
    "range_wide": {"pitch_range": 1.5},
    "range_flat": {"pitch_range": 0.6},
    "energy_up": {"energy": 20.0},
    "energy_down": {"energy": -20.0},
    "rate_fast": {"rate": 1.2},
    "rate_slow": {"rate": 0.8},
    "pauses_long": {"pause": 2.0},
    "pauses_short": {"pause": 0.5},
    **{f"preset_{key}": dict(spec["params"]) for key, spec in PRESETS.items() if key not in ("neutral", "custom")},
}


def run_experiment(input_path: Path, output_dir: Path) -> dict[str, dict]:
    audio, sample_rate = read_audio(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics = analyze_prosody(audio, sample_rate)
    results: dict[str, dict] = {}

    for name, settings in VARIANTS.items():
        params = {**NEUTRAL_PARAMS, **settings}
        processed = ProsodyController.from_params(params).apply(audio, sample_rate)
        output_path = output_dir / f"{input_path.stem}_{name}.wav"
        save_audio(output_path, processed, sample_rate)
        metrics = analyze_prosody(processed, sample_rate)
        results[name] = {
            "params": params,
            "output_path": str(output_path),
            "metrics": metrics,
            "control_accuracy": compute_control_accuracy(baseline_metrics, metrics, params),
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
    for name, entry in report.items():
        acc = entry["control_accuracy"]
        print(
            f"{name:16s} pitch {acc['pitch']['measured'] or 0:+6.1f}%  range {acc['pitch_range']['measured'] or 0:4.2f}x  "
            f"energy {acc['energy']['measured'] or 0:+6.1f}%  rate {acc['rate']['measured'] or 0:4.2f}x  "
            f"pause {acc['pause']['measured'] if acc['pause']['measured'] is not None else float('nan'):4.2f}x"
        )
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
