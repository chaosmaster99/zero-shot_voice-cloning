"""Figures for the study results (spec section 39, phase 8)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"up": "#e11d48", "down": "#2563eb"}
CONTROL_PANELS = [
    ("pitch", "pitch", "Requested pitch change (%)", "Measured median-F0 change (%)"),
    ("energy", "energy", "Requested energy change (%)", "Measured RMS change (%)"),
    ("rate", "rate", "Requested rate (x)", "Measured rate (x)"),
    ("pause", "pause", "Requested pause multiplier (x)", "Measured pause multiplier (x)"),
    ("pitch_range", "pitch_range", "Requested intonation range (x)", "Measured F0 spread ratio (x)"),
]


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def control_accuracy(frame: pd.DataFrame, out: Path) -> None:
    figure, axes = plt.subplots(1, len(CONTROL_PANELS), figsize=(4.2 * len(CONTROL_PANELS), 4))
    for axis, (control, column, xlabel, ylabel) in zip(axes, CONTROL_PANELS):
        data = frame[[f"req_{column}", f"meas_{control}"]].dropna()
        if data.empty:
            axis.set_visible(False)
            continue
        grouped = data.groupby(f"req_{column}")[f"meas_{control}"].agg(["mean", "std"]).reset_index()
        lo = min(grouped[f"req_{column}"].min(), grouped["mean"].min())
        hi = max(grouped[f"req_{column}"].max(), grouped["mean"].max())
        axis.plot([lo, hi], [lo, hi], "--", color="#94a3b8", label="ideal")
        axis.errorbar(grouped[f"req_{column}"], grouped["mean"], yerr=grouped["std"].fillna(0), fmt="o", color="#2563eb", capsize=3)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    _save(figure, out / "control_accuracy.png")


def tradeoff(frame: pd.DataFrame, out: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for direction in ("up", "down"):
        data = frame[frame["experiment"] == f"tradeoff_{direction}"]
        if data.empty:
            continue
        grouped = data.groupby("level")[["speaker_similarity", "wer", "identity_drop"]].agg(["mean", "std"])
        levels = grouped.index.values
        for axis, column, label in zip(axes, ["speaker_similarity", "wer", "identity_drop"], ["Speaker similarity (ECAPA cosine)", "WER", "Identity drop vs baseline"]):
            axis.errorbar(levels, grouped[(column, "mean")], yerr=grouped[(column, "std")].fillna(0), marker="o",
                          color=COLORS[direction], capsize=3, label=f"strong {direction}")
            axis.set_xlabel("Prosody modification level (0 = natural, 3 = strong)")
            axis.set_ylabel(label)
            axis.set_xticks(levels)
            axis.grid(alpha=0.25)
    axes[0].legend()
    _save(figure, out / "identity_prosody_tradeoff.png")


def ablation(frame: pd.DataFrame, out: Path) -> None:
    data = frame[frame["experiment"] == "ablation"]
    if data.empty:
        return
    grouped = data.groupby("condition", sort=False)[["speaker_similarity", "wer", "err_pitch", "err_energy", "err_rate"]].mean()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    grouped["speaker_similarity"].plot.bar(ax=axes[0], color="#2563eb", rot=30)
    axes[0].set_ylabel("Speaker similarity")
    grouped["wer"].plot.bar(ax=axes[1], color="#e11d48", rot=30)
    axes[1].set_ylabel("WER")
    grouped[["err_pitch", "err_energy", "err_rate"]].plot.bar(ax=axes[2], rot=30)
    axes[2].set_ylabel("Control error (abs)")
    for axis in axes:
        axis.set_xlabel("")
        axis.grid(alpha=0.25, axis="y")
    _save(figure, out / "ablation.png")


def presets(frame: pd.DataFrame, out: Path) -> None:
    data = frame[frame["experiment"].isin(["baseline", "preset"])]
    if data.empty:
        return
    grouped = data.groupby("condition", sort=False)[["speaker_similarity", "wer", "f0_median", "duration"]].mean()
    figure, axes = plt.subplots(1, 4, figsize=(18, 4))
    for axis, column, label in zip(axes, grouped.columns, ["Speaker similarity", "WER", "Median F0 (Hz)", "Duration (s)"]):
        grouped[column].plot.bar(ax=axis, color="#7c3aed", rot=30)
        axis.set_ylabel(label)
        axis.set_xlabel("")
        axis.grid(alpha=0.25, axis="y")
    _save(figure, out / "presets.png")


def single_controls(frame: pd.DataFrame, out: Path) -> None:
    experiments = ["pitch", "energy", "rate", "pause", "pitch_range"]
    figure, axes = plt.subplots(1, len(experiments), figsize=(4 * len(experiments), 4), sharey=True)
    base = frame[frame["experiment"] == "baseline"]["speaker_similarity"].mean()
    for axis, experiment in zip(axes, experiments):
        data = frame[frame["experiment"] == experiment]
        if data.empty:
            axis.set_visible(False)
            continue
        grouped = data.groupby("level")["speaker_similarity"].agg(["mean", "std"]).reset_index()
        axis.errorbar(grouped["level"], grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", color="#0f766e", capsize=3)
        axis.axhline(base, color="#94a3b8", linestyle="--", label="baseline")
        axis.set_title(experiment.replace("_", " "))
        axis.set_xlabel("control value")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Speaker similarity")
    axes[0].legend()
    _save(figure, out / "single_controls_similarity.png")


def make_figures(frame: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    control_accuracy(frame, out)
    tradeoff(frame, out)
    ablation(frame, out)
    presets(frame, out)
    single_controls(frame, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-create figures from a results.csv produced by run_study.py")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_figures(pd.read_csv(args.results), args.output_dir)


if __name__ == "__main__":
    main()
