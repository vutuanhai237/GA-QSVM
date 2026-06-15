#!/usr/bin/env python3
"""Plot readout-noise benchmark curves."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


FONTSIZE = 23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/reviewer/noise/readout_noise_curves.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reviewer/noise"),
    )
    return parser.parse_args()


def load_curves(path: Path) -> dict[float, list[tuple[int, float]]]:
    curves: dict[float, list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            curves[float(row["epsilon"])].append(
                (int(row["generation"]), float(row["best_fitness"]))
            )
    return {
        epsilon: sorted(points, key=lambda item: item[0])
        for epsilon, points in curves.items()
    }


def main() -> None:
    args = parse_args()
    curves = load_curves(args.input)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    epsilons = sorted(curves)
    colors = plt.cm.viridis_r(
        [idx / float(max(len(epsilons) - 1, 1)) for idx in range(len(epsilons))]
    )

    for color, epsilon in zip(colors, epsilons):
        points = curves[epsilon]
        generations = [point[0] for point in points]
        fitness = [point[1] for point in points]
        label = rf"$\epsilon = {epsilon:g}$"
        ax.plot(generations, fitness, linestyle="-", linewidth=2.5, color=color, label=label)

    ax.set_title("Readout Noise", y=0.92, fontsize=FONTSIZE)
    ax.set_xlabel("Generation", fontsize=FONTSIZE)
    ax.set_ylabel("Best Fitness", fontsize=FONTSIZE)
    ax.tick_params(axis="x", labelsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.set_ylim(0.0, 0.85)

    legend = ax.legend(
        loc=4,
        bbox_to_anchor=(1.0, 0.13),
        fontsize=FONTSIZE,
        frameon=True,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("0.8")
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "readout_noise.svg")
    fig.savefig(args.output_dir / "readout_noise.png", dpi=300)
    fig.savefig(args.output_dir / "readout_noise.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
