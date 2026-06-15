#!/usr/bin/env python3
"""Plot currently available reviewer benchmark results as PDF previews."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASET_LABELS = {
    "cancer": "Breast Cancer",
    "digits": "Digits",
    "wine": "Wine",
}
DATASET_ORDER = ["cancer", "digits", "wine"]
MODEL_ORDER = ["GA-FQK", "GA-PQK", "FQK", "PQK", "RBF"]
FONTSIZE = 23
FIGSIZE_1X3 = (19, 7)
QUBITS = [3, 4, 5, 6, 7]
MODEL_MARKERS = {"GA-FQK": "o", "GA-PQK": "D", "FQK": "^", "PQK": "v", "RBF": "s"}
MODEL_DASHES = {
    "GA-FQK": (3, 4, 1, 4),
    "GA-PQK": (6, 2),
    "FQK": (1, 1),
    "PQK": (6, 2),
    "RBF": (2, 1),
}
MODEL_COLORS = dict(zip(MODEL_ORDER, sns.color_palette("tab10", n_colors=len(MODEL_ORDER))))


def style_qubit_axis(ax: plt.Axes, *, title: str, ylabel: bool = False, ylim: tuple[float, float] = (45, 101)) -> None:
    ax.set_title(title, fontsize=FONTSIZE, y=0.94)
    ax.set_xlabel("Number of Qubits", fontsize=FONTSIZE)
    ax.set_ylabel("Accuracy (%)" if ylabel else "", fontsize=FONTSIZE)
    ax.set_xlim(2.8, 7.2)
    ax.set_xticks(QUBITS)
    ax.set_xticklabels(QUBITS, fontsize=FONTSIZE)
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(ylim[0], ylim[1], 10))
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.grid(axis="both", linestyle="--", alpha=0.7)


def boxed_legend(ax: plt.Axes, *args, **kwargs) -> plt.Legend:
    legend = ax.legend(*args, frameon=True, fancybox=False, **kwargs)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("0.8")
    legend.get_frame().set_facecolor("white")
    return legend


def read_summaries(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/summary.csv")):
        frame = pd.read_csv(path)
        frame["source_dir"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["accuracy_percent"] = df["mean_accuracy"] * 100.0
    df["accuracy_std_percent"] = df["std_accuracy"] * 100.0
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    return df


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_readout_noise(input_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_path)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    epsilons = sorted(df["epsilon"].unique())
    palette = sns.color_palette("viridis_r", n_colors=len(epsilons))

    for color, epsilon in zip(palette, epsilons):
        subset = df[df["epsilon"] == epsilon].sort_values("generation")
        ax.plot(
            subset["generation"],
            subset["best_fitness"],
            color=color,
            linewidth=2.5,
            label=rf"$\epsilon = {epsilon:g}$",
        )

    ax.set_title("Readout Noise", y=0.92, fontsize=FONTSIZE)
    ax.set_xlabel("Generation", fontsize=FONTSIZE)
    ax.set_ylabel("Best Fitness", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    ax.set_ylim(0.0, 0.85)
    legend = ax.legend(
        loc=4,
        bbox_to_anchor=(1.0, 0.15),
        fontsize=FONTSIZE,
        frameon=True,
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("0.8")
    legend.get_frame().set_facecolor("white")
    fig.tight_layout()
    save(fig, output_dir, "readout_noise_seaborn")


def plot_figure6_holdout(summary_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_path)
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_1X3, sharex=True, sharey=True)

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = (
            df[df["dataset"] == dataset]
            .set_index("model_label")
            .reindex(MODEL_ORDER)
            .reset_index()
        )
        x = np.arange(len(MODEL_ORDER))
        for xpos, row in zip(x, subset.itertuples(index=False)):
            model = row.model_label
            ax.errorbar(
                xpos,
                row.accuracy_mean,
                yerr=row.accuracy_std,
                fmt=MODEL_MARKERS[model],
                markersize=13,
                color=MODEL_COLORS[model],
                ecolor=MODEL_COLORS[model],
                elinewidth=2.5,
                capsize=6,
                capthick=2.5,
                label=model,
            )
        ax.set_title(DATASET_LABELS[dataset], y=0.94, fontsize=FONTSIZE)
        ax.set_xlabel("Model", fontsize=FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, fontsize=17, rotation=25, ha="right")
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.set_ylim(0, 103)
        ax.grid(axis="both", linestyle="--", alpha=0.7)

    axes[0].set_ylabel("Accuracy (%)", fontsize=FONTSIZE)
    handles, labels = axes[2].get_legend_handles_labels()
    boxed_legend(axes[2], handles, labels, loc="lower center", fontsize=17, ncol=1)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save(fig, output_dir, "figure6_holdout_q7_with_std")


def plot_figure6_holdout_lines(summary_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_path)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 7))
    ordered_datasets = [DATASET_LABELS[d] for d in DATASET_ORDER]
    for model in MODEL_ORDER:
        subset = (
            df[df["model_label"] == model]
            .assign(dataset_label=lambda x: pd.Categorical(x["dataset_label"], ordered_datasets, ordered=True))
            .sort_values("dataset_label")
        )
        x = np.arange(len(subset))
        ax.errorbar(
            x,
            subset["accuracy_mean"],
            yerr=subset["accuracy_std"],
            marker=MODEL_MARKERS[model],
            linestyle=(0, MODEL_DASHES[model]),
            linewidth=2.5,
            markersize=12,
            capsize=5,
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.set_xticks(np.arange(len(ordered_datasets)))
    ax.set_xticklabels(ordered_datasets, fontsize=FONTSIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=FONTSIZE)
    ax.set_xlabel("Dataset", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.set_ylim(0, 103)
    ax.grid(axis="both", linestyle="--", alpha=0.7)
    boxed_legend(ax, loc="lower center", fontsize=17, ncol=2)
    fig.tight_layout()
    save(fig, output_dir, "figure6_holdout_q7_dataset_lines")


def plot_legacy_qubit_sweep(input_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_path)
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_1X3, sharex=True, sharey=True)

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = df[df["dataset"] == dataset]
        for model in MODEL_ORDER:
            part = subset[subset["model_label"] == model].sort_values("qubits")
            ax.plot(
                part["qubits"],
                part["accuracy_percent"],
                marker=MODEL_MARKERS[model],
                linestyle=(0, MODEL_DASHES[model]),
                linewidth=2.5,
                markersize=11,
                color=MODEL_COLORS[model],
                label=model,
            )
        style_qubit_axis(ax, title=f"{DATASET_LABELS[dataset]} Dataset", ylabel=ax is axes[0])

    handles, labels = axes[0].get_legend_handles_labels()
    boxed_legend(axes[1], handles, labels, loc="lower center", fontsize=17, ncol=1)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    save(fig, output_dir, "figure6_legacy_qubit_sweep")


def plot_predefined_pqk(predefined: pd.DataFrame, output_dir: Path) -> None:
    df = predefined[predefined["model"].isin(["efficient-su2-pqk", "two-local-pqk"])].copy()
    if df.empty:
        return
    df["model_label"] = df["model"].map(
        {
            "efficient-su2-pqk": "EfficientSU2-PQK",
            "two-local-pqk": "TwoLocal-PQK",
        }
    )
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_1X3, sharex=True, sharey=True)
    palette = {
        "EfficientSU2-PQK": sns.color_palette("tab10")[0],
        "TwoLocal-PQK": sns.color_palette("tab10")[1],
    }
    markers = {"EfficientSU2-PQK": "o", "TwoLocal-PQK": "D"}
    dashes = {"EfficientSU2-PQK": (3, 4, 1, 4), "TwoLocal-PQK": (6, 2)}

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = df[df["dataset"] == dataset].sort_values(["model_label", "qubits"])
        for model, part in subset.groupby("model_label"):
            part = part.sort_values("qubits")
            ax.errorbar(
                part["qubits"],
                part["accuracy_percent"],
                yerr=part["accuracy_std_percent"],
                marker=markers[model],
                linestyle=(0, dashes[model]),
                linewidth=2.5,
                markersize=11,
                capsize=4,
                color=palette[model],
                label=model,
            )
        style_qubit_axis(ax, title=f"{DATASET_LABELS[dataset]} Dataset", ylabel=ax is axes[0])

    handles, labels = axes[0].get_legend_handles_labels()
    boxed_legend(axes[1], handles, labels, loc="lower center", fontsize=17, ncol=1)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    save(fig, output_dir, "predefined_feature_maps_pqk")


def plot_available_fqk(predefined: pd.DataFrame, output_dir: Path) -> None:
    df = predefined[predefined["model"].isin(["efficient-su2-fqk", "two-local-fqk"])].copy()
    if df.empty:
        return
    df["model_label"] = df["model"].map(
        {
            "efficient-su2-fqk": "EfficientSU2-FQK",
            "two-local-fqk": "TwoLocal-FQK",
        }
    )
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_1X3, sharex=True, sharey=True)
    palette = {
        "EfficientSU2-FQK": sns.color_palette("tab10")[0],
        "TwoLocal-FQK": sns.color_palette("tab10")[1],
    }
    markers = {"EfficientSU2-FQK": "o", "TwoLocal-FQK": "D"}
    dashes = {"EfficientSU2-FQK": (3, 4, 1, 4), "TwoLocal-FQK": (6, 2)}

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = df[df["dataset"] == dataset].sort_values(["model_label", "qubits"])
        for model, part in subset.groupby("model_label"):
            part = part.sort_values("qubits")
            ax.errorbar(
                part["qubits"],
                part["accuracy_percent"],
                yerr=part["accuracy_std_percent"],
                marker=markers[model],
                linestyle=(0, dashes[model]),
                linewidth=2.5,
                markersize=11,
                capsize=4,
                color=palette[model],
                label=model,
            )
        style_qubit_axis(ax, title=f"{DATASET_LABELS[dataset]} Dataset", ylabel=ax is axes[0])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        boxed_legend(axes[1], handles, labels, loc="lower center", fontsize=17, ncol=1)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    save(fig, output_dir, "predefined_feature_maps_fqk_available_partial")


def plot_random_pqk(random_df: pd.DataFrame, output_dir: Path) -> None:
    df = random_df[random_df["model"] == "random-pqk"].copy()
    if df.empty:
        return
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_1X3, sharex=True, sharey=True)
    color = sns.color_palette("tab10")[2]

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = df[df["dataset"] == dataset].sort_values("qubits")
        ax.errorbar(
            subset["qubits"],
            subset["accuracy_percent"],
            yerr=subset["accuracy_std_percent"],
            marker="o",
            linestyle=(0, (3, 4, 1, 4)),
            linewidth=2.5,
            markersize=11,
            capsize=4,
            color=color,
            label="Random-PQK",
        )
        style_qubit_axis(ax, title=f"{DATASET_LABELS[dataset]} Dataset", ylabel=ax is axes[0])

    boxed_legend(axes[0], loc=4, fontsize=FONTSIZE)
    fig.tight_layout()
    save(fig, output_dir, "random_search_pqk")


def plot_runtime(predefined: pd.DataFrame, random_df: pd.DataFrame, output_dir: Path) -> None:
    frames = []
    if not predefined.empty:
        frames.append(predefined[["dataset", "dataset_label", "model", "qubits", "mean_runtime_seconds"]])
    if not random_df.empty:
        frames.append(random_df[["dataset", "dataset_label", "model", "qubits", "mean_runtime_seconds"]])
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df["runtime_minutes"] = df["mean_runtime_seconds"] / 60.0

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_1X3, sharex=True, sharey=False)
    models = sorted(df["model"].unique())
    palette = dict(zip(models, sns.color_palette("tab10", n_colors=len(models))))
    markers = ["o", "D", "^", "v", "s", "P", "X"]

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = df[df["dataset"] == dataset].sort_values(["model", "qubits"])
        for idx, (model, part) in enumerate(subset.groupby("model")):
            ax.plot(
                part["qubits"],
                part["runtime_minutes"],
                marker=markers[idx % len(markers)],
                linewidth=2.5,
                markersize=9,
                color=palette[model],
                label=model,
            )
        ax.set_title(f"{DATASET_LABELS[dataset]} Dataset", fontsize=FONTSIZE, y=0.94)
        ax.set_xlabel("Number of Qubits", fontsize=FONTSIZE)
        ax.set_ylabel("Mean Runtime (min)" if ax is axes[0] else "", fontsize=FONTSIZE)
        ax.set_xticks(QUBITS)
        ax.set_xticklabels(QUBITS, fontsize=FONTSIZE)
        ax.tick_params(axis="y", labelsize=17)
        ax.grid(axis="both", linestyle="--", alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    boxed_legend(axes[1], handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=13, ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    save(fig, output_dir, "available_baseline_runtime")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reviewer/plots_pdf"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_readout_noise(
        repo_root / "results/reviewer/noise/readout_noise_curves.csv",
        output_dir,
    )
    plot_figure6_holdout(
        repo_root / "results/reviewer/final/figure6_holdout_summary.csv",
        output_dir,
    )
    plot_figure6_holdout_lines(
        repo_root / "results/reviewer/final/figure6_holdout_summary.csv",
        output_dir,
    )
    plot_legacy_qubit_sweep(
        repo_root / "results/reviewer/final/figure6_legacy_qubit_sweep.csv",
        output_dir,
    )

    predefined = read_summaries(repo_root / "results/reviewer/predefined_baselines")
    random_df = read_summaries(repo_root / "results/reviewer/random_search_baselines")
    plot_predefined_pqk(predefined, output_dir)
    plot_available_fqk(predefined, output_dir)
    plot_random_pqk(random_df, output_dir)
    plot_runtime(predefined, random_df, output_dir)

    for path in sorted(output_dir.glob("*.pdf")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
