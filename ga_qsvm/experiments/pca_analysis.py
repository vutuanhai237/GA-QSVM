from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ga_qsvm.experiments.artifacts import write_csv
from ga_qsvm.experiments.datasets import load_dataset


def _threshold_key(threshold: float) -> str:
    return f"components_for_{int(round(threshold * 100))}"


def compute_pca_summary(dataset: str, x: np.ndarray, *, thresholds: list[float]) -> dict:
    scaled = StandardScaler().fit_transform(x)
    pca = PCA().fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    row = {
        "dataset": dataset,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
    }
    for threshold in thresholds:
        row[_threshold_key(threshold)] = int(np.searchsorted(cumulative, threshold) + 1)
    return row


def run_pca_analysis(
    *,
    datasets: list[str],
    thresholds: list[float],
    output_dir: str | Path,
    skip_missing: bool = False,
) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    for dataset_name in datasets:
        bundle = load_dataset(dataset_name, skip_missing=skip_missing)
        if bundle is None:
            continue
        scaled = StandardScaler().fit_transform(bundle.x)
        pca = PCA().fit(scaled)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        summary_rows.append(compute_pca_summary(bundle.name, bundle.x, thresholds=thresholds))
        for index, value in enumerate(cumulative, start=1):
            curve_rows.append({"dataset": bundle.name, "component": index, "cumulative_explained_variance": float(value)})
        plt.plot(range(1, len(cumulative) + 1), cumulative, label=bundle.name)
    for threshold in thresholds:
        plt.axhline(threshold, linestyle="--", linewidth=0.8)
    plt.xlabel("PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "figure3_pca_explained_variance.png", dpi=200)
    plt.close()
    write_csv(output / "pca_summary.csv", summary_rows)
    write_csv(output / "pca_curve.csv", curve_rows)
    return summary_rows
