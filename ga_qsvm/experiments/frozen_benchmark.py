from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Callable

from ga_qsvm.experiments.artifacts import CircuitArtifact, load_manifest, write_csv
from ga_qsvm.experiments.datasets import load_dataset, make_holdout_split
from ga_qsvm.experiments.kernels import (
    fit_predict_fixed_fqk,
    fit_predict_fixed_pqk,
    fit_predict_ga_fqk,
    fit_predict_ga_pqk,
    fit_predict_rbf,
    qpy_feature_dimension,
    score_predictions,
)

Predictor = Callable[..., object]


MODEL_FUNCTIONS: dict[str, Predictor] = {
    "rbf": fit_predict_rbf,
    "fixed-fqk": fit_predict_fixed_fqk,
    "fixed-pqk": fit_predict_fixed_pqk,
    "ga-fqk": fit_predict_ga_fqk,
    "ga-pqk": fit_predict_ga_pqk,
}


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(float(row["accuracy"]))
    summary = []
    for (dataset, model), values in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "n": len(values),
                "mean_accuracy": mean(values),
                "std_accuracy": stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return summary


def _artifact_for(artifacts: list[CircuitArtifact], dataset: str, model: str) -> CircuitArtifact | None:
    if not model.startswith("ga-"):
        return None
    for artifact in artifacts:
        if artifact.dataset == dataset and artifact.kernel == model:
            return artifact
    raise ValueError(f"No artifact found for dataset={dataset}, model={model}")


def _feature_dimension_for(
    *,
    selected_artifact: CircuitArtifact | None,
    default_n_features: int,
    feature_dim_mode: str,
) -> int:
    if feature_dim_mode == "global" or selected_artifact is None:
        return default_n_features
    if feature_dim_mode == "circuit-parameters":
        return qpy_feature_dimension(selected_artifact.qpy_path)
    raise ValueError(f"Unsupported feature_dim_mode: {feature_dim_mode}")


def run_frozen_benchmark(
    *,
    manifest: str | Path,
    seeds: list[int],
    test_size: float,
    preprocess: str,
    models: list[str],
    output_dir: str | Path,
    n_features: int = 7,
    feature_dim_mode: str = "global",
    datasets: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    artifacts = load_manifest(manifest)
    rows: list[dict] = []
    dataset_names = sorted({artifact.dataset for artifact in artifacts})
    if datasets is not None:
        requested = set(datasets)
        dataset_names = [dataset for dataset in dataset_names if dataset in requested]
    for dataset_name in dataset_names:
        dataset_bundle = load_dataset(dataset_name)
        if dataset_bundle is None:
            continue
        for seed in seeds:
            for model in models:
                selected_artifact = _artifact_for(artifacts, dataset_name, model)
                model_n_features = _feature_dimension_for(
                    selected_artifact=selected_artifact,
                    default_n_features=n_features,
                    feature_dim_mode=feature_dim_mode,
                )
                split = make_holdout_split(
                    dataset_bundle.x,
                    dataset_bundle.y,
                    test_size=test_size,
                    n_features=model_n_features,
                    seed=seed,
                    preprocess=preprocess,
                )
                predictor = MODEL_FUNCTIONS[model]
                start = time.perf_counter()
                if selected_artifact is not None:
                    y_pred = predictor(split.x_train, split.y_train, split.x_test, qpy_path=selected_artifact.qpy_path)
                    circuit_path = str(selected_artifact.qpy_path)
                else:
                    y_pred = predictor(split.x_train, split.y_train, split.x_test)
                    circuit_path = ""
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model,
                        "seed": seed,
                        "accuracy": score_predictions(split.y_test, y_pred),
                        "runtime_seconds": elapsed,
                        "circuit_path": circuit_path,
                        "preprocess": preprocess,
                        "n_features": model_n_features,
                        "feature_dim_mode": feature_dim_mode,
                    }
                )
    summary = summarize_rows(rows)
    output = Path(output_dir)
    write_csv(output / "per_seed_results.csv", rows)
    write_csv(output / "summary.csv", summary)
    return rows, summary
