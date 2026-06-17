from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Callable

from ga_qsvm.experiments.artifacts import write_csv
from ga_qsvm.experiments.datasets import load_dataset, make_holdout_split
from ga_qsvm.experiments.kernels import (
    fit_predict_predefined_fqk,
    fit_predict_predefined_pqk,
    score_predictions,
)

Predictor = Callable[..., object]


PREDEFINED_MODEL_FUNCTIONS: dict[str, Predictor] = {
    "efficient-su2-fqk": fit_predict_predefined_fqk,
    "efficient-su2-pqk": fit_predict_predefined_pqk,
    "two-local-fqk": fit_predict_predefined_fqk,
    "two-local-pqk": fit_predict_predefined_pqk,
}


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], int(row["qubits"]))].append(row)
    summary = []
    for (dataset, model, qubits), group in sorted(grouped.items()):
        accuracies = [float(row["accuracy"]) for row in group]
        runtimes = [float(row["runtime_seconds"]) for row in group]
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "qubits": qubits,
                "n": len(group),
                "mean_accuracy": mean(accuracies),
                "std_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "mean_runtime_seconds": mean(runtimes),
                "std_runtime_seconds": stdev(runtimes) if len(runtimes) > 1 else 0.0,
            }
        )
    return summary


def _ansatz_for_model(model: str) -> str:
    if model.startswith("efficient-su2-"):
        return "efficient-su2"
    if model.startswith("two-local-"):
        return "two-local"
    raise ValueError(f"Unsupported predefined model: {model}")


def run_predefined_benchmark(
    *,
    datasets: list[str],
    qubits: list[int],
    seeds: list[int],
    test_size: float,
    preprocess: str,
    models: list[str],
    output_dir: str | Path,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    for dataset_name in datasets:
        dataset_bundle = load_dataset(dataset_name)
        if dataset_bundle is None:
            continue
        for n_features in qubits:
            for seed in seeds:
                split = make_holdout_split(
                    dataset_bundle.x,
                    dataset_bundle.y,
                    test_size=test_size,
                    n_features=n_features,
                    seed=seed,
                    preprocess=preprocess,
                )
                for model in models:
                    ansatz = _ansatz_for_model(model)
                    predictor = PREDEFINED_MODEL_FUNCTIONS[model]
                    print(
                        (
                            "Start predefined benchmark: "
                            f"dataset={dataset_name} model={model} "
                            f"qubits={n_features} seed={seed}"
                        ),
                        flush=True,
                    )
                    start = time.perf_counter()
                    y_pred = predictor(
                        split.x_train,
                        split.y_train,
                        split.x_test,
                        ansatz=ansatz,
                        n_features=n_features,
                    )
                    elapsed = time.perf_counter() - start
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "qubits": n_features,
                            "seed": seed,
                            "accuracy": score_predictions(split.y_test, y_pred),
                            "runtime_seconds": elapsed,
                            "preprocess": preprocess,
                            "n_features": n_features,
                        }
                    )
                    print(
                        (
                            "Done predefined benchmark: "
                            f"dataset={dataset_name} model={model} "
                            f"qubits={n_features} seed={seed} "
                            f"accuracy={rows[-1]['accuracy']:.6f} "
                            f"runtime_seconds={elapsed:.2f}"
                        ),
                        flush=True,
                    )
    summary = summarize_rows(rows)
    output = Path(output_dir)
    write_csv(output / "per_seed_results.csv", rows)
    write_csv(output / "summary.csv", summary)
    return rows, summary
