from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Callable

import numpy as np

from ga_qsvm.experiments.artifacts import write_csv
from ga_qsvm.experiments.datasets import load_dataset, make_holdout_split
from ga_qsvm.experiments.kernels import (
    fit_predict_random_fqk,
    fit_predict_random_pqk,
    score_predictions,
)
from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.generator import by_num_rotations_and_cnot

Predictor = Callable[..., object]


RANDOM_MODEL_FUNCTIONS: dict[str, Predictor] = {
    "random-fqk": fit_predict_random_fqk,
    "random-pqk": fit_predict_random_pqk,
}


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["model"],
                int(row["qubits"]),
                int(row["random_budget"]),
            )
        ].append(row)
    summary = []
    for (dataset, model, qubits, random_budget), group in sorted(grouped.items()):
        accuracies = [float(row["accuracy"]) for row in group]
        runtimes = [float(row["runtime_seconds"]) for row in group]
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "qubits": qubits,
                "random_budget": random_budget,
                "n": len(group),
                "mean_accuracy": mean(accuracies),
                "std_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "mean_runtime_seconds": mean(runtimes),
                "std_runtime_seconds": stdev(runtimes) if len(runtimes) > 1 else 0.0,
            }
        )
    return summary


def _metadata_for(
    *,
    n_features: int,
    depth_multiplier: int,
    num_cnot_multiplier: int,
) -> MetadataSynthesis:
    return MetadataSynthesis(
        num_qubits=n_features,
        num_cnot=num_cnot_multiplier * n_features,
        depth=depth_multiplier * n_features,
        num_circuit=1,
        num_generation=1,
        prob_mutate=0.0,
    )


def _seed_candidate(seed: int, candidate_index: int) -> None:
    candidate_seed = seed * 100_000 + candidate_index
    random.seed(candidate_seed)
    np.random.seed(candidate_seed % (2**32 - 1))


def _evaluate_random_candidates(
    *,
    predictor: Predictor,
    split,
    metadata: MetadataSynthesis,
    random_budget: int,
    seed: int,
) -> tuple[float, int]:
    best_accuracy = -1.0
    best_candidate_index = -1
    for candidate_index in range(random_budget):
        _seed_candidate(seed, candidate_index)
        circuit = by_num_rotations_and_cnot(metadata)
        y_pred = predictor(
            split.x_train,
            split.y_train,
            split.x_test,
            circuit=circuit,
        )
        accuracy = score_predictions(split.y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_candidate_index = candidate_index
    return best_accuracy, best_candidate_index


def run_random_search_benchmark(
    *,
    datasets: list[str],
    qubits: list[int],
    seeds: list[int],
    test_size: float,
    preprocess: str,
    models: list[str],
    output_dir: str | Path,
    random_budget: int,
    depth_multiplier: int = 5,
    num_cnot_multiplier: int = 2,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    for dataset_name in datasets:
        dataset_bundle = load_dataset(dataset_name)
        if dataset_bundle is None:
            continue
        for n_features in qubits:
            metadata = _metadata_for(
                n_features=n_features,
                depth_multiplier=depth_multiplier,
                num_cnot_multiplier=num_cnot_multiplier,
            )
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
                    predictor = RANDOM_MODEL_FUNCTIONS[model]
                    start = time.perf_counter()
                    best_accuracy, best_candidate_index = _evaluate_random_candidates(
                        predictor=predictor,
                        split=split,
                        metadata=metadata,
                        random_budget=random_budget,
                        seed=seed,
                    )
                    elapsed = time.perf_counter() - start
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "qubits": n_features,
                            "seed": seed,
                            "accuracy": best_accuracy,
                            "runtime_seconds": elapsed,
                            "preprocess": preprocess,
                            "n_features": n_features,
                            "random_budget": random_budget,
                            "best_candidate_index": best_candidate_index,
                            "depth": metadata.depth,
                            "num_cnot": metadata.num_cnot,
                        }
                    )
    summary = summarize_rows(rows)
    output = Path(output_dir)
    write_csv(output / "per_seed_results.csv", rows)
    write_csv(output / "summary.csv", summary)
    return rows, summary
