from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ga_qsvm.experiments.artifacts import load_manifest, write_csv
from ga_qsvm.experiments.datasets import load_dataset, prepare_split
from ga_qsvm.experiments.frozen_benchmark import MODEL_FUNCTIONS, _artifact_for, summarize_rows
from ga_qsvm.experiments.kernels import score_predictions


def iter_stratified_folds(y: np.ndarray, *, folds: int, seed: int):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    placeholder_x = np.zeros((len(y), 1))
    yield from splitter.split(placeholder_x, y)


def run_kfold_benchmark(
    *,
    manifest: str | Path,
    seeds: list[int],
    folds: int,
    preprocess: str,
    models: list[str],
    output_dir: str | Path,
    n_features: int = 7,
    max_folds: int | None = None,
) -> tuple[list[dict], list[dict]]:
    artifacts = load_manifest(manifest)
    rows: list[dict] = []
    seen_datasets = sorted({artifact.dataset for artifact in artifacts})
    for dataset_name in seen_datasets:
        bundle = load_dataset(dataset_name)
        if bundle is None:
            continue
        for seed in seeds:
            for fold_index, (train_idx, test_idx) in enumerate(iter_stratified_folds(bundle.y, folds=folds, seed=seed)):
                if max_folds is not None and fold_index >= max_folds:
                    break
                split = prepare_split(
                    bundle.x[train_idx],
                    bundle.x[test_idx],
                    bundle.y[train_idx],
                    bundle.y[test_idx],
                    n_features=n_features,
                    seed=seed,
                    preprocess=preprocess,
                )
                for model in models:
                    selected_artifact = _artifact_for(artifacts, dataset_name, model)
                    predictor = MODEL_FUNCTIONS[model]
                    start = time.perf_counter()
                    if selected_artifact is not None:
                        y_pred = predictor(split.x_train, split.y_train, split.x_test, qpy_path=selected_artifact.qpy_path)
                        circuit_path = str(selected_artifact.qpy_path)
                    else:
                        y_pred = predictor(split.x_train, split.y_train, split.x_test)
                        circuit_path = ""
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "seed": seed,
                            "fold": fold_index,
                            "accuracy": score_predictions(split.y_test, y_pred),
                            "runtime_seconds": time.perf_counter() - start,
                            "circuit_path": circuit_path,
                            "preprocess": preprocess,
                        }
                    )
    summary = summarize_rows(rows)
    output = Path(output_dir)
    write_csv(output / "per_fold_results.csv", rows)
    write_csv(output / "summary.csv", summary)
    return rows, summary
