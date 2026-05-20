import csv
import json

import numpy as np
import pytest


def test_paper_holdout_split_fits_preprocessing_on_train_only(monkeypatch):
    from ga_qsvm.experiments import datasets

    calls = []

    class RecordingScaler:
        def fit_transform(self, x):
            calls.append(("scaler_fit", x.copy()))
            return x + 10

        def transform(self, x):
            calls.append(("scaler_transform", x.copy()))
            return x + 20

    class RecordingPCA:
        def __init__(self, n_components, random_state=None):
            self.n_components = n_components
            self.random_state = random_state

        def fit_transform(self, x):
            calls.append(("pca_fit", x.copy()))
            return x[:, : self.n_components]

        def transform(self, x):
            calls.append(("pca_transform", x.copy()))
            return x[:, : self.n_components]

    monkeypatch.setattr(datasets, "MinMaxScaler", RecordingScaler)
    monkeypatch.setattr(datasets, "PCA", RecordingPCA)

    x = np.arange(40, dtype=float).reshape(10, 4)
    y = np.array([0, 1] * 5)
    split = datasets.make_holdout_split(
        x,
        y,
        test_size=0.3,
        n_features=2,
        seed=7,
        preprocess="paper",
    )

    assert split.x_train.shape[1] == 2
    assert split.x_test.shape[1] == 2
    assert [name for name, _ in calls] == [
        "scaler_fit",
        "scaler_transform",
        "pca_fit",
        "pca_transform",
    ]
    np.testing.assert_array_equal(calls[2][1], split.raw_x_train + 10)
    np.testing.assert_array_equal(calls[3][1], split.raw_x_test + 20)


def test_manifest_validation_rejects_missing_required_artifacts(tmp_path):
    from ga_qsvm.experiments.artifacts import ManifestValidationError, load_manifest

    missing_dir = tmp_path / "missing-artifact"
    missing_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "circuits": [
                    {
                        "id": "digits-fqk",
                        "dataset": "digits",
                        "kernel": "ga-fqk",
                        "path": str(missing_dir),
                    }
                ]
            }
        )
    )

    with pytest.raises(ManifestValidationError, match="best_circuit.qpy"):
        load_manifest(manifest)


def test_frozen_benchmark_aggregates_mean_and_sample_std():
    from ga_qsvm.experiments.frozen_benchmark import summarize_rows

    summary = summarize_rows(
        [
            {"dataset": "digits", "model": "rbf", "accuracy": 0.5},
            {"dataset": "digits", "model": "rbf", "accuracy": 0.7},
            {"dataset": "wine", "model": "rbf", "accuracy": 1.0},
        ]
    )

    digits = next(row for row in summary if row["dataset"] == "digits")
    assert digits["model"] == "rbf"
    assert digits["n"] == 2
    assert digits["mean_accuracy"] == pytest.approx(0.6)
    assert digits["std_accuracy"] == pytest.approx(0.1414213562)


def test_kfold_runner_creates_stratified_non_overlapping_folds():
    from ga_qsvm.experiments.kfold_benchmark import iter_stratified_folds

    y = np.array([0, 1] * 15)
    folds = list(iter_stratified_folds(y, folds=3, seed=100))

    assert len(folds) == 3
    seen_test = set()
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert set(np.unique(y[test_idx])) == {0, 1}
        seen_test.update(test_idx.tolist())
    assert seen_test == set(range(len(y)))


def test_pca_analysis_reports_threshold_component_counts():
    from ga_qsvm.experiments.pca_analysis import compute_pca_summary

    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    summary = compute_pca_summary("toy", x, thresholds=[0.5, 0.95])

    assert summary["dataset"] == "toy"
    assert "components_for_50" in summary
    assert "components_for_95" in summary
    assert summary["n_features"] == 3


def test_hyperparameter_summary_records_missing_sweeps(tmp_path):
    from ga_qsvm.experiments.hyperparameter_summary import summarize_hyperparameter_sources

    config = tmp_path / "sources.json"
    config.write_text(
        json.dumps(
            {
                "sweeps": [
                    {
                        "name": "missing-depth",
                        "variable": "depth",
                        "value": 35,
                        "path": str(tmp_path / "missing.csv"),
                    }
                ]
            }
        )
    )
    output_dir = tmp_path / "out"

    result = summarize_hyperparameter_sources(config, output_dir)

    assert result.missing[0]["name"] == "missing-depth"
    missing_csv = output_dir / "missing_hyperparameter_sweeps.csv"
    with missing_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["path"].endswith("missing.csv")


def test_transfer_manifest_keeps_source_and_target_datasets(tmp_path):
    from ga_qsvm.experiments.transfer_benchmark import load_transfer_manifest

    circuit_dir = tmp_path / "digits-source"
    circuit_dir.mkdir()
    (circuit_dir / "best_circuit.qpy").write_bytes(b"qpy")
    (circuit_dir / "metadata.json").write_text("{}")
    (circuit_dir / "funcs.json").write_text("{}")
    manifest = tmp_path / "transfer.json"
    manifest.write_text(
        json.dumps(
            {
                "transfers": [
                    {
                        "id": "digits-to-wine-fqk",
                        "source_dataset": "digits",
                        "target_dataset": "wine",
                        "kernel": "ga-fqk",
                        "path": str(circuit_dir),
                    }
                ]
            }
        )
    )

    entries = load_transfer_manifest(manifest)

    assert entries[0].source_dataset == "digits"
    assert entries[0].target_dataset == "wine"


def test_reviewer_cli_parsers_accept_expected_arguments():
    from ga_qsvm.cli.frozen_benchmark import build_parser as frozen_parser
    from ga_qsvm.cli.kfold_benchmark import build_parser as kfold_parser
    from ga_qsvm.cli.pca_analysis import build_parser as pca_parser

    frozen = frozen_parser().parse_args(
        [
            "--manifest",
            "manifest.json",
            "--seeds",
            "100",
            "101",
            "--models",
            "rbf",
            "ga-fqk",
            "--output-dir",
            "out",
        ]
    )
    assert frozen.seeds == [100, 101]
    assert frozen.models == ["rbf", "ga-fqk"]

    kfold = kfold_parser().parse_args(
        ["--manifest", "manifest.json", "--folds", "5", "--max-folds", "1", "--output-dir", "out"]
    )
    assert kfold.folds == 5
    assert kfold.max_folds == 1

    pca = pca_parser().parse_args(
        ["--datasets", "digits", "fashion", "--thresholds", "0.90", "0.95", "--output-dir", "out"]
    )
    assert pca.datasets == ["digits", "fashion"]
    assert pca.thresholds == [0.9, 0.95]


def test_default_figure6_manifest_validates_real_artifacts():
    from ga_qsvm.experiments.artifacts import load_manifest

    artifacts = load_manifest("configs/reviewer/main_figure6_n7_manifest.json")

    assert len(artifacts) == 6
    assert {(artifact.dataset, artifact.kernel) for artifact in artifacts} == {
        ("digits", "ga-fqk"),
        ("digits", "ga-pqk"),
        ("wine", "ga-fqk"),
        ("wine", "ga-pqk"),
        ("cancer", "ga-fqk"),
        ("cancer", "ga-pqk"),
    }


def test_frozen_cli_dispatches_to_runner(monkeypatch):
    from ga_qsvm.cli import frozen_benchmark

    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(frozen_benchmark, "run_frozen_benchmark", fake_runner)

    frozen_benchmark.main(
        [
            "--manifest",
            "manifest.json",
            "--seeds",
            "100",
            "--models",
            "rbf",
            "--output-dir",
            "out",
        ]
    )

    assert calls == [
        {
            "manifest": "manifest.json",
            "seeds": [100],
            "test_size": 0.3,
            "preprocess": "legacy",
            "models": ["rbf"],
            "output_dir": "out",
            "n_features": 7,
        }
    ]
