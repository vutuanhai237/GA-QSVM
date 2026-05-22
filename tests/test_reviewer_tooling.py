import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


def test_cloud_ga_dataset_runs_holdout_for_fresh_ga_artifact(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "uv_calls.txt"
    manifest_copy = tmp_path / "holdout_manifest.json"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_UV_CALLS"
if [[ "$*" == *"ga_qsvm.cli.train"* ]]; then
  artifact="7qubits_train_pqk_qsvm_fake"
  mkdir -p "$artifact"
  touch "$artifact/best_circuit.qpy"
  printf '{}\n' > "$artifact/metadata.json"
  printf '{}\n' > "$artifact/funcs.json"
elif [[ "$*" == *"ga_qsvm.cli.frozen_benchmark"* ]]; then
  manifest=""
  output_dir=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --manifest)
        manifest="$2"
        shift 2
        ;;
      --output-dir)
        output_dir="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  cp "$manifest" "$FAKE_HOLDOUT_MANIFEST_COPY"
  mkdir -p "$output_dir"
  printf 'ran\n' > "$output_dir/ran.txt"
else
  echo "unexpected uv invocation: $*" >&2
  exit 1
fi
"""
    )
    fake_uv.chmod(0o755)

    run_id = "pytest-holdout"
    run_dir = repo_root / "results" / "reviewer" / "ga_reruns" / f"digits_pqk_n7_{run_id}"
    log_paths = [
        repo_root / "logs" / "reviewer" / f"ga_digits_pqk_n7_{run_id}.log",
        repo_root / "logs" / "reviewer" / f"holdout_digits_pqk_n7_{run_id}.log",
    ]
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_UV_CALLS": str(calls_path),
        "FAKE_HOLDOUT_MANIFEST_COPY": str(manifest_copy),
        "RUN_ID": run_id,
        "QUBITS": "7",
        "NUM_CIRCUIT": "2",
        "NUM_GENERATION": "1",
        "KERNEL": "pqk",
        "HOLDOUT_SEEDS": "100 101",
    }

    try:
        subprocess.run(
            ["bash", "scripts/reviewer/cloud_ga_dataset.sh", "digits", "0"],
            cwd=repo_root,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        calls = calls_path.read_text()
        assert "ga_qsvm.cli.train" in calls
        assert "ga_qsvm.cli.frozen_benchmark" in calls
        assert "--models rbf fixed-pqk ga-pqk" in calls
        assert "--seeds 100 101" in calls
        assert "--wandb-project GA-QSVM-digits-pqk-holdout" in calls
        assert "--wandb-name holdout-digits-pqk-n7-pytest-holdout" in calls

        manifest = json.loads(manifest_copy.read_text())
        [circuit] = manifest["circuits"]
        assert circuit["dataset"] == "digits"
        assert circuit["kernel"] == "ga-pqk"
        assert circuit["path"].endswith("7qubits_train_pqk_qsvm_fake")
        assert (run_dir / f"holdout_digits_pqk_n7_{run_id}" / "ran.txt").is_file()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        for log_path in log_paths:
            log_path.unlink(missing_ok=True)


def test_cloud_ga_dataset_defaults_to_pqk_then_fqk(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "uv_calls.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_UV_CALLS"
"""
    )
    fake_uv.chmod(0o755)

    run_id = "pytest-kernels"
    run_dirs = [
        repo_root / "results" / "reviewer" / "ga_reruns" / f"digits_pqk_n7_{run_id}",
        repo_root / "results" / "reviewer" / "ga_reruns" / f"digits_fqk_n7_{run_id}",
    ]
    log_paths = [
        repo_root / "logs" / "reviewer" / f"ga_digits_pqk_n7_{run_id}.log",
        repo_root / "logs" / "reviewer" / f"ga_digits_fqk_n7_{run_id}.log",
    ]
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_UV_CALLS": str(calls_path),
        "RUN_ID": run_id,
        "QUBITS": "7",
        "NUM_CIRCUIT": "2",
        "NUM_GENERATION": "1",
        "RUN_HOLDOUT": "0",
    }
    env.pop("KERNEL", None)
    env.pop("KERNELS", None)

    try:
        subprocess.run(
            ["bash", "scripts/reviewer/cloud_ga_dataset.sh", "digits", "0"],
            cwd=repo_root,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        calls = calls_path.read_text().splitlines()
        assert len(calls) == 2
        assert "--kernel pqk" in calls[0]
        assert "--kernel fqk" in calls[1]
    finally:
        for run_dir in run_dirs:
            shutil.rmtree(run_dir, ignore_errors=True)
        for log_path in log_paths:
            log_path.unlink(missing_ok=True)


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


def test_frozen_benchmark_runs_each_dataset_model_seed_once(monkeypatch, tmp_path):
    from ga_qsvm.experiments import frozen_benchmark
    from ga_qsvm.experiments.artifacts import CircuitArtifact
    from ga_qsvm.experiments.datasets import DatasetBundle, PreparedSplit

    artifacts = [
        CircuitArtifact(
            id="digits-fqk",
            dataset="digits",
            kernel="ga-fqk",
            path=tmp_path / "fqk",
            qpy_path=tmp_path / "fqk" / "best_circuit.qpy",
            metadata_path=tmp_path / "fqk" / "metadata.json",
            funcs_path=tmp_path / "fqk" / "funcs.json",
            metadata={},
            funcs={},
        ),
        CircuitArtifact(
            id="digits-pqk",
            dataset="digits",
            kernel="ga-pqk",
            path=tmp_path / "pqk",
            qpy_path=tmp_path / "pqk" / "best_circuit.qpy",
            metadata_path=tmp_path / "pqk" / "metadata.json",
            funcs_path=tmp_path / "pqk" / "funcs.json",
            metadata={},
            funcs={},
        ),
    ]
    calls = []

    monkeypatch.setattr(frozen_benchmark, "load_manifest", lambda manifest: artifacts)
    monkeypatch.setattr(frozen_benchmark, "qpy_feature_dimension", lambda path: 7)
    monkeypatch.setattr(
        frozen_benchmark,
        "load_dataset",
        lambda dataset: DatasetBundle(dataset, np.arange(20).reshape(10, 2), np.array([0, 1] * 5)),
    )
    monkeypatch.setattr(
        frozen_benchmark,
        "make_holdout_split",
        lambda *args, **kwargs: PreparedSplit(
            x_train=np.zeros((4, 2)),
            x_test=np.zeros((2, 2)),
            y_train=np.array([0, 1, 0, 1]),
            y_test=np.array([0, 1]),
            raw_x_train=np.zeros((4, 2)),
            raw_x_test=np.zeros((2, 2)),
            seed=kwargs["seed"],
            preprocess=kwargs["preprocess"],
        ),
    )

    def fake_predictor(x_train, y_train, x_test, **kwargs):
        calls.append(kwargs.get("qpy_path", "baseline"))
        return np.array([0, 1])

    monkeypatch.setitem(frozen_benchmark.MODEL_FUNCTIONS, "rbf", fake_predictor)
    monkeypatch.setitem(frozen_benchmark.MODEL_FUNCTIONS, "ga-fqk", fake_predictor)

    rows, summary = frozen_benchmark.run_frozen_benchmark(
        manifest="unused.json",
        seeds=[100],
        test_size=0.3,
        preprocess="legacy",
        models=["rbf", "ga-fqk"],
        output_dir=tmp_path / "out",
    )

    assert len(rows) == 2
    assert len(summary) == 2
    assert calls == ["baseline", artifacts[0].qpy_path]


def test_frozen_benchmark_logs_wandb_tables_when_configured(monkeypatch, tmp_path):
    from ga_qsvm.experiments import frozen_benchmark
    from ga_qsvm.experiments.artifacts import CircuitArtifact
    from ga_qsvm.experiments.datasets import DatasetBundle, PreparedSplit

    artifact = CircuitArtifact(
        id="digits-pqk",
        dataset="digits",
        kernel="ga-pqk",
        path=tmp_path / "pqk",
        qpy_path=tmp_path / "pqk" / "best_circuit.qpy",
        metadata_path=tmp_path / "pqk" / "metadata.json",
        funcs_path=tmp_path / "pqk" / "funcs.json",
        metadata={},
        funcs={},
    )
    logs = []
    init_calls = []
    finish_calls = []

    class FakeTable:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    class FakeWandb:
        Table = FakeTable

        @staticmethod
        def init(**kwargs):
            init_calls.append(kwargs)
            return object()

        @staticmethod
        def log(payload):
            logs.append(payload)

        @staticmethod
        def finish():
            finish_calls.append(True)

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    monkeypatch.setattr(frozen_benchmark, "load_manifest", lambda manifest: [artifact])
    monkeypatch.setattr(frozen_benchmark, "qpy_feature_dimension", lambda path: 7)
    monkeypatch.setattr(
        frozen_benchmark,
        "load_dataset",
        lambda dataset: DatasetBundle(dataset, np.arange(20).reshape(10, 2), np.array([0, 1] * 5)),
    )
    monkeypatch.setattr(
        frozen_benchmark,
        "make_holdout_split",
        lambda *args, **kwargs: PreparedSplit(
            x_train=np.zeros((4, 2)),
            x_test=np.zeros((2, 2)),
            y_train=np.array([0, 1, 0, 1]),
            y_test=np.array([0, 1]),
            raw_x_train=np.zeros((4, 2)),
            raw_x_test=np.zeros((2, 2)),
            seed=kwargs["seed"],
            preprocess=kwargs["preprocess"],
        ),
    )
    monkeypatch.setitem(
        frozen_benchmark.MODEL_FUNCTIONS,
        "ga-pqk",
        lambda x_train, y_train, x_test, **kwargs: np.array([0, 1]),
    )

    frozen_benchmark.run_frozen_benchmark(
        manifest="fresh_manifest.json",
        seeds=[100],
        test_size=0.3,
        preprocess="paper",
        models=["ga-pqk"],
        output_dir=tmp_path / "out",
        wandb_config={"project": "holdout-project", "name": "holdout-run"},
    )

    assert init_calls == [{"project": "holdout-project", "name": "holdout-run"}]
    assert len(logs) == 1
    payload = logs[0]
    assert payload["benchmark/num_per_seed_rows"] == 1
    assert payload["benchmark/num_summary_rows"] == 1
    assert payload["benchmark/manifest"] == "fresh_manifest.json"
    assert "benchmark/per_seed_results" in payload
    assert "benchmark/summary" in payload
    assert payload["summary/digits/ga-pqk/mean_accuracy"] == 1.0
    assert finish_calls == [True]


def test_frozen_benchmark_can_filter_datasets(monkeypatch, tmp_path):
    from ga_qsvm.experiments import frozen_benchmark
    from ga_qsvm.experiments.artifacts import CircuitArtifact
    from ga_qsvm.experiments.datasets import DatasetBundle, PreparedSplit

    artifacts = []
    for dataset in ["digits", "wine"]:
        artifacts.append(
            CircuitArtifact(
                id=f"{dataset}-fqk",
                dataset=dataset,
                kernel="ga-fqk",
                path=tmp_path / dataset,
                qpy_path=tmp_path / dataset / "best_circuit.qpy",
                metadata_path=tmp_path / dataset / "metadata.json",
                funcs_path=tmp_path / dataset / "funcs.json",
                metadata={},
                funcs={},
            )
        )
    loaded_datasets = []
    monkeypatch.setattr(frozen_benchmark, "load_manifest", lambda manifest: artifacts)
    monkeypatch.setattr(frozen_benchmark, "qpy_feature_dimension", lambda path: 7)
    monkeypatch.setattr(
        frozen_benchmark,
        "load_dataset",
        lambda dataset: loaded_datasets.append(dataset)
        or DatasetBundle(dataset, np.arange(20).reshape(10, 2), np.array([0, 1] * 5)),
    )
    monkeypatch.setattr(
        frozen_benchmark,
        "make_holdout_split",
        lambda *args, **kwargs: PreparedSplit(
            x_train=np.zeros((4, 2)),
            x_test=np.zeros((2, 2)),
            y_train=np.array([0, 1, 0, 1]),
            y_test=np.array([0, 1]),
            raw_x_train=np.zeros((4, 2)),
            raw_x_test=np.zeros((2, 2)),
            seed=kwargs["seed"],
            preprocess=kwargs["preprocess"],
        ),
    )
    monkeypatch.setitem(
        frozen_benchmark.MODEL_FUNCTIONS,
        "ga-fqk",
        lambda x_train, y_train, x_test, **kwargs: np.array([0, 1]),
    )

    rows, _ = frozen_benchmark.run_frozen_benchmark(
        manifest="unused.json",
        seeds=[100],
        test_size=0.3,
        preprocess="legacy",
        models=["ga-fqk"],
        output_dir=tmp_path / "out",
        datasets=["wine"],
    )

    assert loaded_datasets == ["wine"]
    assert [row["dataset"] for row in rows] == ["wine"]


def test_frozen_benchmark_can_use_circuit_parameter_feature_dimension(monkeypatch, tmp_path):
    from ga_qsvm.experiments import frozen_benchmark
    from ga_qsvm.experiments.artifacts import CircuitArtifact
    from ga_qsvm.experiments.datasets import DatasetBundle, PreparedSplit

    artifact = CircuitArtifact(
        id="wine-fqk",
        dataset="wine",
        kernel="ga-fqk",
        path=tmp_path / "fqk",
        qpy_path=tmp_path / "fqk" / "best_circuit.qpy",
        metadata_path=tmp_path / "fqk" / "metadata.json",
        funcs_path=tmp_path / "fqk" / "funcs.json",
        metadata={},
        funcs={},
    )
    requested_features = []
    monkeypatch.setattr(frozen_benchmark, "load_manifest", lambda manifest: [artifact])
    monkeypatch.setattr(
        frozen_benchmark,
        "load_dataset",
        lambda dataset: DatasetBundle(dataset, np.arange(20).reshape(10, 2), np.array([0, 1] * 5)),
    )
    monkeypatch.setattr(frozen_benchmark, "qpy_feature_dimension", lambda path: 9)

    def fake_split(*args, **kwargs):
        requested_features.append(kwargs["n_features"])
        return PreparedSplit(
            x_train=np.zeros((4, kwargs["n_features"])),
            x_test=np.zeros((2, kwargs["n_features"])),
            y_train=np.array([0, 1, 0, 1]),
            y_test=np.array([0, 1]),
            raw_x_train=np.zeros((4, 2)),
            raw_x_test=np.zeros((2, 2)),
            seed=kwargs["seed"],
            preprocess=kwargs["preprocess"],
        )

    monkeypatch.setattr(frozen_benchmark, "make_holdout_split", fake_split)
    monkeypatch.setitem(
        frozen_benchmark.MODEL_FUNCTIONS,
        "ga-fqk",
        lambda x_train, y_train, x_test, **kwargs: np.array([0, 1]),
    )

    rows, _ = frozen_benchmark.run_frozen_benchmark(
        manifest="unused.json",
        seeds=[100],
        test_size=0.3,
        preprocess="legacy",
        models=["ga-fqk"],
        output_dir=tmp_path / "out",
        n_features=7,
        feature_dim_mode="circuit-parameters",
    )

    assert requested_features == [9]
    assert rows[0]["n_features"] == 9


def test_frozen_benchmark_rejects_global_feature_dimension_mismatch(monkeypatch, tmp_path):
    from ga_qsvm.experiments import frozen_benchmark
    from ga_qsvm.experiments.artifacts import CircuitArtifact
    from ga_qsvm.experiments.datasets import DatasetBundle

    artifact = CircuitArtifact(
        id="wine-ga-fqk-n7",
        dataset="wine",
        kernel="ga-fqk",
        path=tmp_path / "fqk",
        qpy_path=tmp_path / "fqk" / "best_circuit.qpy",
        metadata_path=tmp_path / "fqk" / "metadata.json",
        funcs_path=tmp_path / "fqk" / "funcs.json",
        metadata={},
        funcs={},
    )
    monkeypatch.setattr(frozen_benchmark, "load_manifest", lambda manifest: [artifact])
    monkeypatch.setattr(
        frozen_benchmark,
        "load_dataset",
        lambda dataset: DatasetBundle(dataset, np.arange(20).reshape(10, 2), np.array([0, 1] * 5)),
    )
    monkeypatch.setattr(frozen_benchmark, "qpy_feature_dimension", lambda path: 9)

    def fail_if_split(*args, **kwargs):
        raise AssertionError("benchmark split should not be prepared for invalid artifact")

    monkeypatch.setattr(frozen_benchmark, "make_holdout_split", fail_if_split)

    with pytest.raises(ValueError, match="wine-ga-fqk-n7.*9.*expected 7"):
        frozen_benchmark.run_frozen_benchmark(
            manifest="unused.json",
            seeds=[100],
            test_size=0.3,
            preprocess="legacy",
            models=["ga-fqk"],
            output_dir=tmp_path / "out",
            n_features=7,
            feature_dim_mode="global",
        )


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


def test_default_figure6_artifact_parameter_audit_flags_only_wine_fqk():
    from ga_qsvm.experiments.artifacts import load_manifest
    from ga_qsvm.experiments.kernels import summarize_qpy_circuit

    artifacts = load_manifest("configs/reviewer/main_figure6_n7_manifest.json")
    audit = {artifact.id: summarize_qpy_circuit(artifact.qpy_path) for artifact in artifacts}

    invalid = {
        artifact_id: summary.num_parameters
        for artifact_id, summary in audit.items()
        if summary.num_parameters != 7
    }
    assert invalid == {"wine-ga-fqk-n7": 9}
    assert {summary.constant_rotations for summary in audit.values()} == {0}


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
            "feature_dim_mode": "global",
            "datasets": None,
            "wandb_config": None,
        }
    ]


def test_frozen_cli_passes_wandb_config_to_runner(monkeypatch):
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
            "101",
            "--models",
            "ga-pqk",
            "--datasets",
            "digits",
            "--output-dir",
            "out",
            "--wandb-project",
            "project",
            "--wandb-name",
            "run-name",
            "--wandb-group",
            "group",
        ]
    )

    assert calls[0]["wandb_config"] == {
        "project": "project",
        "name": "run-name",
        "group": "group",
        "job_type": "holdout-benchmark",
        "config": {
            "manifest": "manifest.json",
            "seeds": [100, 101],
            "test_size": 0.3,
            "preprocess": "legacy",
            "models": ["ga-pqk"],
            "datasets": ["digits"],
            "output_dir": "out",
            "n_features": 7,
            "feature_dim_mode": "global",
        },
    }
