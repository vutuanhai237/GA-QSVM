import csv
import json
import zipfile


def test_hypothetical_patience_stop_reports_lost_late_improvement():
    from ga_qsvm.experiments.early_stop_analysis import summarize_series

    values = [0.50, 0.60, 0.60, 0.60, 0.60, 0.60, 0.61]

    summary = summarize_series(
        generation_best=values,
        patience=3,
        checkpoints=[3, 7],
        tolerance=0.0,
    )

    assert summary["early_stop_generation"] == 6
    assert summary["early_stop_best"] == 0.60
    assert summary["final_best"] == 0.61
    assert summary["delta_final_minus_early_stop"] == 0.01
    assert summary["safe_to_stop"] is False
    assert summary["checkpoint_best_3"] == 0.60
    assert summary["improvement_after_3"] == 0.01


def test_hypothetical_patience_stop_marks_plateau_safe():
    from ga_qsvm.experiments.early_stop_analysis import summarize_series

    values = [0.50, 0.60, 0.60, 0.60, 0.60, 0.60]

    summary = summarize_series(
        generation_best=values,
        patience=3,
        checkpoints=[3, 6],
        tolerance=0.0,
    )

    assert summary["early_stop_generation"] == 6
    assert summary["early_stop_best"] == summary["final_best"] == 0.60
    assert summary["safe_to_stop"] is True
    assert summary["first_final_best_generation"] == 2


def test_analyze_metadata_roots_writes_csv_and_markdown(tmp_path):
    from ga_qsvm.experiments.early_stop_analysis import analyze_metadata_roots

    artifact = tmp_path / "results" / "wine_fqk_n3" / "3qubits_train_fqk_qsvm_2026-05-24"
    artifact.mkdir(parents=True)
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "num_qubits": 3,
                "depth": 15,
                "num_cnot": 6,
                "num_circuit": 20,
                    "num_generation": 7,
                    "current_generation": 7,
                    "best_fitnesss": [0.50, 0.60, 0.60, 0.60, 0.60, 0.60, 0.61],
            }
        )
    )

    output = tmp_path / "analysis"
    result = analyze_metadata_roots(
        roots=[tmp_path / "results"],
        output_dir=output,
        patience=3,
        checkpoints=[3, 7],
        tolerance=0.0,
    )

    assert len(result.rows) == 1
    assert (output / "early_stop_analysis.csv").is_file()
    assert (output / "early_stop_analysis.md").is_file()

    with (output / "early_stop_analysis.csv").open(newline="") as handle:
        [row] = list(csv.DictReader(handle))
    assert row["dataset"] == "wine"
    assert row["kernel"] == "fqk"
    assert row["num_qubits"] == "3"
    assert row["safe_to_stop"] == "False"


def test_analyze_metadata_roots_reads_zip_files(tmp_path):
    from ga_qsvm.experiments.early_stop_analysis import analyze_metadata_roots

    archive = tmp_path / "result (1).zip"
    metadata_name = (
        "results/reviewer/ga_reruns/digits_pqk_n3/"
        "3qubits_train_pqk_qsvm_2026-05-24/metadata.json"
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            metadata_name,
            json.dumps(
                {
                    "num_qubits": 3,
                    "depth": 30,
                    "num_cnot": 30,
                    "num_circuit": 20,
                    "num_generation": 6,
                    "current_generation": 6,
                    "best_fitnesss": [0.50, 0.60, 0.60, 0.60, 0.60, 0.60],
                }
            ),
        )

    result = analyze_metadata_roots(
        roots=[archive],
        output_dir=tmp_path / "analysis",
        patience=3,
        checkpoints=[3, 6],
        tolerance=0.0,
    )

    assert len(result.rows) == 1
    [row] = result.rows
    assert row["path"] == f"{archive}!{metadata_name}"
    assert row["dataset"] == "digits"
    assert row["kernel"] == "pqk"
    assert row["safe_to_stop"] is True
