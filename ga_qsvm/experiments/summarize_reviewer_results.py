from __future__ import annotations

import csv
from pathlib import Path

from ga_qsvm.experiments.artifacts import write_csv


def summarize_reviewer_results(*, inputs: list[str | Path], output_dir: str | Path) -> list[dict]:
    rows: list[dict] = []
    for input_path in inputs:
        root = Path(input_path)
        for csv_path in root.rglob("summary.csv"):
            with csv_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append({"source": str(csv_path), **row})
    output = Path(output_dir)
    write_csv(output / "reviewer_summary.csv", rows)
    return rows
