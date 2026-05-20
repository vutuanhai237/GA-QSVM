from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ga_qsvm.experiments.artifacts import read_json, write_csv


@dataclass(frozen=True)
class HyperparameterSummaryResult:
    summaries: list[dict[str, Any]]
    missing: list[dict[str, Any]]


def _read_numeric_series(path: Path) -> list[float]:
    values: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for cell in row:
                try:
                    values.append(float(cell))
                    break
                except ValueError:
                    continue
    return values


def summarize_hyperparameter_sources(config_path: str | Path, output_dir: str | Path) -> HyperparameterSummaryResult:
    config = read_json(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for sweep in config.get("sweeps", []):
        path = Path(sweep["path"]).expanduser()
        if not path.is_file():
            missing.append({**sweep, "path": str(path), "reason": "missing"})
            continue
        values = _read_numeric_series(path)
        if not values:
            missing.append({**sweep, "path": str(path), "reason": "no_numeric_series"})
            continue
        best = max(values)
        generation = values.index(best)
        summaries.append(
            {
                "name": sweep.get("name", path.stem),
                "variable": sweep.get("variable", ""),
                "value": sweep.get("value", ""),
                "path": str(path),
                "final_best_fitness": values[-1],
                "best_fitness": best,
                "generation_reached_best": generation,
            }
        )
        plt.plot(range(len(values)), values, label=sweep.get("name", path.stem))
    if summaries:
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "figure4_hyperparameter_convergence.png", dpi=200)
        plt.close()
    write_csv(output / "hyperparameter_summary.csv", summaries)
    write_csv(output / "missing_hyperparameter_sweeps.csv", missing)
    return HyperparameterSummaryResult(summaries=summaries, missing=missing)
