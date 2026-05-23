from __future__ import annotations

import math
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ga_qsvm.experiments.artifacts import read_json, write_csv


@dataclass(frozen=True)
class EarlyStopAnalysisResult:
    rows: list[dict[str, Any]]
    output_dir: Path


def _generation_best_from_metadata(metadata: dict[str, Any]) -> list[float]:
    best_fitnesss = metadata.get("best_fitnesss")
    if isinstance(best_fitnesss, list) and best_fitnesss:
        return [float(value) for value in best_fitnesss]

    fitnessss = metadata.get("fitnessss")
    if isinstance(fitnessss, list) and fitnessss:
        return [float(max(generation)) for generation in fitnessss if generation]

    return []


def _cumulative_best(values: list[float]) -> list[float]:
    cumulative: list[float] = []
    best = -math.inf
    for value in values:
        best = max(best, value)
        cumulative.append(best)
    return cumulative


def _first_generation_reaching(values: list[float], target: float, tolerance: float) -> int:
    for index, value in enumerate(values, start=1):
        if value >= target - tolerance:
            return index
    return len(values)


def _hypothetical_stop_generation(values: list[float], patience: int, min_delta: float = 0.0) -> int:
    best = -math.inf
    last_improvement_generation = 0
    for generation, value in enumerate(values, start=1):
        if value > best + min_delta:
            best = value
            last_improvement_generation = generation
        if generation - last_improvement_generation > patience:
            return generation
    return len(values)


def summarize_series(
    generation_best: list[float],
    patience: int,
    checkpoints: Iterable[int],
    tolerance: float,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    if not generation_best:
        return {
            "generations_recorded": 0,
            "final_best": "",
            "first_final_best_generation": "",
            "early_stop_generation": "",
            "early_stop_best": "",
            "delta_final_minus_early_stop": "",
            "safe_to_stop": "",
        }

    cumulative = _cumulative_best(generation_best)
    final_best = cumulative[-1]
    stop_generation = _hypothetical_stop_generation(generation_best, patience, min_delta=min_delta)
    early_stop_best = cumulative[stop_generation - 1]
    delta = final_best - early_stop_best
    row: dict[str, Any] = {
        "generations_recorded": len(generation_best),
        "final_best": round(final_best, 12),
        "first_final_best_generation": _first_generation_reaching(cumulative, final_best, tolerance),
        "patience": patience,
        "early_stop_generation": stop_generation,
        "early_stop_best": round(early_stop_best, 12),
        "delta_final_minus_early_stop": round(delta, 12),
        "safe_to_stop": delta <= tolerance,
    }

    for checkpoint in checkpoints:
        effective_checkpoint = min(int(checkpoint), len(cumulative))
        checkpoint_best = cumulative[effective_checkpoint - 1]
        row[f"checkpoint_best_{checkpoint}"] = round(checkpoint_best, 12)
        row[f"improvement_after_{checkpoint}"] = round(final_best - checkpoint_best, 12)
    return row


def _infer_dataset_kernel(path: Path) -> tuple[str, str]:
    text = " ".join(part.lower() for part in path.parts)
    dataset = ""
    for candidate in ("digits", "wine", "cancer", "fashion"):
        if candidate in text:
            dataset = candidate
            break
    kernel = ""
    if "fqk" in text:
        kernel = "fqk"
    elif "pqk" in text:
        kernel = "pqk"
    return dataset, kernel


def _metadata_row(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    dataset, kernel = _infer_dataset_kernel(path)
    return {
        "path": str(path),
        "artifact": path.parent.name,
        "dataset": dataset,
        "kernel": kernel,
        "num_qubits": metadata.get("num_qubits", ""),
        "depth": metadata.get("depth", ""),
        "num_cnot": metadata.get("num_cnot", ""),
        "num_circuit": metadata.get("num_circuit", ""),
        "configured_generations": metadata.get("num_generation", ""),
        "current_generation": metadata.get("current_generation", ""),
    }


def _metadata_row_from_zip(archive: Path, member: str, metadata: dict[str, Any]) -> dict[str, Any]:
    dataset, kernel = _infer_dataset_kernel(Path(member))
    return {
        "path": f"{archive}!{member}",
        "artifact": Path(member).parent.name,
        "dataset": dataset,
        "kernel": kernel,
        "num_qubits": metadata.get("num_qubits", ""),
        "depth": metadata.get("depth", ""),
        "num_cnot": metadata.get("num_cnot", ""),
        "num_circuit": metadata.get("num_circuit", ""),
        "configured_generations": metadata.get("num_generation", ""),
        "current_generation": metadata.get("current_generation", ""),
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "kernel",
        "num_qubits",
        "generations_recorded",
        "final_best",
        "early_stop_generation",
        "early_stop_best",
        "delta_final_minus_early_stop",
        "safe_to_stop",
        "path",
    ]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def analyze_metadata_roots(
    roots: Iterable[str | Path],
    output_dir: str | Path,
    patience: int,
    checkpoints: Iterable[int],
    tolerance: float,
    min_delta: float = 0.0,
) -> EarlyStopAnalysisResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for root in roots:
        root_path = Path(root).expanduser()
        if root_path.is_file() and root_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(root_path) as archive:
                for member in sorted(archive.namelist()):
                    if not member.endswith("metadata.json") or "/wandb/" in member:
                        continue
                    metadata = json.loads(archive.read(member))
                    generation_best = _generation_best_from_metadata(metadata)
                    if not generation_best:
                        continue
                    rows.append(
                        {
                            **_metadata_row_from_zip(root_path, member, metadata),
                            **summarize_series(
                                generation_best=generation_best,
                                patience=patience,
                                checkpoints=checkpoints,
                                tolerance=tolerance,
                                min_delta=min_delta,
                            ),
                        }
                    )
            continue

        for metadata_path in sorted(root_path.rglob("metadata.json")):
            metadata = read_json(metadata_path)
            generation_best = _generation_best_from_metadata(metadata)
            if not generation_best:
                continue
            rows.append(
                {
                    **_metadata_row(metadata_path, metadata),
                    **summarize_series(
                        generation_best=generation_best,
                        patience=patience,
                        checkpoints=checkpoints,
                        tolerance=tolerance,
                        min_delta=min_delta,
                    ),
                }
            )

    write_csv(output / "early_stop_analysis.csv", rows)
    _write_markdown(output / "early_stop_analysis.md", rows)
    return EarlyStopAnalysisResult(rows=rows, output_dir=output)
