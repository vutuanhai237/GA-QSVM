from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CircuitArtifact:
    id: str
    dataset: str
    kernel: str
    path: Path
    qpy_path: Path
    metadata_path: Path
    funcs_path: Path
    metadata: dict[str, Any]
    funcs: dict[str, Any]


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ManifestValidationError(f"Missing required artifact file: {path}")


def _infer_from_name(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    kernel = "ga-pqk" if "pqk" in name else "ga-fqk" if "fqk" in name else "ga"
    dataset = "unknown"
    for candidate in ("digits", "fashion", "wine", "cancer"):
        if candidate in name:
            dataset = candidate
            break
    return dataset, kernel


def load_circuit_artifact(raw: dict[str, Any]) -> CircuitArtifact:
    artifact_dir = Path(raw["path"]).expanduser()
    dataset, kernel = _infer_from_name(artifact_dir)
    dataset = raw.get("dataset", dataset)
    kernel = raw.get("kernel", kernel)
    qpy_path = artifact_dir / raw.get("qpy", "best_circuit.qpy")
    metadata_path = artifact_dir / raw.get("metadata", "metadata.json")
    funcs_path = artifact_dir / raw.get("funcs", "funcs.json")
    for required in (qpy_path, metadata_path, funcs_path):
        _require_file(required)
    metadata = read_json(metadata_path)
    funcs = read_json(funcs_path)
    return CircuitArtifact(
        id=raw.get("id", artifact_dir.name),
        dataset=dataset,
        kernel=kernel,
        path=artifact_dir,
        qpy_path=qpy_path,
        metadata_path=metadata_path,
        funcs_path=funcs_path,
        metadata=metadata,
        funcs=funcs,
    )


def load_manifest(path: str | Path) -> list[CircuitArtifact]:
    manifest = read_json(path)
    circuits = manifest.get("circuits")
    if not isinstance(circuits, list):
        raise ManifestValidationError("Manifest must contain a circuits list")
    return [load_circuit_artifact(raw) for raw in circuits]


def scan_artifact_roots(roots: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for artifact_dir in sorted(Path(root).expanduser().glob("*")):
            if not artifact_dir.is_dir():
                continue
            dataset, kernel = _infer_from_name(artifact_dir)
            qpy = artifact_dir / "best_circuit.qpy"
            metadata_path = artifact_dir / "metadata.json"
            funcs = artifact_dir / "funcs.json"
            metadata = read_json(metadata_path) if metadata_path.is_file() else {}
            rows.append(
                {
                    "id": artifact_dir.name,
                    "path": str(artifact_dir),
                    "dataset": dataset,
                    "kernel": kernel,
                    "has_qpy": qpy.is_file(),
                    "has_metadata": metadata_path.is_file(),
                    "has_funcs": funcs.is_file(),
                    **extract_metadata(metadata),
                }
            )
    return rows


def extract_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    best = metadata.get("best_fitnesss") or metadata.get("best_fitness") or []
    max_fitness = max(best) if isinstance(best, list) and best else ""
    generation = metadata.get("current_generation", metadata.get("num_generation", ""))
    return {
        "qubits": metadata.get("num_qubits", ""),
        "depth": metadata.get("depth", ""),
        "cnot_count": metadata.get("num_cnot", ""),
        "generation": generation,
        "max_search_fitness": max_fitness,
    }


def parse_hyperparameters_from_name(name: str) -> dict[str, Any]:
    patterns = {
        "n_qubits": r"(?:N|n)(\d+)",
        "n_cx": r"(?:Cnot|c)(\d+)",
        "depth": r"(?:D)(\d+)",
        "n_circuit": r"(?:C)(\d+)",
        "p": r"(?:p)([0-9.]+)",
    }
    parsed: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            value = match.group(1).rstrip(".")
            parsed[key] = float(value) if key == "p" else int(value)
    return parsed


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
