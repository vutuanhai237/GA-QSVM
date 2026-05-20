from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ga_qsvm.experiments.artifacts import load_circuit_artifact, read_json
from ga_qsvm.experiments.frozen_benchmark import run_frozen_benchmark


@dataclass(frozen=True)
class TransferArtifact:
    id: str
    source_dataset: str
    target_dataset: str
    kernel: str
    path: Path
    qpy_path: Path
    metadata: dict[str, Any]


def load_transfer_manifest(path: str | Path) -> list[TransferArtifact]:
    manifest = read_json(path)
    entries: list[TransferArtifact] = []
    for raw in manifest.get("transfers", []):
        artifact = load_circuit_artifact({**raw, "dataset": raw["target_dataset"]})
        entries.append(
            TransferArtifact(
                id=raw.get("id", artifact.id),
                source_dataset=raw["source_dataset"],
                target_dataset=raw["target_dataset"],
                kernel=raw["kernel"],
                path=artifact.path,
                qpy_path=artifact.qpy_path,
                metadata=artifact.metadata,
            )
        )
    return entries


def run_transfer_benchmark(
    *,
    manifest: str | Path,
    seeds: list[int],
    test_size: float,
    preprocess: str,
    output_dir: str | Path,
    n_features: int = 7,
):
    transfers = load_transfer_manifest(manifest)
    synthetic_manifest = Path(output_dir) / "_expanded_transfer_manifest.json"
    synthetic_manifest.parent.mkdir(parents=True, exist_ok=True)
    import json

    synthetic_manifest.write_text(
        json.dumps(
            {
                "circuits": [
                    {
                        "id": entry.id,
                        "dataset": entry.target_dataset,
                        "kernel": entry.kernel,
                        "path": str(entry.path),
                    }
                    for entry in transfers
                ]
            },
            indent=2,
        )
    )
    models = sorted({entry.kernel for entry in transfers})
    return run_frozen_benchmark(
        manifest=synthetic_manifest,
        seeds=seeds,
        test_size=test_size,
        preprocess=preprocess,
        models=models,
        output_dir=output_dir,
        n_features=n_features,
    )
