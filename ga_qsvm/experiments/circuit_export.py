from __future__ import annotations

from pathlib import Path

from ga_qsvm.experiments.artifacts import load_manifest, write_csv
from ga_qsvm.experiments.kernels import _load_qpy_circuit


def summarize_circuit(circuit) -> dict:
    counts = circuit.count_ops()
    parameters = getattr(circuit, "parameters", [])
    return {
        "depth": circuit.depth(),
        "num_parameters": len(parameters),
        "H": counts.get("h", counts.get("H", 0)),
        "CX": counts.get("cx", counts.get("CX", 0)),
        "Rx": counts.get("rx", counts.get("Rx", 0)),
        "Ry": counts.get("ry", counts.get("Ry", 0)),
        "Rz": counts.get("rz", counts.get("Rz", 0)),
    }


def export_circuits(*, manifest: str | Path, output_dir: str | Path, formats: list[str] | None = None) -> list[dict]:
    formats = formats or ["txt", "qasm"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for artifact in load_manifest(manifest):
        circuit = _load_qpy_circuit(artifact.qpy_path)
        stem = artifact.id.replace("/", "_")
        if "txt" in formats:
            (output / f"{stem}.txt").write_text(str(circuit.draw(output="text")))
        if "qasm" in formats:
            try:
                from qiskit import qasm2

                (output / f"{stem}.qasm").write_text(qasm2.dumps(circuit))
            except Exception as exc:
                (output / f"{stem}.qasm.error.txt").write_text(str(exc))
        if "png" in formats:
            circuit.draw(output="mpl").savefig(output / f"{stem}.png", dpi=200)
        row = {
            "id": artifact.id,
            "dataset": artifact.dataset,
            "kernel": artifact.kernel,
            "circuit_path": str(artifact.qpy_path),
            **summarize_circuit(circuit),
        }
        rows.append(row)
    write_csv(output / "circuit_gate_counts.csv", rows)
    return rows
