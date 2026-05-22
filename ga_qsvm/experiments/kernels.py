from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC as QiskitQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def fit_predict_rbf(x_train, y_train, x_test, *, c: float = 1.0, gamma: str | float = "scale") -> np.ndarray:
    model = SVC(kernel="rbf", C=c, gamma=gamma)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _load_qpy_circuit(path: str | Path):
    from qiskit import qpy

    with Path(path).open("rb") as handle:
        circuits = qpy.load(handle)
    if not circuits:
        raise ValueError(f"No circuit found in QPY file: {path}")
    return circuits[0]


@dataclass(frozen=True)
class CircuitSummary:
    num_qubits: int
    num_parameters: int
    rotation_count: int
    constant_rotations: int


def summarize_qpy_circuit(path: str | Path) -> CircuitSummary:
    circuit = _load_qpy_circuit(path)
    rotation_names = {"rx", "ry", "rz"}
    rotation_count = 0
    constant_rotations = 0
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name not in rotation_names:
            continue
        rotation_count += 1
        has_parameter = any(getattr(parameter, "parameters", set()) for parameter in operation.params)
        if not has_parameter:
            constant_rotations += 1
    return CircuitSummary(
        num_qubits=circuit.num_qubits,
        num_parameters=circuit.num_parameters,
        rotation_count=rotation_count,
        constant_rotations=constant_rotations,
    )


def qpy_feature_dimension(path: str | Path) -> int:
    return _load_qpy_circuit(path).num_parameters


def _fit_predict_fidelity(x_train, y_train, x_test, *, circuit: Any | None = None) -> np.ndarray:
    feature_map = circuit or ZZFeatureMap(feature_dimension=x_train.shape[1], reps=2)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    model = QiskitQSVC(quantum_kernel=quantum_kernel)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _fit_predict_projected(x_train, y_train, x_test, *, circuit: Any | None = None) -> np.ndarray:
    try:
        from squlearn import Executor
        from squlearn.encoding_circuit import QiskitEncodingCircuit
        from squlearn.kernel import ProjectedQuantumKernel, QSVC
    except Exception as exc:
        raise RuntimeError(
            "Projected quantum kernels require squlearn. Install squlearn before running this benchmark."
        ) from exc

    qiskit_circuit = circuit or ZZFeatureMap(feature_dimension=x_train.shape[1], reps=2)
    encoding_circuit = QiskitEncodingCircuit(qiskit_circuit, mode="features")
    kernel = ProjectedQuantumKernel(
        encoding_circuit=encoding_circuit,
        executor=Executor(),
        initial_parameters=np.random.default_rng(0).random(encoding_circuit.num_parameters),
    )
    model = QSVC(quantum_kernel=kernel)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def fit_predict_fixed_fqk(x_train, y_train, x_test) -> np.ndarray:
    return _fit_predict_fidelity(x_train, y_train, x_test)


def fit_predict_fixed_pqk(x_train, y_train, x_test) -> np.ndarray:
    return _fit_predict_projected(x_train, y_train, x_test)


def fit_predict_ga_fqk(x_train, y_train, x_test, *, qpy_path: str | Path) -> np.ndarray:
    circuit = _load_qpy_circuit(qpy_path)
    return _fit_predict_fidelity(x_train, y_train, x_test, circuit=circuit)


def fit_predict_ga_pqk(x_train, y_train, x_test, *, qpy_path: str | Path) -> np.ndarray:
    circuit = _load_qpy_circuit(qpy_path)
    return _fit_predict_projected(x_train, y_train, x_test, circuit=circuit)


def score_predictions(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))
