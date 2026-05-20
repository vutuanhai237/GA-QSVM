from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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


def _fit_predict_squlearn(kernel_name: str, x_train, y_train, x_test, *, circuit: Any | None = None) -> np.ndarray:
    try:
        from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel, QSVC
    except Exception as exc:
        raise RuntimeError(
            f"{kernel_name} requires squlearn. Install squlearn before running this benchmark."
        ) from exc

    if kernel_name in {"fixed-fqk", "ga-fqk"}:
        kernel = FidelityKernel(encoding_circuit=circuit) if circuit is not None else FidelityKernel()
    elif kernel_name in {"fixed-pqk", "ga-pqk"}:
        kernel = ProjectedQuantumKernel(encoding_circuit=circuit) if circuit is not None else ProjectedQuantumKernel()
    else:
        raise ValueError(f"Unsupported quantum kernel: {kernel_name}")
    model = QSVC(quantum_kernel=kernel)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def fit_predict_fixed_fqk(x_train, y_train, x_test) -> np.ndarray:
    return _fit_predict_squlearn("fixed-fqk", x_train, y_train, x_test)


def fit_predict_fixed_pqk(x_train, y_train, x_test) -> np.ndarray:
    return _fit_predict_squlearn("fixed-pqk", x_train, y_train, x_test)


def fit_predict_ga_fqk(x_train, y_train, x_test, *, qpy_path: str | Path) -> np.ndarray:
    circuit = _load_qpy_circuit(qpy_path)
    return _fit_predict_squlearn("ga-fqk", x_train, y_train, x_test, circuit=circuit)


def fit_predict_ga_pqk(x_train, y_train, x_test, *, qpy_path: str | Path) -> np.ndarray:
    circuit = _load_qpy_circuit(qpy_path)
    return _fit_predict_squlearn("ga-pqk", x_train, y_train, x_test, circuit=circuit)


def score_predictions(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))
