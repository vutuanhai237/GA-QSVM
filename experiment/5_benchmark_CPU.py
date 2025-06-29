import wandb
import time

from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import pennylane as qml

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import itertools
import wandb
import argparse

def create_pennylane_like_qiskit_feature_map(num_qubits: int, num_layers: int, num_repeats: int) -> QuantumCircuit:
    feature_params = ParameterVector('x', length=num_qubits)
    
    qc = QuantumCircuit(num_qubits)
    
    param_idx_counter = 0 # To cycle through feature_params

    for _k in range(num_layers):
        # Single qubit rotations (parameterized)
        for _r in range(num_repeats):
            for i in range(num_qubits):
                # Instead of fixed angles 1, 2, 3, we use parameters from feature_params.
                # We cycle through the feature_params vector.
                qc.rx(feature_params[param_idx_counter % num_qubits], i)
                param_idx_counter += 1
                qc.ry(feature_params[param_idx_counter % num_qubits], i)
                param_idx_counter += 1
                qc.rz(feature_params[param_idx_counter % num_qubits], i)
                param_idx_counter += 1

        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(num_qubits - 1, 0) # Circular CNOT

    return qc

def prepare_digits_data_split(train_size, test_size, n_features, binary=False, random_state=23):
    digits = load_digits()
    X, y = shuffle(digits.data, digits.target, random_state=random_state)

    # Filter for binary classification if requested
    if binary:
        mask = (y == 0) | (y == 1)
        X = X[mask]
        y = y[mask]
        # Convert to binary labels (-1 for class 0, 1 for class 1)
        y = 2 * (y == 1) - 1  # Converts 0 -> -1 and 1 -> 1
        print(f"Filtered for binary classification (0 vs 1). Data shape: {X.shape}")
    else:
        print(f"Using multiclass classification (0-9). Data shape: {X.shape}")

    # Split data into training and testing sets BEFORE scaling/PCA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=random_state, shuffle=True # Ensure split is shuffled
    )

    print(f"Split complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale the features (Fit on training data only!)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Transform test data using training scaler

    # Reduce dimensionality using PCA (Fit on training data only!)
    # Add random_state to PCA if using randomized solvers like 'arpack' or 'randomized'
    pca = PCA(n_components=n_features, random_state=random_state) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test) # Transform test data using training PCA

    print(f"PCA complete. Number of features: {X_train.shape[1]}")
    print(f"Final Training size: {len(X_train)}, Final Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test



def train_qsvm(quantum_circuit, Xw_train, yw_train, Xw_test, yw_test):
    """
    Train Quantum SVM using the Wine dataset
    
    Args:
        quantum_circuit: Quantum circuit to use as feature map
    
    Returns:
        Classification accuracy
    """
    quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(Xw_train, yw_train)
    y_pred = qsvc.predict(Xw_test)
    return accuracy_score(yw_test, y_pred)

run = wandb.init(
    project="GPU-QSVM",  # Specify your project
    name=f"CPU_QSVM",
    config={                        # Track hyperparameters and metadata
        "num_layers": 1,
        "num_repeats": 1,
    },
)

for num_qubits in range(3, 16):
    print(num_qubits)
    NUM_QUBITS_FOR_CIRCUIT = num_qubits
    NUM_LAYERS = 1
    NUM_REPEATS = 1

    start = time.time()
    qiskit_feature_map_qc = create_pennylane_like_qiskit_feature_map(
        num_qubits=NUM_QUBITS_FOR_CIRCUIT,
        num_layers=NUM_LAYERS,
        num_repeats=NUM_REPEATS,
        # num_features=NUM_QUBITS_FOR_CIRCUIT # Length of ParameterVector 'x'
    )
    for i in range(100):
        print(f"Iteration {i}")
        X_train, X_test, y_train, y_test = prepare_digits_data_split(100, 50, NUM_QUBITS_FOR_CIRCUIT, binary=False, random_state=i)

        # pennylane_template = qml.from_qiskit(qiskit_feature_map_qc)

        accuracy_test = train_qsvm(
            qiskit_feature_map_qc, # Pass the converted PennyLane template
            X_train, y_train, X_test, y_test,
        )

        end = time.time()
        time_taken = end - start
        print(f"Time taken: {time_taken:.4f} seconds")
        print(f"Accuracy: {accuracy_test:.4f}")
        wandb.log({"accuracy_test": accuracy_test, "time_taken": time_taken, "num_qubits": num_qubits})
