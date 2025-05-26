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
    """
    Prepare Digits dataset with a standard train/test split and preprocessing.

    Args:
        train_size (float or int): If float, should be between 0.0 and 1.0 and represent the
                                 proportion of the dataset to include in the train split.
                                 If int, represents the absolute number of train samples.
        n_features (int): Number of features to reduce to using PCA.
        binary (bool): If True, filter for digits 0 and 1, convert labels to -1 and 1.
                       If False (default), use all digits 0-9.
        random_state (int): Controls the shuffling applied to the data before splitting and
                           the split itself for reproducibility.

    Returns:
        tuple: Preprocessed training and testing datasets (X_train, X_test, y_train, y_test)
    """
    # Load Digits Dataset
    digits = load_digits()
    
    # Shuffle dataset once initially (optional, as train_test_split can shuffle)
    # Using shuffle here ensures the same shuffling logic as the original if needed downstream,
    # but train_test_split's shuffle=True is generally sufficient.
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

def make_qsvm_circuit(
    pennylane_embedding_template, # This will be the result of qml.from_qiskit(qiskit_circuit)
    num_qubits         # Number of qubits, needed for device and QNode wire mapping
):

    dev = qml.device("lightning.gpu", wires=num_qubits)

    qnode_wires = qml.wires.Wires(range(num_qubits))

    @qml.qnode(dev)
    def circuit(x1, x2):
        pennylane_embedding_template(x1)
        qml.adjoint(pennylane_embedding_template)(x2)
        return qml.probs(wires=qnode_wires)

    return lambda x1, x2: circuit(x1, x2)[0]  # Tại sao lại có [0] ở đây?

def classify_with_qsvm(
    pennylane_embedding_template, # This will be the result of qml.from_qiskit(qiskit_circuit)
    X_train, y_train,
    X_test, y_test,
    num_qubits         # Number of qubits, needed for device and QNode wire mapping
):
    """
    Trains a QSVM classifier and evaluates it.
    """
    kernel = make_qsvm_circuit(pennylane_embedding_template, num_qubits)

    kernel_matrix_fn = lambda X, Z: qml.kernels.kernel_matrix(X, Z, kernel)
    svc = SVC(kernel=kernel_matrix_fn).fit(X_train, y_train)

    # Train/test accuracy
    accuracy_tr = svc.score(X_train, y_train)
    accuracy_te = svc.score(X_test, y_test)

    return accuracy_tr, accuracy_te

run = wandb.init(
    project="GPU-QSVM",  # Specify your project
    name=f"GPU_QSVM",
    config={                        # Track hyperparameters and metadata
        "num_layers": 1,
        "num_repeats": 1,
    },
)

for num_qubits in range(16,35):
    print(num_qubits)
    NUM_QUBITS_FOR_CIRCUIT = num_qubits
    NUM_LAYERS = 1
    NUM_REPEATS = 1

    start = time.time()

    X_train, X_test, y_train, y_test = prepare_digits_data_split(100, 50, NUM_QUBITS_FOR_CIRCUIT, binary=True)

    qiskit_feature_map_qc = create_pennylane_like_qiskit_feature_map(
        num_qubits=NUM_QUBITS_FOR_CIRCUIT,
        num_layers=NUM_LAYERS,
        num_repeats=NUM_REPEATS,
        # num_features=NUM_QUBITS_FOR_CIRCUIT # Length of ParameterVector 'x'
    )

    pennylane_template = qml.from_qiskit(qiskit_feature_map_qc)

    accuracy_train, accuracy_test = classify_with_qsvm(
        pennylane_template, # Pass the converted PennyLane template
        X_train, y_train, X_test, y_test,
        num_qubits=NUM_QUBITS_FOR_CIRCUIT
    )

    end = time.time()
    time_taken = end - start
    print(f"Time taken: {time_taken:.4f} seconds")

    wandb.log({"accuracy_train": accuracy_train, "accuracy_test": accuracy_test, "time_taken": time_taken, "num_qubits": num_qubits})
