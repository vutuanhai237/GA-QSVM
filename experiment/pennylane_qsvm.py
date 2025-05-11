import time
import matplotlib.pyplot as plt
import pennylane as qml
from matplotlib.colors import ListedColormap
from pennylane import numpy as np
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import numpy as np

from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import classification_report, accuracy_score
from qiskit_machine_learning.kernels import FidelityQuantumKernel

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

# Use lightning.gpu for GPU-accelerated simulations with PennyLane-Lightning
# Ensure you have PennyLane-Lightning[GPU] and cuQuantum installed,
# and a compatible CUDA toolkit and NVIDIA driver.
QML_DEVICE = "lightning.gpu"

def get_kernel_circuit(n_wires):
    """
    Defines the quantum kernel circuit using PennyLane.
    """
    print(f"Wires: {n_wires}")
    # Uses lightning.gpu for GPU acceleration
    dev = qml.device(QML_DEVICE, wires=n_wires, shots=None)

    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.IQPEmbedding(x1, wires=range(n_wires), n_repeats=4)
        qml.adjoint(qml.IQPEmbedding)(x2, wires=range(n_wires), n_repeats=4)
        return qml.probs(wires=range(n_wires))

    return lambda x1, x2: circuit(x1, x2)[0]  # Return probability of |0...0> state

def get_split_data(n_samples=18, test_size=0.2):
    """
    Generates and splits a synthetic dataset.
    """
    centers = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    X, y = make_blobs(n_samples, n_features=10, centers=centers, cluster_std=0.25, shuffle=False, random_state=42) # Added random_state for reproducibility
    # Rescale labels to be -1, 1
    mapping = {0: -1, 1: 1, 2: -1, 3: 1, 4: -1, 5: 1, 6: -1, 7: 1, 8: -1}
    y = np.array([mapping[i] for i in y])
    X = X.astype(np.float32)
    y = y.astype(int)

    return train_test_split(X, y, test_size=test_size, random_state=3)

DISP_SETTINGS = {
    "grid_resolution": 50,
    "response_method": "predict",
    "alpha": 0.5,
    "cmap": plt.cm.RdBu,
}

def classify_with_qsvm(Xtr, Xte, ytr, yte, n_wires_for_kernel):
    """
    Trains a QSVM classifier and evaluates it.
    """
    kernel = get_kernel_circuit(n_wires=n_wires_for_kernel)

    kernel_matrix_fn = lambda X, Z: qml.kernels.kernel_matrix(X, Z, kernel)
    svc = SVC(kernel=kernel_matrix_fn).fit(Xtr, ytr)

    # Train/test accuracy
    accuracy_tr = svc.score(Xtr, ytr)
    accuracy_te = svc.score(Xte, yte)

    return accuracy_tr, accuracy_te

def run_qsvm_local(n_samples, test_size):
    """
    Main workflow to run the QSVM classification.
    """
    Xtr, Xte, ytr, yte = prepare_digits_data_split(train_size=300, test_size=60, n_features=10, binary=True)
    # Xtr, Xte, ytr, yte = get_split_data(n_samples, test_size)
    # The number of wires should match the number of features in the data
    n_wires_for_kernel = Xtr.shape[1]
    return classify_with_qsvm(Xtr, Xte, ytr, yte, n_wires_for_kernel)

if __name__ == '__main__':
    # Parameters for the QSVM
    num_samples = 300
    test_set_size = 0.2

    print(f'Wires: 20')
    print(f"Running QSVM with {num_samples} samples and {test_set_size*100}% test data.")
    print(f"Attempting to use PennyLane device: {QML_DEVICE}")
    start_time = time.time()
    train_acc, test_acc = run_qsvm_local(
        n_samples=num_samples,
        test_size=test_set_size
    )
    execution_time = time.time() - start_time

    print(f"Train accuracy: {train_acc * 100:.1f}%")
    print(f"Test accuracy: {test_acc * 100:.1f}%")
    print(f"Execution GPU time: {execution_time:.2f} seconds")
    
    Xtr, Xte, ytr, yte = prepare_digits_data_split(train_size=300, test_size=60, n_features=10, binary=True)
    start_time = time.time()
    quantum_kernel = FidelityQuantumKernel()
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(Xtr, ytr)
    y_pred = qsvc.predict(Xte)
    execution_time = time.time() - start_time
    print('CPU:')
    print(f"Accuracy: {accuracy_score(yte, y_pred)}")
    print(f"Execution CPU time: {execution_time:.2f} seconds")