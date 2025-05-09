import time
import matplotlib.pyplot as plt
import pennylane as qml
from matplotlib.colors import ListedColormap
from pennylane import numpy as np
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Use lightning.gpu for GPU-accelerated simulations with PennyLane-Lightning
# Ensure you have PennyLane-Lightning[GPU] and cuQuantum installed,
# and a compatible CUDA toolkit and NVIDIA driver.
QML_DEVICE = "lightning.gpu"

def get_kernel_circuit(n_wires):
    """
    Defines the quantum kernel circuit using PennyLane.
    """
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
    X, y = make_blobs(n_samples, centers=centers, cluster_std=0.25, shuffle=False, random_state=42) # Added random_state for reproducibility
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

    # Decision boundary plot
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    fig, ax = plt.subplots() # Create a figure and an axes
    disp = DecisionBoundaryDisplay.from_estimator(svc, Xte, ax=ax, **DISP_SETTINGS)
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cm_bright, edgecolors='k') # Added edgecolors for clarity
    ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cm_bright, marker="$\u25EF$", edgecolors='k') # Added edgecolors

    return accuracy_tr, accuracy_te, fig # Return the figure object

def run_qsvm_local(n_samples, test_size):
    """
    Main workflow to run the QSVM classification.
    """
    Xtr, Xte, ytr, yte = get_split_data(n_samples, test_size)
    # The number of wires should match the number of features in the data
    n_wires_for_kernel = Xtr.shape[1]
    return classify_with_qsvm(Xtr, Xte, ytr, yte, n_wires_for_kernel)

if __name__ == '__main__':
    # Parameters for the QSVM
    num_samples = 64
    test_set_size = 0.2

    print(f"Running QSVM with {num_samples} samples and {test_set_size*100}% test data.")
    print(f"Attempting to use PennyLane device: {QML_DEVICE}")
    start_time = time.time()
    train_acc, test_acc, decision_boundary_figure = run_qsvm_local(
        n_samples=num_samples,
        test_size=test_set_size
    )
    execution_time = time.time() - start_time

    print(f"Train accuracy: {train_acc * 100:.1f}%")
    print(f"Test accuracy: {test_acc * 100:.1f}%")
    print(f"Execution time: {execution_time:.2f} seconds")
    # Show the plot
    plt.title("QSVM Decision Boundary")
    plt.show()