import sys
# sys.path.append("..")
import wandb
from sklearn.svm import SVC
# from data.cv import prepare_wine_data
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import accuracy_score

def prepare_wine_data_split(training_size, test_size, n_features, binary=False, random_state=20):
    wine = load_wine()
    X, y = wine.data, wine.target

    # Filter for binary classification if requested
    if binary:
        mask = (y == 0) | (y == 1)
        X = X[mask]
        y = y[mask]
        # Convert to binary labels (-1 for class 0, 1 for class 1)
        y = 2 * (y == 1) - 1  # Converts 0 -> -1 and 1 -> 1
        print(f"Filtered for binary classification (0 vs 1). Data shape: {X.shape}")
    else:
        print(f"Using multiclass classification (0-3). Data shape: {X.shape}")

    # Split data into training and testing sets BEFORE scaling/PCA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=training_size, test_size=test_size, random_state=20, shuffle=True, stratify=y
    )

    print(f"Split complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale the features (Fit on training data only!)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Transform test data using training scaler

    pca = PCA(n_components=n_features) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test) # Transform test data using training PCA

    print(f"PCA complete. Number of features: {X_train.shape[1]}")
    print(f"Final Training size: {len(X_train)}, Final Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def train_qsvm(quantum_circuit, X_train, y_train, X_test, y_test):
    """
    Train Quantum SVM using the provided quantum circuit as feature map
    
    Args:
        quantum_circuit: Quantum circuit to use as feature map
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
    
    Returns:
        Classification accuracy
    """
    quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    y_pred = qsvc.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Initialize wandb
wandb.init(project="SVM-PCA", name="wine-QSVM-100-78")

# Test different numbers of qubits (which determines feature dimensions)
max_qubits = 13
training_size = 100
test_size = 78
num_machines = 10

# Define hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 1, 6),
    'gamma': np.logspace(-4, 1, 6),
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

for num_qubits in range(3, max_qubits + 1):
    # Prepare data with PCA reduction based on num_qubits
    Xw_train, Xw_test, yw_train, yw_test = prepare_wine_data_split(
        training_size=training_size,
        test_size=test_size,
        n_features=num_qubits,
        random_state=20
    )
    
            # Train and evaluate quantum SVM
    from qiskit.circuit.library import ZZFeatureMap
    quantum_circuit = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
    qsvm_accuracy = train_qsvm(quantum_circuit, Xw_train, yw_train, Xw_test, yw_test)
    
    print(f"Num qubits: {num_qubits}, Features: {num_qubits}")
    print(f"Quantum SVM - Testing accuracy: {qsvm_accuracy:.4f}")
    print("-" * 50)
            
    # Log to wandb
    wandb.log({
        "num_qubits": num_qubits,
        "test_accuracy": qsvm_accuracy,
        "n_features": num_qubits,
    })

wandb.finish()