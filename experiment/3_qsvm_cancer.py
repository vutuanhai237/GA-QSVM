import sys
# sys.path.append("..")
import wandb
from sklearn.svm import SVC
# from data.cv import prepare_cancer_data
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

def prepare_cancer_data_split(training_size, test_size, n_features, random_state=52):
    digits = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=test_size,train_size=training_size, random_state=52, stratify=digits.target)
    
    # Scale the features 
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print(f"Training size: {len(X_train)}, Test size: {len(X_test)}")

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

# Test different numbers of qubits (which determines feature dimensions)
max_qubits = 8  
training_size = 100
test_size = 100

# Initialize wandb
wandb.init(project="SVM-PCA", name=f"cancer-SVM-100-100")

# Define hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 1, 6),
    'gamma': np.logspace(-4, 1, 6),
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

for num_qubits in range(3, max_qubits + 1):
    Xw_train, Xw_test, yw_train, yw_test = prepare_cancer_data_split(
        training_size=training_size,
        test_size=test_size,
        n_features=num_qubits,
        random_state=52,
    )
    
    # Perform grid search
    # svm = SVC()
    # grid_search = GridSearchCV(
    #     svm, 
    #     param_grid, 
    #     cv=5,
    #     n_jobs=-1,
    #     scoring='accuracy'
    # )
    # grid_search.fit(Xw_train, yw_train)
    
    # Get best parameters
    # best_params = grid_search.best_params_
    # best_params_list.append(best_params)
    
    # Train SVM with best parameters
    # clf = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma']).fit(Xw_train, yw_train)
    from qiskit.circuit.library import ZZFeatureMap
    quantum_circuit = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
    qsvm_accuracy = train_qsvm(quantum_circuit, Xw_train, yw_train, Xw_test, yw_test)
    
    # train_accuracies.append(train_accuracy)
    # test_accuracies.append(test_accuracy)
    
    print(f"Num qubits: {num_qubits}, Features: {num_qubits}")
    # print(f"Best parameters: C={best_params['C']:.6f}, gamma={best_params['gamma']:.6f}, kernel={best_params['kernel']}")
    print(f"Quantum SVM - Testing accuracy: {qsvm_accuracy:.4f}")
    print("-" * 30)
    
    # Calculate average best parameters
    # avg_C = np.mean([params['C'] for params in best_params_list])
    # avg_gamma = np.mean([params['gamma'] for params in best_params_list])
    
    # Log to wandb
    wandb.log({
        "num_qubits": num_qubits,
        "test_accuracy": qsvm_accuracy,
        # "std_train_accuracy": std_train_accuracy,
        # "std_test_accuracy": std_test_accuracy,
        "n_features": num_qubits,
        # "avg_best_C": avg_C,
        # "avg_best_gamma": avg_gamma
    })
    
    # print(f"\nAverages for {num_qubits} qubits:")
    # print(f"Average Best C: {avg_C:.6f}")
    # print(f"Average Best gamma: {avg_gamma:.6f}")
    # print(f"Average Training accuracy: {avg_train_accuracy:.4f} ± {std_train_accuracy:.4f}")
    # print(f"Average Testing accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
    # print("=" * 50)

wandb.finish()