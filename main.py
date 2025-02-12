# Import required libraries
import qiskit 
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Qiskit-specific imports
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit import qpy

# QOOP-specific imports
from qoop.evolution import normalizer
from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.generator import by_num_rotations, by_num_rotations_and_cnot
from qoop.evolution.environment import EEnvironment
from qoop.evolution.crossover import onepoint
from qoop.evolution.mutate import bitflip_mutate_with_normalizer
from qoop.evolution.divider import by_num_cnot
from qoop.evolution.threshold import synthesis_threshold
from qoop.backend.constant import operations_with_rotations
from qoop.evolution import divider
from qoop.backend.utilities import load_circuit

# Set NumPy display options
np.set_printoptions(suppress=True)  # Suppress scientific notation

def generate_data(n_samples, n_features, centers, random_state):
    """
    Generate synthetic data for binary classification using make_blobs
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features for each sample
        centers: Number of centers/clusters
        random_state: Random seed for reproducibility
    
    Returns:
        Preprocessed training and testing datasets
    """
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    y = 2 * y - 1  # Convert labels to {-1, +1} for QSVC compatibility
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def load_circuits(fitness_levels):
    """
    Load quantum circuits from QPY files based on fitness levels
    
    Args:
        fitness_levels: Number of fitness levels to load
    
    Returns:
        List of loaded quantum circuits
    """
    circuits = []
    for fitness in range(1, fitness_levels + 1):
        print(fitness)
        file_name = f'4qubits_FM{fitness}_fitness_2024-12-12/best_circuit.qpy'
        with open(file_name, 'rb') as fd:
            circuit = qpy.load(fd)[0]
            print(circuit)
            circuits.append(circuit)
    return circuits

def prepare_wine_data():
    """
    Prepare Wine dataset for binary classification
    
    Returns:
        Preprocessed training and testing datasets
    """
    # Load Wine Dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # Filter for binary classification (only classes 0 and 1)
    X = X[y != 2]
    y = y[y != 2]
    
    # Split and preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def train_qsvm_with_wine(quantum_circuit):
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

def plot_data_distribution(X_train, y_train, X_test, y_test):
    """
    Plot the distribution of training and testing data
    """
    plt.figure()
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
               color='blue', label='Class 1 (Train)', alpha=0.7)
    plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
               color='red', label='Class -1 (Train)', alpha=0.7)
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
               color='lightblue', label='Class 1 (Test)', marker='x', alpha=0.7)
    plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], 
               color='lightcoral', label='Class -1 (Test)', marker='x', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Train and Test Data Points')
    plt.legend()

# Main execution
if __name__ == "__main__":
    # Generate synthetic data
    X_train, X_test, y_train, y_test = generate_data(
        n_samples=120, n_features=4, centers=2, random_state=41
    )
    
    # Prepare Wine dataset
    Xw_train, Xw_test, yw_train, yw_test = prepare_wine_data()
    
    # Setup feature map
    feature_dimension = 2
    FeatureM = ZZFeatureMap(feature_dimension=4, reps=1)
    
    # Plot data distribution
    # plot_data_distribution(X_train, y_train, X_test, y_test)
    
    # Define hyperparameter search space using ranges
    hyperparameter_space = {
        'num_cnot': list(range(2, 5)),          # [2, 3, 4]
        'num_rx': list(range(1, 3)),            # [1, 2]
        'num_ry': list(range(1, 3)),            # [1, 2]
        'num_rz': list(range(1, 4)),            # [1, 2, 3]
        'depth': list(range(4, 7)),             # [4, 5, 6]
        'num_circuit': [4], #[2**i for i in range(2, 5)],  # [4, 8, 16]
        'num_generation': [4], #[2**i for i in range(2, 5)],  # [4, 8, 16]
        'prob_mutate': [0.01 * (2**i) for i in range(3)]  # [0.01, 0.02, 0.04]
    }

    # Print search space size
    total_combinations = np.prod([len(v) for v in hyperparameter_space.values()])
    print(f"Total number of combinations to search: {total_combinations}")
    print("Hyperparameter ranges:")
    for param, values in hyperparameter_space.items():
        print(f"{param}: {values}")

    # Generate all combinations of hyperparameters
    import itertools
    keys, values = zip(*hyperparameter_space.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for run_id, params in enumerate(hyperparameter_combinations):
        wandb_config = {
            "project": "quantum-circuit-evolution",
            "name": f"hyperparam-search-{run_id}",
            "config": {
                "num_qubits": 4,  # Fixed parameter
                **params  # Include all hyperparameters in wandb config
            }
        }

        # Define evolution environment metadata with current hyperparameters
        env_metadata = MetadataSynthesis(
            num_qubits=4,  # Fixed parameter
            num_cnot=params['num_cnot'],
            num_rx=params['num_rx'],
            num_ry=params['num_ry'],
            num_rz=params['num_rz'],
            depth=params['depth'],
            num_circuit=params['num_circuit'],
            num_generation=params['num_generation'],
            prob_mutate=params['prob_mutate']
        )
        
        # Setup evolution environment
        env = EEnvironment(
            metadata=env_metadata,
            fitness_func=train_qsvm_with_wine,
            generator_func=by_num_rotations_and_cnot,
            crossover_func=onepoint(
                divider.by_num_rotation_gate(int(env_metadata.num_qubits / 2)),
                normalizer.by_num_rotation_gate(env_metadata.num_qubits)
            ),
            mutate_func=bitflip_mutate_with_normalizer(
                operations_with_rotations, 
                normalizer_func=normalizer.by_num_rotation_gate(env_metadata.num_qubits)
            ),
            threshold_func=synthesis_threshold,
            wandb_config=wandb_config
        )
        
        # Run evolution
        env.evol(verbose=False, mode="noparallel")
        
        # Finish the wandb run
        wandb.finish()
    
    # Classical SVM comparison
    clf = SVC(gamma=0.877551020408163, kernel="rbf").fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    
    # Print results
    print("Classical SVM Training Score:", clf.score(X_train, y_train))
    print("Classical SVM Testing Score:", clf.score(X_test, y_test))
    print("\nTraining Classification Report:")
    print(classification_report(y_train, train_pred))
    print("\nTesting Classification Report:")
    print(classification_report(y_test, test_pred))