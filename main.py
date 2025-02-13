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

import itertools
import wandb
import numpy as np

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

def prepare_wine_data(n):
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
    pca = PCA(n_components=n)
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

def find_permutations_sum_n(n):
    """
    Generates all permutations of three non-negative integers that sum up to a given value N.

    Args:
        n: The target sum (integer).

    Returns:
        A list of tuples, where each tuple represents a permutation (x, y, z)
        such that x + y + z = n and x, y, z are non-negative integers.
        Returns an empty list if n is negative.
    """
    if n < 0:
        return []  # No permutations of non-negative numbers will sum to a negative number

    permutations_list = []
    for x in range(n + 1):  # Iterate through possible values for the first number (x)
        for y in range(n - x + 1): # Iterate through possible values for the second number (y), ensuring x + y <= n
            z = n - x - y      # Calculate the third number (z) to make the sum equal to n
            permutations_list.append((x, y, z)) # Add the permutation (x, y, z) to the list

    return permutations_list

# Main execution
if __name__ == "__main__":
    
    # Define hyperparameter search space using ranges
    base_hyperparameter_space = {
        'depth': list(range(4, 7)),
        'num_circuit': list(range(4, 33, 4)),
        'num_generation': list(range(10, 101, 10)),
        'prob_mutate': list(np.linspace(-2, -1, 10))
    }

    # Iterate through different numbers of qubits
    for num_qubits in range(2, 8):  # [2, 3, 4, 5, 6, 7]
        # Does the n_features of the synthetic data match the num_qubits?
        # Generate synthetic data 
        X_train, X_test, y_train, y_test = generate_data(
            n_samples=120, n_features=4, centers=2, random_state=41
        )
        
        # Prepare Wine dataset
        Xw_train, Xw_test, yw_train, yw_test = prepare_wine_data(num_qubits)
        
        # Setup feature map
        FeatureM = ZZFeatureMap(feature_dimension=num_qubits, reps=1)

        print(f"\nExploring configurations for {num_qubits} qubits:")
        
        # Get all possible rotation gate combinations that sum to num_qubits
        rotation_combinations = find_permutations_sum_n(num_qubits)
        
        # Create the complete hyperparameter space for this num_qubits
        current_hyperparameter_space = base_hyperparameter_space.copy()
        
        # Generate all combinations of hyperparameters
        keys, values = zip(*base_hyperparameter_space.items())
        base_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # For each base combination, create variants with different rotation combinations
        for base_params in base_combinations:
            i = 0
            for rx, ry, rz in rotation_combinations:
                params = base_params.copy()
                params.update({
                    'num_qubits': num_qubits,
                    'num_rx': rx,
                    'num_ry': ry,
                    'num_rz': rz
                })
                
                wandb_config = {
                    "project": "quantum-circuit-evolution",
                    "name": f"N:{num_qubits}-x{rx}-y{ry}-z{rz}-run{i}",
                    "config": params
                }

                # Define evolution environment metadata with current hyperparameters
                env_metadata = MetadataSynthesis(
                    num_qubits=num_qubits,
                    # num_cnot=num_cnot,
                    num_rx=rx,
                    num_ry=ry,
                    num_rz=rz,
                    depth=params['depth'],
                    num_circuit=params['num_circuit'],
                    num_generation=params['num_generation'],
                    prob_mutate=params['prob_mutate']
                )
                
                # Print current configuration
                print(f"\nTesting configuration:")
                print(f"Qubits: {num_qubits}, RX: {rx}, RY: {ry}, RZ: {rz}")
                print(f"Other params: {base_params}")
                
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
                env.evol(verbose=False)#, mode="noparallel")
                
                # Finish the wandb run
                wandb.finish()
                i += 1
    
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