
import qiskit 
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit import qpy

from qoop.evolution import normalizer
from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.generator import by_num_rotations,by_num_rotations_and_cnot
from qoop.evolution.environment import EEnvironment
from qoop.evolution.crossover import onepoint
from qoop.evolution.mutate import bitflip_mutate_with_normalizer
from qoop.evolution.divider import by_num_cnot
from qoop.evolution.threshold import synthesis_threshold
from qoop.backend.constant import operations_with_rotations
from qoop.evolution import divider
from qoop.backend.utilities import load_circuit

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from datetime import datetime
import pandas as pd
import json

def load_circuits(fitness_levels):
    circuits = []
    for fitness in range(1, fitness_levels+1):
        print(fitness)
        file_name = f'4qubits_FM{fitness}_fitness_2024-12-12/best_circuit.qpy'
        with open(file_name, 'rb') as fd:
            circuit = qpy.load(fd)[0]
            print(circuit)  # Print if needed
            circuits.append(circuit)
    return circuits

def prepare_wine_data_case(n_features, random_seed, scale=None):
    """
    Prepare Wine dataset for binary classification
    
    Returns:
        Preprocessed training and testing datasets
    """
    # Load Wine Dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # Split and preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    
    # Scale the features
    if scale == "minmax":
        scaler = MinMaxScaler()
    elif scale == "standard":
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train, X_test, y_train, y_test

env_metadata = MetadataSynthesis(
    num_qubits=3,            # Number of qubits
    num_rx=1,                # Number of RX gates # Can't change
    num_ry=2,                # Number of RY gates # Can't change
    num_rz=0,                # Number of RZ gates # Can't change
    depth=4,                 # Depth of the circuit # Can't change this with num_circuit, num_generation
    num_circuit=8,           # Number of circuits in the population
    num_generation=5,       # Number of generations 
    prob_mutate=0.02783         # Mutation probability
)

def train_qsvm_with_wine(quantum_circuit):
    quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(Xw_train, yw_train)
    y_pred = qsvc.predict(Xw_test)
    accuracy = accuracy_score(yw_test, y_pred)

    return accuracy

results = []

# First run quantum experiments in parallel
quantum_results = {}
for i in range(40,50,1):
    print("Random seed: ", i)
    Xw_train, Xw_test, yw_train, yw_test = prepare_wine_data_case(n_features=3, random_seed=i, scale="minmax")

    env = None
    # Define the environment
    env = EEnvironment(
        metadata=env_metadata,
        fitness_func=train_qsvm_with_wine,
        generator_func=by_num_rotations_and_cnot,  # Use the new generator function
        crossover_func=onepoint(
            divider.by_num_rotation_gate(int((env_metadata.num_qubits)/ 2)),
            normalizer.by_num_rotation_gate(env_metadata.num_qubits)
        ),
        mutate_func=bitflip_mutate_with_normalizer(operations_with_rotations, 
                                                normalizer_func=normalizer.by_num_rotation_gate(env_metadata.num_qubits)),
        threshold_func=synthesis_threshold
    )

    # Run the evolution process
    env.evol(verbose=True, mode="parallel")
    quantum_results[i] = env.best_fitness

# Then run classical experiments sequentially
for i in range(40,50,1):
    print("Random seed: ", i)
    Xw_train, Xw_test, yw_train, yw_test = prepare_wine_data_case(n_features=3, random_seed=i, scale="standard")

    clf = SVC(gamma=0.877551020408163, kernel="rbf").fit(Xw_train, yw_train)
    train_pred = clf.predict(Xw_train)
    test_pred = clf.predict(Xw_test)
    classical_train_acc = clf.score(Xw_train, yw_train)
    classical_test_acc = clf.score(Xw_test, yw_test)
    
    print("Training Accuracy:", classical_train_acc)
    print("Testing Accuracy:", classical_test_acc)

    result = {
        'seed': i,
        'classical_train_accuracy': classical_train_acc,
        'classical_test_accuracy': classical_test_acc,
        'quantum_test_accuracy': quantum_results[i]
    }
    results.append(result)

    print("--------------------------------\n\n")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df = pd.DataFrame(results)

# Save to CSV
csv_filename = f'svm_comparison_results_{timestamp}.csv'
df.to_csv(csv_filename, index=False)

# Save to JSON
json_filename = f'svm_comparison_results_{timestamp}.json'
with open(json_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to:")
print(f"- CSV: {csv_filename}")
print(f"- JSON: {json_filename}")