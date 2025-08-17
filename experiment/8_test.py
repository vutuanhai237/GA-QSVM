# Import required libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import itertools
import wandb
import argparse
import random

# QOOP-specific imports
from qoop.evolution import normalizer
from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.generator import by_num_rotations_and_cnot
from qoop.evolution.environment import EEnvironment
from qoop.evolution.crossover import onepoint
from qoop.evolution.mutate import bitflip_mutate_with_normalizer
from qoop.evolution.threshold import synthesis_threshold
from qoop.backend.constant import operations_with_rotations
from qoop.evolution import divider


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import numpy as np
from qiskit.qpy import dump, load
n_features = 10
data_name = "digits"
home = '/home/ducto/workspace/temp/GA-QSVM/'
folder = 'digits-N10-Cnot5-D10-C16-g200-p0.01'
# Load cancer data using the function from data/split.py
# Number of features should match the number of qubits in the quantum circuit
training_size = 100
test_size = 100


def prepare_cancer_data_split(training_size, test_size, n_features, random_state):
    digits = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=100,train_size=100, random_state=random_state, stratify=digits.target)
    
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

def prepare_wine_data_split(training_size, test_size, n_features, random_state=20):
    wine = load_wine()
    X, y = wine.data, wine.target

    # Split data into training and testing sets BEFORE scaling/PCA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=100, test_size=78, random_state=random_state, shuffle=True, stratify=y
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

def prepare_digits_data_split(training_size, test_size, n_features, binary=False, random_state=55):
    digits = load_digits()
    X, y = shuffle(digits.data, digits.target, random_state=random_state)

    # Split data into training and testing sets BEFORE scaling/PCA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=100, test_size=100, random_state=random_state, shuffle=True, stratify=y
    )

    # print(f"Split complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)

    # Scale the features (Fit on training data only!)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Transform test data using training scaler

    # Reduce dimensionality using PCA (Fit on training data only!)
    # Add random_state to PCA if using randomized solvers like 'arpack' or 'randomized'
    pca = PCA(n_components=n_features, random_state=random_state) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test) # Transform test data using training PCA

    # print(f"PCA complete. Number of features: {X_train.shape[1]}")
    # print(f"Final Training size: {len(X_train)}, Final Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

dataset = {'digits': prepare_digits_data_split, 'wine': prepare_wine_data_split, 'cancer': prepare_cancer_data_split}
data = dataset[data_name]

for number in range(19, 23):  # Assuming you want to test circuits 1 through 22
    try:
        with open(home + folder + f'/best_circuit_{number}.qpy', 'rb') as f:
            loaded_circuits = load(f) # qpy.load returns a list of circuits

        qc_loaded = loaded_circuits[0]

        results = []
        for i in range(100,110):
        # for i in range(1000, 1020):

            # Load and prepare cancer dataset
            X_train, X_test, y_train, y_test = data(
                training_size=training_size, 
                test_size=test_size, 
                n_features=n_features, 
                random_state=i
            )

            # Training function adapted from main.py to work with our loaded data
            def train_qsvm(quantum_circuit):
                """
                Train Quantum SVM using the cancer dataset
                
                Args:
                    quantum_circuit: Quantum circuit to use as feature map
                
                Returns:
                    Classification accuracy
                """
                quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)
                qsvc = QSVC(quantum_kernel=quantum_kernel)
                qsvc.fit(X_train, y_train)
                y_pred = qsvc.predict(X_test)
                return accuracy_score(y_test, y_pred)

            # Train the QSVM with the loaded quantum circuit
            # print("Training Quantum SVM with the loaded 7-qubit circuit...")
            accuracy = train_qsvm(qc_loaded)
            results.append(accuracy)
            # print(f"Classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # print(results)
        mean_accuracy = np.mean(results)
        print(f"Circuit {number} - Mean accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"Error with circuit {number}")
        continue