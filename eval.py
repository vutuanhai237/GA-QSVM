# Import required libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import itertools
import wandb
import argparse

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
from data import prepare_wine_data_split, prepare_digits_data_split, prepare_cancer_data_split

from sklearn.model_selection import train_test_split

# Set NumPy display options
np.set_printoptions(suppress=True)  # Suppress scientific notation

dataset = {'digits': prepare_digits_data_split, 'wine': prepare_wine_data_split, 'cancer': prepare_cancer_data_split}

# Parse command line arguments
parser = argparse.ArgumentParser(description='GA-QSVM Evaluation')
parser.add_argument('--rx', type=int, default=4, help='Number of RX rotations')
parser.add_argument('--ry', type=int, default=1, help='Number of RY rotations')
parser.add_argument('--rz', type=int, default=2, help='Number of RZ rotations')
parser.add_argument('--num_qubits', type=int, default=7, help='Number of qubits')
parser.add_argument('--prob_mutate', type=float, default=0.027825594022071243, help='Mutation probability')
parser.add_argument('--data', type=str, default='digits', choices=['digits', 'wine', 'cancer'], help='Dataset to use')

args = parser.parse_args()

# Set parameters from arguments
rx = args.rx
ry = args.ry
rz = args.rz
num_qubits = args.num_qubits
prob_mutate = args.prob_mutate
data = dataset[args.data]

training_size = 100
test_size = 50
num_circuit = 8
num_generation = 100
depth = 4

def train_qsvm(quantum_circuit):
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
    y_pred_val = qsvc.predict(Xw_val)
    y_pred_eval = qsvc.predict(Xw_eval)
    # wandb.log({
    #     "accuracy_on_eval": accuracy_score(yw_eval, y_pred_eval)
    # })
    return accuracy_score(yw_val, y_pred_val), accuracy_score(yw_eval, y_pred_eval)

# Main execution
if __name__ == "__main__":
    Xw_train, Xw_test, yw_train, yw_test = data(training_size=training_size, test_size=test_size, n_features=num_qubits, random_state=55)
    Xw_val, Xw_eval, yw_val, yw_eval = train_test_split(
        Xw_test, yw_test, train_size=0.5, shuffle=True, stratify=yw_test, random_state=55
    )
    
    wandb_config = {
        "project": f"GA-QSVM-eval",
        "name": f"{args.data}-x{rx}-y{ry}-z{rz}-c{num_circuit}-p{round(prob_mutate, 5)}-d{depth}",
        "config": {
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "num_circuit": num_circuit,
            "num_generation": num_generation,
            "prob_mutate": prob_mutate,
            "data": args.data
        }
    }

    # Define evolution environment metadata with current hyperparameters
    env_metadata = MetadataSynthesis(
        num_qubits=num_qubits,
        num_rx=rx,
        num_ry=ry,
        num_rz=rz,
        depth=depth,
        num_circuit=num_circuit,
        num_generation=num_generation,
        prob_mutate=prob_mutate
    )
    
    # Print current configuration
    print(f"\nTesting configuration:")
    print(f"Qubits: {num_qubits}, RX: {rx}, RY: {ry}, RZ: {rz}")
    
    # Setup evolution environment
    env = EEnvironment(
        metadata=env_metadata,
        fitness_func=train_qsvm,
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
    
    env.evol(verbose=False, mode="parallel")