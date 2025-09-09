# Import required libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from qiskit.circuit.library import ZZFeatureMap
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
from data import prepare_wine_data_split, prepare_digits_data_split, prepare_cancer_data_split, prepare_fashion_mnist_data_split
from utils import find_permutations_sum_n
import datetime
from squlearn.kernel.qsvc import QSVC as PQSVC
from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel
from squlearn import Executor
from squlearn.encoding_circuit import QiskitEncodingCircuit


# Set NumPy display options
np.set_printoptions(suppress=True)  # Suppress scientific notation

def parse_args():
    parser = argparse.ArgumentParser(description='GA-QSVM Training Parameters')
    parser.add_argument('--num-circuit', type=int, nargs='+', default=[16],#range(4, 33, 4),
                      help='List of number of circuits to try, ie. num parallel')
    parser.add_argument('--num-generation', type=int, nargs='+', default=[200],
                      help='List of generation numbers to try')
    parser.add_argument('--prob-mutate', type=float, nargs='+', 
                    #   default=list(np.logspace(-2, -1, 10)),
                      default=[0.1],
                      help='List of mutation probabilities to try')
    parser.add_argument('--qubits', type=int, nargs='+', default=[5],
                      help='List of number of qubits to try')
    parser.add_argument('--training-size', type=int, default=100,
                      help='Size of training dataset')
    parser.add_argument('--test-size', type=int, default=50,
                      help='Size of test dataset')
    parser.add_argument('--data', type=str, default='digits',
                      help='Dataset to use (digits or wine or cancer)')
    parser.add_argument('--start-index', type=int, default=0,
                      help='Index to start from in the base combinations, ie. when the running fail, use this to continue the benchmarking')
    parser.add_argument('--kernel', type=str, default='pqk',
                      help='Kernel to use (fqk or pqk)')
    parser.add_argument('--num_cnot', type=int, nargs='+', default=[14],
                      help='Number of CNOT gates')
    parser.add_argument('--depth', type=int, nargs='+', default=[35],
                      help='Depth of the circuit')
    return parser.parse_args()

# Define hyperparameter search space using ranges
args = parse_args()
base_hyperparameter_space = {
    'depth': args.depth,
    'num_cnot': args.num_cnot,
    'num_circuit': args.num_circuit,
    'num_generation': args.num_generation,
    'prob_mutate': args.prob_mutate
}

dataset = {'digits': prepare_digits_data_split, 'wine': prepare_wine_data_split, 'cancer': prepare_cancer_data_split, 'fashion': prepare_fashion_mnist_data_split}

range_num_qubits = args.qubits
data = dataset[args.data]
training_size = args.training_size
test_size = args.test_size
start_index = args.start_index

print(f"Starting with dataset: {data}, training size: {training_size}, test size: {test_size}")

def train_fidelity_qsvm(quantum_circuit):
    """
    Train Fidelity Quantum SVM
    
    Args:
        quantum_circuit: Quantum circuit to use as feature map
    
    Returns:
        Classification accuracy and a custom metric
    """
    quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(Xw_train, yw_train)
    y_pred = qsvc.predict(Xw_test)
    return accuracy_score(yw_test, y_pred), 0.0

def train_projected_qsvm(quantum_circuit):
    """
    Train Projected Quantum SVM
    
    Args:
        quantum_circuit: Quantum circuit to use as feature map
    
    Returns:
        Classification accuracy and a custom metric
    """
    encoding_circuit = QiskitEncodingCircuit(quantum_circuit, mode='features')
    quantum_kernel = ProjectedQuantumKernel(
        encoding_circuit=encoding_circuit,
        executor=Executor(),
        initial_parameters=np.random.rand(encoding_circuit.num_parameters)
    )
    qsvc = PQSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(Xw_train, yw_train)
    y_pred = qsvc.predict(Xw_test)
    return accuracy_score(yw_test, y_pred), 0.0

if args.kernel == 'fqk':
    train_qsvm = train_fidelity_qsvm
elif args.kernel == 'pqk':
    train_qsvm = train_projected_qsvm
else:
    raise ValueError(f"Invalid kernel: {args.kernel}")

# Main execution
if __name__ == "__main__":
    # Iterate through different numbers of qubits
    i = 0
    for num_qubits in range_num_qubits:
        Xw_train, Xw_test, yw_train, yw_test = data(training_size=training_size, test_size=test_size, n_features=num_qubits, random_state=55)

        current_hyperparameter_space = base_hyperparameter_space.copy()
        keys, values = zip(*current_hyperparameter_space.items())
        base_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Calculate total combinations for this num_qubits
        total_combinations_for_qubits = len(base_combinations)
        
        # Skip if we haven't reached our start index yet
        if i + total_combinations_for_qubits <= start_index:
            i += total_combinations_for_qubits
            print(f"Skipping {total_combinations_for_qubits} combinations for {num_qubits} qubits (total skipped: {i})")
            continue

        for base_params in base_combinations:
            if i < start_index:
                i += 1
                continue
            print(f"\nExploring configurations for {num_qubits} qubits:")
            for k in range(10):
                        
                params = base_params.copy()
                params.update({
                    'num_qubits': num_qubits,
                })
                
                wandb_config = {
                    "project": f"Benchmark-PQK-GA-QSVM-{args.data}-N{num_qubits}-Cnot{params['num_cnot']}-D{params['depth']}-C{params['num_circuit']}",
                    "name": f"n{num_qubits}-c{params['num_cnot']}-D{params['depth']}-C{params['num_circuit']}-g{params['num_generation']}-p{round(params['prob_mutate'], 5)}",
                    "config": {
                        **params,
                        "i": i,
                        "k": k
                    }
                }

                # Define evolution environment metadata with current hyperparameters
                env_metadata = MetadataSynthesis(
                    num_qubits=num_qubits,
                    num_cnot=params['num_cnot'],
                    depth=params['depth'],
                    num_circuit=params['num_circuit'],
                    num_generation=params['num_generation'],
                    prob_mutate=params['prob_mutate']
                )
                
                # Print current configuration
                print(f"\nTesting configuration:")
                print(f"Qubits: {num_qubits}")
                print(f"Other params: {params}")
                
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
                    wandb_config=wandb_config,
                    file_name=f"PQK-{args.data}-N{num_qubits}-Cnot{params['num_cnot']}-D{params['depth']}-C{params['num_circuit']}-g{params['num_generation']}-p{round(params['prob_mutate'], 5)}"
                )
                
                # Run evolution
                env.evol(verbose=False, mode="parallel")
                
                # Finish the wandb run
                wandb.finish()
            i += 1