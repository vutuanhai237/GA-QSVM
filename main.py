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
from datasets import prepare_wine_data, prepare_digits_data
from utils import find_permutations_sum_n

# Set NumPy display options
np.set_printoptions(suppress=True)  # Suppress scientific notation

def parse_args():
    parser = argparse.ArgumentParser(description='GA-QSVM Training Parameters')
    parser.add_argument('--depth', type=int, nargs='+', default=[4, 5, 6],
                      help='List of circuit depths to try')
    parser.add_argument('--num-circuit', type=int, nargs='+', default=range(4, 33, 4),
                      help='List of number of circuits to try, ie. num parallel')
    parser.add_argument('--num-generation', type=int, nargs='+', default=[100],
                      help='List of generation numbers to try')
    parser.add_argument('--prob-mutate', type=float, nargs='+', 
                      default=list(np.logspace(-2, -1, 10)),
                      help='List of mutation probabilities to try')
    parser.add_argument('--qubits', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8],
                      help='List of number of qubits to try')
    parser.add_argument('--training-size', type=int, default=100,
                      help='Size of training dataset')
    parser.add_argument('--test-size', type=int, default=50,
                      help='Size of test dataset')
    parser.add_argument('--num-machines', type=int, default=3,
                      help='Number of machines')
    parser.add_argument('--id', type=int, default=0,
                      help='ID for the run')
    return parser.parse_args()

# Define hyperparameter search space using ranges
args = parse_args()
base_hyperparameter_space = {
    'depth': args.depth,
    'num_circuit': args.num_circuit,
    'num_generation': args.num_generation,
    'prob_mutate': args.prob_mutate
}

range_num_qubits = args.qubits
data = prepare_wine_data
training_size = args.training_size
test_size = args.test_size
num_machines = args.num_machines
id = args.id

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

# Main execution
if __name__ == "__main__":
    # Iterate through different numbers of qubits
    for num_qubits in range_num_qubits:  # [2, 3, 4, 5, 6, 7]
        while True:
            Xw_train, Xw_test, yw_train, yw_test = data(training_size, test_size, num_qubits, id, num_machines)
            if Xw_train is not None:
                break
        
        # Setup feature map
        FeatureM = ZZFeatureMap(feature_dimension=num_qubits, reps=1)

        print(f"\nExploring configurations for {num_qubits} qubits:")
        
        # Get all possible rotation gate combinations that sum to num_qubits
        rotation_combinations = find_permutations_sum_n(num_qubits)
        
        # Create the complete hyperparameter space for this num_qubits
        current_hyperparameter_space = base_hyperparameter_space.copy()
        
        # Generate all combinations of hyperparameters
        keys, values = zip(*current_hyperparameter_space.items())
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
                    "project": f"GA-QSVM-N{num_qubits}-D{params['depth']}-C{params['num_circuit']}",
                    "name": f"x{rx}-y{ry}-z{rz}-c{params['num_circuit']}-g{params['num_generation']}-p{round(params['prob_mutate'], 5)}",
                    "id": id,
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
                env.evol(verbose=False, mode="parallel")
                
                # Finish the wandb run
                wandb.finish()
                i += 1
    
    # Classical SVM comparison
    clf = SVC(gamma=0.877551020408163, kernel="rbf").fit(Xw_train, yw_train)
    train_pred = clf.predict(Xw_train)
    test_pred = clf.predict(Xw_test)
    
    # Print results
    print("Classical SVM Training Score:", clf.score(Xw_train, yw_train))
    print("Classical SVM Testing Score:", clf.score(Xw_test, yw_test))
    print("\nTraining Classification Report:")
    print(classification_report(yw_train, train_pred))
    print("\nTesting Classification Report:")
    print(classification_report(yw_test, test_pred))