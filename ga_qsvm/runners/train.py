from sklearn.metrics import accuracy_score
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import wandb

from ga_qsvm.datasets import get_dataset_loader
from ga_qsvm.search.space import build_base_hyperparameter_space, iter_parameter_sets
from ga_qsvm.tracking.wandb import build_train_wandb_config
from qoop.backend.constant import operations_with_rotations
from qoop.evolution import divider, normalizer
from qoop.evolution.crossover import onepoint
from qoop.evolution.environment import EEnvironment
from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.generator import by_num_rotations_and_cnot
from qoop.evolution.mutate import bitflip_mutate_with_normalizer
from qoop.evolution.threshold import synthesis_threshold


def build_train_environment(dataset_name, params, machine_id, index, dataset_split):
    x_train, x_test, y_train, y_test = dataset_split

    def train_qsvm(quantum_circuit):
        quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)
        qsvc = QSVC(quantum_kernel=quantum_kernel)
        qsvc.fit(x_train, y_train)
        y_pred = qsvc.predict(x_test)
        return accuracy_score(y_test, y_pred), 0.0

    env_metadata = MetadataSynthesis(
        num_qubits=params["num_qubits"],
        num_rx=params["num_rx"],
        num_ry=params["num_ry"],
        num_rz=params["num_rz"],
        depth=params["depth"],
        num_circuit=params["num_circuit"],
        num_generation=params["num_generation"],
        prob_mutate=params["prob_mutate"],
    )
    return EEnvironment(
        metadata=env_metadata,
        fitness_func=train_qsvm,
        generator_func=by_num_rotations_and_cnot,
        crossover_func=onepoint(
            divider.by_num_rotation_gate(int(env_metadata.num_qubits / 2)),
            normalizer.by_num_rotation_gate(env_metadata.num_qubits),
        ),
        mutate_func=bitflip_mutate_with_normalizer(
            operations_with_rotations,
            normalizer_func=normalizer.by_num_rotation_gate(env_metadata.num_qubits),
        ),
        threshold_func=synthesis_threshold,
        wandb_config=build_train_wandb_config(dataset_name, params, machine_id, index),
    )


def build_train_runner(dataset_loader, environment_factory):
    def run_train(
        dataset_name,
        depths,
        num_circuits,
        num_generations,
        prob_mutations,
        qubits,
        training_size,
        test_size,
        num_machines,
        machine_id,
        start_index,
    ):
        hyperparameter_space = build_base_hyperparameter_space(
            depths=depths,
            num_circuits=num_circuits,
            num_generations=num_generations,
            prob_mutations=prob_mutations,
        )
        current_index = 0
        for num_qubits in qubits:
            dataset_split = dataset_loader(
                training_size=training_size,
                test_size=test_size,
                n_features=num_qubits,
                random_state=55,
            )
            for params in iter_parameter_sets(num_qubits, hyperparameter_space):
                if current_index < start_index:
                    current_index += 1
                    continue
                env = environment_factory(
                    dataset_name=dataset_name,
                    params=params,
                    machine_id=machine_id,
                    index=current_index,
                    dataset_split=dataset_split,
                )
                env.evol(verbose=False, mode="parallel")
                wandb.finish()
                current_index += 1

    return run_train


def create_train_runner(dataset_lookup=get_dataset_loader, environment_factory=build_train_environment):
    def run_train(**kwargs):
        dataset_loader = dataset_lookup(kwargs["dataset_name"])
        return build_train_runner(
            dataset_loader=dataset_loader,
            environment_factory=environment_factory,
        )(**kwargs)

    return run_train
