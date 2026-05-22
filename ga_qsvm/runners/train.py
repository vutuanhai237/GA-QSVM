from sklearn.metrics import accuracy_score
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import numpy as np
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


class TrainFidelityQSVMFitness:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.__name__ = "train_fqk_qsvm"
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def __call__(self, quantum_circuit):
        quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)
        qsvc = QSVC(quantum_kernel=quantum_kernel)
        qsvc.fit(self.x_train, self.y_train)
        y_pred = qsvc.predict(self.x_test)
        return accuracy_score(self.y_test, y_pred), 0.0


class TrainProjectedQSVMFitness:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.__name__ = "train_pqk_qsvm"
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def __call__(self, quantum_circuit):
        try:
            from squlearn import Executor
            from squlearn.encoding_circuit import QiskitEncodingCircuit
            from squlearn.kernel import ProjectedQuantumKernel, QSVC as ProjectedQSVC
        except Exception as exc:
            raise RuntimeError(
                "Projected quantum kernels require squlearn. Install squlearn before running GA with --kernel pqk."
            ) from exc

        encoding_circuit = QiskitEncodingCircuit(quantum_circuit, mode="features")
        quantum_kernel = ProjectedQuantumKernel(
            encoding_circuit=encoding_circuit,
            executor=Executor(),
            initial_parameters=np.random.default_rng(0).random(encoding_circuit.num_parameters),
        )
        qsvc = ProjectedQSVC(quantum_kernel=quantum_kernel)
        qsvc.fit(self.x_train, self.y_train)
        y_pred = qsvc.predict(self.x_test)
        return accuracy_score(self.y_test, y_pred), 0.0


def build_train_fitness(kernel, dataset_split):
    x_train, x_test, y_train, y_test = dataset_split
    if kernel == "fqk":
        return TrainFidelityQSVMFitness(x_train, x_test, y_train, y_test)
    if kernel == "pqk":
        return TrainProjectedQSVMFitness(x_train, x_test, y_train, y_test)
    raise ValueError(f"Unsupported kernel: {kernel}")


def build_train_environment(dataset_name, params, machine_id, index, dataset_split):
    train_qsvm = build_train_fitness(params.get("kernel", "pqk"), dataset_split)

    env_metadata = MetadataSynthesis(
        num_qubits=params["num_qubits"],
        num_cnot=params["num_cnot"],
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
            [gate for gate in operations_with_rotations if gate["num_params"] == 0],
            normalizer_func=normalizer.by_num_rotation_gate(env_metadata.num_qubits),
            prob_mutate=env_metadata.prob_mutate,
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
        kernel,
    ):
        current_index = 0
        for num_qubits in qubits:
            depth_values = depths if depths is not None else [10 * num_qubits]
            hyperparameter_space = build_base_hyperparameter_space(
                depths=depth_values,
                num_circuits=num_circuits,
                num_generations=num_generations,
                prob_mutations=prob_mutations,
                kernels=[kernel],
            )
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
