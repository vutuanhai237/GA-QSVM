from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import wandb

from ga_qsvm.datasets import get_dataset_loader
from qoop.backend.constant import operations_with_rotations
from qoop.evolution import divider, normalizer
from qoop.evolution.crossover import onepoint
from qoop.evolution.environment import EEnvironment
from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.generator import by_num_rotations_and_cnot
from qoop.evolution.mutate import bitflip_mutate_with_normalizer
from qoop.evolution.threshold import synthesis_threshold


def build_eval_environment(num_qubits, params, dataset_split):
    x_train, x_val, x_eval, y_train, y_val, y_eval = dataset_split

    def train_qsvm(quantum_circuit):
        quantum_kernel = FidelityQuantumKernel(feature_map=quantum_circuit)
        qsvc = QSVC(quantum_kernel=quantum_kernel)
        qsvc.fit(x_train, y_train)
        y_pred_val = qsvc.predict(x_val)
        y_pred_eval = qsvc.predict(x_eval)
        return accuracy_score(y_val, y_pred_val), accuracy_score(y_eval, y_pred_eval)

    env_metadata = MetadataSynthesis(
        num_qubits=num_qubits,
        num_rx=params["rx"],
        num_ry=params["ry"],
        num_rz=params["rz"],
        depth=4,
        num_circuit=8,
        num_generation=100,
        prob_mutate=params["prob_mutate"],
    )
    wandb_config = {
        "project": "GA-QSVM-eval",
        "name": (
            f"{params['data']}-x{params['rx']}-y{params['ry']}-z{params['rz']}"
            f"-c8-p{round(params['prob_mutate'], 5)}-d4"
        ),
        "config": {
            "rx": params["rx"],
            "ry": params["ry"],
            "rz": params["rz"],
            "num_circuit": 8,
            "num_generation": 100,
            "prob_mutate": params["prob_mutate"],
            "data": params["data"],
        },
    }
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
        wandb_config=wandb_config,
    )


def build_eval_runner(dataset_loader, environment_factory):
    def run_eval(num_qubits, training_size, test_size, random_state=55, **params):
        x_train, x_test, y_train, y_test = dataset_loader(
            training_size=training_size,
            test_size=test_size,
            n_features=num_qubits,
            random_state=random_state,
        )
        x_val, x_eval, y_val, y_eval = train_test_split(
            x_test,
            y_test,
            train_size=0.5,
            shuffle=True,
            stratify=y_test,
            random_state=random_state,
        )
        env = environment_factory(
            num_qubits=num_qubits,
            params=params,
            dataset_split=(x_train, x_val, x_eval, y_train, y_val, y_eval),
        )
        env.evol(verbose=False, mode="parallel")
        wandb.finish()
        return env

    return run_eval


def create_eval_runner(dataset_lookup=get_dataset_loader, environment_factory=build_eval_environment):
    def run_eval(**kwargs):
        dataset_loader = dataset_lookup(kwargs["data"])
        return build_eval_runner(
            dataset_loader=dataset_loader,
            environment_factory=environment_factory,
        )(
            num_qubits=kwargs["num_qubits"],
            training_size=kwargs["training_size"],
            test_size=kwargs["test_size"],
            rx=kwargs["rx"],
            ry=kwargs["ry"],
            rz=kwargs["rz"],
            prob_mutate=kwargs["prob_mutate"],
            data=kwargs["data"],
        )

    return run_eval
