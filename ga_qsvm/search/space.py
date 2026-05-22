import itertools


def build_base_hyperparameter_space(
    depths, num_circuits, num_generations, prob_mutations, kernels=("pqk",)
):
    return {
        "depth": list(depths),
        "num_circuit": list(num_circuits),
        "num_generation": list(num_generations),
        "prob_mutate": list(prob_mutations),
        "kernel": list(kernels),
    }


def iter_parameter_sets(num_qubits, hyperparameter_space):
    keys, values = zip(*hyperparameter_space.items())
    for base_values in itertools.product(*values):
        base_params = dict(zip(keys, base_values))
        yield {
            **base_params,
            "num_qubits": num_qubits,
            "num_cnot": base_params.get("num_cnot", 10 * num_qubits),
        }
