import itertools

from ga_qsvm.utils.combinatorics import find_permutations_sum_n


def build_base_hyperparameter_space(
    depths, num_circuits, num_generations, prob_mutations
):
    return {
        "depth": list(depths),
        "num_circuit": list(num_circuits),
        "num_generation": list(num_generations),
        "prob_mutate": list(prob_mutations),
    }


def iter_parameter_sets(num_qubits, hyperparameter_space):
    keys, values = zip(*hyperparameter_space.items())
    for base_values in itertools.product(*values):
        base_params = dict(zip(keys, base_values))
        for rx, ry, rz in find_permutations_sum_n(num_qubits):
            yield {
                **base_params,
                "num_qubits": num_qubits,
                "num_rx": rx,
                "num_ry": ry,
                "num_rz": rz,
            }
