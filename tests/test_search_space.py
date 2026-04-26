from ga_qsvm.search.space import (
    build_base_hyperparameter_space,
    iter_parameter_sets,
)


def test_build_base_hyperparameter_space_preserves_cli_values():
    hyperparameter_space = build_base_hyperparameter_space(
        depths=[4],
        num_circuits=[8, 16],
        num_generations=[100],
        prob_mutations=[0.1, 0.2],
    )

    assert hyperparameter_space == {
        "depth": [4],
        "num_circuit": [8, 16],
        "num_generation": [100],
        "prob_mutate": [0.1, 0.2],
    }


def test_iter_parameter_sets_combines_base_space_with_rotation_counts():
    parameter_sets = list(
        iter_parameter_sets(
            num_qubits=2,
            hyperparameter_space=build_base_hyperparameter_space(
                depths=[4],
                num_circuits=[8],
                num_generations=[10],
                prob_mutations=[0.1],
            ),
        )
    )

    assert parameter_sets[0] == {
        "depth": 4,
        "num_circuit": 8,
        "num_generation": 10,
        "prob_mutate": 0.1,
        "num_qubits": 2,
        "num_rx": 0,
        "num_ry": 0,
        "num_rz": 2,
    }
    assert len(parameter_sets) == 6
