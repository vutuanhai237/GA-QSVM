from ga_qsvm.search.space import build_base_hyperparameter_space, iter_parameter_sets


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
            dataset_loader(
                training_size=training_size,
                test_size=test_size,
                n_features=num_qubits,
                random_state=55,
            )
            for params in iter_parameter_sets(num_qubits, hyperparameter_space):
                if current_index < start_index:
                    current_index += 1
                    continue
                environment_factory(
                    dataset_name=dataset_name,
                    params=params,
                    machine_id=machine_id,
                    index=current_index,
                )
                current_index += 1

    return run_train
