def build_eval_runner(dataset_loader, environment_factory):
    def run_eval(num_qubits, training_size, test_size, random_state=55, **params):
        dataset_loader(
            training_size=training_size,
            test_size=test_size,
            n_features=num_qubits,
            random_state=random_state,
        )
        return environment_factory(num_qubits=num_qubits, params=params)

    return run_eval
