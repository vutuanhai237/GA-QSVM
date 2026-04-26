from unittest.mock import Mock

from ga_qsvm.runners.train import build_train_runner


def test_build_train_runner_uses_dataset_loader_and_environment_factory():
    dataset_loader = Mock(return_value=("x_train", "x_test", "y_train", "y_test"))
    environment_factory = Mock()

    run_train = build_train_runner(
        dataset_loader=dataset_loader,
        environment_factory=environment_factory,
    )

    run_train(
        dataset_name="digits",
        depths=[4],
        num_circuits=[8],
        num_generations=[10],
        prob_mutations=[0.1],
        qubits=[3],
        training_size=20,
        test_size=10,
        num_machines=1,
        machine_id=0,
        start_index=0,
    )

    dataset_loader.assert_called_once_with(
        training_size=20,
        test_size=10,
        n_features=3,
        random_state=55,
    )
    assert environment_factory.called
