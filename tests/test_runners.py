from unittest.mock import Mock
import numpy as np

from ga_qsvm.runners.eval import create_eval_runner
from ga_qsvm.runners.train import build_train_runner, create_train_runner


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


def test_create_train_runner_looks_up_dataset_loader(monkeypatch):
    calls = []
    fake_env = Mock()

    def fake_dataset_loader(**kwargs):
        calls.append(kwargs)
        return ("x_train", "x_test", "y_train", "y_test")

    run_train = create_train_runner(
        dataset_lookup=lambda name: fake_dataset_loader,
        environment_factory=lambda **kwargs: fake_env,
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

    assert calls == [
        {
            "training_size": 20,
            "test_size": 10,
            "n_features": 3,
            "random_state": 55,
        }
    ]
    assert fake_env.evol.call_count == 10
    fake_env.evol.assert_called_with(verbose=False, mode="parallel")


def test_create_eval_runner_looks_up_dataset_loader(monkeypatch):
    calls = []
    fake_env = Mock()
    x_train = np.arange(400).reshape(100, 4)
    x_test = np.arange(200).reshape(50, 4)
    y_train = np.array([0, 1] * 50)
    y_test = np.array([0, 1] * 25)

    def fake_dataset_loader(**kwargs):
        calls.append(kwargs)
        return (x_train, x_test, y_train, y_test)

    run_eval = create_eval_runner(
        dataset_lookup=lambda name: fake_dataset_loader,
        environment_factory=lambda **kwargs: fake_env,
    )
    result = run_eval(
        num_qubits=4,
        training_size=100,
        test_size=50,
        rx=1,
        ry=2,
        rz=3,
        prob_mutate=0.1,
        data="wine",
    )

    assert calls == [
        {
            "training_size": 100,
            "test_size": 50,
            "n_features": 4,
            "random_state": 55,
        }
    ]
    fake_env.evol.assert_called_once_with(verbose=False, mode="parallel")
    assert result is fake_env
