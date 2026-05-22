import pickle
from unittest.mock import Mock
import numpy as np

from ga_qsvm.runners.eval import create_eval_runner
from ga_qsvm.runners.train import (
    TrainFidelityQSVMFitness,
    TrainProjectedQSVMFitness,
    build_train_runner,
    create_train_runner,
)


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
        kernel="pqk",
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
        kernel="pqk",
    )

    assert calls == [
        {
            "training_size": 20,
            "test_size": 10,
            "n_features": 3,
            "random_state": 55,
        }
    ]
    assert fake_env.evol.call_count == 1
    fake_env.evol.assert_called_with(verbose=False, mode="parallel")


def test_train_environment_fitness_function_is_picklable(monkeypatch):
    from ga_qsvm.runners import train as train_module

    captured = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train_module, "EEnvironment", FakeEnvironment)
    monkeypatch.setattr(train_module, "build_train_wandb_config", lambda *args: None)

    train_module.build_train_environment(
        dataset_name="digits",
        params={
            "num_qubits": 2,
            "num_cnot": 20,
            "depth": 4,
            "num_circuit": 4,
            "num_generation": 1,
            "prob_mutate": 0.1,
            "kernel": "fqk",
        },
        machine_id=0,
        index=0,
        dataset_split=(
            np.zeros((4, 2)),
            np.zeros((2, 2)),
            np.array([0, 1, 0, 1]),
            np.array([0, 1]),
        ),
    )

    pickle.dumps(captured["fitness_func"])
    assert captured["fitness_func"].__name__ == "train_fqk_qsvm"


def test_train_environment_selects_projected_kernel(monkeypatch):
    from ga_qsvm.runners import train as train_module

    captured = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train_module, "EEnvironment", FakeEnvironment)
    monkeypatch.setattr(train_module, "build_train_wandb_config", lambda *args: None)

    params = {
        "num_qubits": 2,
        "num_cnot": 20,
        "depth": 4,
        "num_circuit": 4,
        "num_generation": 1,
        "prob_mutate": 0.1,
        "kernel": "pqk",
    }
    train_module.build_train_environment(
        dataset_name="digits",
        params=params,
        machine_id=0,
        index=0,
        dataset_split=(
            np.zeros((4, 2)),
            np.zeros((2, 2)),
            np.array([0, 1, 0, 1]),
            np.array([0, 1]),
        ),
    )

    assert isinstance(captured["fitness_func"], TrainProjectedQSVMFitness)
    assert captured["fitness_func"].__name__ == "train_pqk_qsvm"


def test_train_environment_selects_fidelity_kernel(monkeypatch):
    from ga_qsvm.runners import train as train_module

    captured = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train_module, "EEnvironment", FakeEnvironment)
    monkeypatch.setattr(train_module, "build_train_wandb_config", lambda *args: None)

    params = {
        "num_qubits": 2,
        "num_cnot": 20,
        "depth": 4,
        "num_circuit": 4,
        "num_generation": 1,
        "prob_mutate": 0.1,
        "kernel": "fqk",
    }
    train_module.build_train_environment(
        dataset_name="digits",
        params=params,
        machine_id=0,
        index=0,
        dataset_split=(
            np.zeros((4, 2)),
            np.zeros((2, 2)),
            np.array([0, 1, 0, 1]),
            np.array([0, 1]),
        ),
    )

    assert isinstance(captured["fitness_func"], TrainFidelityQSVMFitness)


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
