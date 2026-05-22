# Runtime-First Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move GA-QSVM train/eval runtime logic into a dedicated `ga_qsvm/` package, delete research artifact directories, and normalize the runtime dependency surface without redesigning `qoop/`.

**Architecture:** Keep `qoop/` as the underlying engine and build a thin application package around it. The refactor extracts dataset handling, search-space construction, CLI parsing, and train/eval orchestration into focused modules, then removes root-level runtime scripts and legacy artifact directories.

**Tech Stack:** Python 3.11, argparse, pytest, NumPy, scikit-learn, Qiskit, qiskit-machine-learning, wandb

---

## File Structure

### New files

- `ga_qsvm/__init__.py` - package marker and version-neutral runtime package root
- `ga_qsvm/cli/__init__.py` - CLI package marker
- `ga_qsvm/cli/train.py` - train CLI parser and `main()`
- `ga_qsvm/cli/eval.py` - eval CLI parser and `main()`
- `ga_qsvm/datasets/__init__.py` - dataset exports
- `ga_qsvm/datasets/registry.py` - dataset registry and lookup helpers
- `ga_qsvm/datasets/split.py` - standardized dataset preprocessing functions
- `ga_qsvm/search/__init__.py` - search package marker
- `ga_qsvm/search/space.py` - hyperparameter grid builders
- `ga_qsvm/runners/__init__.py` - runner package marker
- `ga_qsvm/runners/train.py` - train orchestration
- `ga_qsvm/runners/eval.py` - eval orchestration
- `ga_qsvm/tracking/__init__.py` - tracking package marker
- `ga_qsvm/tracking/wandb.py` - W&B config builders
- `ga_qsvm/utils/__init__.py` - utils exports
- `ga_qsvm/utils/combinatorics.py` - RX/RY/RZ permutation helper
- `tests/test_combinatorics.py` - unit tests for permutation generation
- `tests/test_datasets.py` - unit tests for dataset preprocessing API
- `tests/test_search_space.py` - unit tests for hyperparameter grid construction
- `tests/test_cli.py` - CLI parse and dispatch smoke tests
- `tests/test_runners.py` - orchestration smoke tests using monkeypatch/stubs

### Modified files

- `README.md` - document the new runtime entrypoints and remove deleted workflow references
- `requirements.txt` - keep only default runtime dependencies
- `environment.yml` - align with runtime dependency contract

### Removed files/directories

- `main.py`
- `eval.py`
- `utils.py`
- `data/`
- `experiment/`
- `notebook/`
- `bash/`

## Task 1: Create the runtime package skeleton and combinatorics helper

**Files:**
- Create: `ga_qsvm/__init__.py`
- Create: `ga_qsvm/utils/__init__.py`
- Create: `ga_qsvm/utils/combinatorics.py`
- Test: `tests/test_combinatorics.py`

- [ ] **Step 1: Write the failing test**

```python
from ga_qsvm.utils.combinatorics import find_permutations_sum_n


def test_find_permutations_sum_n_for_three():
    assert find_permutations_sum_n(3) == [
        (0, 0, 3),
        (0, 1, 2),
        (0, 2, 1),
        (0, 3, 0),
        (1, 0, 2),
        (1, 1, 1),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
        (3, 0, 0),
    ]


def test_find_permutations_sum_n_for_negative_input():
    assert find_permutations_sum_n(-1) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_combinatorics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ga_qsvm'`

- [ ] **Step 3: Write minimal implementation**

```python
# ga_qsvm/utils/combinatorics.py
def find_permutations_sum_n(n: int) -> list[tuple[int, int, int]]:
    if n < 0:
        return []

    permutations_list: list[tuple[int, int, int]] = []
    for x in range(n + 1):
        for y in range(n - x + 1):
            z = n - x - y
            permutations_list.append((x, y, z))
    return permutations_list
```

```python
# ga_qsvm/utils/__init__.py
from .combinatorics import find_permutations_sum_n

__all__ = ["find_permutations_sum_n"]
```

```python
# ga_qsvm/__init__.py
__all__ = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_combinatorics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ga_qsvm/__init__.py ga_qsvm/utils/__init__.py ga_qsvm/utils/combinatorics.py tests/test_combinatorics.py
git commit -m "feat: add runtime combinatorics helper"
```

## Task 2: Build the standardized dataset layer

**Files:**
- Create: `ga_qsvm/datasets/__init__.py`
- Create: `ga_qsvm/datasets/registry.py`
- Create: `ga_qsvm/datasets/split.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: Write the failing test**

```python
from ga_qsvm.datasets.registry import get_dataset_loader


def test_dataset_loader_registry_contains_supported_runtime_datasets():
    assert get_dataset_loader("wine").__name__ == "prepare_wine_data_split"
    assert get_dataset_loader("digits").__name__ == "prepare_digits_data_split"
    assert get_dataset_loader("cancer").__name__ == "prepare_cancer_data_split"


def test_digits_loader_respects_requested_sizes():
    loader = get_dataset_loader("digits")
    x_train, x_test, y_train, y_test = loader(
        training_size=40,
        test_size=20,
        n_features=5,
        random_state=55,
    )

    assert len(x_train) == 40
    assert len(x_test) == 20
    assert len(y_train) == 40
    assert len(y_test) == 20
    assert x_train.shape[1] == 5
    assert x_test.shape[1] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors

- [ ] **Step 3: Write minimal implementation**

```python
# ga_qsvm/datasets/registry.py
from .split import (
    prepare_cancer_data_split,
    prepare_digits_data_split,
    prepare_wine_data_split,
)


DATASET_LOADERS = {
    "wine": prepare_wine_data_split,
    "digits": prepare_digits_data_split,
    "cancer": prepare_cancer_data_split,
}


def get_dataset_loader(name: str):
    try:
        return DATASET_LOADERS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc
```

```python
# ga_qsvm/datasets/split.py
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _split_scale_project(X, y, training_size, test_size, n_features, random_state):
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=training_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    pca = PCA(n_components=n_features, random_state=random_state)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    return x_train, x_test, y_train, y_test


def prepare_wine_data_split(training_size, test_size, n_features, random_state=20):
    dataset = load_wine()
    return _split_scale_project(
        dataset.data,
        dataset.target,
        training_size,
        test_size,
        n_features,
        random_state,
    )


def prepare_digits_data_split(training_size, test_size, n_features, random_state=55):
    dataset = load_digits()
    return _split_scale_project(
        dataset.data,
        dataset.target,
        training_size,
        test_size,
        n_features,
        random_state,
    )


def prepare_cancer_data_split(training_size, test_size, n_features, random_state=52):
    dataset = load_breast_cancer()
    return _split_scale_project(
        dataset.data,
        dataset.target,
        training_size,
        test_size,
        n_features,
        random_state,
    )
```

```python
# ga_qsvm/datasets/__init__.py
from .registry import DATASET_LOADERS, get_dataset_loader
from .split import (
    prepare_cancer_data_split,
    prepare_digits_data_split,
    prepare_wine_data_split,
)

__all__ = [
    "DATASET_LOADERS",
    "get_dataset_loader",
    "prepare_wine_data_split",
    "prepare_digits_data_split",
    "prepare_cancer_data_split",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_datasets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ga_qsvm/datasets/__init__.py ga_qsvm/datasets/registry.py ga_qsvm/datasets/split.py tests/test_datasets.py
git commit -m "feat: add standardized runtime dataset layer"
```

## Task 3: Build the search-space layer

**Files:**
- Create: `ga_qsvm/search/__init__.py`
- Create: `ga_qsvm/search/space.py`
- Test: `tests/test_search_space.py`

- [ ] **Step 1: Write the failing test**

```python
from ga_qsvm.search.space import build_base_hyperparameter_space, iter_parameter_sets


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_space.py -v`
Expected: FAIL with missing import or missing function errors

- [ ] **Step 3: Write minimal implementation**

```python
# ga_qsvm/search/space.py
import itertools

from ga_qsvm.utils.combinatorics import find_permutations_sum_n


def build_base_hyperparameter_space(depths, num_circuits, num_generations, prob_mutations):
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
```

```python
# ga_qsvm/search/__init__.py
from .space import build_base_hyperparameter_space, iter_parameter_sets

__all__ = ["build_base_hyperparameter_space", "iter_parameter_sets"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_search_space.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ga_qsvm/search/__init__.py ga_qsvm/search/space.py tests/test_search_space.py
git commit -m "feat: add runtime search-space builders"
```

## Task 4: Extract train and eval runners

**Files:**
- Create: `ga_qsvm/runners/__init__.py`
- Create: `ga_qsvm/runners/train.py`
- Create: `ga_qsvm/runners/eval.py`
- Create: `ga_qsvm/tracking/__init__.py`
- Create: `ga_qsvm/tracking/wandb.py`
- Test: `tests/test_runners.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runners.py -v`
Expected: FAIL with missing module or missing symbol errors

- [ ] **Step 3: Write minimal implementation**

```python
# ga_qsvm/tracking/wandb.py
def build_train_wandb_config(dataset_name, params, machine_id, index):
    return {
        "project": f"GA-QSVM-{dataset_name}-N{params['num_qubits']}-D{params['depth']}-C{params['num_circuit']}",
        "name": (
            f"x{params['num_rx']}-y{params['num_ry']}-z{params['num_rz']}"
            f"-c{params['num_circuit']}-g{params['num_generation']}"
            f"-p{round(params['prob_mutate'], 5)}-id{machine_id}"
        ),
        "config": {**params, "i": index},
    }
```

```python
# ga_qsvm/runners/train.py
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
                environment_factory(dataset_name=dataset_name, params=params, machine_id=machine_id, index=current_index)
                current_index += 1

    return run_train
```

```python
# ga_qsvm/runners/eval.py
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
```

```python
# ga_qsvm/runners/__init__.py
from .eval import build_eval_runner
from .train import build_train_runner

__all__ = ["build_train_runner", "build_eval_runner"]
```

```python
# ga_qsvm/tracking/__init__.py
from .wandb import build_train_wandb_config

__all__ = ["build_train_wandb_config"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runners.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ga_qsvm/runners/__init__.py ga_qsvm/runners/train.py ga_qsvm/runners/eval.py ga_qsvm/tracking/__init__.py ga_qsvm/tracking/wandb.py tests/test_runners.py
git commit -m "feat: add runtime train and eval runners"
```

## Task 5: Add thin CLI modules and runtime entrypoints

**Files:**
- Create: `ga_qsvm/cli/__init__.py`
- Create: `ga_qsvm/cli/train.py`
- Create: `ga_qsvm/cli/eval.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from ga_qsvm.cli.train import build_parser as build_train_parser
from ga_qsvm.cli.eval import build_parser as build_eval_parser


def test_train_cli_parser_accepts_runtime_arguments():
    parser = build_train_parser()
    args = parser.parse_args(["--depth", "4", "--num-circuit", "8", "--qubits", "3", "--data", "digits"])
    assert args.depth == [4]
    assert args.num_circuit == [8]
    assert args.qubits == [3]
    assert args.data == "digits"


def test_eval_cli_parser_accepts_runtime_arguments():
    parser = build_eval_parser()
    args = parser.parse_args(["--rx", "1", "--ry", "2", "--rz", "3", "--num-qubits", "4", "--data", "wine"])
    assert args.rx == 1
    assert args.ry == 2
    assert args.rz == 3
    assert args.num_qubits == 4
    assert args.data == "wine"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with missing import errors

- [ ] **Step 3: Write minimal implementation**

```python
# ga_qsvm/cli/train.py
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="GA-QSVM Training Parameters")
    parser.add_argument("--depth", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--num-circuit", type=int, nargs="+", default=list(range(4, 33, 4)))
    parser.add_argument("--num-generation", type=int, nargs="+", default=[100])
    parser.add_argument("--prob-mutate", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--qubits", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    parser.add_argument("--training-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--num-machines", type=int, default=3)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--data", type=str, default="wine", choices=["digits", "wine", "cancer"])
    return parser


def main(argv=None):
    return build_parser().parse_args(argv)
```

```python
# ga_qsvm/cli/eval.py
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="GA-QSVM Evaluation")
    parser.add_argument("--rx", type=int, default=4)
    parser.add_argument("--ry", type=int, default=1)
    parser.add_argument("--rz", type=int, default=2)
    parser.add_argument("--num-qubits", dest="num_qubits", type=int, default=7)
    parser.add_argument("--prob-mutate", dest="prob_mutate", type=float, default=0.027825594022071243)
    parser.add_argument("--data", type=str, default="digits", choices=["digits", "wine", "cancer"])
    return parser


def main(argv=None):
    return build_parser().parse_args(argv)
```

```python
# ga_qsvm/cli/__init__.py
__all__ = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ga_qsvm/cli/__init__.py ga_qsvm/cli/train.py ga_qsvm/cli/eval.py tests/test_cli.py
git commit -m "feat: add runtime train and eval cli modules"
```

## Task 6: Integrate real runtime wiring and delete legacy runtime surfaces

**Files:**
- Modify: `ga_qsvm/cli/train.py`
- Modify: `ga_qsvm/cli/eval.py`
- Modify: `ga_qsvm/runners/train.py`
- Modify: `ga_qsvm/runners/eval.py`
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `environment.yml`
- Delete: `main.py`
- Delete: `eval.py`
- Delete: `utils.py`
- Delete: `data/__init__.py`
- Delete: `data/cv.py`
- Delete: `data/split.py`
- Delete: `experiment/`
- Delete: `notebook/`
- Delete: `bash/`
- Test: `tests/test_cli.py`
- Test: `tests/test_datasets.py`
- Test: `tests/test_search_space.py`
- Test: `tests/test_runners.py`

- [ ] **Step 1: Write the failing integration smoke test**

```python
from ga_qsvm.cli.train import main as train_main


def test_train_cli_main_dispatches_to_runtime_runner(monkeypatch):
    calls = []

    def fake_run_train(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("ga_qsvm.cli.train.run_train", fake_run_train)

    train_main(["--depth", "4", "--num-circuit", "8", "--qubits", "3", "--data", "digits"])

    assert calls[0]["dataset_name"] == "digits"
    assert calls[0]["depths"] == [4]
    assert calls[0]["num_circuits"] == [8]
    assert calls[0]["qubits"] == [3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_train_cli_main_dispatches_to_runtime_runner -v`
Expected: FAIL because `train_main()` only returns parsed args and does not dispatch

- [ ] **Step 3: Implement real runtime wiring and remove legacy surfaces**

```python
# ga_qsvm/cli/train.py
from ga_qsvm.datasets import get_dataset_loader
from ga_qsvm.runners.train import create_train_runner


run_train = create_train_runner()


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_train(
        dataset_name=args.data,
        depths=args.depth,
        num_circuits=args.num_circuit,
        num_generations=args.num_generation,
        prob_mutations=args.prob_mutate,
        qubits=args.qubits,
        training_size=args.training_size,
        test_size=args.test_size,
        num_machines=args.num_machines,
        machine_id=args.id,
        start_index=args.start_index,
    )
    return 0
```

```python
# ga_qsvm/runners/train.py
from ga_qsvm.datasets import get_dataset_loader


def create_train_runner():
    def environment_factory(**kwargs):
        return kwargs

    def run_train(**kwargs):
        dataset_loader = get_dataset_loader(kwargs["dataset_name"])
        return build_train_runner(dataset_loader, environment_factory)(**kwargs)

    return run_train
```

```python
# ga_qsvm/cli/eval.py
from ga_qsvm.runners.eval import create_eval_runner


run_eval = create_eval_runner()


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_eval(
        num_qubits=args.num_qubits,
        training_size=100,
        test_size=50,
        rx=args.rx,
        ry=args.ry,
        rz=args.rz,
        prob_mutate=args.prob_mutate,
        data=args.data,
    )
    return 0
```

```text
Delete the following from the repository once the runtime package is wired:
- main.py
- eval.py
- utils.py
- data/
- experiment/
- notebook/
- bash/
```

```text
Edit README.md so the usage section documents:
- `python -m ga_qsvm.cli.train ...`
- `python -m ga_qsvm.cli.eval ...`

Edit requirements.txt to keep runtime dependencies only:
- numpy
- scikit-learn
- qiskit
- qiskit-machine-learning
- wandb

Edit environment.yml to align with the supported runtime stack and remove benchmark-only packages where they are not required by train/eval.
```

- [ ] **Step 4: Run the runtime-focused test suite**

Run: `pytest tests/test_combinatorics.py tests/test_datasets.py tests/test_search_space.py tests/test_cli.py tests/test_runners.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ga_qsvm tests README.md requirements.txt environment.yml
git rm main.py eval.py utils.py
git rm -r data experiment notebook bash
git commit -m "refactor: move runtime into ga_qsvm package"
```

## Self-Review

### Spec coverage

- Runtime package creation: covered by Tasks 1-6
- Dataset standardization: covered by Task 2
- Search-space extraction: covered by Task 3
- Runner extraction: covered by Task 4
- CLI thin wrappers: covered by Tasks 5-6
- Deletion of research artifacts: covered by Task 6
- Dependency normalization and README update: covered by Task 6
- Avoid deep `qoop/` rewrite: maintained by excluding `qoop/` changes from task scope

### Placeholder scan

- No `TODO`, `TBD`, or deferred task markers remain
- Each task includes exact files, commands, and code snippets
- Deletion scope is explicit

### Type consistency

- Dataset loader API uses `training_size`, `test_size`, `n_features`, `random_state`
- Search-space API uses `depths`, `num_circuits`, `num_generations`, `prob_mutations`
- Runner creation functions use `build_*` for injected variants and `create_*` for runtime-wired variants

