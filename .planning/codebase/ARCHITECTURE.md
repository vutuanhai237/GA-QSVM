# Architecture

**Analysis Date:** 2026-04-26

## Pattern Overview

**Overall:** Research-oriented script orchestration over a local quantum optimization library.

**Key Characteristics:**
- CLI scripts (`main.py`, `eval.py`) assemble datasets, QOOP genetic operators, QSVM fitness functions, hyperparameter grids, W&B logging, and persistence directly.
- Reusable quantum optimization logic lives under `qoop/`, split into backend utilities, circuit/core math, genetic evolution, compilation, Hamiltonian, and VQE modules.
- Dataset preparation lives in `data/`, but several `experiment/*.py` files duplicate preprocessing and benchmark logic inline.
- Experiment scripts and notebooks under `experiment/` and `notebook/` are execution/research artifacts, not clean library boundaries.
- Qiskit `QuantumCircuit` is the central object crossing package boundaries: QOOP generators/mutators produce circuits, QSVM fitness functions consume circuits, and QPY persistence stores circuits.

## Layers

**Command-Line Experiment Drivers:**
- Purpose: Run GA-QSVM searches and fixed-parameter evaluation.
- Location: `main.py`, `eval.py`
- Contains: argument parsing, hyperparameter construction, dataset selection, closure-based fitness functions, `MetadataSynthesis` creation, `EEnvironment` assembly, W&B run configuration.
- Depends on: `data/`, `utils.py`, `qoop/evolution/`, `qoop/backend/constant.py`, `qiskit_machine_learning`, `sklearn`, `numpy`, `wandb`.
- Used by: direct shell/PBS execution through `python main.py ...`, `python eval.py ...`, `bash/digits.sh`, `bash/eval0.sh`, `bash/eval1.sh`, and `bash/eval2.sh`.

**Dataset Preparation:**
- Purpose: Load Wine, Digits, Breast Cancer, MNIST, and synthetic datasets; split, scale, and apply PCA.
- Location: `data/`
- Contains: `data/split.py`, `data/cv.py`, `data/__init__.py`.
- Depends on: `sklearn.datasets`, `sklearn.model_selection`, `sklearn.preprocessing`, `sklearn.decomposition`, `sklearn.utils`, `numpy`, and TensorFlow inside `prepare_mnist_data()`.
- Used by: `main.py`, `eval.py`, and selected `experiment/*.py`.

**QOOP Evolution Engine:**
- Purpose: Generic genetic algorithm loop over quantum circuits.
- Location: `qoop/evolution/`
- Contains: environment state/lifecycle (`qoop/evolution/environment.py`), metadata dataclasses (`qoop/evolution/environment_parent.py`, `qoop/evolution/environment_synthesis.py`), circuit generation (`qoop/evolution/generator.py`), selection (`qoop/evolution/selection.py`), crossover (`qoop/evolution/crossover.py`), mutation (`qoop/evolution/mutate.py`), normalization (`qoop/evolution/normalizer.py`), splitting (`qoop/evolution/divider.py`), thresholds (`qoop/evolution/threshold.py`), and utilities.
- Depends on: `qiskit`, `numpy`, `concurrent.futures`, `wandb`, `qoop/backend/`, and `qoop/core/`.
- Used by: `main.py`, `eval.py`, notebooks, and QOOP compilation modules.

**QOOP Core Quantum Math:**
- Purpose: Build ansatz circuits, evaluate metrics, measure circuits, calculate gradients, optimize parameters, generate random circuits, visualize, and provide parallel measurement helpers.
- Location: `qoop/core/`
- Contains: `qoop/core/ansatz.py`, `qoop/core/metric.py`, `qoop/core/measure.py`, `qoop/core/gradient.py`, `qoop/core/optimizer.py`, `qoop/core/random_circuit.py`, `qoop/core/state.py`, `qoop/core/parallel.py`, `qoop/core/dag.py`, `qoop/core/visualize.py`.
- Depends on: `qiskit`, `qiskit.quantum_info`, `qiskit.primitives.Sampler`, `scipy`, `numpy`, `pandas`, `matplotlib`, optional `pennylane` and `torch` in `qoop/core/dag.py`.
- Used by: `qoop/evolution/`, `qoop/compilation/`, and library consumers.

**QOOP Backend Support:**
- Purpose: Constants, gate pools, enum names, utility functions for circuit layering, composition, serialization, and progress.
- Location: `qoop/backend/`
- Contains: `qoop/backend/constant.py`, `qoop/backend/utilities.py`, empty `qoop/backend/__init__.py`.
- Depends on: `qiskit`, `scipy`, `tqdm`, `numpy`.
- Used by: `qoop/core/`, `qoop/evolution/`, `qoop/compilation/`, `main.py`, and `eval.py`.

**QOOP Compilation/VQE/Hamiltonian Modules:**
- Purpose: General QOOP library features outside the GA-QSVM driver path.
- Location: `qoop/compilation/`, `qoop/vqe/`, `qoop/hamiltonian/`
- Contains: `qoop/compilation/qcompilation.py`, `qoop/compilation/qsp.py`, `qoop/compilation/qst.py`, `qoop/vqe/vqe.py`, `qoop/vqe/utilities.py`, `qoop/hamiltonian/model.py`.
- Depends on: `qiskit`, `qiskit_nature`, `qiskit_algorithms`, `numpy`, `matplotlib`, QOOP core/backend modules.
- Used by: library users and notebooks; not part of the main GA-QSVM execution path.

**Research Scripts and Notebooks:**
- Purpose: Standalone baselines, one-off comparisons, GPU benchmarks, and external cloud demos.
- Location: `experiment/`, `notebook/`
- Contains: SVC/QSVC dataset scripts, PennyLane GPU kernel benchmarks, Covalent Cloud notebook, and exploratory notebooks.
- Depends on: `sklearn`, `qiskit_machine_learning`, `pennylane`, `wandb`, optional `cupy`, `thundersvm`, `covalent`, and `covalent_cloud`.
- Used by: manual execution only.

## Data Flow

**GA-QSVM Search (`main.py`):**

1. `main.py` parses CLI hyperparameters and selects a data loader from `data/__init__.py`.
2. For each qubit count, `data/split.py` returns train/test arrays after train/test split, `MinMaxScaler`, and PCA.
3. `utils.find_permutations_sum_n()` enumerates RX/RY/RZ count combinations for a given qubit count.
4. `main.py` builds `MetadataSynthesis` and passes QOOP operator functions into `qoop/evolution/environment.py:EEnvironment`.
5. `EEnvironment.init()` calls `qoop/evolution/generator.py:by_num_rotations_and_cnot()` until the population contains parameterized circuits.
6. `EEnvironment.evol()` evaluates each circuit through the `train_qsvm()` closure, which trains `qiskit_machine_learning.algorithms.QSVC` with `FidelityQuantumKernel`.
7. `EEnvironment.evol()` logs generation metrics to W&B, performs selection/crossover/mutation, normalizes circuits, saves QPY/JSON outputs, and exits on threshold or no improvement.

**Fixed-Configuration Evaluation (`eval.py`):**

1. `eval.py` parses RX/RY/RZ, qubits, mutation probability, and dataset arguments.
2. `data/split.py` returns train/test arrays, then `eval.py` splits the test set into validation and evaluation subsets.
3. `train_qsvm()` returns both validation accuracy and held-out evaluation accuracy for each generated circuit.
4. `EEnvironment` uses validation accuracy for selection while tracking evaluation accuracy for the best validation circuit.

**Classical and GPU Benchmarks (`experiment/`):**

1. SVC baseline scripts load/scales/PCA datasets in-place or through `data/cv.py`.
2. QSVC scripts create Qiskit feature maps and train `QSVC`.
3. GPU benchmark scripts create Qiskit feature maps, convert them to PennyLane templates with `qml.from_qiskit()`, evaluate kernels on `lightning.gpu`, and optionally fit ThunderSVM with CuPy arrays.
4. Benchmark metrics are logged to W&B in `experiment/5_benchmark_CPU.py`, `experiment/5_benchmark_GPU.py`, and `experiment/5_benchmark_GPU_testSVC.py`.

**State Management:**
- Runtime search state is held in `EEnvironment` fields in `qoop/evolution/environment.py`: `metadata`, `fitnesss`, `eval_fitnesss`, `circuits`, `circuitss`, `best_circuit`, `best_circuits`, `best_fitness`, and `best_eval_fitness`.
- Persistent experiment state is written by `EEnvironment.save()` to local JSON and QPY files; `.gitignore` excludes these generated artifacts.
- Dataset state is not persisted; loaders recompute split/scale/PCA on each run.
- W&B state is external and driven by `wandb.init()`, `wandb.log()`, `wandb.finish()`, and alerts.

## Key Abstractions

**`EEnvironment`:**
- Purpose: Owns the genetic algorithm lifecycle for quantum circuit populations.
- Examples: `qoop/evolution/environment.py`, constructed in `main.py` and `eval.py`.
- Pattern: Strategy injection through `fitness_func`, `generator_func`, `crossover_func`, `mutate_func`, `selection_func`, and `threshold_func`.

**`Metadata` and `MetadataSynthesis`:**
- Purpose: Carry GA search parameters and generation history.
- Examples: `qoop/evolution/environment_parent.py`, `qoop/evolution/environment_synthesis.py`.
- Pattern: dataclass configuration object passed into generators and environment.

**Quantum Circuit Generators:**
- Purpose: Produce candidate `qiskit.QuantumCircuit` feature maps.
- Examples: `qoop/evolution/generator.py:by_depth()`, `qoop/evolution/generator.py:by_num_rotations()`, `qoop/evolution/generator.py:by_num_rotations_and_cnot()`, `qoop/evolution/generator.py:by_num_rotations_and_cnot_gpu()`.
- Pattern: metadata-driven factory functions over Qiskit gate pools from `qoop/backend/constant.py`.

**Genetic Operators:**
- Purpose: Transform candidate circuits between generations.
- Examples: `qoop/evolution/selection.py`, `qoop/evolution/crossover.py`, `qoop/evolution/mutate.py`, `qoop/evolution/normalizer.py`, `qoop/evolution/divider.py`.
- Pattern: function closures configured in `main.py` and `eval.py`.

**Fitness Function Closure:**
- Purpose: Converts a circuit into validation/test accuracy by training a QSVM against in-scope dataset arrays.
- Examples: `main.py:train_qsvm()`, `eval.py:train_qsvm()`.
- Pattern: closure over `Xw_train`, `Xw_test`, `yw_train`, `yw_test`; this creates a hidden dependency between data loading and `EEnvironment.evol()`.

**Gate Pools and Constants:**
- Purpose: Define allowed gates, metrics, enum names, and algorithm constants.
- Examples: `qoop/backend/constant.py`.
- Pattern: module-level mutable configuration consumed throughout QOOP modules.

## Entry Points

**GA-QSVM Training CLI:**
- Location: `main.py`
- Triggers: `python main.py ...`, `bash/digits.sh`.
- Responsibilities: Hyperparameter sweep, dataset loading, W&B config, QOOP environment construction, full GA run.

**GA-QSVM Evaluation CLI:**
- Location: `eval.py`
- Triggers: `python eval.py ...`, `bash/eval0.sh`, `bash/eval1.sh`, `bash/eval2.sh`.
- Responsibilities: Fixed RX/RY/RZ search with validation/evaluation split and W&B logging.

**Classical/QSVC Baselines:**
- Location: `experiment/0_svc_wine_0.py`, `experiment/0_svc_wine_1.py`, `experiment/1_svc_mnist.py`, `experiment/2_svc_digits.py`, `experiment/3_svc_cancer.py`, `experiment/3_svc_cancer_test_holdout.py`, `experiment/3_svc_cancer_test_holdout_weird_setup.py`, `experiment/0_qsvm_wine_1.py`, `experiment/2_qsvm_digits.py`, `experiment/3_qsvm_cancer.py`.
- Triggers: direct `python experiment/<script>.py`.
- Responsibilities: Dataset-specific baseline fitting, grid search, QSVC fitting, W&B metrics.

**CPU/GPU Benchmark Scripts:**
- Location: `experiment/5_benchmark_CPU.py`, `experiment/5_benchmark_GPU.py`, `experiment/5_benchmark_GPU_testSVC.py`, `experiment/5_pennylane_qsvm.py`.
- Triggers: direct `python experiment/<script>.py`.
- Responsibilities: Compare Qiskit QSVC and PennyLane/Lightning GPU kernel computation.

**Notebook Entry Points:**
- Location: `notebook/*.ipynb`, `experiment/*.ipynb`.
- Triggers: Jupyter execution.
- Responsibilities: Exploratory analysis, cloud GPU demo, and experimental scratch work.

## Error Handling

**Strategy:** Exceptions are mostly allowed to propagate; validation is local and minimal.

**Patterns:**
- Argument constraints are limited to `argparse` type declarations and `choices` in `eval.py`.
- Generator validation raises `ValueError` for impossible gate counts in `qoop/evolution/generator.py`.
- `EEnvironment.init()` raises `ValueError` when no fitness function is set in `qoop/evolution/environment.py`.
- `EEnvironment.evol()` uses threshold and no-improvement early stopping rather than exception-based flow.
- `EEnvironment.load()` catches all exceptions and returns `None`, obscuring load failures in `qoop/evolution/environment.py`.
- W&B alert at the end of `main.py` assumes `wandb.run` is available after finishing runs.

## Cross-Cutting Concerns

**Logging:** `print()` statements across `main.py`, `eval.py`, `data/`, and `qoop/evolution/environment.py`; W&B metrics in experiment scripts and `EEnvironment`; root logger setup in `qoop/backend/constant.py`.

**Validation:** Dataset and parameter validation is ad hoc. `data/split.py` accepts `training_size`, `test_size`, and `random_state` parameters but some functions use hard-coded sizes and seeds internally.

**Authentication:** External SDK auth only. W&B authentication is implicit. Covalent Cloud API key appears only in notebook text in `experiment/4_covalent_cloud_gpu.ipynb`.

## Refactor-Relevant Architectural Concerns

- `main.py` executes `parse_args()` and initializes global variables at import time; keep it as a CLI script or move orchestration into a `main()` function before importing it from tests or other modules.
- `main.py:train_qsvm()` and `eval.py:train_qsvm()` depend on global variables populated inside `if __name__ == "__main__"`, which makes the fitness function hard to reuse and hard to parallelize safely.
- `qoop/evolution/environment.py` mixes library logic with W&B initialization/logging, plotting, filesystem persistence, multiprocessing, and early stopping; extract adapters before changing GA behavior.
- `data/split.py` exposes size parameters but uses fixed train/test sizes in several functions, which makes CLI parameters in `main.py` partially ineffective.
- `experiment/*.py` duplicates dataset preparation and feature-map construction that already exists in `data/` or can be promoted into reusable benchmark helpers.
- `qoop/setup.py` does not package `qoop/`, so installable-package refactors should start by correcting packaging metadata.

---

*Architecture analysis: 2026-04-26*
