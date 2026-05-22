# Codebase Structure

**Analysis Date:** 2026-04-26

## Directory Layout

```text
GA-QSVM/
├── main.py                 # Primary GA-QSVM hyperparameter search CLI
├── eval.py                 # Fixed-configuration GA-QSVM evaluation CLI
├── utils.py                # Small top-level helper functions and QPY loading helper
├── requirements.txt        # Minimal pip dependency pins for the GA-QSVM project
├── environment.yml         # Full Conda environment pin set
├── README.md               # Project overview, install, and CLI usage
├── data/                   # Reusable dataset loading, splitting, scaling, and PCA helpers
├── qoop/                   # Local quantum object optimizer library
├── experiment/             # Standalone experiments, baselines, GPU benchmarks, notebooks
├── notebook/               # Exploratory notebooks
├── bash/                   # PBS/manual shell execution wrappers
├── images/                 # README/diagram assets
├── wandb/                  # Local W&B run artifacts; ignored by .gitignore
└── .planning/codebase/     # Generated codebase mapping documents
```

## Directory Purposes

**Root Scripts:**
- Purpose: User-facing execution surfaces for GA-QSVM.
- Contains: `main.py`, `eval.py`, `utils.py`.
- Key files: `main.py` for full hyperparameter search; `eval.py` for fixed RX/RY/RZ evaluation; `utils.py` for rotation-count permutations and QPY circuit loading.

**`data/`:**
- Purpose: Shared dataset preparation.
- Contains: `data/split.py` for fixed split helpers used by `main.py` and `eval.py`; `data/cv.py` for cross-validation-style and MNIST helpers; `data/__init__.py` for public imports.
- Key files: `data/split.py`, `data/cv.py`, `data/__init__.py`.

**`qoop/`:**
- Purpose: Local reusable quantum optimization package.
- Contains: Backend utilities/constants, core quantum math, evolution engine, compilation helpers, Hamiltonian helpers, VQE helpers, package metadata, README, license, and project docs.
- Key files: `qoop/evolution/environment.py`, `qoop/evolution/generator.py`, `qoop/evolution/crossover.py`, `qoop/evolution/mutate.py`, `qoop/evolution/selection.py`, `qoop/backend/constant.py`, `qoop/backend/utilities.py`, `qoop/core/ansatz.py`, `qoop/core/metric.py`, `qoop/setup.py`.

**`qoop/backend/`:**
- Purpose: Shared low-level constants and Qiskit circuit utilities.
- Contains: gate pools, enum names, metric names, QPY save/load, circuit layering/composition helpers, progress bar.
- Key files: `qoop/backend/constant.py`, `qoop/backend/utilities.py`.

**`qoop/core/`:**
- Purpose: Quantum circuit construction and numerical primitives.
- Contains: ansatz builders, metrics, measurement, gradients, optimizers, random circuits, state helpers, DAG/ML helper, visualization, parallel measurement helper.
- Key files: `qoop/core/ansatz.py`, `qoop/core/metric.py`, `qoop/core/measure.py`, `qoop/core/gradient.py`, `qoop/core/optimizer.py`, `qoop/core/random_circuit.py`, `qoop/core/state.py`.

**`qoop/evolution/`:**
- Purpose: Genetic algorithm over Qiskit circuits.
- Contains: environment lifecycle, metadata dataclasses, generator, crossover, mutation, normalizer, divider, selection, threshold, and evolution utilities.
- Key files: `qoop/evolution/environment.py`, `qoop/evolution/environment_parent.py`, `qoop/evolution/environment_synthesis.py`, `qoop/evolution/generator.py`, `qoop/evolution/crossover.py`, `qoop/evolution/mutate.py`, `qoop/evolution/normalizer.py`, `qoop/evolution/divider.py`.

**`qoop/compilation/`:**
- Purpose: QOOP quantum compilation and state preparation/tomography support.
- Contains: `qoop/compilation/qcompilation.py`, `qoop/compilation/qsp.py`, `qoop/compilation/qst.py`.
- Key files: `qoop/compilation/qcompilation.py`, `qoop/compilation/qsp.py`.

**`qoop/vqe/` and `qoop/hamiltonian/`:**
- Purpose: Additional QOOP library domains for VQE and Hamiltonian modeling.
- Contains: `qoop/vqe/vqe.py`, `qoop/vqe/utilities.py`, `qoop/hamiltonian/model.py`.
- Key files: `qoop/vqe/vqe.py`, `qoop/hamiltonian/model.py`.

**`experiment/`:**
- Purpose: Standalone research scripts and notebooks outside the reusable library path.
- Contains: classical SVC baselines, QSVC experiments, CPU/GPU benchmarks, Covalent Cloud notebook, PennyLane notebooks.
- Key files: `experiment/5_benchmark_CPU.py`, `experiment/5_benchmark_GPU.py`, `experiment/5_benchmark_GPU_testSVC.py`, `experiment/5_pennylane_qsvm.py`, `experiment/0_qsvm_wine_1.py`, `experiment/2_qsvm_digits.py`, `experiment/3_qsvm_cancer.py`.

**`notebook/`:**
- Purpose: Exploratory notebooks.
- Contains: `notebook/test.ipynb`, `notebook/test_classical.ipynb`.
- Key files: `notebook/test.ipynb`, `notebook/test_classical.ipynb`.

**`bash/`:**
- Purpose: Shell wrappers for batch/manual experiment execution.
- Contains: PBS job script and grouped evaluation commands.
- Key files: `bash/digits.sh`, `bash/eval0.sh`, `bash/eval1.sh`, `bash/eval2.sh`.

**`images/`:**
- Purpose: Static diagram/example assets.
- Contains: SVG, PDF, and PNG files.
- Key files: `images/example.svg`, `images/example.pdf`, `images/example.png`, `images/diagram.svg`, `images/diagram.pdf`.

**`wandb/`:**
- Purpose: Local W&B run artifacts.
- Contains: run directories created by W&B.
- Key files: ignored generated artifacts; do not place source code here.

## Key File Locations

**Entry Points:**
- `main.py`: Primary GA-QSVM search CLI with hyperparameter grid and W&B run naming.
- `eval.py`: Fixed RX/RY/RZ GA-QSVM evaluation CLI with validation/evaluation split.
- `bash/digits.sh`: PBS cluster wrapper for `main.py`.
- `bash/eval0.sh`, `bash/eval1.sh`, `bash/eval2.sh`: Manual grouped `eval.py` commands.
- `experiment/*.py`: Standalone baselines and benchmarks.

**Configuration:**
- `requirements.txt`: Top-level pip dependency pins.
- `environment.yml`: Full Conda environment.
- `.gitignore`: Generated artifacts, virtualenvs, W&B runs, QPY files, JSON outputs, and secrets exclusions.
- `qoop/setup.py`: Local QOOP packaging metadata.
- `qoop/requirements.txt`: QOOP-specific dependency pins.

**Core Logic:**
- `qoop/evolution/environment.py`: GA lifecycle, multiprocessing fitness evaluation, W&B logging, early stopping, persistence.
- `qoop/evolution/generator.py`: Circuit population generation by depth, rotation counts, and GPU-oriented parameter vectors.
- `qoop/evolution/crossover.py`: Crossover operators.
- `qoop/evolution/mutate.py`: Mutation operators.
- `qoop/evolution/selection.py`: Circuit validity and selection operators.
- `qoop/evolution/normalizer.py`: Circuit shape/count normalization helpers.
- `qoop/evolution/divider.py`: Circuit splitting helpers for crossover.
- `qoop/backend/constant.py`: Gate pools and global algorithm constants.
- `qoop/backend/utilities.py`: QPY persistence, circuit layer extraction, composition, progress helpers.
- `qoop/core/metric.py`: Loss and quantum metric calculations.
- `qoop/core/measure.py`: Qiskit Sampler measurement helper.
- `data/split.py`: Fixed split helpers for primary GA-QSVM runs.
- `data/cv.py`: Cross-validation-style data helpers and MNIST loader.

**Testing:**
- No committed test directory or `*.test.py`/`*_test.py` files detected.
- No pytest config detected.
- Exploratory notebooks under `notebook/` are not structured tests.

## Naming Conventions

**Files:**
- Top-level execution scripts use concise names: `main.py`, `eval.py`, `utils.py`.
- Dataset helpers use functional names: `data/split.py`, `data/cv.py`.
- Experiment scripts use numbered prefixes and dataset/model names: `experiment/0_qsvm_wine_1.py`, `experiment/2_svc_digits.py`, `experiment/5_benchmark_GPU.py`.
- QOOP modules use domain names: `qoop/evolution/environment.py`, `qoop/core/metric.py`, `qoop/backend/utilities.py`.
- Shell scripts use command/domain names: `bash/digits.sh`, `bash/eval0.sh`.

**Directories:**
- Domain directories under `qoop/` are lowercase nouns: `backend`, `core`, `evolution`, `compilation`, `hamiltonian`, `vqe`.
- Research and execution artifacts are grouped separately: `experiment/`, `notebook/`, `bash/`.

## Where to Add New Code

**New GA-QSVM Feature:**
- Primary code: add reusable GA behavior under `qoop/evolution/`.
- CLI wiring: update `main.py` or `eval.py` only after the reusable behavior exists.
- Dataset handling: add shared loaders/splits in `data/`, then import through `data/__init__.py`.
- Tests: add a new test tree such as `tests/` because no test structure exists.

**New Experiment Script:**
- Implementation: add one-off or paper-reproduction scripts under `experiment/`.
- Shared logic: extract duplicated preprocessing or feature-map code into `data/` or a reusable module before adding another copied helper.
- Batch wrapper: add shell/PBS command files under `bash/` when needed.

**New QOOP Library Module:**
- Implementation: place core circuit/math code under `qoop/core/`, GA operators under `qoop/evolution/`, gate/constants/filesystem helpers under `qoop/backend/`, and compilation-specific code under `qoop/compilation/`.
- Public import: update the relevant `__init__.py` only when a stable public surface is intended; most `qoop/**/__init__.py` files are empty.

**Utilities:**
- Top-level experiment-only helpers: `utils.py`.
- Circuit persistence/composition helpers used by QOOP modules: `qoop/backend/utilities.py`.
- Dataset preprocessing helpers: `data/`.

**Configuration:**
- Python dependency changes for active GA-QSVM runs: `requirements.txt` and `environment.yml`.
- Local QOOP package metadata: `qoop/setup.py` and `qoop/requirements.txt`, after reconciling package naming and dependency versions.

## Special Directories

**`wandb/`:**
- Purpose: Local W&B run output.
- Generated: Yes.
- Committed: No; `.gitignore` excludes `wandb/` and `wandb/latest-run`.

**Generated QPY/JSON Result Directories:**
- Purpose: Circuit and GA state output from `qoop/evolution/environment.py`.
- Generated: Yes.
- Committed: No; `.gitignore` excludes `*.qpy` and `*.json`.

**`experiment/`:**
- Purpose: Research scripts and notebooks.
- Generated: No.
- Committed: Yes.

**`notebook/`:**
- Purpose: Exploratory notebooks.
- Generated: No.
- Committed: Yes.

**`images/`:**
- Purpose: Diagram and README assets.
- Generated: No.
- Committed: Yes.

## Package Boundaries

- `qoop/` is the reusable library boundary, but its packaging metadata in `qoop/setup.py` names `qsee` instead of `qoop`.
- `data/` is shared application support code for dataset preparation.
- `main.py` and `eval.py` are orchestration scripts and should remain thin entry points after refactoring.
- `experiment/`, `notebook/`, and `bash/` are operational/research surfaces; avoid importing from them in reusable modules.
- W&B integration crosses the boundary today because `qoop/evolution/environment.py` calls `wandb.init()` and `wandb.log()` directly.

## Refactor Placement Guidance

- Put pure circuit generation, mutation, crossover, and selection changes in `qoop/evolution/` and pass them into `EEnvironment` from CLI scripts.
- Put QSVM fitness construction in a dedicated reusable module rather than keeping it as a closure in `main.py` and `eval.py`.
- Put W&B-specific code in an experiment tracking adapter used by `main.py` and `eval.py`; keep `qoop/` usable without W&B for library contexts.
- Put reusable CPU/GPU benchmark helpers in a new module under `experiment/` or a clearly named benchmark package; keep `qoop/` free of ThunderSVM/CuPy/Covalent requirements.
- Put any new tests in `tests/`, mirroring `data/`, `qoop/evolution/`, and CLI orchestration behavior.

---

*Structure analysis: 2026-04-26*
