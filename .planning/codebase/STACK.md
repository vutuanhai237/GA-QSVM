# Technology Stack

**Analysis Date:** 2026-04-26

## Languages

**Primary:**
- Python 3.11.11 - Primary implementation language for experiment drivers, data preparation, and the local quantum optimization package. Version is pinned in `environment.yml`.

**Secondary:**
- Bash - PBS/HPC job wrappers in `bash/digits.sh`, `bash/eval0.sh`, `bash/eval1.sh`, and `bash/eval2.sh`.
- Jupyter Notebook JSON - Exploratory and benchmark notebooks in `notebook/test.ipynb`, `notebook/test_classical.ipynb`, `experiment/4_cutn-qsvm.ipynb`, `experiment/4_covalent_cloud_gpu.ipynb`, `experiment/5_pennylane.ipynb`, and `experiment/6_test.ipynb`.

## Runtime

**Environment:**
- Conda environment `ga-qsvm` with Python 3.11.11, declared in `environment.yml`.
- PBS cluster runtime for `bash/digits.sh`, which loads `python3.10` and runs `main.py`.
- CUDA/NVIDIA GPU runtime is assumed by PennyLane Lightning GPU experiments in `experiment/5_benchmark_GPU.py`, `experiment/5_benchmark_GPU_testSVC.py`, and `experiment/5_pennylane_qsvm.py`.

**Package Manager:**
- Conda - Full environment captured in `environment.yml`.
- pip - Lightweight dependency install via `requirements.txt`.
- Lockfile: missing. `environment.yml` is pinned but no `poetry.lock`, `uv.lock`, or pip lockfile is present.

## Frameworks

**Core:**
- Qiskit 1.3.1 - Quantum circuit construction, QPY serialization, primitives, and feature maps in `main.py`, `eval.py`, `utils.py`, and `qoop/`.
- qiskit-machine-learning 0.8.2 - `FidelityQuantumKernel` and `QSVC` in `main.py`, `eval.py`, and `experiment/*qsvm*.py`.
- scikit-learn 1.6.0 - Dataset loading, PCA, scaling, train/test split, SVC baselines, grid search, and metrics in `data/`, `main.py`, `eval.py`, and `experiment/`.
- NumPy 1.26.4 - Numeric arrays and random generation across `main.py`, `eval.py`, `data/`, and `qoop/`.
- PennyLane 0.36.0 - GPU QSVM kernel experiments and Qiskit-to-PennyLane conversion in `experiment/5_benchmark_GPU.py`, `experiment/5_benchmark_GPU_testSVC.py`, and `experiment/5_pennylane_qsvm.py`.
- PennyLane-Lightning-GPU 0.36.0 - GPU simulator backend selected as `lightning.gpu` in `experiment/5_benchmark_GPU.py`, `experiment/5_benchmark_GPU_testSVC.py`, and `experiment/5_pennylane_qsvm.py`.
- qoop 2.0 local package - Vendored quantum object optimizer under `qoop/`, used by `main.py` and `eval.py`.

**Testing:**
- pytest appears only in `qoop/setup.py` install requirements. No pytest config or committed test files are detected.

**Build/Dev:**
- setuptools - `qoop/setup.py` defines a package distribution.
- matplotlib - Plotting and circuit drawing in `qoop/evolution/environment.py`, `qoop/compilation/qcompilation.py`, and GPU benchmark scripts.
- tqdm - Progress bar wrapper in `qoop/backend/utilities.py`.

## Key Dependencies

**Critical:**
- `qiskit==1.3.1` in `requirements.txt` and `environment.yml` - Circuit objects are the shared data model between GA generation, mutation, crossover, kernel training, and QPY persistence.
- `qiskit_machine_learning==0.8.2` in `requirements.txt` and `environment.yml` - Provides the QSVM classifier path used by `main.py`, `eval.py`, and QSVM experiments.
- `scikit-learn==1.6.0` in `requirements.txt` - Provides SVC baselines and all bundled dataset preparation in `data/cv.py` and `data/split.py`.
- `wandb==0.19.8` in `requirements.txt` - Experiment tracking in `main.py`, `eval.py`, `qoop/evolution/environment.py`, and most files under `experiment/`.
- `pennylane==0.36.0`, `pennylane-qiskit==0.36.0`, `PennyLane-Lightning-GPU==0.36.0` in `requirements.txt` - GPU quantum kernel experiments in `experiment/5_benchmark_GPU.py`, `experiment/5_benchmark_GPU_testSVC.py`, and `experiment/5_pennylane_qsvm.py`.

**Infrastructure:**
- `qoop/requirements.txt` pins `qiskit==1.1.0`, `qiskit-algorithms==0.2.1`, and `qiskit-machine-learning==0.7.1`, which conflicts with top-level `requirements.txt` and `environment.yml`.
- `qoop/setup.py` lists legacy/deprecated or misspelled dependencies including `qiskit-ignis` and `tdqm`, and declares `packages=['qsee']` even though the code package is `qoop/`.
- `environment.yml` pins `wandb=0.19.6`, while `requirements.txt` pins `wandb==0.19.8`.
- `data/cv.py` imports TensorFlow inside `prepare_mnist_data()`, while `tensorflow` is mentioned in `README.md` but absent from `requirements.txt`.
- `experiment/5_benchmark_GPU_testSVC.py` imports `thundersvm` and `cupy`, but those packages are not listed in `requirements.txt`.
- `qoop/core/dag.py` imports `pennylane`, `torch`, and `torch.nn`, but `torch` is not listed in `requirements.txt`.
- `qoop/vqe/vqe.py` imports `qiskit_nature`, `qiskit_algorithms`, and PySCF driver components; `qiskit-nature` is only listed in `qoop/setup.py`.

## Configuration

**Environment:**
- Primary reproducible environment is `environment.yml`.
- Minimal pip install path is `requirements.txt`.
- Local QOOP package metadata is in `qoop/setup.py`, but its package declaration does not match the `qoop/` directory.
- `.gitignore` excludes `.env`, virtual environments, W&B artifacts, QPY files, JSON outputs, cluster `error/` and `output/` directories, and generated result folders.

**Build:**
- `qoop/setup.py` is the only packaging/build config detected.
- No `pyproject.toml`, `setup.cfg`, `tox.ini`, `pytest.ini`, `Makefile`, or lint/format config is detected.

## Platform Requirements

**Development:**
- Use Python 3.11.11 with Conda from `environment.yml` for closest parity.
- Use `pip install -r requirements.txt` for the lean runtime documented in `README.md`, then install missing optional packages as needed for MNIST, ThunderSVM/CuPy GPU SVC, Torch DAG code, or QOOP VQE modules.
- Use a W&B login or offline mode for scripts that call `wandb.init()` in `main.py`, `eval.py`, `qoop/evolution/environment.py`, and `experiment/`.
- Use CUDA-compatible NVIDIA drivers for `lightning.gpu` experiments in `experiment/5_benchmark_GPU.py`, `experiment/5_benchmark_GPU_testSVC.py`, and `experiment/5_pennylane_qsvm.py`.

**Production:**
- Not detected. The repository is a research/experiment codebase with CLI scripts, notebooks, and local result persistence.

## Refactor-Relevant Stack Notes

- Treat top-level `requirements.txt` plus `environment.yml` as the active application environment; treat `qoop/requirements.txt` and `qoop/setup.py` as stale package metadata until reconciled.
- Keep Qiskit version changes scoped and tested across `main.py`, `eval.py`, `qoop/core/metric.py`, `qoop/core/measure.py`, `qoop/backend/utilities.py`, and `qoop/vqe/vqe.py`, because these modules use APIs from different Qiskit generations.
- Separate optional experiment-only dependencies from reusable library dependencies: `cupy`, `thundersvm`, Covalent Cloud, and `PennyLane-Lightning-GPU` belong to benchmark/remote execution paths under `experiment/`, not to the core `qoop/` library.
- Move reusable preprocessing out of duplicated experiment scripts and into `data/` before broad dependency cleanup.

---

*Stack analysis: 2026-04-26*
