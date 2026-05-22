# External Integrations

**Analysis Date:** 2026-04-26

## APIs & External Services

**Experiment Tracking:**
- Weights & Biases - Tracks GA-QSVM, SVC, CPU QSVM, and GPU QSVM experiment metrics.
  - SDK/Client: `wandb` from `requirements.txt`.
  - Auth: W&B standard local credentials/environment; no explicit env var is read in code.
  - Implementations: `main.py`, `eval.py`, `qoop/evolution/environment.py`, `experiment/0_qsvm_wine_1.py`, `experiment/0_svc_wine_1.py`, `experiment/1_svc_mnist.py`, `experiment/2_qsvm_digits.py`, `experiment/2_svc_digits.py`, `experiment/3_qsvm_cancer.py`, `experiment/3_svc_cancer.py`, `experiment/3_svc_cancer_test_holdout.py`, `experiment/3_svc_cancer_test_holdout_weird_setup.py`, `experiment/5_benchmark_CPU.py`, `experiment/5_benchmark_GPU.py`, and `experiment/5_benchmark_GPU_testSVC.py`.
  - Projects: `GA-QSVM-*`, `GA-QSVM-eval`, `SVM-PCA`, and `GPU-QSVM`.

**Quantum Simulation and ML Libraries:**
- Qiskit local simulators/primitives - Circuit construction, statevector/density matrix operations, Sampler, Estimator, and QPY persistence.
  - SDK/Client: `qiskit`, `qiskit_machine_learning`, `qiskit_algorithms`, `qiskit_nature`.
  - Auth: None.
  - Implementations: `main.py`, `eval.py`, `utils.py`, `qoop/backend/utilities.py`, `qoop/core/metric.py`, `qoop/core/measure.py`, `qoop/core/parallel.py`, `qoop/vqe/vqe.py`.
- PennyLane GPU simulator - GPU-accelerated kernel evaluation.
  - SDK/Client: `pennylane`, `pennylane-qiskit`, `PennyLane-Lightning-GPU`.
  - Auth: None.
  - Implementations: `experiment/5_benchmark_GPU.py`, `experiment/5_benchmark_GPU_testSVC.py`, `experiment/5_pennylane_qsvm.py`.

**Remote Compute:**
- Covalent Cloud - Notebook-only cloud GPU workflow example.
  - SDK/Client: `covalent`, `covalent_cloud` referenced in `experiment/4_covalent_cloud_gpu.ipynb`.
  - Auth: Covalent Cloud API key via `cc.save_api_key(...)` shown in notebook text.
  - Production code usage: Not detected.

**Dataset Downloads:**
- TensorFlow Keras MNIST loader - Downloads/loads MNIST in `data/cv.py`.
  - SDK/Client: `tensorflow.keras.datasets.mnist`.
  - Auth: None.
  - Implementation: `data/cv.py`.
- scikit-learn bundled datasets - Wine, Digits, Breast Cancer, and synthetic blobs.
  - SDK/Client: `sklearn.datasets`.
  - Auth: None.
  - Implementations: `data/cv.py`, `data/split.py`, and repeated copies in `experiment/`.

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only.
  - QPY circuit persistence uses `qiskit.qpy` in `utils.py` and `qoop/backend/utilities.py`.
  - GA environment save output writes `metadata.json`, `funcs.json`, `circuit_*.qpy`, and `best_circuit*.qpy` from `qoop/evolution/environment.py`.
  - `.gitignore` excludes `*.qpy` and `*.json`, so generated circuit artifacts and metadata are intentionally untracked.

**Caching:**
- W&B local run cache/artifacts under `wandb/`; the directory exists in the working tree and is ignored by `.gitignore`.
- Python/Jupyter caches are ignored through `.gitignore`.
- No application cache service is detected.

## Authentication & Identity

**Auth Provider:**
- W&B CLI/session credentials for `wandb.init()` in `main.py`, `eval.py`, `qoop/evolution/environment.py`, and `experiment/`.
  - Implementation: External SDK authentication, not explicit application code.
- Covalent Cloud API key in `experiment/4_covalent_cloud_gpu.ipynb`.
  - Implementation: Notebook demonstration text uses `cc.save_api_key(...)`; no secret values are committed in code inspected.

## Monitoring & Observability

**Error Tracking:**
- None detected.

**Logs:**
- `print()` logging throughout `main.py`, `eval.py`, `data/split.py`, `data/cv.py`, and `qoop/evolution/environment.py`.
- W&B metric logging through `wandb.log()` in `qoop/evolution/environment.py` and experiment scripts.
- Python root logger is configured at DEBUG level in `qoop/backend/constant.py`, but structured logging usage is minimal.
- PBS stdout/stderr paths are configured in `bash/digits.sh`.

## CI/CD & Deployment

**Hosting:**
- Not detected.

**CI Pipeline:**
- None detected. No GitHub Actions, GitLab CI, tox, nox, or test runner config is present.

**Batch Execution:**
- PBS cluster job in `bash/digits.sh` runs `main.py` with a fixed digits search.
- Manual shell command groups in `bash/eval0.sh`, `bash/eval1.sh`, and `bash/eval2.sh` run selected `eval.py` configurations.

## Environment Configuration

**Required env vars:**
- None are read explicitly in source files.
- W&B can use standard SDK configuration such as login credentials or offline mode for scripts that call `wandb.init()`.
- Covalent Cloud credentials are notebook-local in `experiment/4_covalent_cloud_gpu.ipynb`.

**Secrets location:**
- No `.env` files or credential files were detected at repository depth inspected.
- `.gitignore` excludes `.env`, `.pypirc`, W&B artifacts, generated JSON, and QPY outputs.
- W&B and Covalent credentials should remain outside source-controlled files.

## Webhooks & Callbacks

**Incoming:**
- None detected.

**Outgoing:**
- W&B metric and alert calls from `main.py`, `qoop/evolution/environment.py`, and `experiment/`.
- Covalent Cloud dispatch from `experiment/4_covalent_cloud_gpu.ipynb`.

## Refactor-Relevant Integration Notes

- Centralize W&B initialization and logging behind a small adapter before refactoring `main.py`, `eval.py`, and `qoop/evolution/environment.py`; the library package currently has a direct W&B dependency.
- Keep notebook-only Covalent Cloud code out of reusable modules; it belongs under `experiment/` or a dedicated remote-execution adapter.
- Keep QPY save/load paths explicit and configurable. `qoop/evolution/environment.py` writes generated result directories from inside library code, while `.gitignore` excludes the resulting artifacts.
- Separate local simulator paths (`qiskit_machine_learning` QSVC) from GPU simulator paths (`pennylane`, `cupy`, `thundersvm`) to avoid making optional GPU dependencies required for CPU-only runs.

---

*Integration audit: 2026-04-26*
