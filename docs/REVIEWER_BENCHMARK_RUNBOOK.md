# Reviewer Benchmark Runbook

This runbook is for generating the reviewer-response results without rerunning full GA by default.

## Current Environment

- Local machine: 20 CPU cores.
- Required Python environment is managed by `uv`.
- PQK requires:
  - `squlearn==0.8.4`
  - `pennylane==0.34.0`
  - `autoray==0.6.7`
- Use these environment variables on local and cloud VMs to avoid CPU oversubscription:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Cheap Local Jobs

Run these on the 20-core local machine first. They are not the bottleneck.

```bash
uv run pytest tests/test_reviewer_tooling.py -q
```

```bash
uv run python -m ga_qsvm.cli.audit_artifacts \
  --roots /home/qsvm/benchmark/GA-QSVM /home/qsvm/test/GA-QSVM /home/qsvm/GA-QSVM \
  --output results/reviewer/artifact_audit.csv
```

```bash
uv run python -m ga_qsvm.cli.pca_analysis \
  --datasets digits wine cancer \
  --thresholds 0.90 0.95 \
  --output-dir results/reviewer/figure3_pca
```

```bash
uv run python -m ga_qsvm.cli.circuit_export \
  --manifest configs/reviewer/figure5_circuits_manifest.json \
  --output-dir results/reviewer/figure5_circuits \
  --formats txt qasm
```

```bash
uv run python -m ga_qsvm.cli.hyperparameter_summary \
  --config configs/reviewer/figure4_hyperparameter_sources.json \
  --output-dir results/reviewer/figure4_hyperparameters
```

## Figure 6 Holdout Jobs

The holdout benchmark is the main reviewer-critical run.

Default seed set:

```bash
SEEDS="100 101 102 103 104 105 106 107 108 109"
```

Run cheap/classical and PQK jobs locally first:

```bash
uv run python -m ga_qsvm.cli.frozen_benchmark \
  --manifest configs/reviewer/main_figure6_n7_manifest.json \
  --datasets digits wine cancer \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess legacy \
  --models rbf ga-pqk fixed-pqk \
  --output-dir results/reviewer/figure6_holdout_pqk_rbf \
  --n-features 7
```

FQK is the bottleneck. Run it per dataset, preferably on separate cloud VMs:

```bash
uv run python -m ga_qsvm.cli.frozen_benchmark \
  --manifest configs/reviewer/main_figure6_n7_manifest.json \
  --datasets wine \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess legacy \
  --models ga-fqk fixed-fqk \
  --output-dir results/reviewer/figure6_holdout_fqk_wine \
  --n-features 7 \
  --feature-dim-mode circuit-parameters
```

```bash
uv run python -m ga_qsvm.cli.frozen_benchmark \
  --manifest configs/reviewer/main_figure6_n7_manifest.json \
  --datasets cancer \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess legacy \
  --models ga-fqk fixed-fqk \
  --output-dir results/reviewer/figure6_holdout_fqk_cancer \
  --n-features 7 \
  --feature-dim-mode circuit-parameters
```

```bash
uv run python -m ga_qsvm.cli.frozen_benchmark \
  --manifest configs/reviewer/main_figure6_n7_manifest.json \
  --datasets digits \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess legacy \
  --models ga-fqk fixed-fqk \
  --output-dir results/reviewer/figure6_holdout_fqk_digits \
  --n-features 7 \
  --feature-dim-mode circuit-parameters
```

Important: `wine-ga-fqk-n7` has 7 qubits but 9 trainable feature parameters. `--feature-dim-mode circuit-parameters` prevents dimension mismatch by using 9 PCA features for that frozen circuit. If the paper must keep exactly 7 PCA features for every model, replace the Wine FQK artifact with a 7-parameter circuit before running full FQK.

## Figure 6 K-Fold Jobs

K-fold multiplies runtime by `folds`. Do not start full k-fold until holdout finishes.

Smoke test:

```bash
uv run python -m ga_qsvm.cli.kfold_benchmark \
  --manifest configs/reviewer/main_figure6_n7_manifest.json \
  --datasets wine \
  --seeds 100 \
  --folds 5 \
  --max-folds 1 \
  --preprocess legacy \
  --models rbf ga-pqk \
  --output-dir results/reviewer/figure6_kfold_smoke \
  --n-features 7
```

Full k-fold, only if runtime budget allows:

```bash
uv run python -m ga_qsvm.cli.kfold_benchmark \
  --manifest configs/reviewer/main_figure6_n7_manifest.json \
  --datasets digits wine cancer \
  --seeds 100 \
  --folds 5 \
  --preprocess legacy \
  --models rbf fixed-fqk fixed-pqk ga-fqk ga-pqk \
  --output-dir results/reviewer/figure6_kfold \
  --n-features 7 \
  --feature-dim-mode circuit-parameters
```

## Figure 7 Transfer Jobs

Run after Figure 6 holdout. Current transfer manifest covers Digits-to-Wine and Digits-to-Cancer. Fashion needs a separate manifest decision because its artifact source is not the same as Figure 6.

```bash
uv run python -m ga_qsvm.cli.transfer_benchmark \
  --manifest configs/reviewer/figure7_transfer_manifest.json \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess legacy \
  --output-dir results/reviewer/figure7_transfer \
  --n-features 7
```

## Summarize

After copying all cloud result folders back into `results/reviewer`, run:

```bash
uv run python -m ga_qsvm.cli.summarize_reviewer_results \
  --inputs \
    results/reviewer/figure6_holdout_pqk_rbf \
    results/reviewer/figure6_holdout_fqk_wine \
    results/reviewer/figure6_holdout_fqk_cancer \
    results/reviewer/figure6_holdout_fqk_digits \
    results/reviewer/figure6_kfold \
    results/reviewer/figure7_transfer \
  --output-dir results/reviewer/final
```

## Cloud VM Recommendation

Use CPU VMs. GPU is not expected to help this Qiskit/squlearn CPU path.

Recommended starting layout:

- Local 20-core machine: cheap jobs, RBF, PQK, summaries.
- 1 VM for `wine` FQK: 8-16 vCPU, 32 GB RAM.
- 1 VM for `cancer` FQK: 16-32 vCPU, 64 GB RAM.
- 2-4 VMs for `digits` FQK split by seed ranges: 32-64 vCPU, 64-128 GB RAM.

For Digits FQK, split seeds manually:

```bash
--seeds 100 101 102
--seeds 103 104 105
--seeds 106 107
--seeds 108 109
```

Keep one process per VM at first. If CPU utilization is low, run two processes with disjoint seed ranges.

