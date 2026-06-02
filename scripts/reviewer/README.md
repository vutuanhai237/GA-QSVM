# Reviewer Benchmark Scripts

This folder contains shell entrypoints for the reviewer-response benchmark setup.

## Seed Meaning

There are two different concepts:

- **GA rerun**: runs the genetic algorithm search and produces new circuit artifacts.
- **Benchmark seeds 100-109**: keep selected circuits frozen and repeat final train/test splits.

The scripts here use `100-109` only for the final repeated-holdout benchmark. The current
`ga_qsvm.cli.train` command does not expose a GA random seed argument.

## Cloud GA Rerun

Use one VM per dataset:

```bash
scripts/reviewer/cloud_ga_dataset.sh digits 0
scripts/reviewer/cloud_ga_dataset.sh wine 0
scripts/reviewer/cloud_ga_dataset.sh cancer 0
```

Default GA config:

```text
qubits=3 4 5 6 7
depth=5 * qubits
num_circuit=20
num_generation=200
prob_mutate=0.1
kernels=pqk fqk
training_size=100
test_size=baseline dataset split
run_holdout=1
holdout_qubits=7
holdout_seeds=100 101 102 103 104 105 106 107 108 109
```

Each VM runs one dataset across five GA searches per kernel, one for each qubit count. The
train CLI no longer sweeps fixed `(n_Rx, n_Ry, n_Rz)` allocations; each generated
circuit randomly allocates RX/RY/RZ gates while keeping the total rotation
parameter count equal to the qubit count.

After GA completes, this script automatically creates a manifest for the fresh
`holdout_qubits` artifact and runs repeated-holdout benchmark seeds against that
circuit. By default, PQK runs first and FQK runs second. PQK holdout models are
`rbf fixed-pqk ga-pqk`; FQK holdout models are `fixed-fqk ga-fqk`.
The GA stage logs through the evolution environment, and the holdout stage logs
its per-seed and summary tables to a separate W&B run. Set `RUN_HOLDOUT=0` to
skip the holdout stage.

## Local Jobs

Run cheap provenance and audit jobs:

```bash
scripts/reviewer/local_cheap_jobs.sh
```

Run repeated-holdout benchmark for cheap local models:

```bash
scripts/reviewer/local_benchmark_repeated_holdout.sh
```

Run FQK repeated-holdout benchmark on cloud:

```bash
scripts/reviewer/cloud_fqk_benchmark_dataset.sh digits 100 101 102 103 104
scripts/reviewer/cloud_fqk_benchmark_dataset.sh digits 105 106 107 108 109
scripts/reviewer/cloud_fqk_benchmark_dataset.sh wine
scripts/reviewer/cloud_fqk_benchmark_dataset.sh cancer
```

After copying cloud result folders back into `results/reviewer`, summarize:

```bash
scripts/reviewer/local_summarize_results.sh
```

## Predefined Ansatz Baselines

These jobs do not run GA and do not load GA artifacts. They build fixed
`EfficientSU2` and `TwoLocal` circuits, use the same train-only scaler/PCA
holdout flow as the Figure 6 rerun, and evaluate both FQK and PQK.

Run one shard:

```bash
scripts/reviewer/predefined_baseline_job.sh wine efficient-su2 pqk 7
SEEDS="100" scripts/reviewer/predefined_baseline_job.sh digits two-local fqk 3
```

Run the full serial sweep:

```bash
scripts/reviewer/run_predefined_baseline_sweep.sh
```

The full sweep covers datasets `digits wine cancer`, ansatzes
`efficient-su2 two-local`, kernels `pqk fqk`, qubits `3 4 5 6 7`, and
holdout seeds `100-109`. For cloud execution, shard FQK jobs by seed using the
first command instead of running the full serial sweep on one VM.

Outputs are written below:

```text
results/reviewer/predefined_baselines/
```

## Random Circuit Search Baseline

This job does not run GA. It samples random circuits from the same generator and
structural constraints used by GA-QSVM, evaluates each candidate on the same
holdout split, and reports the best candidate for that split.

Run one shard:

```bash
RANDOM_BUDGET=20 scripts/reviewer/random_search_baseline_job.sh digits pqk 7
SEEDS="100" RANDOM_BUDGET=20 scripts/reviewer/random_search_baseline_job.sh wine pqk 7
```

Run the default sweep:

```bash
scripts/reviewer/run_random_search_baseline_sweep.sh
```

Defaults are datasets `digits wine cancer`, kernel `pqk`, qubits `7`, seeds
`100-109`, and random budget `20`. Increase `RANDOM_BUDGET` to compare against a
larger search budget, or shard by seed/dataset on cloud VMs.

Outputs are written below:

```text
results/reviewer/random_search_baselines/
```

## Early-Stop Evidence

Use this after at least one full FQK GA run finishes. It does not change training;
it reads `metadata.json` and asks: if we had stopped after 50 generations without
a new best validation score, would the final best score be the same?

```bash
scripts/reviewer/analyze_early_stop.sh
```

Default output:

```text
results/reviewer/early_stop_analysis/early_stop_analysis.csv
results/reviewer/early_stop_analysis/early_stop_analysis.md
```

For the slow q3 FQK decision, inspect rows with `kernel=fqk` and `num_qubits=3`.
If `safe_to_stop=True` and `delta_final_minus_early_stop=0`, patience 50 did not
lose accuracy for that run. If the delta is positive, keep full 200 generations
for that dataset/kernel/qubit.
