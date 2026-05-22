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
depth=10 * qubits
num_circuit=20
num_generation=200
prob_mutate=0.1
kernel=pqk
training_size=100
test_size=50
run_holdout=1
holdout_qubits=7
holdout_seeds=100 101 102 103 104 105 106 107 108 109
```

Each VM runs one dataset across five GA searches, one for each qubit count. The
train CLI no longer sweeps fixed `(n_Rx, n_Ry, n_Rz)` allocations; each generated
circuit randomly allocates RX/RY/RZ gates while keeping the total rotation
parameter count equal to the qubit count.

After GA completes, this script automatically creates a manifest for the fresh
`holdout_qubits` artifact and runs repeated-holdout benchmark seeds against that
circuit. For the default PQK rerun, the holdout models are `rbf fixed-pqk ga-pqk`.
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
