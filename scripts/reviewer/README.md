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
qubits=7
depth=35
num_circuit=20
num_generation=200
prob_mutate=0.1
training_size=100
test_size=50
```

With `--qubits 7`, the train CLI sweeps all 36 `(n_Rx, n_Ry, n_Rz)` allocations
whose sum is 7.

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

