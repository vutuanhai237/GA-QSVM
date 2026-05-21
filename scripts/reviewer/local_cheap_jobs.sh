#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "Running reviewer tooling tests"
uv run pytest tests/test_reviewer_tooling.py -q

echo "Auditing existing artifacts"
uv run python -m ga_qsvm.cli.audit_artifacts \
  --roots /home/qsvm/benchmark/GA-QSVM /home/qsvm/test/GA-QSVM /home/qsvm/GA-QSVM results/reviewer/ga_reruns \
  --output results/reviewer/artifact_audit.csv

echo "Generating PCA provenance"
uv run python -m ga_qsvm.cli.pca_analysis \
  --datasets digits wine cancer \
  --thresholds 0.90 0.95 \
  --output-dir results/reviewer/figure3_pca

echo "Exporting configured Figure 5 circuits"
uv run python -m ga_qsvm.cli.circuit_export \
  --manifest configs/reviewer/figure5_circuits_manifest.json \
  --output-dir results/reviewer/figure5_circuits \
  --formats txt qasm

echo "Summarizing Figure 4 hyperparameter sources"
uv run python -m ga_qsvm.cli.hyperparameter_summary \
  --config configs/reviewer/figure4_hyperparameter_sources.json \
  --output-dir results/reviewer/figure4_hyperparameters

echo "Done: cheap local reviewer jobs"

