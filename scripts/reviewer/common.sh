#!/usr/bin/env bash
set -euo pipefail

# Source this file from reviewer benchmark scripts.
# Defaults are chosen for CPU-only Qiskit/squlearn workloads.

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# Avoid blocking on wandb login on fresh cloud VMs. Override with
# WANDB_MODE=online if you explicitly want cloud logging.
export WANDB_MODE="${WANDB_MODE:-offline}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$UV_CACHE_DIR" results/reviewer logs/reviewer
