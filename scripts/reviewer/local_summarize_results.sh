#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "Summarizing reviewer benchmark folders under results/reviewer"

mapfile -t INPUTS < <(find results/reviewer -maxdepth 1 -type d \( \
  -name 'figure6_holdout_*' -o \
  -name 'figure7_transfer*' \
\) | sort)

if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "No benchmark result folders found under results/reviewer" >&2
  exit 1
fi

printf 'Inputs:\n'
printf '  %s\n' "${INPUTS[@]}"

uv run python -m ga_qsvm.cli.summarize_reviewer_results \
  --inputs "${INPUTS[@]}" \
  --output-dir results/reviewer/final

echo "Done: results/reviewer/final/reviewer_summary.csv"
echo "LaTeX table and revised Figure 6 image generation are the next step after we confirm which result folders are accepted."

