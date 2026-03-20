#!/bin/bash
# Run full analysis pipeline after protocol + SMOTE + drift results are ready.
# Usage: ./scripts/run_full_analysis.sh

set -e
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$$}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true
PYTHON="${PYTHON:-python}"
if [ -x .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
elif [ -x venv/bin/python ]; then
  PYTHON="venv/bin/python"
fi

echo "=== 1. Normalize protocol CSV (if mixed schemas) ==="
$PYTHON scripts/normalize_protocol_csv.py 2>/dev/null || true

echo ""
echo "=== 2. Unified analysis (merge results, drift-harm, stat tests) ==="
$PYTHON -m src.run_unified_analysis

echo ""
echo "=== 3. Canonical analysis (run mapping) ==="
$PYTHON -m src.run_canonical_analysis || echo "  (canonical analysis had issues, continuing)"

echo ""
echo "=== 4. Generate paper figures ==="
$PYTHON -m src.paper_figures

echo ""
echo "=== 5. Aggregate summary ==="
$PYTHON -m src.run_aggregate_results

echo ""
echo "Done. Check paper/tables/, paper/figures/, results/SUMMARY.md"
