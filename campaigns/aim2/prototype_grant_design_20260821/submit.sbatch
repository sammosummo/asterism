#!/usr/bin/env bash
#SBATCH --job-name=aim2grantproto
#SBATCH --partition=highmemq
#SBATCH --array=0-71
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# PROTOTYPE — one Medusa array task per grant-design cell.
# Override the work directory with:
#   sbatch --export=ALL,WORK=/path/to/work submit.sbatch
# Override or resume array cells with:
#   sbatch --array=0,17,35-41 --export=ALL,WORK=/path/to/work submit.sbatch
set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
CELL="${SLURM_ARRAY_TASK_ID:?submit this file with sbatch --array}"

mkdir -p "$WORK/results"
cd "$WORK"

echo "prototype cell $CELL on $(hostname), $(nproc) cores"
.venv/bin/python -u run_cell.py \
  --cells "$WORK/cells.json" \
  --cell "$CELL" \
  --design-dir "$WORK/designs" \
  --out-dir "$WORK/results" \
  --replicates 200 \
  --bootstrap 50 \
  --workers 120
