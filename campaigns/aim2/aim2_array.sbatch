#!/usr/bin/env bash
#SBATCH --job-name=aim2
#SBATCH --partition=highmemq
#SBATCH --array=0-11
#SBATCH --cpus-per-task=120
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/net/lus01/home/smathias/asterism-campaign/logs/aim2_%a.out
#SBATCH --error=/net/lus01/home/smathias/asterism-campaign/logs/aim2_%a.err
#
# One array task per cell of the grid: two measurement arms by six truths.
# Each task gets its own node and its own 120 workers, so the twelve cells run
# beside each other rather than queueing behind each other. A single fit on
# 1,600 people is expensive; the answer is breadth, which is what this cluster
# has and a laptop does not.
#
# Each task writes its own result file. They are combined afterwards rather
# than during, so a task that dies loses only its own cell.
set -euo pipefail

WORK="$HOME/asterism-campaign"
cd "$WORK"

export CELL="$SLURM_ARRAY_TASK_ID"
export REPLICATES="${REPLICATES:-200}"
export WORKERS=120
export BOOTSTRAP=50
export DESIGN="$WORK/design.npz"
export OUT="$WORK/results/aim2_cell_${SLURM_ARRAY_TASK_ID}.json"

mkdir -p "$WORK/results" "$WORK/logs"

echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $(nproc) cores"
"$WORK/.venv/bin/python" -u campaign.py
