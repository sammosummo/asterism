#!/usr/bin/env bash
#SBATCH --job-name=aim2lev
#SBATCH --partition=highmemq
#SBATCH --array=0-5
#SBATCH --cpus-per-task=120
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/net/lus01/home/smathias/asterism-campaign/logs/aim2lev_%a.out
#SBATCH --error=/net/lus01/home/smathias/asterism-campaign/logs/aim2lev_%a.err
#
# The horizontal test's level, which the power campaign never measures.
#
# That campaign fixes the direct path at 0.2 in every cell including both of
# its nulls, so the vertical estimand is nought somewhere and the horizontal one
# is nought nowhere. A power figure was being read off a test whose level had
# never been established. These six cells set the direct path to nought: two
# arms by three truths.
#
# It runs `campaign_level.py`, a separate file, rather than a changed
# `campaign.py`. Cells of the power array were still pending when this was
# written, and they load their script when they start, so editing it would have
# had the last cells compute something different from the first with nothing to
# show that it had happened.
set -euo pipefail

WORK="$HOME/asterism-campaign"
cd "$WORK"

export CELL="$SLURM_ARRAY_TASK_ID"
export REPLICATES="${REPLICATES:-200}"
export WORKERS=120
export BOOTSTRAP=50
export DESIGN="$WORK/design.npz"
export OUT="$WORK/results/aim2_level_cell_${SLURM_ARRAY_TASK_ID}.json"

mkdir -p "$WORK/results" "$WORK/logs"

echo "level task $SLURM_ARRAY_TASK_ID on $(hostname), $(nproc) cores"
"$WORK/.venv/bin/python" -u campaign_level.py
