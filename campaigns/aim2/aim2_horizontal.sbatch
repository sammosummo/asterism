#!/usr/bin/env bash
#SBATCH --job-name=aim2hor
#SBATCH --partition=highmemq
#SBATCH --array=0-7
#SBATCH --cpus-per-task=120
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/net/lus01/home/smathias/asterism-campaign/logs/aim2hor_%a.out
#SBATCH --error=/net/lus01/home/smathias/asterism-campaign/logs/aim2hor_%a.err
#
# What size of direct path could this study detect?
#
# The power campaign found the direct-path test rejecting between 0.06 and 0.12
# where the true direct path is 0.2 -- close to no power. That says the study
# cannot detect a path of that size. It does not say what size it could detect,
# and that is the number worth quoting: "no power at 0.2" invites explaining
# away, "detectable from 0.5" can be weighed.
#
# The mediated effect is held real and large in every cell, because separating
# the two paths is the point of the aim and a test that only behaves when there
# is nothing to separate would not answer it.
set -euo pipefail

WORK="$HOME/asterism-campaign"
cd "$WORK"

export CELL="$SLURM_ARRAY_TASK_ID"
export REPLICATES="${REPLICATES:-200}"
export WORKERS=120
export BOOTSTRAP=50
export DESIGN="$WORK/design.npz"
export OUT="$WORK/results/aim2_horizontal_cell_${SLURM_ARRAY_TASK_ID}.json"

mkdir -p "$WORK/results" "$WORK/logs"

echo "direct-path task $SLURM_ARRAY_TASK_ID on $(hostname), $(nproc) cores"
"$WORK/.venv/bin/python" -u campaign_horizontal.py
