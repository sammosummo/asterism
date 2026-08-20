#!/usr/bin/env bash
#SBATCH --job-name=aim2lev2
#SBATCH --partition=highmemq
#SBATCH --array=12-13
#SBATCH --cpus-per-task=120
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/net/lus01/home/smathias/asterism-campaign/logs/aim2lev2_%a.out
#SBATCH --error=/net/lus01/home/smathias/asterism-campaign/logs/aim2lev2_%a.err
#
# Which design change would buy power?
#
# The campaign's answer so far is "about 1.7 times the sample", which is true
# and expensive and only right if people are what is short. Counting the design
# suggests they are not: of about 257 cases, 200 are the adjudicated probands
# and the other 1,400 people supply roughly 57 between them. If the binding
# constraint is the binary outcome rather than the hearing measurement, then
# adjudicated families are worth far more per person than anybody else.
#
# Six levers, each changing exactly one thing, each with its own level cell
# under the identical change. Two arms of the grid are the same design with a
# different truth, so twelve tasks.
set -euo pipefail

WORK="$HOME/asterism-campaign"
cd "$WORK"

export CELL="$SLURM_ARRAY_TASK_ID"
export REPLICATES="${REPLICATES:-200}"
export WORKERS=120
export BOOTSTRAP=50
export DESIGN="$WORK/design.npz"
export OUT="$WORK/results/aim2_levers_cell_${SLURM_ARRAY_TASK_ID}.json"

mkdir -p "$WORK/results" "$WORK/logs"

echo "lever task $SLURM_ARRAY_TASK_ID on $(hostname), $(nproc) cores"
"$WORK/.venv/bin/python" -u campaign_levers.py
