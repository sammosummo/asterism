#!/usr/bin/env bash
#SBATCH --job-name=aim2off
#SBATCH --partition=highmemq
#SBATCH --cpus-per-task=120
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/net/lus01/home/smathias/asterism-campaign/logs/aim2off.out
#SBATCH --error=/net/lus01/home/smathias/asterism-campaign/logs/aim2off.err
#
# How many offspring per adjudicated case?
#
# This is the one design decision the grant records as genuinely open: whether
# to enrol several offspring per case. Doing so cuts adjudicated cases from 250
# to about 184, because adjudication is what the budget buys, while making each
# family larger. Fewer larger families carry more information per case and less
# independent information overall, and which way that trades cannot be argued
# from first principles.
#
# It was blocked on a generator that could put more than two people in a family.
# That is no longer true, so it can be answered.
#
# 3 designs x 2 sampling arms x 3 truths x 400 replicates. It runs as one job
# rather than an array because the script sweeps its own grid, and on 120
# workers rather than the 6 a laptop gets.
set -euo pipefail

WORK="$HOME/asterism-campaign"
cd "$WORK/asterism"

export ASTERISM_REPLICATES=400
export ASTERISM_WORKERS=120
export ASTERISM_BOOTSTRAP=50

echo "offspring sweep on $(hostname), $(nproc) cores"
"$WORK/.venv/bin/python" -u checks/mediation_power.py
