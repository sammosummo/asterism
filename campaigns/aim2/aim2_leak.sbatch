#!/usr/bin/env bash
#SBATCH --job-name=aim2leak
#SBATCH --partition=highmemq
#SBATCH --cpus-per-task=120
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/net/lus01/home/smathias/asterism-campaign/logs/aim2leak.out
#SBATCH --error=/net/lus01/home/smathias/asterism-campaign/logs/aim2leak.err
#
# Does pleiotropy leak into the mediation estimate?
#
# The genetic covariance sees only `a (ab + c')`, the sum, so genetics alone
# cannot separate mediation from pleiotropy. The separation comes from the
# residual within-person covariance `b sigma_m2`. If that separation is weak
# enough, some of the direct path could be read as mediation -- which would
# attack the primary claim, not the secondary one.
#
# A first run on 150 families of three was too noisy to answer: the means
# climbed but nothing reached two standard errors and the estimates were
# heavy-tailed. This is the real design, with medians beside means and a count
# of runaway fits, because a few runaway fits and a systematic leak look
# identical in an average and are fixed differently.
set -euo pipefail

WORK="$HOME/asterism-campaign"
cd "$WORK"

export REPLICATES="${REPLICATES:-200}"
export WORKERS=120
export DESIGN="$WORK/design.npz"
export OUT="$WORK/results/aim2_leak.json"

mkdir -p "$WORK/results" "$WORK/logs"

echo "leak check on $(hostname), $(nproc) cores"
"$WORK/.venv/bin/python" -u campaign_leak.py
