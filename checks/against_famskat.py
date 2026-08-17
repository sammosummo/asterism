"""Does Asterism's variant-set score test agree with famSKAT?

famSKAT is the family form of SKAT: it fits a null model carrying the kinship,
then scores each variant set against it and reads the statistic against a
weighted sum of chi-squares. Asterism fits the same model and forms the same
statistic, so the two should agree to the precision each can hold.

This is the comparison that matters for the whole gene scan, because the
score test's reference distribution is computed rather than assumed, and
assuming it is exactly what goes wrong for a variant-set kernel: measured under
the null, a likelihood ratio against the usual 50:50 reference rejected 0.020
at a nominal 0.05 on a rank-one burden kernel.

Requires R with `SKAT` 2.2.5, and fails rather than skips when it is absent.

Run with:

    uv run --locked --no-sync python checks/against_famskat.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import asterism

ACCEPTED_SKAT_VERSION = "2.2.5"
# SKAT reads its own tail by Davies' inversion at an accuracy near 1e-6, so
# asking the two to agree more closely than that is asking for better than
# either computes. Measured agreement is 1e-6 to 2e-6; this leaves room for
# that without admitting a real disagreement, which would be orders larger.
RELATIVE_TOLERANCE = 1e-4
FAMILIES, SIBS, VARIANTS = 50, 4, 20
REPLICATES = 12

SCRIPT = """
suppressMessages(library(SKAT))
a <- commandArgs(trailingOnly=TRUE)
stopifnot(identical(as.character(packageVersion("SKAT")), "%s"))
G <- as.matrix(read.table(a[1])); Phi <- as.matrix(read.table(a[2]))
y <- scan(a[3], quiet=TRUE); w <- scan(a[4], quiet=TRUE)
obj <- SKAT_NULL_emmaX(y ~ 1, K = Phi)
out <- SKAT(G, obj, kernel="linear.weighted", weights=w, method="davies",
            r.corr=as.numeric(a[5]), is_check_genotype=FALSE)
cat(out$p.value, "\\n")
""" % ACCEPTED_SKAT_VERSION


def simulate(replicate: int, effect: float):
    rng = np.random.default_rng(30_000 + replicate)
    family = np.repeat(np.arange(FAMILIES), SIBS)
    people = len(family)
    relationship = (
        np.where(family[:, None] == family[None, :], 0.5, 0.0) + 0.5 * np.eye(people)
    )
    mafs = rng.uniform(0.02, 0.10, VARIANTS)
    genotypes = np.zeros((people, VARIANTS))
    for variant, maf in enumerate(mafs):
        founders = rng.random((FAMILIES, 4)) < maf
        for person, f in enumerate(family):
            genotypes[person, variant] = (
                founders[f, rng.integers(2)] + founders[f, 2 + rng.integers(2)]
            )
    weights = np.array([25.0 * (1 - m) ** 24 for m in mafs])
    y = np.linalg.cholesky(
        0.4 * relationship + 0.6 * np.eye(people)
    ) @ rng.standard_normal(people)
    if effect:
        beta = np.zeros(VARIANTS)
        beta[:5] = rng.standard_normal(5) * effect
        y = y + genotypes @ beta
    return genotypes, relationship, y, weights, people


def main() -> int:
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on the path. This check fails rather than skips.")
    probe = subprocess.run(
        ["Rscript", "-e", 'cat(requireNamespace("SKAT", quietly=TRUE))'],
        capture_output=True, text=True,
    )
    if probe.stdout.strip() != "TRUE":
        raise SystemExit(
            "the SKAT comparison cannot run: R package 'SKAT' is not installed. "
            "Install it, then rerun checks/against_famskat.py."
        )

    failures: list[str] = []
    compared = []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "famskat.R").write_text(SCRIPT)
        for replicate in range(REPLICATES):
            effect = 0.0 if replicate < REPLICATES - 4 else 0.35
            genotypes, relationship, y, weights, people = simulate(replicate, effect)
            ours = asterism.VariantSetModel(
                [relationship], np.ones((people, 1)), y
            ).test(genotypes * weights)

            np.savetxt(directory / "G.txt", genotypes)
            np.savetxt(directory / "Phi.txt", relationship)
            np.savetxt(directory / "y.txt", y)
            np.savetxt(directory / "w.txt", weights)
            finished = subprocess.run(
                ["Rscript", str(directory / "famskat.R"),
                 str(directory / "G.txt"), str(directory / "Phi.txt"),
                 str(directory / "y.txt"), str(directory / "w.txt"), "0"],
                capture_output=True, text=True, timeout=600,
            )
            if finished.returncode != 0:
                raise SystemExit(f"famSKAT failed: {finished.stderr.strip()[:300]}")
            theirs = float(finished.stdout.strip())
            relative = abs(ours["p_value"] - theirs) / max(theirs, 1e-300)
            compared.append(
                {"replicate": replicate, "asterism": ours["p_value"],
                 "famskat": theirs, "relative": relative,
                 "eigenvalues_kept": len(ours["eigenvalues"])}
            )
            if relative > RELATIVE_TOLERANCE:
                failures.append(
                    f"replicate {replicate}: {ours['p_value']:.6e} against "
                    f"{theirs:.6e}, relative {relative:.2e}"
                )

    worst = max(c["relative"] for c in compared)
    if failures:
        print("NOT AGREED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(json.dumps({
        "what": "the variant-set score test against famSKAT",
        "skat_version": ACCEPTED_SKAT_VERSION,
        "null_model": "SKAT_NULL_emmaX, a kinship-carrying null",
        "kernel": "linear.weighted with Beta(1,25) weights",
        "people": FAMILIES * SIBS,
        "variants": VARIANTS,
        "replicates": REPLICATES,
        "worst_relative_difference": worst,
        "comparisons": compared,
    }, indent=2))
    print(
        f"\nThe two implementations agree to {worst:.1e} relative across "
        f"{REPLICATES} data sets.\nSKAT shares no code with Asterism."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
