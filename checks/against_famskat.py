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

import asterism
import numpy as np

ACCEPTED_SKAT_VERSION: str = "2.2.5"
"""Exact SKAT release whose independent outputs qualify this comparison."""

# SKAT reads its own tail by Davies' inversion at an accuracy near 1e-6, so
# asking the two to agree more closely than that is asking for better than
# either computes. Measured agreement is 1e-6 to 2e-6; this leaves room for
# that without admitting a real disagreement, which would be orders larger.
RELATIVE_TOLERANCE: float = 1e-4
"""Maximum accepted relative p-value difference from famSKAT."""

FAMILIES: int = 50
"""Independent full-sibling families in each simulated comparison."""

SIBS: int = 4
"""Full siblings represented within every simulated family."""

VARIANTS: int = 20
"""Rare variants included in each tested set."""

REPLICATES: int = 12
"""Fixed null-and-effect data sets compared with famSKAT."""

SCRIPT: str = f"""
suppressMessages(library(SKAT))
a <- commandArgs(trailingOnly=TRUE)
stopifnot(identical(as.character(packageVersion("SKAT")), "{ACCEPTED_SKAT_VERSION}"))
G <- as.matrix(read.table(a[1])); Phi <- as.matrix(read.table(a[2]))
y <- scan(a[3], quiet=TRUE); w <- scan(a[4], quiet=TRUE)
obj <- SKAT_NULL_emmaX(y ~ 1, K = Phi)
out <- SKAT(G, obj, kernel="linear.weighted", weights=w, method="davies",
            r.corr=as.numeric(a[5]), is_check_genotype=FALSE)
cat(out$p.value, "\\n")
"""
"""R program fitting famSKAT with the accepted package and model settings."""


def simulate(
    replicate: int,
    effect: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Simulate one related rare-variant data set.

    Args:
        replicate: Index added to the fixed random-number seed base.
        effect: Scale of effects applied to the first five variants.

    Returns:
        Genotypes, relationship matrix, outcome, weights and sample size.
    """
    rng: np.random.Generator = np.random.default_rng(30_000 + replicate)
    """Selected the fixed random-number stream for this replicate."""

    family: np.ndarray = np.repeat(np.arange(FAMILIES), SIBS)
    """Assigned each simulated person to a full-sibling family."""

    people: int = len(family)
    """Counted simulated people in family order."""

    relationship: np.ndarray = np.where(
        family[:, None] == family[None, :], 0.5, 0.0
    ) + 0.5 * np.eye(people)
    """Constructed the fixed full-sibling additive relationship matrix."""

    mafs: np.ndarray = rng.uniform(0.02, 0.10, VARIANTS)
    """Drew variant-specific minor-allele frequencies in the stated range."""

    genotypes: np.ndarray = np.zeros((people, VARIANTS))
    """Initialised the person-by-variant genotype matrix."""

    for variant, maf in enumerate(mafs):
        founders: np.ndarray = rng.random((FAMILIES, 4)) < maf
        """Drew two parental diploid genotypes independently for each family."""

        for person, f in enumerate(family):
            genotypes[person, variant] = (
                founders[f, rng.integers(2)] + founders[f, 2 + rng.integers(2)]
            )
            """Transmitted one random allele from each simulated parent."""

    weights: np.ndarray = np.array([25.0 * (1 - m) ** 24 for m in mafs])
    """Calculated the historical Beta(1, 25) rare-variant weights."""

    y: np.ndarray = np.linalg.cholesky(
        0.4 * relationship + 0.6 * np.eye(people)
    ) @ rng.standard_normal(people)
    """Drew the baseline polygenic Gaussian outcome."""

    if effect:
        beta: np.ndarray = np.zeros(VARIANTS)
        """Initialised variant effects at nought outside the causal subset."""

        beta[:5] = rng.standard_normal(5) * effect
        """Drew effects for the fixed five-variant causal subset."""

        y = y + genotypes @ beta
        """Added the rare-variant contribution to the polygenic outcome."""

    return genotypes, relationship, y, weights, people


def main() -> int:
    """Run null, effect and correlation-family comparisons against famSKAT.

    Returns:
        Zero when every p-value agrees within tolerance, otherwise one.

    Raises:
        SystemExit: If R, SKAT or a live famSKAT fit is unavailable.
    """
    if shutil.which("Rscript") is None:
        raise SystemExit(
            "Rscript is not on the path. This check fails rather than skips."
        )
    probe: subprocess.CompletedProcess[str] = subprocess.run(
        ["Rscript", "-e", 'cat(requireNamespace("SKAT", quietly=TRUE))'],
        capture_output=True,
        text=True,
    )
    """Checked that the required SKAT namespace is available to R."""

    if probe.stdout.strip() != "TRUE":
        raise SystemExit(
            "the SKAT comparison cannot run: R package 'SKAT' is not installed. "
            "Install it, then rerun checks/against_famskat.py."
        )

    failures: list[str] = []
    """Collected every relative-agreement failure."""

    compared: list[dict[str, int | float]] = []
    """Accumulated per-replicate scalar comparison evidence."""

    with tempfile.TemporaryDirectory() as directory:
        directory: Path = Path(directory)
        """Converted the isolated temporary workspace to a path object."""

        (directory / "famskat.R").write_text(SCRIPT)
        for replicate in range(REPLICATES):
            effect: float = 0.0 if replicate < REPLICATES - 4 else 0.35
            """Selected null or fixed non-null effect status for this replicate."""

            genotypes, relationship, y, weights, people = simulate(replicate, effect)
            """Regenerated the exact model inputs for both implementations."""

            ours: dict[str, float | list[float]] = asterism.VariantSetModel(
                [relationship], np.ones((people, 1)), y
            ).test(genotypes * weights)
            """Ran Asterism's weighted linear-kernel score test."""

            np.savetxt(directory / "G.txt", genotypes)
            np.savetxt(directory / "Phi.txt", relationship)
            np.savetxt(directory / "y.txt", y)
            np.savetxt(directory / "w.txt", weights)
            finished: subprocess.CompletedProcess[str] = subprocess.run(
                [
                    "Rscript",
                    str(directory / "famskat.R"),
                    str(directory / "G.txt"),
                    str(directory / "Phi.txt"),
                    str(directory / "y.txt"),
                    str(directory / "w.txt"),
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            """Ran famSKAT on the identical data and zero correlation setting."""

            if finished.returncode != 0:
                raise SystemExit(f"famSKAT failed: {finished.stderr.strip()[:300]}")
            theirs: float = float(finished.stdout.strip())
            """Parsed famSKAT's independently calculated p-value."""

            relative: float = abs(ours["p_value"] - theirs) / max(theirs, 1e-300)
            """Calculated the protected relative p-value difference."""

            compared.append(
                {
                    "replicate": replicate,
                    "asterism": ours["p_value"],
                    "famskat": theirs,
                    "relative": relative,
                    "eigenvalues_kept": len(ours["eigenvalues"]),
                }
            )
            if relative > RELATIVE_TOLERANCE:
                failures.append(
                    f"replicate {replicate}: {ours['p_value']:.6e} against "
                    f"{theirs:.6e}, relative {relative:.2e}"
                )

    # Every member of the correlation family must reproduce SKAT's own
    # r.corr, which is the identical quantity under a different name.
    family_compared: list[dict[str, float]] = []
    """Accumulated comparisons across the predeclared correlation family."""

    correlations: list[float] = [0.0, 0.04, 0.25, 0.5, 0.9]
    """Fixed the complete burden-to-SKAT correlation family being checked."""

    with tempfile.TemporaryDirectory() as directory:
        directory: Path = Path(directory)
        """Converted the second isolated workspace to a path object."""

        (directory / "famskat.R").write_text(SCRIPT)
        genotypes, relationship, y, weights, people = simulate(77, 0.30)
        """Generated the fixed non-null data shared by every correlation test."""

        family: dict[str, float | list[float]] = asterism.VariantSetModel(
            [relationship], np.ones((people, 1)), y
        ).test_family(genotypes * weights, correlations)
        """Ran Asterism's complete correlation-family test and combination."""

        np.savetxt(directory / "G.txt", genotypes)
        np.savetxt(directory / "Phi.txt", relationship)
        np.savetxt(directory / "y.txt", y)
        np.savetxt(directory / "w.txt", weights)
        for correlation, ours in zip(correlations, family["p_values"], strict=True):
            finished: subprocess.CompletedProcess[str] = subprocess.run(
                [
                    "Rscript",
                    str(directory / "famskat.R"),
                    str(directory / "G.txt"),
                    str(directory / "Phi.txt"),
                    str(directory / "y.txt"),
                    str(directory / "w.txt"),
                    str(correlation),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            """Ran famSKAT at this fixed member of the correlation family."""

            if finished.returncode != 0:
                raise SystemExit(f"famSKAT failed at rho={correlation}")
            theirs: float = float(finished.stdout.strip())
            """Parsed famSKAT's p-value for this correlation member."""

            relative: float = abs(ours - theirs) / max(theirs, 1e-300)
            """Calculated the protected relative family-member difference."""

            family_compared.append(
                {
                    "correlation": correlation,
                    "asterism": ours,
                    "famskat": theirs,
                    "relative": relative,
                }
            )
            if relative > RELATIVE_TOLERANCE:
                failures.append(
                    f"rho={correlation}: {ours:.6e} against {theirs:.6e}, "
                    f"relative {relative:.2e}"
                )
    # Combining cannot beat the best test it combines: that would be the
    # inflation the combination exists to prevent.
    if family["p_value"] < min(family["p_values"]) - 1e-12:
        failures.append(
            f"the combination {family['p_value']:.6e} beat its best member "
            f"{min(family['p_values']):.6e}"
        )

    worst: float = max(comparison["relative"] for comparison in compared)
    """Selected the largest replicate-level relative difference for reporting."""
    if failures:
        print("NOT AGREED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        json.dumps(
            {
                "what": "the variant-set score test against famSKAT",
                "skat_version": ACCEPTED_SKAT_VERSION,
                "null_model": "SKAT_NULL_emmaX, a kinship-carrying null",
                "kernel": "linear.weighted with Beta(1,25) weights",
                "people": FAMILIES * SIBS,
                "variants": VARIANTS,
                "replicates": REPLICATES,
                "worst_relative_difference": worst,
                "comparisons": compared,
                "correlation_family": {
                    "compared": family_compared,
                    "combined": family["p_value"],
                    "best_single": min(family["p_values"]),
                    "strongest_correlation": family["strongest_correlation"],
                    "note": "the combination sits above its best member, which is the price of looking",
                },
            },
            indent=2,
        )
    )
    print(
        f"\nThe two implementations agree to {worst:.1e} relative across "
        f"{REPLICATES} data sets.\nSKAT shares no code with Asterism."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
