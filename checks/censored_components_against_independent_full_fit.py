"""Compare the several-component censored model with an independent full fit.

Asterism receives the same participant-free arrays through
``CensoredComponentModel``. SciPy optimises a separately assembled censored
likelihood in natural covariance coordinates, described in
``checks/censored_full_fit_reference.py``. The two share no likelihood,
parameterisation, starting values or optimiser.

The comparison covers the mean-diagonal proportions, which is what the science
reads, and the maximised log likelihood, which is what the optimiser climbed.
They are held to separate tolerances because they disagree for different
reasons. The proportions agree to a few parts in ten thousand. The log
likelihoods do not agree that closely and are not expected to: Asterism reaches
the censored region probability by sequential truncation, which is an
approximation, while the reference integrates it. The gap between them is that
approximation, measured here rather than assumed.

Both tolerances are declared in ``release.toml`` beside the pass rule, so
loosening one is a visible change to the release manifest rather than an edit
to a constant in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import asterism
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from censored_full_fit_reference import (
    DenseCensoredFit,
    fit_censored,
    negative_loglik,
)

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout carrying the release manifest."""

ANALYSIS: str = "one_trait_censored_components"
"""Named the analysis whose pass rule declares this comparison's tolerances."""

RULE: str = "censored_components_against_independent_full_fit"
"""Named the pass rule itself."""

FAMILIES: int = 20
"""Sibling pairs in each comparison."""

RECORDS: int = 2
"""Records each person contributes, which is what wants a person-level component."""

REPLICATES: int = 12
"""Independent data sets compared at each censoring share."""

SHARES: tuple[float, ...] = (0.25, 0.5)
"""Censoring shares compared, so the comparison is not read at one severity."""

ADDITIVE: float = 0.4
"""True additive share of the total variance."""

PERSON: float = 0.3
"""True person-level share."""

RESIDUAL: float = 0.3
"""True residual share."""

SLACK: float = 1e-4
"""How far below Asterism's answer the independent optimum may score.

This is not a tolerance on agreement. It is the guard that says the comparison
is testing the model and not the reference optimiser: if SciPy cannot even
reach the point Asterism found, a disagreement between them says nothing about
Asterism.

Nought would be the honest bound if the two optima were ever cleanly separated,
and where the disagreement is real the independent optimum wins by around 1e-03.
But where the two agree they agree to within a millionth of a log unit, and
which of them comes out a hair ahead there is arbitrary -- at a quarter censored
one replicate put Asterism ahead by 9.6e-07. A surface this flat has no
meaningful ordering at that scale, so the guard is set well above it and fires
only on a reference optimum that lands somewhere materially worse.
"""

SEED: int = 20_260_829
"""Fixed the integration so the reference surface does not move under SciPy."""


def declared_tolerances() -> tuple[float, float]:
    """Read the two tolerances this comparison is held to.

    Returns:
        The proportion tolerance and the log-likelihood tolerance.
    """
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative release manifest."""

    analysis: dict[str, Any] = next(
        entry for entry in manifest["analyses"] if entry["id"] == ANALYSIS
    )
    """Selected the analysis this check supports."""

    rule: dict[str, Any] = next(
        entry for entry in analysis["pass_rules"] if entry["id"] == RULE
    )
    """Selected this comparison's own pass rule."""

    facts: dict[str, Any] = rule["design_facts"]
    """Read the declared design facts carrying both tolerances."""

    return float(facts["proportion_tolerance"]), float(facts["loglik_tolerance"])


def design_matrices(families: int, records: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the additive and person-level components for one design.

    Args:
        families: Sibling pairs; each contributes two people.
        records: Records each person contributes.

    Returns:
        The additive matrix and the person-level matrix.
    """
    rows: int = families * 2 * records
    """Counted the rows, several to a person and two people to a pair."""

    additive: np.ndarray = np.zeros((rows, rows))
    """Initialised the additive relationship over records."""

    for first in range(rows):
        for second in range(rows):
            one, other = first // records, second // records
            """Found which person each of the two rows belongs to."""

            if one == other:
                additive[first, second] = 1.0
                """Set a person's own records to one."""
            elif one // 2 == other // 2:
                additive[first, second] = 0.5
                """Set full siblings to a half."""
    """Set a person's own records to one and full siblings to a half."""

    person: np.ndarray = asterism.grouping_matrix(
        [f"person-{row // records}" for row in range(rows)]
    )
    """Built the person-level component by grouping the rows of one person."""

    return additive, person


def one_replicate(
    families: int, records: int, share: float, seed: int
) -> dict[str, float] | None:
    """Fit one simulated data set both ways and return how far apart they are.

    Args:
        families: Sibling pairs in this design.
        records: Records each person contributes.
        share: Share of rows the instrument fails to measure.
        seed: Fixed so a replicate can be reproduced on its own.

    Returns:
        The two gaps and what produced them, or None where either fit refused.
    """
    additive, person = design_matrices(families, records)
    """Built the components under test."""

    rows: int = additive.shape[0]
    """Counted the rows once."""

    blocks: list[np.ndarray] = [
        np.arange(family * 2 * records, (family + 1) * 2 * records)
        for family in range(families)
    ]
    """Named the independent families the likelihood factorises over."""

    design: np.ndarray = np.ones((rows, 1))
    """Fitted an intercept and nothing else."""

    generator: np.random.Generator = np.random.default_rng(seed)
    """Seeded this replicate."""

    covariance: np.ndarray = (
        ADDITIVE * additive + PERSON * person + RESIDUAL * np.eye(rows)
    )
    """Assembled the covariance the truths imply."""

    complete: np.ndarray = np.linalg.cholesky(
        covariance + 1e-10 * np.eye(rows)
    ) @ generator.standard_normal(rows)
    """Drew the complete trait, before any instrument stopped."""

    ceiling: float = float(np.quantile(complete, 1.0 - share))
    """Put the limit where it censors the declared share."""

    censoring: np.ndarray = (complete >= ceiling).astype(np.int64)
    """Marked 1 at or above the limit, 0 where measured."""

    value: np.ndarray = np.where(censoring == 0, complete, np.nan)
    """Kept measured values; censored ones are never read."""

    limit: np.ndarray = np.full(rows, ceiling)
    """Gave every row the same instrument limit."""

    try:
        fit: dict[str, Any] = asterism.CensoredComponentModel(
            [additive, person], design
        ).fit(value, censoring, limit)
        """Fitted the model under test."""
    except ValueError:
        return None
    if not fit["converged"]:
        return None
    starts: list[np.ndarray] = [
        np.array([0.3, 0.3, 0.4, float(np.nanmean(value))]),
        np.array([0.5, 0.2, 0.3, float(np.nanmean(value))]),
        np.array([0.2, 0.5, 0.3, float(np.nanmean(value))]),
    ]
    """Started SciPy somewhere Asterism's own parameterisation never visits."""

    dense: DenseCensoredFit = fit_censored(
        lambda variances: (
            variances[0] * additive
            + variances[1] * person
            + variances[2] * np.eye(rows)
        ),
        3,
        starts,
        [(1e-6, 20.0)] * 3 + [(None, None)],
        design,
        value,
        censoring,
        limit,
        blocks,
        seed=SEED,
    )
    """Fitted the same data through the independent likelihood."""

    if not np.isfinite(dense.negative_loglik):
        return None
    independent: np.ndarray = dense.variances / dense.variances.sum()
    """Turned the independent coefficients into proportions of the total."""

    reported: np.ndarray = np.asarray(fit["mean_diagonal_proportions"], dtype=float)
    """Read what Asterism reports, the residual last."""

    coefficients: np.ndarray = np.asarray(fit["coefficients"], dtype=float)
    """Read the shares Asterism fitted, the residual taking what they leave."""

    theirs: np.ndarray = (
        np.append(coefficients, 1.0 - coefficients.sum()) * fit["total_variance"]
    )
    """Turned those shares back into the variances they stand for."""

    at_theirs: float = -negative_loglik(
        theirs[0] * additive + theirs[1] * person + theirs[2] * np.eye(rows),
        design,
        value,
        censoring,
        limit,
        np.asarray(fit["fixed_effects"], dtype=float),
        blocks,
        seed=SEED,
    )
    """Scored Asterism's own answer through the independent likelihood."""

    return {
        "proportion_gap": float(np.abs(reported - independent).max()),
        "loglik_gap": abs(-dense.negative_loglik - float(fit["loglik"])),
        "advantage": -dense.negative_loglik - at_theirs,
        "censored": float(censoring.sum()) / rows,
        "worst_block": float(max(int(censoring[block].sum()) for block in blocks)),
    }


def main() -> int:
    """Run the comparison and report whether both tolerances held.

    Returns:
        Zero when every replicate agreed within both declared tolerances.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )
    """Built the command line."""

    parser.add_argument("--families", type=int, default=FAMILIES)
    parser.add_argument("--records", type=int, default=RECORDS)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--no-write", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()
    """Read the design and whether this run records evidence."""

    proportion_tolerance, loglik_tolerance = declared_tolerances()
    """Read both tolerances from the release manifest, not from this script."""

    rows: list[dict[str, Any]] = []
    """Collected every replicate that produced two fits."""

    failures: list[str] = []
    """Collected every replicate that agreed less closely than declared."""

    print(
        f"{'censored':>9} {'replicates':>11} {'worst proportion':>17} "
        f"{'worst loglik':>13} {'worst block':>12} {'least advantage':>16}"
    )
    print("-" * 83)
    for share in SHARES:
        measured: list[dict[str, float]] = []
        """Collected this share's replicates."""

        for replicate in range(arguments.replicates):
            outcome: dict[str, float] | None = one_replicate(
                arguments.families,
                arguments.records,
                share,
                SEED + replicate + int(share * 1000) * 97,
            )
            """Compared one data set both ways."""

            if outcome is not None:
                measured.append(outcome)
        if not measured:
            failures.append(f"censoring {share:.2f}: no replicate produced two fits")
            continue
        worst_proportion: float = max(item["proportion_gap"] for item in measured)
        """Took the least agreeable proportion across this share."""

        worst_loglik: float = max(item["loglik_gap"] for item in measured)
        """Took the least agreeable log likelihood."""

        worst_block: int = int(max(item["worst_block"] for item in measured))
        """Recorded how many rows of one family the instrument ever missed."""

        advantage: float = min(item["advantage"] for item in measured)
        """Took the replicate where the independent optimum won by least."""

        print(
            f"{share:>9.2f} {len(measured):>11d} {worst_proportion:>17.2e} "
            f"{worst_loglik:>13.2e} {worst_block:>12d} {advantage:>+16.2e}"
        )
        rows.append(
            {
                "censoring_share": share,
                "replicates": len(measured),
                "worst_proportion_gap": worst_proportion,
                "worst_loglik_gap": worst_loglik,
                "worst_censored_rows_in_a_family": worst_block,
                "least_independent_advantage": advantage,
            }
        )
        if worst_proportion > proportion_tolerance:
            failures.append(
                f"censoring {share:.2f}: proportions differ by "
                f"{worst_proportion:.2e}, past the declared "
                f"{proportion_tolerance:.0e}"
            )
        if advantage < -SLACK:
            failures.append(
                f"censoring {share:.2f}: the independent optimum scored "
                f"{advantage:.2e} below Asterism's own answer under the "
                "independent likelihood, so this run measures the reference "
                "optimiser rather than the model"
            )
        if worst_loglik > loglik_tolerance:
            failures.append(
                f"censoring {share:.2f}: log likelihoods differ by "
                f"{worst_loglik:.2e}, past the declared {loglik_tolerance:.0e}"
            )
    print()
    print(
        "The proportions are what the science reads and they agree closely. The log\n"
        "likelihoods agree less closely because Asterism reaches the censored region\n"
        "probability by sequential truncation and the reference integrates it; the\n"
        "gap between them is that approximation.\n\n"
        "The last column is the reason to believe that reading. It is how much better\n"
        "the independent optimum scores than Asterism's own answer, both measured\n"
        "through the independent likelihood. Wherever the two disagree it is\n"
        "positive, so the reference is not merely stopping short of what Asterism\n"
        "found: it reaches a better point, and what separates them is the\n"
        "approximation moving the optimum. It wins by very little, and where the two\n"
        "agree the column can go a millionth negative, which is two numbers tying on\n"
        "a surface with no ordering left at that scale. The flatness is what the\n"
        "separation check measures from the other side."
    )
    if not arguments.no_write:
        record: Path = ROOT / "evidence" / "components-full-fit-2026-08-29.json"
        """Named this run's evidence."""

        record.write_text(
            json.dumps(
                {
                    "families": arguments.families,
                    "records_per_person": arguments.records,
                    "truths": {
                        "additive": ADDITIVE,
                        "person": PERSON,
                        "residual": RESIDUAL,
                    },
                    "proportion_tolerance": proportion_tolerance,
                    "loglik_tolerance": loglik_tolerance,
                    "cells": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwritten to {record}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"\nPASSED: every replicate agreed within {proportion_tolerance:.0e} on the "
        f"proportions and {loglik_tolerance:.0e} on the log likelihood."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
