"""How many relatives before an additive and a person-level component separate?

Both are one within a person. What tells them apart is that additive kinship is
also non-zero between relatives and a person-level matrix is not, so the
information comes from **related pairs** rather than from records: a person's
second record sharpens their sum and says nothing about the split.

That makes the question one about the design, and this measures where a design
is enough.

**The question is how many, not whether.** A person-level component is only ever
added because a person contributes more than one record -- two ears, two
occasions -- so the sweep is over related pairs at two and three records each,
which is what a real design varies. (With one record a person's matrix is the
identity and the term is the residual, which is why nobody writes that model;
it is not a trap anyone falls into and is not swept here.)

components exchange variance, an over-estimate of one arrives with an
under-estimate of the other, so the correlation between their estimates across
replicates goes towards minus one while the standard deviation of their sum
stays small. Both are reported. A wide estimate uncorrelated with its neighbour
is an imprecise answer; a wide estimate correlated with it is half of one answer
reported twice, and the sum is the part that was actually measured.

**The pass rule is consistency, not unbiasedness, and the difference is the
point.** Maximum likelihood variance components are biased in finite samples --
that is what REML exists to remedy, and this model cannot use REML because a
censored observation has no residual to project onto the null space of the
design. Measured here at two records each: the additive share comes back at
0.358 against a truth of 0.40 over thirty sibling pairs, and at 0.398 over two
hundred and forty, while the person-level share falls from 0.327 to 0.303
against 0.30. The smaller component absorbs what the design cannot attribute to
the larger one, and the two find their truths as relatives accumulate.

So the rule is that the bias shrinks with the design and is small at the largest
one swept. Demanding it be absent everywhere would fail this estimator for
having a property every maximum-likelihood variance component has.

How precisely a given pedigree separates the two is a fact about that pedigree,
and the table is the answer to it rather than a verdict on it.

Run with:

    uv run --no-project python checks/component_separation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import asterism
import numpy as np

ADDITIVE: float = 0.40
"""Generating additive share."""

PERSON: float = 0.30
"""Generating person-level share."""

RESIDUAL: float = 0.30
"""What the two leave, and what the model estimates as the remainder."""

FAMILIES: list[int] = [30, 60, 120, 240]
"""Sibling pairs swept, which is what carries the additive information.

The range is chosen to bracket a real extended-family study rather than to make
a point: the smallest cell is smaller than anything one would run and the
largest is bigger, so the boundary falls inside the table rather than off its
edge."""

RECORDS: list[int] = [2, 3]
"""Records per person -- two ears, or three occasions.

One record is not swept. A person-level component exists because a person was
measured more than once; with one record its matrix is the identity and the term
is the residual, so the model nobody writes is not a boundary worth spending a
third of the sweep on."""

REPLICATES: int = 200
"""Replicates per cell, which fixes how finely bias can be resolved."""

CENSORING_SHARE: float = 0.4
"""Share of records censored, this being the censored model."""

BIAS_AT_LARGEST: float = 0.03
"""How far the mean estimate may sit from its truth at the largest design swept.

An estimator that is merely consistent must still have arrived by the time a
design is realistic, and this is where "arrived" is drawn. It is not a claim
that smaller designs are wrong -- they are biased, measurably and in the
direction theory predicts -- only that the bias is gone by the top of the
sweep."""

SPLIT_COST: float = 3.0
"""How many times less precisely each component may be known than their sum
before the split is the thing limiting what can be said.

**Not the correlation between the estimates, which was the first attempt and
measures the wrong thing.** That correlation goes towards minus one as a design
grows -- measured at -0.920, -0.959 and -0.973 over thirty, sixty and a hundred
and twenty sibling pairs -- because the sum becomes precisely determined and the
split is then the only uncertainty left. It describes the shape of the
uncertainty, which is always a trade-off here, and says nothing about whether a
design is adequate. What does say so is how much worse each part is known than
the whole."""


def design_matrices(families: int, records: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the additive and person-level components for one design.

    Args:
        families: Sibling pairs; each contributes two people.
        records: Records each person contributes.

    Returns:
        The additive matrix, the person-level matrix, and the row count.
    """
    people: int = families * 2
    """Counted the people, two to a sibling pair."""

    rows: int = people * records
    """Counted the rows, several to a person."""

    additive: np.ndarray = np.zeros((rows, rows))
    """Initialised the additive relationship over records."""

    for i in range(rows):
        for j in range(rows):
            first, second = i // records, j // records
            """Found which person each of the two rows belongs to."""

            if first == second:
                additive[i, j] = 1.0
                """Set a person's own records to one."""
            elif first // 2 == second // 2:
                additive[i, j] = 0.5
                """Set full siblings to a half."""
    """Set a person's own records to one and full siblings to a half."""

    person: np.ndarray = asterism.grouping_matrix(
        [f"person-{row // records}" for row in range(rows)]
    )
    """Built the person-level component by grouping the rows of one person."""
    return additive, person, rows


def one_replicate(families: int, records: int, seed: int) -> tuple[float, float] | None:
    """Fit one simulated data set and return the two estimates.

    Args:
        families: Sibling pairs in this design.
        records: Records each person contributes.
        seed: Fixed so a cell can be reproduced on its own.

    Returns:
        The additive and person-level proportions, or None where no fit
        converged.
    """
    additive, person, rows = design_matrices(families, records)
    """Built the components this design implies."""

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

    ceiling: float = float(np.quantile(complete, 1.0 - CENSORING_SHARE))
    """Put the limit where it censors the declared share."""

    censoring: np.ndarray = (complete >= ceiling).astype(np.int64)
    """Marked 1 at or above the limit, 0 where measured."""

    value: np.ndarray = np.where(censoring == 0, complete, np.nan)
    """Kept measured values; censored ones are never read."""

    model: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
        [additive, person], np.ones((rows, 1))
    )
    """Built the model with the two components under test."""

    try:
        fit: dict[str, Any] = model.fit(value, censoring, np.full(rows, ceiling))
        """Fitted this replicate."""
    except ValueError:
        return None
    """Refused fits are counted rather than retried."""

    if not fit["converged"]:
        return None
    shares: list[float] = fit["mean_diagonal_proportions"]
    """Took the comparable quantity, the residual's share last."""
    return float(shares[0]), float(shares[1])


def main() -> int:
    """Sweep the design and report where the two components separate."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Created the parser for the sweep's extent."""

    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--families", type=int, nargs="+", default=FAMILIES)
    parser.add_argument("--records", type=int, nargs="+", default=RECORDS)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--no-write", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()
    """Read how far to sweep and how hard."""

    report: dict[str, Any] = {}
    """Collected one record per design cell."""

    failures: list[str] = []
    """Collected every cell where an identified component came back biased."""

    print(
        f"{'records':>8} {'families':>9} {'additive':>18} {'person':>18} "
        f"{'corr':>7} {'sd(sum)':>8}  split costs"
    )
    print("-" * 88)
    for records in arguments.records:
        for families in arguments.families:
            pairs: list[tuple[float, float]] = [
                got
                for replicate in range(arguments.replicates)
                if (
                    got := one_replicate(
                        families,
                        records,
                        arguments.seed + 7919 * replicate + families + records,
                    )
                )
                is not None
            ]
            """Fitted every replicate this cell asked for, dropping none silently."""

            if len(pairs) < arguments.replicates // 2:
                failures.append(
                    f"records {records}, families {families}: only "
                    f"{len(pairs)} of {arguments.replicates} replicates fitted"
                )
                continue

            additive_hat: np.ndarray = np.array([p[0] for p in pairs])
            """Collected the additive estimate from every replicate."""

            person_hat: np.ndarray = np.array([p[1] for p in pairs])
            """Separated the two estimates across replicates."""

            correlation: float = (
                float(np.corrcoef(additive_hat, person_hat)[0, 1])
                if len(pairs) > 2
                else float("nan")
            )
            """Measured the trade-off directly: two components exchanging
            variance move opposite ways across replicates."""

            summary: dict[str, Any] = {
                "records_per_person": records,
                "families": families,
                "replicates_fitted": len(pairs),
                "additive": {
                    "mean": float(additive_hat.mean()),
                    "sd": float(additive_hat.std(ddof=1)),
                    "truth": ADDITIVE,
                },
                "person": {
                    "mean": float(person_hat.mean()),
                    "sd": float(person_hat.std(ddof=1)),
                    "truth": PERSON,
                },
                "correlation_between_estimates": correlation,
                "sd_of_their_sum": float((additive_hat + person_hat).std(ddof=1)),
                "split_costs_this_many_times_the_precision": float(
                    max(additive_hat.std(ddof=1), person_hat.std(ddof=1))
                    / max((additive_hat + person_hat).std(ddof=1), 1e-12)
                ),
            }
            """Retained enough facts to recompute this cell's verdict."""

            # Bias is recorded at every design and scored only at the
            # largest, because a maximum-likelihood variance component is
            # biased in small samples by construction. What has to hold is that
            # it goes away.
            for name, truth in (("additive", ADDITIVE), ("person", PERSON)):
                values: np.ndarray = additive_hat if name == "additive" else person_hat
                """Selected the estimates for this component."""

                bias: float = float(values.mean()) - truth
                """Measured how far this cell's mean sits from its truth."""

                summary[f"{name}_bias"] = bias
                """Recorded the bias at this design, scored or not."""

                # **Allowing for how well the bias itself is measured.** The
                # mean of a cell carries its own Monte Carlo error, and a run
                # with few replicates cannot resolve a bias of this size: at
                # forty replicates that error is about 0.026, so a reading of
                # -0.037 is not distinguishable from none. Scoring the raw
                # difference would fail a short run for being short.
                resolvable: float = BIAS_AT_LARGEST + 2.0 * float(
                    values.std(ddof=1) / np.sqrt(len(values))
                )
                """Widened the tolerance by twice this cell's Monte Carlo error."""

                summary[f"{name}_bias_resolvable_beyond"] = resolvable
                """Recorded what this run could have resolved, so a later reader
                can tell a small bias from a short run."""

                if families == max(arguments.families) and abs(bias) > resolvable:
                    failures.append(
                        f"records {records}, families {families}: {name} came "
                        f"back at {values.mean():.3f} where the truth is "
                        f"{truth}, a bias of {bias:+.3f} at the largest design "
                        f"swept -- past {resolvable:.3f}, which is the "
                        f"tolerance widened by this run's own Monte Carlo error"
                    )
            """Recorded the bias everywhere and scored it where it must be gone."""

            report[f"records={records}/families={families}"] = summary
            """Kept this cell's record."""

            print(
                f"{records:>8} {families:>9} "
                f"{additive_hat.mean():>8.3f} ({additive_hat.std(ddof=1):.3f}) "
                f"{person_hat.mean():>8.3f} ({person_hat.std(ddof=1):.3f}) "
                f"{correlation:>7.3f} "
                f"{(additive_hat + person_hat).std(ddof=1):>8.3f}  "
                f"{max(additive_hat.std(ddof=1), person_hat.std(ddof=1)) / max((additive_hat + person_hat).std(ddof=1), 1e-12):>5.1f}x",
                flush=True,
            )

    receipt: dict[str, Any] = {
        "what": (
            "whether an additive component and a person-level one can be told "
            "apart, swept over records per person and related pairs"
        ),
        "date": date.today().isoformat(),
        "truths": {
            "additive": ADDITIVE,
            "person": PERSON,
            "residual": RESIDUAL,
        },
        "censoring_share": CENSORING_SHARE,
        "replicates_per_cell": arguments.replicates,
        "split_cost_threshold": SPLIT_COST,
        "cells": report,
        "passed": not failures,
        "failures": failures,
        "note": (
            "A design that does not separate the two is reported and not "
            "failed. The pass rule is that the estimator is correct where the "
            "components are identified; where they are not, that is a fact "
            "about the design."
        ),
    }
    """Built the complete evidence record for the sweep."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / f"component-separation-{receipt['date']}.json"
    )
    """Selected the evidence path used outside release mode."""
    if not arguments.no_write:
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"\nwritten to {out}")

    print("\nWhat the split costs, as a multiple of how well the sum is known:")
    for name, cell in report.items():
        cost: float = cell["split_costs_this_many_times_the_precision"]
        """Took how much worse each part is known than the whole."""

        limiting: str = (
            "  <- the split is what limits this design" if cost > SPLIT_COST else ""
        )
        """Marked the designs where the split, not the data, is the constraint."""

        print(f"  {name:>28}: {cost:>5.1f}x{limiting}")
    print(
        "\nThe correlation between the two estimates approaches minus one as a "
        "design grows.\nThat is the sum becoming certain while the split stays "
        "open, not a design getting worse."
    )

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"\nPASSED: the bias is gone by the largest design swept, within "
        f"{BIAS_AT_LARGEST} and this run's own Monte Carlo error. It is present "
        f"at the smaller ones, which is what a maximum-likelihood variance "
        f"component does and not a fault."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
