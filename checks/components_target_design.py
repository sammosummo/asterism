"""Does the several-component model work at the design it will actually be used on?

**This analysis had no target-design rule, and it is the only one of the nine
that had none.** Its two existing rules simulate four hundred people in sibling
pairs, so the supported design range they justify is a largest family of two --
which refuses every SAFS pedigree, where families reach a hundred and sixty
five. A household model that cannot be run on the households it was built for
is not a supported analysis.

The design comes from the reviewed, participant-free aggregate already used by
the spatial target layout: 1,792 analysed people, 190 relationship components
with the largest at 165, and 1,140 shared locations sized 1 to 7. Nothing here
is a participant location or a reconstructed pedigree; the structure is matched
and the values are simulated.

**Why the household matrix is not the family matrix.** A household that is
exactly a sibling pair makes twice the kinship the identity plus the within-pair
pattern and the household matrix the identity plus that same pattern, so the
residual identity is their exact combination and the three bases have rank two.
Nothing separates additive from shared household there. Real households cross
families -- that is what the 1,140 locations record -- so this design identifies
what the sibling-pair simulations cannot.

Run with:

    uv run --no-project python checks/components_target_design.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asterism
from spatial_target_layout import (
    expanded_histogram,
    integer_histogram,
    load_fixture,
    mapping_field,
    synthetic_location_sizes,
    synthetic_relationship,
)

FIXTURE: Path = (
    Path(__file__).resolve().parent
    / "design_fixtures"
    / "spatial_component_presence_target_layout.json"
)
"""Named the reviewed aggregate this design is matched to."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "200"))
"""Replicates scored, overridable for a shorter exploratory run."""

TRUE_ADDITIVE: float = 0.45
"""Generating additive share, close to the aggregate's own profile."""

TRUE_HOUSEHOLD: float = 0.15
"""Generating shared-household share, the quantity this rule exists to recover."""

NOMINAL_COVERAGE: float = 0.95
"""Coverage the household interval claims and is scored against."""

BASE_SEED: int = 970_000
"""Base seed separating this rule's replicates from every other check."""


def clopper_pearson(hits: int, n: int) -> tuple[float, float]:
    """The exact interval on a binomial proportion."""
    low: float = beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0
    """Took the lower limit, or nought where nothing hit."""

    high: float = beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    """Took the upper limit, or one where everything hit."""
    return float(low), float(high)


def target_matrices() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Build the structure-matched relationship and household matrices.

    Returns:
        The additive relationship matrix and the shared-household matrix.
    """
    fixture: dict[str, object] = load_fixture(FIXTURE)
    """Read the reviewed aggregate without reconstructing any participant."""

    relationship_record: dict[str, object] = mapping_field(
        fixture, "relationship", "COMPONENTS_TARGET_RELATIONSHIP_INVALID"
    )
    """Selected the relationship-component aggregate."""

    sizes: tuple[int, ...] = expanded_histogram(
        integer_histogram(
            relationship_record.get("component_size_histogram"),
            "COMPONENTS_TARGET_COMPONENT_HISTOGRAM_INVALID",
        )
    )
    """Expanded the component-size histogram into one entry per family."""

    generator_settings: dict[str, object] = mapping_field(
        relationship_record,
        "synthetic_generator",
        "COMPONENTS_TARGET_PEDIGREE_GENERATOR_INVALID",
    )
    """Selected the reviewed pedigree-generation settings the aggregate fixes."""

    relationship, _ = synthetic_relationship(sizes, generator_settings)
    """Generated the additive relationship matrix matching the real structure."""

    shared: dict[str, object] = mapping_field(
        fixture, "shared_locations", "COMPONENTS_TARGET_SHARED_LOCATIONS_INVALID"
    )
    """Selected the shared-location aggregate."""

    location_sizes: tuple[int, ...] = synthetic_location_sizes(
        integer_histogram(
            shared.get("location_size_histogram"),
            "COMPONENTS_TARGET_LOCATION_HISTOGRAM_INVALID",
        )
    )
    """Expanded the location-size histogram into one entry per household."""

    people: int = relationship.shape[0]
    """Counted the people the relationship structure carries."""

    labels: npt.NDArray[np.int64] = np.concatenate(
        [
            np.full(size, index, dtype=np.int64)
            for index, size in enumerate(location_sizes)
        ]
    )[:people]
    """Assigned each person to a household, in the aggregate's own size profile."""

    household: npt.NDArray[np.float64] = (labels[:, None] == labels[None, :]).astype(
        np.float64
    )
    """Built the shared-household covariance from those memberships."""
    return relationship, household, max(sizes)


def one(
    relationship: npt.NDArray[np.float64],
    household: npt.NDArray[np.float64],
    replicate: int,
) -> dict[str, Any]:
    """Fit and score one replicate at the target design."""
    people: int = relationship.shape[0]
    """Counted the people in this design."""

    rng: np.random.Generator = np.random.default_rng(BASE_SEED + 7919 * replicate)
    """Created the generator this replicate owns alone."""

    residual_share: float = 1.0 - TRUE_ADDITIVE - TRUE_HOUSEHOLD
    """Left the remaining variance to the residual."""

    covariance: npt.NDArray[np.float64] = (
        TRUE_ADDITIVE * relationship
        + TRUE_HOUSEHOLD * household
        + residual_share * np.eye(people)
    )
    """Assembled the generating covariance from the three components."""

    y: npt.NDArray[np.float64] = np.linalg.cholesky(
        covariance + 1e-10 * np.eye(people)
    ) @ rng.standard_normal(people)
    """Drew one response with a known household share."""

    design: npt.NDArray[np.float64] = np.ones((people, 1))
    """Built the intercept-only fixed-effect design."""

    model: asterism.ComponentModel = asterism.ComponentModel(
        [relationship, household], design
    )
    """Prepared additive, household and residual components."""

    try:
        interval: dict[str, Any] = model.interval(y, component=1)
        """Profiled the household component's contribution proportion."""
    except ValueError as refusal:
        return {"replicate": replicate, "refusal": str(refusal)}

    truth: float = TRUE_HOUSEHOLD
    """Named the generating household share this interval must contain."""
    return {
        "replicate": replicate,
        "estimate": interval["estimate"],
        "lower": interval["lower"],
        "upper": interval["upper"],
        "profile_failures": interval["profile_failures"],
        "covered": bool(
            interval["profile_failures"] == 0
            and interval["lower"] <= truth <= interval["upper"]
        ),
    }


def main() -> int:
    """Score the several-component model at its intended design."""
    relationship, household, largest_component = target_matrices()
    """Built the structure-matched matrices once for every replicate."""

    people: int = relationship.shape[0]
    """Counted the analysed people this design carries."""

    largest_family: int = largest_component
    """Took the largest pedigree component, which is what a design range bounds.

    Counting each person's nonzero relationships instead gives a smaller number
    -- 115 here -- because not everyone in a hundred and sixty five person
    pedigree is related to everyone else in it. The block the likelihood
    factorises over is the component, so that is the size a supported design
    range makes a statement about.
    """

    print(
        f"{people} people, largest family {largest_family}, "
        f"{REPLICATES} replicates, true household share {TRUE_HOUSEHOLD}"
    )

    rows: list[dict[str, Any]] = [
        one(relationship, household, replicate) for replicate in range(REPLICATES)
    ]
    """Scored every replicate, refusals included."""

    refused: int = sum(1 for row in rows if "refusal" in row)
    """Counted replicates the model refused outright."""

    hits: int = sum(1 for row in rows if row.get("covered"))
    """Counted replicates whose interval contained the generating share."""

    low, high = clopper_pearson(hits, len(rows))
    """Quantified Monte Carlo uncertainty around the coverage."""

    estimates: list[float] = [
        float(row["estimate"]) for row in rows if "estimate" in row
    ]
    """Collected the point estimates that could be made."""

    mean_estimate: float = float(np.mean(estimates)) if estimates else float("nan")
    """Averaged the recovered household share."""

    passed: bool = low <= NOMINAL_COVERAGE <= high and refused == 0
    """Applied the pre-written acceptance rule."""

    print(f"  refused           {refused}/{len(rows)}")
    print(f"  household share   {mean_estimate:.4f} against a true {TRUE_HOUSEHOLD}")
    print(f"  coverage          {hits / len(rows):.4f} [{low:.4f}, {high:.4f}]")

    record: dict[str, Any] = {
        "check": "components_target_design",
        "participant_free": True,
        "people": people,
        "largest_family": largest_family,
        "replicates": len(rows),
        "true_additive": TRUE_ADDITIVE,
        "true_household": TRUE_HOUSEHOLD,
        "refused": refused,
        "mean_household_estimate": mean_estimate,
        "coverage": hits / len(rows),
        "coverage_interval": [low, high],
        "nominal_coverage": NOMINAL_COVERAGE,
        "passed": passed,
    }
    """Assembled the values-free evidence record."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / f"components-target-design-{date.today().isoformat()}.json"
    )
    """Named the dated evidence file."""

    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nwritten to {out}")
    print("PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
