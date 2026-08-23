"""The compiled profile intervals against the independent Python ones.

Every other quantity Asterism reports is checked against software written by
somebody else: SOLAR for ML, R `regress` for REML. **Neither computes a profile
interval for these models.** SOLAR gives standard errors and `regress` gives
variance components, so the two-trait intervals had no second opinion at all and
rested entirely on calibration.

Calibration and this check answer different questions, and a package wants both.
Calibration says a 95 per cent interval contains the truth 95 per cent of the
time — that the recipe works. It cannot say the compiled arithmetic agrees with
an independent route to the same definition, because a second implementation
with the same coverage could still disagree case by case. This says the two
routes land in the same place.

The reference is `checks/bivariate_reference.py`, written in numpy and scipy
against the same definition but sharing no code with the Rust: its own
optimiser, its own constrained refits, its own bisection.

**It covers the four estimated quantities and not the phenotypic correlation**,
which the reference has no constrained fit for — pinning it means substituting
for the residual correlation rather than holding a parameter, and building that
twice was not judged worth it. The phenotypic interval rests on calibration, on
the chain rule being checked against a central difference, and on the four it is
built from agreeing here.

The tolerance is loose on purpose. Two bisections to different tolerances over
two optimisers will not agree to machine precision, and demanding that they do
would be a statement about the searches rather than about the intervals.

Run with:

    uv run --no-project python checks/bivariate_intervals_against_reference.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import asterism
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bivariate_reference as reference

# Two bisections stopping at different tolerances, over two different
# optimisers. Agreement to a thousandth on a quantity that runs from -1 to 1 is
# what this can honestly demonstrate.
# The worst endpoint gap measured here is 2.6e-5, set by how finely the two
# profiles bisect rather than by any disagreement about the mathematics. A
# tolerance of 2e-3 was eighty times looser than that and would have passed a
# real regression; 2e-4 keeps roughly eight times headroom over the worst
# measured gap.
TOLERANCE: float = 2e-4
"""Maximum accepted absolute disagreement at either interval endpoint."""

QUANTITIES: tuple[tuple[str, int, str], ...] = (
    ("h2_first", 2, "h2_trait_a"),
    ("h2_second", 3, "h2_trait_b"),
    ("rho_g", 4, "rho_g"),
    ("rho_e", 5, "rho_e"),
)
"""Mapped public interval names to independent parameter indices and fit fields."""


def simulate(
    families: int,
    per_family: int,
    seed: int,
) -> tuple[
    np.ndarray,
    list[list[bool]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
    int,
]:
    """Generate unbalanced two-trait data with a known answer.

    Args:
        families: Number of independent full-sibling families.
        per_family: Full siblings represented within each family.
        seed: Reproducible random-number seed.

    Returns:
        Relationship matrix, public observation mask, reference observation mask,
        full reference design and outcome, observed public design and outcome,
        true parameters, and sample size.
    """
    n: int = families * per_family
    """Calculated the number of simulated people."""

    relationship: np.ndarray = np.zeros((n, n))
    """Initialised the block-diagonal additive relationship matrix."""

    for family in range(families):
        block: slice = slice(family * per_family, (family + 1) * per_family)
        """Located this family's contiguous rows and columns."""

        relationship[block, block] = 0.5
        """Set every within-family relationship to the full-sibling coefficient."""

    np.fill_diagonal(relationship, 1.0)

    truth: dict[str, float] = {"h1": 0.6, "h2": 0.35, "rg": 0.55, "re": 0.25}
    """Fixed both heritabilities and covariance correlations for simulation."""

    genetic: np.ndarray = np.array(
        [
            [truth["h1"], truth["rg"] * np.sqrt(truth["h1"] * truth["h2"])],
            [truth["rg"] * np.sqrt(truth["h1"] * truth["h2"]), truth["h2"]],
        ]
    )
    """Constructed the two-trait additive covariance from the fixed truth."""

    residual: np.ndarray = np.array(
        [
            [
                1 - truth["h1"],
                truth["re"] * np.sqrt((1 - truth["h1"]) * (1 - truth["h2"])),
            ],
            [
                truth["re"] * np.sqrt((1 - truth["h1"]) * (1 - truth["h2"])),
                1 - truth["h2"],
            ],
        ]
    )
    """Constructed the two-trait residual covariance from the fixed truth."""

    full: np.ndarray = np.kron(genetic, relationship) + np.kron(
        residual,
        np.eye(n),
    )
    """Expanded trait covariances across the relationship and identity bases."""

    draw: np.ndarray = np.linalg.cholesky(
        full + 1e-10 * np.eye(2 * n)
    ) @ np.random.default_rng(seed).standard_normal(2 * n)
    """Drew the reproducible outcome after the fixed diagonal stabiliser."""

    # Unbalanced, because that is the case eigen-simplification cannot take.
    observed: np.ndarray = np.array([[True, person % 8 != 0] for person in range(n)])
    """Applied the fixed unbalanced second-trait observation pattern."""

    mask: list[list[bool]] = [
        [bool(observed[person, 0]), bool(observed[person, 1])] for person in range(n)
    ]
    """Converted the observation pattern to the public nested-list contract."""

    # The two implementations want the same data laid out differently. The
    # reference takes the full 2n rows with `observed` saying which are real;
    # Asterism takes only the real ones. Getting this backwards is not an error
    # either of them raises — it is a different data set quietly fitted — so the
    # two layouts are built from one draw here rather than separately.
    full_values: np.ndarray = np.zeros(2 * n)
    """Initialised the reference's complete interleaved outcome layout."""

    full_values[0::2] = draw[:n]
    """Placed first-trait values in the reference's even rows."""

    full_values[1::2] = draw[n:]
    """Placed second-trait values in the reference's odd rows."""

    full_design: np.ndarray = np.zeros((2 * n, 2))
    """Initialised the reference's complete trait-specific intercept design."""

    full_design[0::2, 0] = 1.0
    """Activated the first-trait intercept on even rows."""

    full_design[1::2, 1] = 1.0
    """Activated the second-trait intercept on odd rows."""

    rows: list[int] = [
        person * 2 + trait
        for person in range(n)
        for trait in (0, 1)
        if observed[person, trait]
    ]
    """Selected interleaved rows observed by Asterism's compact public layout."""

    return (
        relationship,
        mask,
        observed,
        full_design,
        full_values,
        np.ascontiguousarray(full_design[rows]),
        np.ascontiguousarray(full_values[rows]),
        truth,
        n,
    )


def main() -> int:
    """Compare every supported bivariate profile endpoint with the reference.

    Returns:
        Zero when convergence and all endpoint-agreement checks pass, otherwise one.
    """
    (
        relationship,
        mask,
        observed,
        full_design,
        full_values,
        design,
        values,
        truth,
        n,
    ) = simulate(25, 6, 707)
    """Generated the fixed unbalanced data in both required row layouts."""

    theirs_fit: dict[str, float | bool | str] = reference.fit(
        relationship,
        full_values,
        observed,
        full_design,
        True,
    )
    """Fitted the independent NumPy and SciPy point reference."""

    model: asterism.BivariateModel = asterism.BivariateModel(
        relationship,
        mask,
        design,
    )
    """Built the documented bivariate model for the independently simulated data."""

    fit: dict[str, float | bool | list[float]] = model.fit(values, reml=True)
    """Fitted the point estimates through the same public boundary users call."""

    ours_point: dict[str, float] = {
        "h2_first": fit["h2_first"],
        "h2_second": fit["h2_second"],
        "rho_g": fit["rho_g"],
        "rho_e": fit["rho_e"],
    }
    """Selected the point estimates independently compared with profile endpoints."""

    print(
        f"Two-trait profile intervals, REML, n = {n}, "
        f"{observed[:, 0].sum()} with the first trait and "
        f"{observed[:, 1].sum()} with the second.\n"
    )
    print("The compiled interval against an independent Python one. Neither SOLAR")
    print("nor R computes these, so this is the only second opinion there is.\n")
    print(
        f"{'quantity':<12}{'asterism':>22}{'reference':>22}"
        f"{'lower gap':>11}{'upper gap':>11}"
    )

    failures: list[str] = []
    """Collected convergence, gradient and endpoint-agreement failures."""

    if not fit["converged"]:
        failures.append("Asterism did not declare convergence")
    if not fit["scaled_gradient"] < 1e-7:
        failures.append(
            f"Asterism scaled projected gradient was {fit['scaled_gradient']:.3e}"
        )

    recorded: dict[str, dict[str, dict[str, float] | float]] = {}
    """Accumulated both interval routes and their endpoint differences."""

    started: float = time.perf_counter()
    """Started timing the four independent constrained profile comparisons."""

    for name, index, reference_key in QUANTITIES:
        ours: dict[str, float | bool] = model.interval(values, name, reml=True)
        """Computed one profile interval through its documented named record."""

        theirs: dict[str, float | bool] = reference.profile_interval(
            relationship,
            full_values,
            observed,
            full_design,
            index,
            theirs_fit[reference_key],
            True,
        )
        """Computed the same profile interval through the independent route."""
        low_gap: float = abs(ours["lower"] - theirs["lower"])
        """Measured disagreement at the lower profile endpoint."""

        high_gap: float = abs(ours["upper"] - theirs["upper"])
        """Measured disagreement at the upper profile endpoint."""

        recorded[name] = {
            "asterism": {"lower": ours["lower"], "upper": ours["upper"]},
            "reference": {"lower": theirs["lower"], "upper": theirs["upper"]},
            "lower_gap": low_gap,
            "upper_gap": high_gap,
        }
        """Recorded both routes and their endpoint differences for release evidence."""
        print(
            f"{name:<12}[{ours['lower']:>9.5f},{ours['upper']:>9.5f}]"
            f"[{theirs['lower']:>9.5f},{theirs['upper']:>9.5f}]"
            f"{low_gap:>11.2e}{high_gap:>11.2e}"
        )
        if max(low_gap, high_gap) > TOLERANCE:
            failures.append(
                f"{name} endpoints differ by up to {max(low_gap, high_gap):.3e}"
            )

    if failures:
        print("\nDISAGREEMENT:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(
        f"\nBoth routes agree on every endpoint to {TOLERANCE:.0e}, in "
        f"{time.perf_counter() - started:.0f}s."
    )
    print("Different optimisers, different constrained refits, different")
    print("bisections, same interval. Calibration alone cannot establish this:")
    print("two implementations can both cover at 95 per cent and still disagree")
    print("case by case.")

    print(
        json.dumps(
            {
                "estimator": "reml",
                "people": n,
                "observations_by_trait": [
                    int(observed[:, 0].sum()),
                    int(observed[:, 1].sum()),
                ],
                "balanced": False,
                "truth": truth,
                "asterism_point_estimates": ours_point,
                "intervals": recorded,
                "comparison_tolerance": TOLERANCE,
                "not_covered": (
                    "the phenotypic correlation, which the reference has no "
                    "constrained fit for; it rests on calibration and on the four "
                    "it is built from agreeing here"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
