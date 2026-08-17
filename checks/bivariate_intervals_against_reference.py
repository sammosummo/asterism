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

import numpy as np
from asterism import _core

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
TOLERANCE = 2e-4

QUANTITIES = (
    ("h2_first", 2, "h2_trait_a"),
    ("h2_second", 3, "h2_trait_b"),
    ("rho_g", 4, "rho_g"),
    ("rho_e", 5, "rho_e"),
)


def simulate(families: int, per_family: int, seed: int):
    """Unbalanced two-trait data with a known answer."""
    n = families * per_family
    relationship = np.zeros((n, n))
    for family in range(families):
        block = slice(family * per_family, (family + 1) * per_family)
        relationship[block, block] = 0.5
    np.fill_diagonal(relationship, 1.0)

    truth = {"h1": 0.6, "h2": 0.35, "rg": 0.55, "re": 0.25}
    genetic = np.array(
        [
            [truth["h1"], truth["rg"] * np.sqrt(truth["h1"] * truth["h2"])],
            [truth["rg"] * np.sqrt(truth["h1"] * truth["h2"]), truth["h2"]],
        ]
    )
    residual = np.array(
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
    full = np.kron(genetic, relationship) + np.kron(residual, np.eye(n))
    draw = np.linalg.cholesky(full + 1e-10 * np.eye(2 * n)) @ np.random.default_rng(
        seed
    ).standard_normal(2 * n)

    # Unbalanced, because that is the case eigen-simplification cannot take.
    observed = np.array([[True, person % 8 != 0] for person in range(n)])
    mask = [[bool(observed[p, 0]), bool(observed[p, 1])] for p in range(n)]

    # The two implementations want the same data laid out differently. The
    # reference takes the full 2n rows with `observed` saying which are real;
    # Asterism takes only the real ones. Getting this backwards is not an error
    # either of them raises — it is a different data set quietly fitted — so the
    # two layouts are built from one draw here rather than separately.
    full_values = np.zeros(2 * n)
    full_values[0::2] = draw[:n]
    full_values[1::2] = draw[n:]
    full_design = np.zeros((2 * n, 2))
    full_design[0::2, 0] = 1.0
    full_design[1::2, 1] = 1.0
    rows = [p * 2 + t for p in range(n) for t in (0, 1) if observed[p, t]]
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

    theirs_fit = reference.fit(relationship, full_values, observed, full_design, True)
    theta, _, gradient, converged = _core.bivariate_fit(
        relationship, mask, design, values, True
    )
    ours_point = {
        "h2_first": theta[2],
        "h2_second": theta[3],
        "rho_g": theta[4],
        "rho_e": theta[5],
    }

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

    failures = []
    if not converged:
        failures.append("Asterism did not declare convergence")
    if not gradient < 1e-7:
        failures.append(f"Asterism scaled projected gradient was {gradient:.3e}")

    recorded, started = {}, time.perf_counter()
    for name, index, reference_key in QUANTITIES:
        ours = _core.bivariate_interval(
            relationship, mask, design, values, name, True
        )
        theirs = reference.profile_interval(
            relationship,
            full_values,
            observed,
            full_design,
            index,
            theirs_fit[reference_key],
            True,
        )
        low_gap = abs(ours[0] - theirs["lower"])
        high_gap = abs(ours[1] - theirs["upper"])
        recorded[name] = {
            "asterism": {"lower": ours[0], "upper": ours[1]},
            "reference": {"lower": theirs["lower"], "upper": theirs["upper"]},
            "lower_gap": low_gap,
            "upper_gap": high_gap,
        }
        print(
            f"{name:<12}[{ours[0]:>9.5f},{ours[1]:>9.5f}]"
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

    print(f"\nBoth routes agree on every endpoint to {TOLERANCE:.0e}, in "
          f"{time.perf_counter() - started:.0f}s.")
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
