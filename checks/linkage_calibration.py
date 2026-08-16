"""Does a linkage scan built on the local-IBD matrix hold its level, and what
does an uncertain IBD matrix cost?

The matrix builders arrived with equation checks — the matrices are the right
matrices — but with nothing showing that the model they exist for works. This
is that check. A linkage scan fits

    y = X beta + local + polygenic + e

with `local` scaling `local_ibd_matrix` at one position and `polygenic` scaling
the genome-wide additive matrix, and tests the local component against nought.

**The two matrices are separable because the local one averages to the
polygenic one.** Among full sibs, IBD at a locus is 0, 0.5 or 1 in the ratio
1:2:1, whose mean is the polygenic 0.5. Linkage information is the *variation*
around that mean and nothing else, which is why the spread of the local matrix
is the quantity to look at before running a scan.

**Real IBD is estimated, not known**, so the scan uses a posterior mean. That
shrinks the local matrix towards the polygenic one, taking the variation with
it. This check measures what that costs. It is a loss of power and not of
validity: a shrunken matrix cannot manufacture a signal, it can only fail to
find one, and it does so silently.

Run with:

    ASTERISM_REPLICATES=400 uv run --no-project python checks/linkage_calibration.py

Everything here is simulated; no study material is read.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import asterism

REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "200"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "8"))
FAMILIES, SIBS, DRAWS = 120, 4, 20
LEVELS = (0.01, 0.05, 0.10)
# Among full sibs IBD is 0, 0.5, 1 in the ratio 1:2:1, so its standard
# deviation is sqrt(0.125). A posterior mean that has lost this spread has lost
# the linkage information with it.
FULL_INFORMATION_SPREAD = 0.125**0.5


def sibships(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Founder lineages and family labels. Each child draws one allele a side."""
    rng = np.random.default_rng(seed)
    lineages, family = [], []
    for f in range(FAMILIES):
        paternal, maternal = (4 * f + 1, 4 * f + 2), (4 * f + 3, 4 * f + 4)
        for _ in range(SIBS):
            lineages.append([paternal[rng.integers(2)], maternal[rng.integers(2)]])
            family.append(f)
    return np.array(lineages), np.array(family)


def polygenic(family: np.ndarray) -> np.ndarray:
    """Twice the kinship: one on the diagonal, a half between full sibs."""
    same = family[:, None] == family[None, :]
    return np.where(same, 0.5, 0.0) + 0.5 * np.eye(len(family))


def posterior_matrix(
    lineages: np.ndarray, family: np.ndarray, resolution: float, seed: int
) -> np.ndarray:
    """A posterior mean IBD matrix.

    `resolution` is the chance that a person's lineage is pinned down by the
    markers. The rest is redrawn, which is what finite marker information looks
    like once it is averaged over draws.
    """
    rng = np.random.default_rng(seed)
    stack = np.empty((DRAWS, *lineages.shape), dtype=lineages.dtype)
    for draw in range(DRAWS):
        proposed = lineages.copy()
        for person in range(len(family)):
            if rng.random() > resolution:
                f = family[person]
                proposed[person] = [
                    4 * f + 1 + rng.integers(2),
                    4 * f + 3 + rng.integers(2),
                ]
        stack[draw] = proposed
    return asterism.posterior_local_ibd_matrix(stack)


def spread(matrix: np.ndarray, family: np.ndarray) -> float:
    """The standard deviation of local IBD among relatives."""
    same = np.triu(family[:, None] == family[None, :], 1)
    return float(matrix[same].std())


def one(job: tuple[int, float, float]) -> float | None:
    replicate, share, resolution = job
    lineages, family = sibships(10_000 + replicate)
    truth = asterism.local_ibd_matrix(lineages)
    additive = polygenic(family)
    people = len(family)
    covariance = share * truth + 0.4 * additive + (0.6 - share) * np.eye(people)
    factor = np.linalg.cholesky(covariance + 1e-10 * np.eye(people))
    y = factor @ np.random.default_rng(88_000 + replicate).standard_normal(people)
    used = (
        truth
        if resolution >= 1.0
        else posterior_matrix(lineages, family, resolution, 5_000 + replicate)
    )
    try:
        model = asterism.ComponentModel([used, additive], np.ones((people, 1)))
        return float(model.test(y, component=0)["p_value"])
    except Exception:
        return None


SCENARIOS = (
    # (locus share of variance, IBD resolution, judged as)
    (0.00, 1.0, "level"),
    (0.00, 0.6, "level"),
    (0.00, 0.3, "level"),
    (0.25, 1.0, "power"),
    (0.25, 0.6, "power"),
    (0.25, 0.3, "power"),
)


def main() -> int:
    lineages, family = sibships(1)
    people = len(family)
    truth = asterism.local_ibd_matrix(lineages)
    same = np.triu(family[:, None] == family[None, :], 1)
    states = {v: int((truth[same] == v).sum()) for v in (0.0, 0.5, 1.0)}

    print(
        f"Linkage on {people} people in {FAMILIES} sibships of {SIBS}.\n"
        f"{REPLICATES} replicates per scenario; everything simulated.\n"
        f"IBD states among sibs: {states}, mean {truth[same].mean():.4f} "
        f"against the polygenic 0.5.\n"
    )

    failures: list[str] = []
    recorded: dict = {}
    print(
        f"  {'locus':>7}{'IBD':>7}{'spread':>9}{'of full':>9}"
        + "".join(f"{f'p<={lv:g}':>9}" for lv in LEVELS)
        + "  what it is"
    )
    for share, resolution, judged in SCENARIOS:
        with ProcessPoolExecutor(WORKERS) as pool:
            values = [
                v
                for v in pool.map(
                    one,
                    [(r, share, resolution) for r in range(REPLICATES)],
                    chunksize=1,
                )
                if v is not None
            ]
        if not values:
            failures.append(f"share {share}, resolution {resolution}: nothing computed")
            continue
        values = np.array(values)
        rates = {lv: float((values <= lv).mean()) for lv in LEVELS}
        used = (
            truth
            if resolution >= 1.0
            else posterior_matrix(lineages, family, resolution, 5_001)
        )
        informativeness = spread(used, family) / FULL_INFORMATION_SPREAD
        key = f"share {share}, resolution {resolution}"
        recorded[key] = {
            "judged_as": judged,
            "computed": len(values),
            "informativeness": informativeness,
            "rejection": {str(lv): rates[lv] for lv in LEVELS},
        }
        marks = []
        for lv in LEVELS:
            if judged == "power":
                marks.append(f"{rates[lv]:>9.3f}")
                continue
            ceiling = lv + 1.96 * (lv * (1 - lv) / len(values)) ** 0.5
            marks.append(f"{rates[lv]:>8.3f}{'!' if rates[lv] > ceiling else ' '}")
            if rates[lv] > ceiling:
                failures.append(
                    f"{key} rejects {rates[lv]:.3f} at {lv:g}, above {ceiling:.3f}"
                )
        print(
            f"  {share:>7.2f}{resolution:>7.1f}{spread(used, family):>9.3f}"
            f"{informativeness:>9.2f}" + "".join(marks) + f"  {judged}"
        )

    print(
        "\nA `!` marks a rejection rate above its level's one-sided binomial "
        "ceiling.\n"
        "`spread` is the standard deviation of local IBD among relatives and\n"
        f"`of full` divides it by {FULL_INFORMATION_SPREAD:.3f}, its value when "
        "IBD is known\nexactly. That ratio is the quantity to compute on a real "
        "IBD matrix before\nrunning a scan: it falls with marker information, "
        "and power falls with it."
    )
    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        "\nThe test holds its level whether IBD is known or estimated, and an\n"
        "estimated matrix costs power rather than validity."
    )
    print(
        json.dumps(
            {
                "what": "level and power of a variance-component linkage scan",
                "people": people,
                "families": FAMILIES,
                "sibs_per_family": SIBS,
                "replicates": REPLICATES,
                "posterior_draws": DRAWS,
                "ibd_states_among_sibs": {str(k): v for k, v in states.items()},
                "full_information_spread": FULL_INFORMATION_SPREAD,
                "levels": list(LEVELS),
                "results": recorded,
                "note": (
                    "resolution is the chance a person's lineage is pinned down "
                    "by the markers; the remainder is averaged over draws"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
