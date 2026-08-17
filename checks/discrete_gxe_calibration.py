"""Do the discrete gene-by-environment tests hold their level, and can they
find anything?

Five nulls are tested against one alternative here, and most of them can be
made to reject by something that is not genetic at all. That is what this
checks. The environment is sex, which is the canonical binary environment and
the one whose unbalanced reality the GOBS pedigree supplies.

**The scenario that matters is `noisier`.** One sex measured with more error,
identical genetics, nothing to find. A model holding a single residual variance
would push that extra error into the genetic term and report a genetic
difference. This model holds two, and the claim is that it therefore does not.
A claim of that shape is worth nothing until it has been simulated against.

`any_difference` is expected to reject there and is judged as power. It ties
the two residual variances, so a noisier sex is a real departure from its null.
It is in the table to show plainly that it is not a genetic test, which is why
`gene_by_environment` exists beside it.

**On the real pedigree, with the real sexes.** GOBS is not sex-balanced and its
families are not of one shape, so a balanced simulation would say nothing about
the correlation across sexes, which rests entirely on opposite-sex pairs.

**The criterion is validity, not uniformity.** The correlation is bounded above
at one, so under the null a share of fits sit on that bound and the deviance has
an atom at nought. What must hold is that the rejection rate does not exceed its
level.

Run with:

    ASTERISM_REPLICATES=400 uv run --no-project python checks/discrete_gxe_calibration.py

It reads the GOBS pedigree, which is study material, and simulates every
response it uses. No observed phenotype is read.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import asterism
import numpy as np

DATABASE = Path("~/MathiasLab/staging/studies/existing/safs/data/SAFS.db").expanduser()
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "400"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "6"))
LEVELS = (0.01, 0.05, 0.10)
NULLS = ("gene_by_environment", "any_difference", "correlation", "genetic", "residual")

# The null: one genetic standard deviation, one residual, same genes in both
# sexes. A heritability of 0.5, near what the real GOBS traits give.
GENETIC = (0.707, 0.707)
RESIDUAL = (0.707, 0.707)
CORRELATION = 1.0

SCENARIOS = {
    # Nothing to find. Every test must hold its level.
    "null": (GENETIC, RESIDUAL, 1.0),
    # The genes differ across the sexes: the genetic finding proper.
    "different genes": (GENETIC, RESIDUAL, 0.4),
    # The genetic variance differs: a difference of scale, not of genes.
    "different scale": ((0.95, 0.45), RESIDUAL, 1.0),
    # One sex measured twice as noisily, identical genetics. The genetic tests
    # must not reject. This is the scenario the two free residual variances
    # exist for.
    "noisier": (GENETIC, (0.5, 1.0), 1.0),
}

STATE: dict = {}


def _start(payload: dict) -> None:
    STATE.update(payload)


def structure():
    """The real GOBS pedigree, its real sexes, and an age design."""
    db = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    pedigree = list(db.execute("select id, fa, mo, sex from pedigree"))
    ids = [str(p) for p, _, _, _ in pedigree]
    fathers = [None if str(f) in ("0", "", "None") else str(f) for _, f, _, _ in pedigree]
    mothers = [None if str(m) in ("0", "", "None") else str(m) for _, _, m, _ in pedigree]
    # Always the pedigree sex.
    sexes = {str(p): str(s) for p, _, _, s in pedigree}
    ages = {}
    for person, age in db.execute(
        "select subject_id, gobs_age_at_consent_years from gobs_demographics"
    ):
        try:
            value = float(str(age).strip())
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and str(person) in sexes:
            ages[str(person)] = value

    keep = sorted(p for p in ages if sexes.get(p) in ("1", "2"))
    relationship, order = asterism.relationship_matrix(ids, fathers, mothers, keep=keep)
    assert order == keep
    group = np.array([1.0 if sexes[p] == "1" else 2.0 for p in order])
    age = np.array([ages[p] for p in order])
    age = (age - age.mean()) / age.std()
    # Sex is in the model as a variance, and it belongs in the mean as well: a
    # mean difference left in the residual is not what any of these tests are
    # about.
    male = (group == 1.0).astype(float)
    design = np.column_stack([np.ones(len(order)), age, age**2, male])
    return np.ascontiguousarray(relationship), group, design


def covariance(relationship, group, genetic, residual, correlation):
    """The simulating covariance, built the way the model reads it."""
    n = len(group)
    first = group == 1.0
    sd = np.where(first, genetic[0], genetic[1])
    across = np.where(first[:, None] == first[None, :], 1.0, correlation)
    v = relationship * np.outer(sd, sd) * across
    v[np.diag_indices(n)] += np.where(first, residual[0], residual[1]) ** 2
    return v


def one(job):
    scenario, index = job
    relationship = STATE["relationship"]
    group = STATE["group"]
    design = STATE["design"]
    factor = STATE["factors"][scenario]
    n = len(group)
    y = factor @ np.random.default_rng(910_000 + index).standard_normal(n)

    model = asterism.DiscreteGxeModel(relationship, group, design)
    out = {"scenario": scenario, "p": {}}
    try:
        fit = model.fit(y)
    except ValueError:
        fit = None
    out["correlation"] = None if fit is None else fit["genetic_correlation"]
    out["converged"] = False if fit is None else fit["converged"]
    for null in NULLS:
        try:
            outcome = model.test(y, null)
        except ValueError:
            outcome = None
        out["p"][null] = None if outcome is None else outcome["p_value"]
    return out


def main() -> int:
    relationship, group, design = structure()
    n = len(group)
    counts = [int((group == 1.0).sum()), int((group == 2.0).sum())]

    factors = {}
    for name, (genetic, residual, correlation) in SCENARIOS.items():
        v = covariance(relationship, group, genetic, residual, correlation)
        factors[name] = np.linalg.cholesky(v + 1e-9 * np.eye(n))

    print(
        f"Discrete gene-by-environment tests on the real GOBS pedigree, with\n"
        f"sex as the environment, REML. {n} people, "
        f"{counts[0]} of sex 1 and {counts[1]} of sex 2.\n"
        f"{REPLICATES} replicates per scenario; only the response is simulated.\n"
        f"Scenarios: " + ", ".join(SCENARIOS) + ".\n"
    )

    shared = {
        "relationship": relationship,
        "group": group,
        "design": design,
        "factors": factors,
    }
    jobs = [(s, i) for s in SCENARIOS for i in range(REPLICATES)]
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS, initializer=_start, initargs=(shared,)) as pool:
        results = list(pool.map(one, jobs, chunksize=1))
    print(
        f"{len(results)} fits in {(time.perf_counter() - started) / 60:.0f} minutes.\n"
    )

    failures: list[str] = []
    recorded: dict = {}
    for scenario in SCENARIOS:
        got = [r for r in results if r["scenario"] == scenario]
        is_null = scenario == "null"
        # A scenario where a test has nothing to find is judged as a level even
        # when the scenario as a whole is an alternative. That is the whole
        # point of `noisier`.
        genetic, residual, correlation = SCENARIOS[scenario]
        nothing_to_find = {
            # The headline test leaves the residuals free, so a noisier sex
            # gives it nothing to find and it is judged as a level there. That
            # is the single most important row in this table.
            "gene_by_environment": genetic[0] == genetic[1] and correlation >= 1.0,
            # This one ties the residuals, so only the pure null leaves it
            # nothing to find.
            "any_difference": is_null,
            "correlation": correlation >= 1.0,
            "genetic": genetic[0] == genetic[1],
            "residual": residual[0] == residual[1],
        }
        print(f"{scenario}")
        print(
            f"  {'test':<16}" + "".join(f"{f'p<={l:g}':>10}" for l in LEVELS)
            + f"{'of':>6}  what it is"
        )
        recorded[scenario] = {}
        for null in NULLS:
            have = [r["p"][null] for r in got if r["p"][null] is not None]
            if not have:
                failures.append(f"{scenario}: nothing computed for {null}")
                continue
            rates = [float(np.mean([v <= level for v in have])) for level in LEVELS]
            level_check = nothing_to_find[null]
            recorded[scenario][null] = {
                "computed": len(have),
                "judged_as": "level" if level_check else "power",
                "rejection": {str(l): r for l, r in zip(LEVELS, rates, strict=True)},
            }
            marks = []
            for level, rate in zip(LEVELS, rates, strict=True):
                if not level_check:
                    marks.append(f"{rate:>10.3f}")
                    continue
                error = np.sqrt(level * (1 - level) / len(have))
                ceiling = level + 1.96 * error
                marks.append(f"{rate:>9.3f}{'!' if rate > ceiling else ' '}")
                if rate > ceiling:
                    failures.append(
                        f"{scenario}/{null} rejects {rate:.3f} at {level:g}, "
                        f"above {ceiling:.3f}"
                    )
            print(
                f"  {null:<16}" + "".join(marks) + f"{len(have):>6}"
                f"  {'level' if level_check else 'power'}"
            )
        # How often the correlation sat on its bound. If that is never, the even
        # mixture is the wrong reference and the correlation test is
        # conservative for a reason nobody would see in the p-values alone.
        bound = [
            r["correlation"] for r in got if r["correlation"] is not None
        ]
        on = float(np.mean([c > 1.0 - 1e-6 for c in bound])) if bound else float("nan")
        recorded[scenario]["correlation_at_bound"] = on
        recorded[scenario]["median_correlation"] = (
            float(np.median(bound)) if bound else None
        )
        print(
            f"  correlation: median {np.median(bound):.3f}, "
            f"on the bound in {on:.1%} of fits\n"
        )

    print("A `!` marks a rejection rate above its level's one-sided binomial ceiling.")
    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        "\nEvery test holds its level wherever it has nothing to find, including\n"
        "the genetic tests under a sex difference in measurement error alone."
    )

    print(
        json.dumps(
            {
                "estimator": "reml",
                "pedigree": "the real GOBS pedigree; only responses simulated",
                "people": n,
                "counts": counts,
                "replicates": REPLICATES,
                "levels": list(LEVELS),
                "scenarios": {
                    name: {
                        "genetic_sd": list(g),
                        "residual_sd": list(r),
                        "correlation": c,
                    }
                    for name, (g, r, c) in SCENARIOS.items()
                },
                "results": recorded,
                "note": (
                    "a test is judged as a level wherever the scenario gives it "
                    "nothing to find, which is why the genetic tests under "
                    "'noisier' are level checks and not power ones"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
