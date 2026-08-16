"""Compare Asterism's REML fit against R's `regress`.

R `regress` independently implements the same published REML algebra with a
different optimiser. This check compares estimates and the identifiable
log-likelihood offset between the two implementations.

Run with:

    uv run --no-project python checks/against_r.py

It needs R with the `regress` package installed, and it fails rather than skips
when R is missing: a check that skips is a check that passes.
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

# The estimates are optimiser-bounded, so the tolerance is 1e-6 relative. The
# log-likelihood is compared as a difference rather than a level for the reason
# given below.
ESTIMATE_TOLERANCE = 1e-6
LOGLIK_DIFFERENCE_TOLERANCE = 1e-8

R_SCRIPT = """
suppressMessages(library(regress))
args <- commandArgs(trailingOnly = TRUE)
k <- as.matrix(read.csv(file.path(args[1], "k.csv"), header = FALSE))
d <- read.csv(file.path(args[1], "data.csv"))
covariates <- setdiff(names(d), "y")
formula <- if (length(covariates) == 0) y ~ 1 else
    as.formula(paste("y ~", paste(covariates, collapse = " + ")))
fit <- regress(formula, ~k, identity = TRUE, data = d, tol = 1e-10)
cat(toJSON <- sprintf(
    '{"sigma_k": %.17g, "sigma_e": %.17g, "llik": %.17g, "beta": [%s]}',
    fit$sigma[["k"]], fit$sigma[["In"]], fit$llik,
    paste(sprintf("%.17g", fit$beta), collapse = ",")
))
"""


def extended_family() -> list[tuple[int | None, int | None]]:
    """The same three-generation family the coverage check uses."""
    family: list[tuple[int | None, int | None]] = [(None, None), (None, None)]
    family += [(0, 1)] * 3
    family += [(None, None)] * 3
    for child in range(3):
        family += [(2 + child, 5 + child)] * 2
    return family


def kinship(people) -> np.ndarray:
    n = len(people)
    phi = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            mother, father = people[i]
            if i == j:
                value = 0.5 if mother is None else 0.5 * (1.0 + phi[mother, father])
            else:
                value = 0.0 if mother is None else 0.5 * (phi[mother, j] + phi[father, j])
            phi[i, j] = phi[j, i] = value
    return phi


def roster(families: int) -> np.ndarray:
    family = extended_family()
    size = len(family)
    phi = kinship(family)
    k = np.zeros((families * size, families * size))
    for f in range(families):
        o = f * size
        k[o : o + size, o : o + size] = 2.0 * phi
    return k


def design(n: int, block: int) -> np.ndarray:
    """Intercept, age, age squared, sex, and the two age-by-sex products —
    `age_years^1,2#sex`, which is what 1,194 of the lab's 1,246 recorded SOLAR
    runs use. An intercept-only comparison would leave the restricted
    likelihood's determinant term unexercised, which is the term most likely to
    differ between two implementations."""
    x = np.zeros((n, 6))
    for row in range(n):
        within = row % block
        decade = 7.0 if within < 2 else (4.5 if within < 8 else 2.0)
        age = (decade - 4.5) / 2.0 + ((row % 7) - 3.0) / 10.0
        sex = float(row % 2 == 0)
        x[row] = [1.0, age, age * age, sex, age * sex, age * age * sex]
    return x


def simulate(k: np.ndarray, x: np.ndarray, beta: np.ndarray, h2: float, seed: int):
    rng = np.random.default_rng(seed)
    n = k.shape[0]
    covariance = h2 * k + (1.0 - h2) * np.eye(n)
    factor = np.linalg.cholesky(covariance)
    return x @ beta + factor @ rng.standard_normal(n)


def fit_in_r(directory: Path, k: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    np.savetxt(directory / "k.csv", k, delimiter=",")
    header = ",".join(["y"] + [f"c{i}" for i in range(1, x.shape[1])])
    columns = np.column_stack([y] + [x[:, i] for i in range(1, x.shape[1])])
    np.savetxt(directory / "data.csv", columns, delimiter=",", header=header, comments="")
    script = directory / "fit.R"
    script.write_text(R_SCRIPT)
    finished = subprocess.run(
        ["Rscript", "--vanilla", str(script), str(directory)],
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        raise SystemExit(f"R failed:\n{finished.stdout}\n{finished.stderr}")
    return json.loads(finished.stdout.strip().splitlines()[-1])


def relative(a: float, b: float) -> float:
    return abs(a - b) / (1.0 + abs(b))


def main() -> int:
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on the path. This check fails rather than skips.")

    families, block = 25, len(extended_family())
    k = roster(families)
    n = k.shape[0]
    x = design(n, block)
    beta = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])
    model = asterism.prepare(x, k)

    cases = [(0.5, 101), (0.25, 102), (0.75, 103)]
    results = []
    print(f"Asterism against R `regress`, REML, n = {n}, {x.shape[1]} fixed effects.\n")
    print(f"{'truth':>6} {'asterism h2':>12} {'regress h2':>12} {'rel':>10} "
          f"{'var rel':>10} {'worst beta rel':>15} {'loglik gap':>13}")

    for h2, seed in cases:
        y = simulate(k, x, beta, h2, seed)
        ours = model.fit(y, "reml")
        with tempfile.TemporaryDirectory() as temporary_directory:
            theirs = fit_in_r(Path(temporary_directory), k, x, y)

        their_total = theirs["sigma_k"] + theirs["sigma_e"]
        their_h2 = theirs["sigma_k"] / their_total
        beta_rel = max(
            relative(a, b) for a, b in zip(ours["beta"], theirs["beta"], strict=True)
        )
        gap = ours["loglik"] - theirs["llik"]
        results.append(
            {
                "true_h2": h2,
                "seed": seed,
                "asterism": {
                    "h2": ours["h2"],
                    "total_variance": ours["total_variance"],
                    "loglik": ours["loglik"],
                    "beta": ours["beta"],
                },
                "regress": {
                    "h2": their_h2,
                    "total_variance": their_total,
                    "llik": theirs["llik"],
                    "beta": theirs["beta"],
                },
                "h2_relative": relative(ours["h2"], their_h2),
                "total_variance_relative": relative(ours["total_variance"], their_total),
                "worst_beta_relative": beta_rel,
                "loglik_gap": gap,
            }
        )
        print(
            f"{h2:>6.2f} {ours['h2']:>12.9f} {their_h2:>12.9f} "
            f"{relative(ours['h2'], their_h2):>10.2e} "
            f"{relative(ours['total_variance'], their_total):>10.2e} "
            f"{beta_rel:>15.2e} {gap:>13.9f}"
        )

    # The two log-likelihoods use different constants: `regress` drops terms
    # Asterism keeps. A level comparison would therefore mean nothing, but the
    # *offset* must be identical across datasets — if it drifts, the two are not
    # computing the same function of the data.
    gaps = [r["loglik_gap"] for r in results]
    drift = max(gaps) - min(gaps)
    # Better than constant: the offset is the constant we can name. `regress`
    # omits the 2*pi term from the restricted likelihood and Asterism keeps it,
    # which is exactly (n - p)/2 * log(2*pi). If the measured offset matches
    # that, the two are computing the same function and the difference is
    # bookkeeping. If it were merely stable but unrecognisable, something else
    # would be going on and it would need finding.
    expected = 0.5 * (n - x.shape[1]) * np.log(2.0 * np.pi)
    unexplained = abs(gaps[0] + expected)
    print(f"\nlog-likelihood offset {gaps[0]:.9f}, drift across cases {drift:.3e}")
    print(f"expected offset -(n - p)/2 * log(2*pi) = {-expected:.9f}, "
          f"unexplained {unexplained:.3e}")

    failures = []
    for r in results:
        for name, value, tolerance in (
            ("h2", r["h2_relative"], ESTIMATE_TOLERANCE),
            ("total variance", r["total_variance_relative"], ESTIMATE_TOLERANCE),
            ("beta", r["worst_beta_relative"], ESTIMATE_TOLERANCE),
        ):
            if not value < tolerance:
                failures.append(f"true h2 {r['true_h2']}: {name} differs by {value:.3e}")
    if not drift < LOGLIK_DIFFERENCE_TOLERANCE:
        failures.append(f"log-likelihood offset drifts by {drift:.3e}")
    if not unexplained < 1e-6:
        failures.append(
            f"the offset is stable but not the expected 2*pi term; "
            f"{unexplained:.3e} is unaccounted for"
        )

    print(
        json.dumps(
            {
                "results": results,
                "loglik_offset": gaps[0],
                "loglik_offset_drift": drift,
                "loglik_offset_expected": -expected,
                "loglik_offset_unexplained": unexplained,
            },
            indent=2,
        )
    )

    if failures:
        print("\nDISAGREEMENT:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nAgreement within tolerance on every quantity.")
    print("The independent REML implementations agree on this comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
