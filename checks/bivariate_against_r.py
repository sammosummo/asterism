"""The two-trait REML fit against R's `regress`.

This compares Asterism's two-trait REML fit with an equivalent construction in
R `regress`. SOLAR covers the separate ML comparison because its `polygenic`
command does not fit REML.

**`regress` has no bivariate response, and does not need one.** The two-trait
model is linear in six variance components — three for the genetic covariance
and three for the residual — so handing `regress` six structure matrices over
the stacked response fits exactly the same model:

    V = a₁₁·A⊗e₁₁ + a₂₂·A⊗e₂₂ + a₁₂·A⊗e₁₂ + r₁₁·I⊗e₁₁ + r₂₂·I⊗e₂₂ + r₁₂·I⊗e₁₂

That makes it a genuinely independent route rather than a second spelling of the
same one: it carries covariances where Asterism carries correlations, its
constraint set is different, and it uses its own optimiser. Where the two agree,
the agreement means something.

Run with:

    uv run --no-project python checks/bivariate_against_r.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from asterism import _core

# `regress` reports variance components; the comparison is on the quantities
# anybody reports, which are the ratios.
# The two routes agree to about 2e-8 on this problem, so a tolerance of 1e-4
# would pass through a regression four orders of magnitude larger than
# anything ever seen here. This is set from the measured agreement with room
# for optimiser drift, not from what looks safe.
TOLERANCE = 1e-6

R_SCRIPT = """
suppressMessages(library(regress))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(file.path(args[1], "data.csv"))
n <- nrow(d)
load(file.path(args[1], "structures.RData"))
# Start from something feasible. `regress` otherwise begins with equal weight
# on every component, and the two cross-trait matrices have zero diagonals and
# are indefinite alone, so that starting point is singular.
start <- c(0.5, 0.5, 0.1, 0.5, 0.5, 0.1)
fit <- regress(y ~ 0 + t1 + t2, ~ Va11 + Va22 + Va12 + Vr11 + Vr22 + Vr12,
               identity = FALSE, data = d, start = start,
               tol = 1e-10, maxcyc = 500)
s <- fit$sigma
cat(sprintf('{"a11": %.17g, "a22": %.17g, "a12": %.17g, "r11": %.17g, "r22": %.17g, "r12": %.17g, "llik": %.17g}',
            s[["Va11"]], s[["Va22"]], s[["Va12"]], s[["Vr11"]], s[["Vr22"]], s[["Vr12"]], fit$llik))
"""


def structures(relationship: np.ndarray, observed: np.ndarray):
    """The six matrices, over the observed rows only."""
    rows = [(p, t) for p in range(len(observed)) for t in (0, 1) if observed[p, t]]
    size = len(rows)
    people = np.array([p for p, _ in rows])
    traits = np.array([t for _, t in rows])
    a = relationship[np.ix_(people, people)]
    same_person = people[:, None] == people[None, :]

    first = (traits[:, None] == 0) & (traits[None, :] == 0)
    second = (traits[:, None] == 1) & (traits[None, :] == 1)
    cross = traits[:, None] != traits[None, :]

    return rows, {
        "Va11": a * first,
        "Va22": a * second,
        "Va12": a * cross,
        "Vr11": np.eye(size) * first,
        "Vr22": np.eye(size) * second,
        "Vr12": same_person * cross * 1.0,
    }


def fit_in_r(directory: Path, matrices: dict, y: np.ndarray, traits: np.ndarray) -> dict:
    lines = ["y,t1,t2"]
    for value, t in zip(y, traits, strict=True):
        lines.append(f"{value:.17g},{1 if t == 0 else 0},{1 if t == 1 else 0}")
    (directory / "data.csv").write_text("\n".join(lines) + "\n")

    # Write the matrices where R can load them without a text round trip.
    save = ["setwd(commandArgs(trailingOnly = TRUE)[1])"]
    for name, matrix in matrices.items():
        np.savetxt(directory / f"{name}.txt", matrix)
        save.append(f'{name} <- as.matrix(read.table("{name}.txt"))')
        save.append(f'dimnames({name}) <- NULL')
    save.append('save(' + ", ".join(matrices) + ', file = "structures.RData")')
    (directory / "save.R").write_text("\n".join(save) + "\n")
    subprocess.run(
        ["Rscript", "--vanilla", str(directory / "save.R"), str(directory)],
        capture_output=True, text=True, check=True,
    )

    (directory / "fit.R").write_text(R_SCRIPT)
    finished = subprocess.run(
        ["Rscript", "--vanilla", str(directory / "fit.R"), str(directory)],
        capture_output=True, text=True, check=False,
    )
    if finished.returncode != 0:
        raise SystemExit(f"R failed:\n{finished.stdout}\n{finished.stderr}")
    return json.loads(finished.stdout.strip().splitlines()[-1])


def main() -> int:
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on the path. This check fails rather than skips.")

    families, per_family = 60, 6
    n = families * per_family
    relationship = np.zeros((n, n))
    for f in range(families):
        block = slice(f * per_family, (f + 1) * per_family)
        relationship[block, block] = 0.5
    np.fill_diagonal(relationship, 1.0)

    truth = {"h1": 0.6, "h2": 0.35, "rg": 0.55, "re": 0.25}
    sa = np.array([[truth["h1"], truth["rg"] * np.sqrt(truth["h1"] * truth["h2"])],
                   [truth["rg"] * np.sqrt(truth["h1"] * truth["h2"]), truth["h2"]]])
    se = np.array([[1 - truth["h1"], truth["re"] * np.sqrt((1 - truth["h1"]) * (1 - truth["h2"]))],
                   [truth["re"] * np.sqrt((1 - truth["h1"]) * (1 - truth["h2"])), 1 - truth["h2"]]])
    rng = np.random.default_rng(31)
    full = np.kron(sa, relationship) + np.kron(se, np.eye(n))
    draw = np.linalg.cholesky(full + 1e-10 * np.eye(2 * n)) @ rng.standard_normal(2 * n)

    # Unbalanced: every ninth person lacks the second trait.
    observed = np.array([[True, i % 9 != 0] for i in range(n)])
    y_full = np.zeros(2 * n)
    y_full[0::2] = draw[:n]
    y_full[1::2] = draw[n:]
    design_full = np.zeros((2 * n, 2))
    design_full[0::2, 0] = 1.0
    design_full[1::2, 1] = 1.0

    rows, matrices = structures(relationship, observed)
    index = [p * 2 + t for p, t in rows]
    theta, ours_loglik, scaled_gradient, converged = _core.bivariate_fit(
        np.ascontiguousarray(relationship),
        observed.tolist(),
        np.ascontiguousarray(design_full[index]),
        np.ascontiguousarray(y_full[index]),
        True,
    )
    ours = {
        "total_variance_a": theta[0],
        "total_variance_b": theta[1],
        "h2_trait_a": theta[2],
        "h2_trait_b": theta[3],
        "rho_g": theta[4],
        "rho_e": theta[5],
        "loglik": ours_loglik,
        "scaled_gradient": scaled_gradient,
        "converged": converged,
    }
    with tempfile.TemporaryDirectory() as temporary:
        theirs = fit_in_r(
            Path(temporary), matrices, y_full[index], np.array([t for _, t in rows])
        )

    # `regress` gives covariances; the ratios are what anybody reports.
    their_h1 = theirs["a11"] / (theirs["a11"] + theirs["r11"])
    their_h2 = theirs["a22"] / (theirs["a22"] + theirs["r22"])
    their_rg = theirs["a12"] / np.sqrt(theirs["a11"] * theirs["a22"])
    their_re = theirs["r12"] / np.sqrt(theirs["r11"] * theirs["r22"])

    print(f"Two traits, REML, n = {n}, "
          f"{observed[:, 0].sum()} with the first and {observed[:, 1].sum()} with the second.\n")
    print(f"{'':<10} {'asterism':>12} {'R regress':>12} {'difference':>12} {'truth':>8}")
    failures = []
    if not ours["converged"]:
        failures.append("Asterism did not declare convergence")
    if not ours["scaled_gradient"] < 1e-7:
        failures.append(
            f"Asterism scaled projected gradient was {ours['scaled_gradient']:.3e}"
        )
    comparisons = (
        ("h2 first", ours["h2_trait_a"], their_h1, truth["h1"]),
        ("h2 second", ours["h2_trait_b"], their_h2, truth["h2"]),
        ("rho_g", ours["rho_g"], their_rg, truth["rg"]),
        ("rho_e", ours["rho_e"], their_re, truth["re"]),
    )
    differences = {}
    for label, ours_value, theirs_value, true_value in comparisons:
        difference = abs(ours_value - theirs_value)
        differences[label] = difference
        print(f"{label:<10} {ours_value:>12.7f} {theirs_value:>12.7f} "
              f"{difference:>12.2e} {true_value:>8.2f}")
        if not difference < TOLERANCE:
            failures.append(f"{label} differs by {difference:.3e}")

    if failures:
        print("\nDISAGREEMENT:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nAgreement within tolerance on every reported quantity.")
    print(f"Asterism scaled projected gradient: {ours['scaled_gradient']:.3e}.")
    print("Different parameterisation, different optimiser, same answer.")
    print("The independently implemented REML calculations agree.")
    print(
        json.dumps(
            {
                "seed": 31,
                "families": families,
                "people_per_family": per_family,
                "people": n,
                "observations_by_trait": [
                    int(observed[:, 0].sum()),
                    int(observed[:, 1].sum()),
                ],
                "estimator": "reml",
                "truth": truth,
                "asterism": ours,
                "regress": {
                    **theirs,
                    "h2_trait_a": their_h1,
                    "h2_trait_b": their_h2,
                    "rho_g": their_rg,
                    "rho_e": their_re,
                },
                "absolute_differences": differences,
                "comparison_tolerance": TOLERANCE,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
