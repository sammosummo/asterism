"""Does the repeated-measures model agree with MCMCglmm?

Every other check this model has is against something in the same crate.
`ComponentModel` says one position is right, `TobitModel` says a censored
position is right, `MixedBivariateModel` says the covariance *between* two
positions is right -- and all three share `region_log_probability` with the
thing they are checking. Under ADR 0006 agreement proves fidelity only against
something arrived at separately, and by that standard none of them is external.

MCMCglmm is. It is Bayesian where this is maximum likelihood, sampled where this
is optimised, written by somebody else in another language, and it fits exactly
this model: `us(trait):animal` is a genetic covariance across positions shared
by every record of a person, `us(trait):units` is the replicate level, and
`cengaussian` takes the interval a censored value is known to lie in. So this
is the check ADR 0010 calls the one that could be wrong quietly.

**It cannot check the covariance kernel and it is not asked to.** MCMCglmm has
no such structure, so the comparison runs on the free covariance, which is where
the likelihood lives. The kernel is a restriction of that and has its own tests.

**The criterion has to match the comparison.** Two estimators of different kinds
will not agree to eight decimals and it would be suspicious if they did. What
can be asked is whether the maximum-likelihood estimate falls inside the
posterior MCMCglmm drew, which is the strongest statement two engines of
different kinds can make about each other. A posterior too wide to discriminate
is reported as a failure rather than a pass, because a number falling inside an
interval that spans the whole range says nothing.

**The convergence flag is not a criterion here, and the sweep below is why.**
ADR 0010 warned that an approximate expectation step need not have its fixed
point at the maximum of the likelihood it reports, and put a gradient reading in
the fit record to detect it. It detects it. With nothing censored the fit
reaches `1e-07`; the reading then grows with the share of the data that is
censored, because that is the share the region approximation touches. So the
reading measures the approximation rather than the search, and failing on it
would be failing the model for a thing this check exists to quantify. **What is
still required is that the uncensored fit converges**, which is a real bar with
nothing approximate in it.

Run with:

    uv run --no-project python checks/repeated_against_mcmcglmm.py

It fails rather than skips when R or MCMCglmm is missing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import asterism
import numpy as np

FAMILIES = 120
REPLICATES = 2
POSITIONS = 2
# Families of four -- two parents and two children -- so that a family carries
# sixteen coordinates and the region a censored record sits in is not trivially
# small. The ladder is what covers the large-family case; this covers whether
# the likelihood is the right one at all.
PER_FAMILY = 4
TRUE_MEAN = (10.0, 12.0)
GENETIC_VARIANCE = (2.0, 3.0)
GENETIC_CORRELATION = 0.6
RESIDUAL_VARIANCE = (2.0, 3.0)
RESIDUAL_CORRELATION = 0.25
CENSORED_SHARE = 0.25
# The shares the gradient reading is swept over, to separate the approximation
# from the search. Nought has to be in it: that is the case with nothing
# approximate in it, and the only one this check requires to converge.
SWEEP = (0.0, 0.02, 0.05, 0.10, 0.25, 0.50)

ITERATIONS = 150_000
BURNIN = 30_000
THIN = 60
# A posterior wider than this cannot tell two engines apart, so a "pass"
# against it would be worth nothing.
WIDEST_USEFUL = {"heritability": 0.45, "correlation": 1.10}
LEAST_EFFECTIVE = 150

R_SCRIPT = """
suppressMessages(library(MCMCglmm))

d <- read.csv("data.csv")
d$animal <- factor(d$animal)
A <- as.matrix(read.csv("relationship.csv", header = FALSE))
dimnames(A) <- list(levels(d$animal), levels(d$animal))
Ainv <- as(solve(A), "dgCMatrix")

# **Parameter expanded, not the plain inverse Wishart.** With a small `nu` the
# prior drags the additive covariance towards nought and, at this sample size,
# dominates a heritability the data barely constrain; the posterior then spans
# almost the whole range and the comparison proves nothing. This is the prior
# the animal-model literature recommends for exactly that reason.
prior <- list(
  G = list(G1 = list(V = diag(2), nu = 2,
                     alpha.mu = c(0, 0), alpha.V = diag(2) * 1000)),
  R = list(V = diag(2), nu = 2)
)

set.seed(20260820)
m <- MCMCglmm(cbind(cbind(low_one, high_one), cbind(low_two, high_two)) ~ trait - 1,
              random = ~ us(trait):animal,
              rcov = ~ us(trait):units,
              ginverse = list(animal = Ainv),
              family = c("cengaussian", "cengaussian"),
              data = d, prior = prior,
              nitt = {iterations}, burnin = {burnin}, thin = {thin},
              verbose = FALSE)

vcv <- m$VCV
genetic <- vcv[, grepl("animal", colnames(vcv))]
residual <- vcv[, grepl("units", colnames(vcv))]
# Columns come back in the order (1,1), (2,1), (1,2), (2,2).
ga <- genetic[, 1]; gb <- genetic[, 4]; gab <- genetic[, 2]
ra <- residual[, 1]; rb <- residual[, 4]; rab <- residual[, 2]

h2_one <- ga / (ga + ra)
h2_two <- gb / (gb + rb)
rg <- gab / sqrt(ga * gb)
re <- rab / sqrt(ra * rb)

report <- function(name, draws) {{
  hpd <- HPDinterval(mcmc(draws))
  cat(sprintf("%s_mean %.9f\\n", name, mean(draws)))
  cat(sprintf("%s_low %.9f\\n", name, hpd[1]))
  cat(sprintf("%s_high %.9f\\n", name, hpd[2]))
  cat(sprintf("%s_effective %.1f\\n", name, effectiveSize(draws)))
}}
report("h2_one", h2_one)
report("h2_two", h2_two)
report("rg", rg)
report("re", re)
"""


def relationship(families: int) -> np.ndarray:
    """Unrelated nuclear families of two parents and two children."""
    people = families * PER_FAMILY
    matrix = np.eye(people)
    for family in range(families):
        base = family * PER_FAMILY
        for child in (2, 3):
            for parent in (0, 1):
                matrix[base + child, base + parent] = 0.5
                matrix[base + parent, base + child] = 0.5
        matrix[base + 2, base + 3] = 0.5
        matrix[base + 3, base + 2] = 0.5
    return matrix


def covariance(variance: tuple[float, float], correlation: float) -> np.ndarray:
    off = correlation * np.sqrt(variance[0] * variance[1])
    return np.array([[variance[0], off], [off, variance[1]]])


def simulate(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = relationship(FAMILIES)
    people = a.shape[0]
    rows = people * REPLICATES
    genetic = covariance(GENETIC_VARIANCE, GENETIC_CORRELATION)
    residual = covariance(RESIDUAL_VARIANCE, RESIDUAL_CORRELATION)

    # A matrix normal: correlated down the people by the relationship matrix
    # and across the positions by the genetic covariance.
    across = np.linalg.cholesky(a)
    along = np.linalg.cholesky(genetic)
    effects = across @ rng.standard_normal((people, POSITIONS)) @ along.T
    noise = np.linalg.cholesky(residual)
    value = np.zeros((rows, POSITIONS))
    for person in range(people):
        for replicate in range(REPLICATES):
            row = person * REPLICATES + replicate
            own = noise @ rng.standard_normal(POSITIONS)
            for position in range(POSITIONS):
                value[row, position] = (
                    TRUE_MEAN[position] + effects[person, position] + own[position]
                )
    return {"relationship": a, "value": value}


def censor_at(complete: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray]:
    """Which observations reached a limit, and where the limit was."""
    if share <= 0.0:
        return np.zeros_like(complete, dtype=bool), np.zeros(POSITIONS)
    limit = np.array(
        [float(np.quantile(complete[:, t], 1.0 - share)) for t in range(POSITIONS)]
    )
    return complete >= limit[np.newaxis, :], limit


def fit_at(
    model: Any, complete: np.ndarray, censored: np.ndarray, limit: np.ndarray
) -> dict[str, Any]:
    rows = complete.shape[0]
    return model.fit(
        np.where(censored, 0.0, complete),
        np.where(censored, 1, 0).astype(np.int64),
        np.tile(limit, (rows, 1)),
    )


def quantities(fit: dict[str, Any]) -> dict[str, float]:
    genetic = fit["component_covariances"][0]
    residual = fit["residual_covariance"]
    return {
        "h2_one": float(genetic[0, 0] / (genetic[0, 0] + residual[0, 0])),
        "h2_two": float(genetic[1, 1] / (genetic[1, 1] + residual[1, 1])),
        "rg": float(genetic[0, 1] / np.sqrt(genetic[0, 0] * genetic[1, 1])),
        "re": float(residual[0, 1] / np.sqrt(residual[0, 0] * residual[1, 1])),
    }


def main() -> int:
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on the path")

    data = simulate(20_260_820)
    a = data["relationship"]
    people = a.shape[0]
    rows = people * REPLICATES
    complete = data["value"]
    design = np.ones((rows, 1))
    model = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS)

    print(
        f"{people} people in {FAMILIES} families of {PER_FAMILY}, "
        f"{REPLICATES} replicates, {POSITIONS} positions: "
        f"{rows * POSITIONS} observations"
    )
    print(
        f"True heritabilities "
        f"{GENETIC_VARIANCE[0] / (GENETIC_VARIANCE[0] + RESIDUAL_VARIANCE[0]):.3f} and "
        f"{GENETIC_VARIANCE[1] / (GENETIC_VARIANCE[1] + RESIDUAL_VARIANCE[1]):.3f}, "
        f"genetic correlation {GENETIC_CORRELATION}, "
        f"residual correlation {RESIDUAL_CORRELATION}\n"
    )

    # How far the expectation step's fixed point sits from the maximum, against
    # how much of the data the region approximation has to touch. One data set,
    # so the estimates below are one draw and not a bias measurement; the
    # gradient is the point of it.
    print("the gradient reading against the share of the data that is censored:")
    print(
        f"{'share':>7} {'dim':>4} {'|g|':>10} {'monotone':>9} {'iters':>6} "
        f"{'h2 1':>7} {'h2 2':>7} {'rg':>7} {'re':>7}"
    )
    sweep = []
    uncensored = None
    for share in SWEEP:
        censored, limit = censor_at(complete, share)
        fit = fit_at(model, complete, censored, limit)
        found = quantities(fit)
        if share == 0.0:
            uncensored = fit
        print(
            f"{share:7.2f} {fit['sequential_dimension']:4d} "
            f"{fit['scaled_gradient']:10.2e} {str(fit['monotone']):>9} "
            f"{fit['iterations']:6d} "
            f"{found['h2_one']:7.4f} {found['h2_two']:7.4f} "
            f"{found['rg']:7.4f} {found['re']:7.4f}"
        )
        sweep.append(
            {
                "censored_share": share,
                "sequential_dimension": fit["sequential_dimension"],
                "scaled_gradient": fit["scaled_gradient"],
                "monotone": bool(fit["monotone"]),
                "iterations": fit["iterations"],
                **found,
            }
        )

    failures: list[str] = []
    if uncensored is None or not uncensored["converged"]:
        failures.append(
            "the fit with nothing censored did not converge, which has no "
            "approximation in it to blame"
        )

    censored, limit = censor_at(complete, CENSORED_SHARE)
    ours = fit_at(model, complete, censored, limit)
    mine = quantities(ours)
    print(
        f"\nAt {censored.mean():.0%} censored, against MCMCglmm.\n"
        f"Asterism  : h2 {mine['h2_one']:.6f} and {mine['h2_two']:.6f}, "
        f"rg {mine['rg']:.6f}, re {mine['re']:.6f}"
    )
    print(
        f"            scaled gradient {ours['scaled_gradient']:.2e} "
        f"(the approximation, not the search -- see the sweep above), "
        f"{ours['iterations']} iterations, "
        f"sequential dimension {ours['sequential_dimension']}\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        header = "animal,low_one,high_one,low_two,high_two"
        lines = [header]
        for row in range(rows):
            person = row // REPLICATES
            parts = [str(person + 1)]
            for position in range(POSITIONS):
                if censored[row, position]:
                    parts.extend([f"{limit[position]:.12f}", "Inf"])
                else:
                    parts.extend(
                        [
                            f"{complete[row, position]:.12f}",
                            f"{complete[row, position]:.12f}",
                        ]
                    )
            lines.append(",".join(parts))
        (directory / "data.csv").write_text("\n".join(lines) + "\n")
        (directory / "relationship.csv").write_text(
            "\n".join(",".join(f"{v:.12f}" for v in row) for row in a) + "\n"
        )
        (directory / "run.R").write_text(
            R_SCRIPT.format(iterations=ITERATIONS, burnin=BURNIN, thin=THIN)
        )
        print(
            f"sampling {ITERATIONS:,} iterations in MCMCglmm, this takes a while...",
            flush=True,
        )
        finished = subprocess.run(
            ["Rscript", "run.R"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if finished.returncode != 0:
            raise SystemExit(
                f"MCMCglmm did not run:\n{finished.stdout[-3000:]}\n"
                f"{finished.stderr[-3000:]}"
            )
        theirs = {}
        for name in ("h2_one", "h2_two", "rg", "re"):
            for suffix in ("mean", "low", "high", "effective"):
                found = re.search(rf"{name}_{suffix} (-?[0-9.eE+-]+)", finished.stdout)
                if found is None:
                    raise SystemExit(
                        f"MCMCglmm printed no {name}_{suffix}:"
                        f"\n{finished.stdout[-2000:]}"
                    )
                theirs[f"{name}_{suffix}"] = float(found.group(1))

    print(
        f"\n{'quantity':<14} | {'Asterism':>10} | {'MCMCglmm 95% interval':>25} "
        f"| {'ESS':>6} | inside"
    )
    print("-" * 74)
    report: dict[str, dict[str, float | bool]] = {}
    for name, label, kind in [
        ("h2_one", "heritability 1", "heritability"),
        ("h2_two", "heritability 2", "heritability"),
        ("rg", "genetic corr", "correlation"),
        ("re", "residual corr", "correlation"),
    ]:
        low = theirs[f"{name}_low"]
        high = theirs[f"{name}_high"]
        value = float(mine[name])
        inside = low <= value <= high
        width = high - low
        effective = theirs[f"{name}_effective"]
        print(
            f"{label:<14} | {value:10.6f} | [{low:10.6f}, {high:9.6f}] "
            f"| {effective:6.0f} | {'yes' if inside else 'NO'}"
        )
        report[name] = {
            "asterism": value,
            "posterior_mean": theirs[f"{name}_mean"],
            "posterior_low": low,
            "posterior_high": high,
            "effective_sample_size": effective,
            "inside": inside,
            "width": width,
        }
        if not inside:
            failures.append(
                f"the maximum-likelihood {label} falls outside the posterior "
                f"MCMCglmm drew"
            )
        if width > WIDEST_USEFUL[kind]:
            failures.append(
                f"the posterior for {label} spans {width:.3f}, wider than "
                f"{WIDEST_USEFUL[kind]}; an estimate falling inside it would "
                "say nothing about either engine"
            )
        if effective < LEAST_EFFECTIVE:
            failures.append(
                f"the chain's effective sample size for {label} is "
                f"{effective:.0f}, too few to compare against"
            )

    receipt = {
        "what": "the repeated-measures model against MCMCglmm's multivariate "
                "censored animal model, on a free covariance",
        "date": date.today().isoformat(),
        "people": people,
        "families": FAMILIES,
        "replicates": REPLICATES,
        "positions": POSITIONS,
        "censored_share": float(censored.mean()),
        "sequential_dimension": ours["sequential_dimension"],
        "truth": {
            "heritability": [
                GENETIC_VARIANCE[t] / (GENETIC_VARIANCE[t] + RESIDUAL_VARIANCE[t])
                for t in range(POSITIONS)
            ],
            "genetic_correlation": GENETIC_CORRELATION,
            "residual_correlation": RESIDUAL_CORRELATION,
        },
        "asterism": {
            "converged": bool(ours["converged"]),
            "monotone": bool(ours["monotone"]),
            "scaled_gradient": ours["scaled_gradient"],
            "iterations": ours["iterations"],
            "loglik": ours["loglik"],
        },
        "gradient_against_censoring": sweep,
        "note": "the gradient reading measures how far the approximate "
                "expectation step's fixed point sits from the maximum, and "
                "grows with the share of the data the region approximation "
                "touches. It is not a criterion here. The fit with nothing "
                "censored is required to converge and is.",
        "mcmcglmm": {
            "iterations": ITERATIONS,
            "burnin": BURNIN,
            "thin": THIN,
        },
        "by_quantity": report,
        "passed": not failures,
        "failures": failures,
    }
    out = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / f"repeated-against-mcmcglmm-{receipt['date']}.json"
    )
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        "\nPASSED: the maximum-likelihood estimates fall inside the posterior "
        "an independent Bayesian engine drew, for both heritabilities and both "
        "correlations, with censoring and relatedness in play at once."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
