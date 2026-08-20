"""Does the censored model agree with MCMCglmm once relatedness is in play?

`checks/tobit_against_censreg.py` compares the censored likelihood on unrelated
people, where the model is an ordinary Tobit regression. That leaves the case
the package actually exists for untested against anything external: censoring
**and** a relationship matrix at the same time. No engine in R does that except
MCMCglmm, whose `cengaussian` family takes a censoring interval per observation
and whose `ginverse` argument takes a relationship matrix.

**This comparison is of a different kind from the others and the criterion has
to match.** MCMCglmm is Bayesian and sampled; Asterism is maximum likelihood and
optimised. They will not agree to eight decimals and it would be suspicious if
they did. What can be asked is whether the maximum-likelihood estimate falls
inside the posterior MCMCglmm draws, which is the strongest statement two
estimators of different kinds can make about each other.

That MCMCglmm is Bayesian is the point rather than a nuisance. Under ADR 0006
agreement proves fidelity only against something arrived at separately, and an
engine using a different inferential framework, a different algorithm and a
different author is about as separate as a comparator gets.

Run with:

    uv run --no-project python checks/tobit_against_mcmcglmm.py

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

import asterism
import numpy as np

PAIRS = 800
TRUE_HERITABILITY = 0.5
TRUE_VARIANCE = 4.0
TRUE_MEAN = 10.0
CENSORED_SHARE = 0.25
# Long enough that the posterior is not the limiting uncertainty. Reported with
# the result, because a chain too short is how two engines come to "disagree".
# A posterior wider than this cannot discriminate between two engines, so a
# "pass" against it would be worth nothing.
WIDEST_USEFUL_INTERVAL = 0.40
ITERATIONS = 120_000
BURNIN = 20_000
THIN = 50

R_SCRIPT = """
suppressMessages(library(MCMCglmm))
suppressMessages(library(Matrix))

d <- read.csv("data.csv")
n <- nrow(d)
# **The levels must be in the matrix's own order, and R will not do that
# by itself.** A factor built from "1".."1600" sorts its levels
# lexicographically -- "1", "10", "100", "1000" -- so naming the matrix
# rows after them labels row 2 as person 10. Every sibling pair is then
# mis-paired, the additive variance collapses, and the heritability comes
# back near nought while everything else still looks right. Stating the
# levels explicitly is what stops it.
d$animal <- factor(as.character(seq_len(n)), levels = as.character(seq_len(n)))
d$upper[!is.finite(d$upper)] <- Inf

A <- as.matrix(read.csv("relationship.csv", header = FALSE))
dimnames(A) <- list(levels(d$animal), levels(d$animal))
Ainv <- as(solve(A), "dgCMatrix")

# **Parameter expanded, not the plain inverse Wishart.** With `nu = 0.002`
# the prior on the additive variance drags it towards nought, and at this
# sample size it dominates a heritability the data barely constrain -- the
# posterior then spans almost the whole range and the comparison proves
# nothing. This is the prior the animal-model literature recommends for
# exactly that reason.
prior <- list(G = list(G1 = list(V = 1, nu = 1, alpha.mu = 0, alpha.V = 1000)),
              R = list(V = 1, nu = 0.002))

set.seed(20260818)
m <- MCMCglmm(cbind(lower, upper) ~ 1,
              random = ~ animal,
              ginverse = list(animal = Ainv),
              family = "cengaussian",
              data = d, prior = prior,
              nitt = {iterations}, burnin = {burnin}, thin = {thin},
              verbose = FALSE)

va <- m$VCV[, "animal"]
ve <- m$VCV[, "units"]
h2 <- va / (va + ve)
total <- va + ve
hpd_h2 <- HPDinterval(mcmc(h2))
hpd_total <- HPDinterval(mcmc(total))
hpd_mean <- HPDinterval(m$Sol[, "(Intercept)"])

cat(sprintf("h2_mean %.9f\\n", mean(h2)))
cat(sprintf("h2_low %.9f\\n", hpd_h2[1]))
cat(sprintf("h2_high %.9f\\n", hpd_h2[2]))
cat(sprintf("total_mean %.9f\\n", mean(total)))
cat(sprintf("total_low %.9f\\n", hpd_total[1]))
cat(sprintf("total_high %.9f\\n", hpd_total[2]))
cat(sprintf("intercept_mean %.9f\\n", mean(m$Sol[, "(Intercept)"])))
cat(sprintf("intercept_low %.9f\\n", hpd_mean[1]))
cat(sprintf("intercept_high %.9f\\n", hpd_mean[2]))
cat(sprintf("effective_h2 %.1f\\n", effectiveSize(h2)))
"""


def main() -> int:
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on the path")

    rng = np.random.default_rng(20_260_818)
    n = 2 * PAIRS
    shared = rng.normal(size=PAIRS) * np.sqrt(TRUE_HERITABILITY / 2.0)
    own = rng.normal(size=n) * np.sqrt(1.0 - TRUE_HERITABILITY / 2.0)
    complete = TRUE_MEAN + np.sqrt(TRUE_VARIANCE) * (np.repeat(shared, 2) + own)

    relationship = np.eye(n)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5

    limit = float(np.quantile(complete, 1.0 - CENSORED_SHARE))
    censored = complete >= limit
    print(f"{n} people in {PAIRS} sibling pairs, {censored.sum()} censored "
          f"({censored.mean():.1%}) at a limit of {limit:.4f}")
    print(f"True heritability {TRUE_HERITABILITY}, variance {TRUE_VARIANCE}, "
          f"mean {TRUE_MEAN}\n")

    ours = asterism.tobit_fit(
        relationship,
        np.where(censored, np.nan, complete),
        np.where(censored, 1, 0).astype(np.int64),
        np.full(n, limit),
        np.ones((n, 1)),
    )
    print(f"Asterism  : h2 {ours['heritability']:.6f}  "
          f"variance {ours['total_variance']:.6f}  "
          f"intercept {ours['fixed_effects'][0]:.6f}")
    print(f"            converged {ours['converged']}, "
          f"scaled gradient {ours['scaled_gradient']:.2e}\n")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        # `cengaussian` takes the interval the value is known to lie in.
        lower = np.where(censored, limit, complete)
        upper = np.where(censored, np.inf, complete)
        rows = ["lower,upper"] + [
            f"{lower[i]:.12f},{'Inf' if not np.isfinite(upper[i]) else f'{upper[i]:.12f}'}"
            for i in range(n)
        ]
        (directory / "data.csv").write_text("\n".join(rows) + "\n")
        (directory / "relationship.csv").write_text(
            "\n".join(",".join(f"{v:.12f}" for v in row) for row in relationship) + "\n"
        )
        (directory / "run.R").write_text(
            R_SCRIPT.format(iterations=ITERATIONS, burnin=BURNIN, thin=THIN)
        )
        print(f"sampling {ITERATIONS:,} iterations in MCMCglmm, this takes a while...",
              flush=True)
        finished = subprocess.run(
            ["Rscript", "run.R"], cwd=directory, capture_output=True,
            text=True, check=False,
        )
        if finished.returncode != 0:
            raise SystemExit(
                f"MCMCglmm did not run:\n{finished.stdout[-3000:]}\n"
                f"{finished.stderr[-3000:]}"
            )
        theirs = {}
        for name in ("h2_mean", "h2_low", "h2_high", "total_mean", "total_low",
                     "total_high", "intercept_mean", "intercept_low",
                     "intercept_high", "effective_h2"):
            found = re.search(rf"{name} (-?[0-9.eE+-]+)", finished.stdout)
            if found is None:
                raise SystemExit(
                    f"MCMCglmm printed no {name}:\n{finished.stdout[-2000:]}"
                )
            theirs[name] = float(found.group(1))

    print(f"\nMCMCglmm  : h2 {theirs['h2_mean']:.6f} "
          f"[{theirs['h2_low']:.6f}, {theirs['h2_high']:.6f}]")
    print(f"            variance {theirs['total_mean']:.6f} "
          f"[{theirs['total_low']:.6f}, {theirs['total_high']:.6f}]")
    print(f"            intercept {theirs['intercept_mean']:.6f} "
          f"[{theirs['intercept_low']:.6f}, {theirs['intercept_high']:.6f}]")
    print(f"            effective sample size for h2: {theirs['effective_h2']:.0f}")

    failures = []
    width = theirs["h2_high"] - theirs["h2_low"]
    if width > WIDEST_USEFUL_INTERVAL:
        failures.append(
            f"the posterior interval for the heritability spans {width:.3f}, "
            f"wider than {WIDEST_USEFUL_INTERVAL}; a maximum-likelihood estimate "
            "falling inside it would say nothing about either engine"
        )
    if theirs["effective_h2"] < 200:
        failures.append(
            f"the chain's effective sample size for the heritability is "
            f"{theirs['effective_h2']:.0f}, too few to compare against"
        )

    print(f"\n{'quantity':<11} | {'Asterism':>11} | {'MCMCglmm 95% interval':>26} | inside")
    print("-" * 66)
    comparisons = [
        ("heritability", ours["heritability"], "h2_low", "h2_high"),
        ("variance", ours["total_variance"], "total_low", "total_high"),
        ("intercept", ours["fixed_effects"][0], "intercept_low", "intercept_high"),
    ]
    inside_all = {}
    for label, value, low, high in comparisons:
        inside = theirs[low] <= value <= theirs[high]
        inside_all[label] = inside
        print(f"{label:<11} | {value:11.6f} | "
              f"[{theirs[low]:11.6f}, {theirs[high]:9.6f}] | {'yes' if inside else 'NO'}")
        if not inside:
            failures.append(
                f"the maximum-likelihood {label} falls outside the posterior "
                f"interval MCMCglmm drew"
            )

    receipt = {
        "what": "the censored model with relatedness, against MCMCglmm's "
                "cengaussian family",
        "date": date.today().isoformat(),
        "people": n,
        "pairs": PAIRS,
        "censored": int(censored.sum()),
        "limit": limit,
        "truth": {
            "heritability": TRUE_HERITABILITY,
            "variance": TRUE_VARIANCE,
            "mean": TRUE_MEAN,
        },
        "asterism": {
            "heritability": ours["heritability"],
            "total_variance": ours["total_variance"],
            "intercept": ours["fixed_effects"][0],
            "converged": ours["converged"],
        },
        "mcmcglmm": theirs,
        "sampling": {"iterations": ITERATIONS, "burnin": BURNIN, "thin": THIN},
        "inside_posterior": inside_all,
        "passed": not failures,
        "failures": failures,
    }
    out = Path(__file__).resolve().parent.parent / "evidence" / (
        f"tobit-against-mcmcglmm-{receipt['date']}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: every maximum-likelihood estimate lies inside the posterior.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
