"""Does the censored likelihood agree with R's `censReg`?

**The comparison is arranged so that only the new part is under test.** With no
relatedness the relationship matrix is the identity, the additive and residual
variances add to one total, and the model is an ordinary Tobit regression --
exactly what `censReg` fits. So a disagreement here is a disagreement about
censoring, not about variance components, which Asterism already checks
elsewhere against `regress` and SOLAR.

That the heritability is unidentified in this design is deliberate rather than
a flaw in it. The likelihood is flat along that coordinate, so the optimiser
must still return the right total variance and the right coefficients while
having nothing to say about the split. A model that cannot do that would be
unusable on any real trait with little family structure.

`censReg` takes the dependent variable with censored values set to the limit
and infers the censoring from equality with it. Asterism takes the status
separately and never infers it, which is the safer contract but means the two
inputs are constructed differently here.

Run with:

    uv run --no-project python checks/tobit_against_censreg.py

It fails rather than skips when R or `censReg` is missing.
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

PEOPLE = 4000
TRUE_INTERCEPT = 5.0
TRUE_SLOPE = 1.5
TRUE_SD = 2.0
# A limit a little above the mean, censoring roughly a third.
LIMIT = 6.5
TOLERANCE = 1.0e-3

R_SCRIPT = """
suppressMessages(library(censReg))
d <- read.csv("data.csv")
m <- censReg(y ~ x, left = -Inf, right = {limit}, data = d)
co <- coef(m)
cat(sprintf("intercept %.12f\\n", co[["(Intercept)"]]))
cat(sprintf("slope %.12f\\n", co[["x"]]))
cat(sprintf("sigma %.12f\\n", exp(co[["logSigma"]])))
cat(sprintf("loglik %.12f\\n", as.numeric(logLik(m))))
"""


def main() -> int:
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on the path")

    rng = np.random.default_rng(20_260_818)
    x = rng.normal(size=PEOPLE)
    complete = TRUE_INTERCEPT + TRUE_SLOPE * x + TRUE_SD * rng.normal(size=PEOPLE)
    censored = complete >= LIMIT
    print(f"{PEOPLE} unrelated people, {censored.sum()} censored "
          f"({censored.mean():.1%}) at a limit of {LIMIT}")

    # Asterism: the status is given, never inferred from the value.
    relationship = np.eye(PEOPLE)
    value = np.where(censored, np.nan, complete)
    status = np.where(censored, 1, 0).astype(np.int64)
    limit = np.full(PEOPLE, LIMIT)
    design = np.column_stack([np.ones(PEOPLE), x])
    ours = asterism.tobit_fit(relationship, value, status, limit, design)
    effects = ours["fixed_effects"]
    total_variance = ours["total_variance"]
    loglik = ours["loglik"]
    print(f"\nAsterism : intercept {effects[0]:.9f} slope {effects[1]:.9f} "
          f"sigma {total_variance ** 0.5:.9f}")
    print(f"           loglik {loglik:.9f}, converged {ours['converged']}, "
          f"scaled gradient {ours['scaled_gradient']:.2e}")
    print(f"           heritability {ours['heritability']:.4f} "
          f"(unidentified here, and expected to be)")

    # censReg: the dependent variable carries the limit where censored.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        observed = np.where(censored, LIMIT, complete)
        rows = ["y,x"] + [f"{observed[i]:.12f},{x[i]:.12f}" for i in range(PEOPLE)]
        (directory / "data.csv").write_text("\n".join(rows) + "\n")
        (directory / "run.R").write_text(R_SCRIPT.format(limit=LIMIT))
        finished = subprocess.run(
            ["Rscript", "run.R"], cwd=directory, capture_output=True,
            text=True, check=False,
        )
        if finished.returncode != 0:
            raise SystemExit(
                f"censReg did not run:\n{finished.stdout}\n{finished.stderr}"
            )
        theirs = {}
        for name in ("intercept", "slope", "sigma", "loglik"):
            found = re.search(rf"{name} (-?[0-9.eE+-]+)", finished.stdout)
            if found is None:
                raise SystemExit(f"censReg printed no {name}:\n{finished.stdout}")
            theirs[name] = float(found.group(1))

    print(f"\ncensReg  : intercept {theirs['intercept']:.9f} "
          f"slope {theirs['slope']:.9f} sigma {theirs['sigma']:.9f}")
    print(f"           loglik {theirs['loglik']:.9f}")

    ours_named = {
        "intercept": effects[0],
        "slope": effects[1],
        "sigma": total_variance ** 0.5,
        "loglik": loglik,
    }
    print(f"\n{'quantity':<11} | {'Asterism':>16} | {'censReg':>16} | difference")
    print("-" * 68)
    worst = 0.0
    failures = []
    for name in ("intercept", "slope", "sigma", "loglik"):
        difference = abs(ours_named[name] - theirs[name])
        relative = difference / max(1.0, abs(theirs[name]))
        worst = max(worst, relative)
        print(f"{name:<11} | {ours_named[name]:16.9f} | {theirs[name]:16.9f} | "
              f"{difference:.3e}")
        if relative > TOLERANCE:
            failures.append(f"{name} differs by {relative:.3e}, over {TOLERANCE:.0e}")

    receipt = {
        "what": "the censored likelihood against R's censReg, with no relatedness",
        "date": date.today().isoformat(),
        "people": PEOPLE,
        "censored": int(censored.sum()),
        "limit": LIMIT,
        "truth": {"intercept": TRUE_INTERCEPT, "slope": TRUE_SLOPE, "sigma": TRUE_SD},
        "asterism": ours_named,
        "censreg": theirs,
        "worst_relative_difference": worst,
        "tolerance": TOLERANCE,
        "passed": not failures,
        "failures": failures,
    }
    out = Path(__file__).resolve().parent.parent / "evidence" / (
        f"tobit-against-censreg-{receipt['date']}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nPASSED: worst relative difference {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
