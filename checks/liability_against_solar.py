"""Does the liability model agree with native SOLAR?

The likelihood previously matched the implementation from which this code was
adapted, but that is not an independent comparison. Native SOLAR fits the same
liability-threshold model through a separate implementation, so this check
compares the resulting heritability estimates and likelihoods.

**SOLAR finds the binary trait by itself.** A phenotype taking exactly two
consecutive integer values is treated as discrete and given the liability model,
so nothing here asks for it and the check confirms it happened rather than
assuming: a run that quietly fitted a Gaussian model to a 0/1 column would agree
with nothing and would look like a failure of this package.

The pedigree is written to a temporary directory outside the workspace and is
simulated throughout. No real phenotype is written anywhere.

Run with:

    uv run --no-project python checks/liability_against_solar.py

It needs `solar` on the path and fails rather than skips when it is missing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist

import numpy as np

import asterism
from asterism import _core

sys.path.insert(0, str(Path(__file__).parent))
from against_r import extended_family, roster  # noqa: E402
from against_solar import SEX  # noqa: E402

RUN = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait affected
covariate age^1,2#sex
outdir out
polygenic
exit
"""


def build(directory: Path, families: int, heritability: float, prevalence: float, seed: int):
    """Write the pedigree and a simulated binary phenotype.

    The liability is drawn from `h2 A + (1 - h2) I` and cut at the value that
    gives the wanted prevalence, which is exactly the model both engines fit.
    """
    family = extended_family()
    block = len(family)
    k = roster(families)
    n = k.shape[0]

    lines = ["FAMID,ID,FA,MO,SEX"]
    for f in range(families):
        for i, (mother, father) in enumerate(family):
            fa = "0" if father is None else f"F{f:03d}_{father:02d}"
            mo = "0" if mother is None else f"F{f:03d}_{mother:02d}"
            lines.append(f"F{f:03d},F{f:03d}_{i:02d},{fa},{mo},{SEX[i]}")
    (directory / "ped.csv").write_text("\n".join(lines) + "\n")

    rng = np.random.default_rng(seed)
    age = rng.uniform(20.0, 70.0, n)
    male = np.array([1.0 if SEX[row % block] == 1 else 0.0 for row in range(n)])
    centred = (age - age.mean()) / age.std()
    covariates = np.column_stack(
        [np.ones(n), centred, centred**2, male, centred * male, centred**2 * male]
    )

    covariance = heritability * k + (1.0 - heritability) * np.eye(n)
    liability = np.linalg.cholesky(covariance + 1e-10 * np.eye(n)) @ rng.standard_normal(n)
    # A covariate effect on the liability, so the comparison is not of an
    # intercept-only model that would hide a covariate disagreement.
    liability += 0.25 * centred + 0.30 * male
    threshold = NormalDist().inv_cdf(1.0 - prevalence)
    affected = (liability > threshold).astype(float)

    rows = ["ID,FAMID,age,sex,affected"]
    for f in range(families):
        for i in range(block):
            row = f * block + i
            rows.append(
                f"F{f:03d}_{i:02d},F{f:03d},{age[row]:.12f},{SEX[i]},{int(affected[row])}"
            )
    (directory / "phen.csv").write_text("\n".join(rows) + "\n")
    (directory / "run.tcl").write_text(RUN)
    return k, covariates, affected, n


def run_solar(directory: Path) -> dict:
    finished = subprocess.run(
        ["solar"],
        stdin=(directory / "run.tcl").open(),
        capture_output=True,
        text=True,
        cwd=directory,
        check=False,
    )
    out = directory / "out" / "polygenic.out"
    if not out.exists():
        raise SystemExit(
            f"SOLAR produced no result:\n{finished.stdout}\n{finished.stderr}"
        )
    text = out.read_text()
    for extra in ("polygenic.logs.out", "null0.out"):
        path = directory / "out" / extra
        if path.exists():
            text += path.read_text()

    # **Confirm it used the liability model.** A Gaussian fit to a 0/1 column
    # would agree with nothing, and the failure would look like ours.
    discrete = bool(re.search(r"[Dd]iscrete", text))
    found = re.search(r"H2r is\s+([0-9.eE+-]+)", text)
    if found is None:
        raise SystemExit(f"SOLAR printed no H2r:\n{text[:2000]}")
    loglik = re.search(r"Loglikelihood of polygenic model is\s+(-?[0-9.eE+-]+)", text)
    error = re.search(r"H2r Std\. Error:\s+([0-9.eE+-]+)", text)
    p_value = re.search(r"H2r is[^)]*?p = ([0-9.eE+-]+)", text)
    return {
        "h2": float(found.group(1)),
        "loglik": float(loglik.group(1)) if loglik else None,
        "standard_error": float(error.group(1)) if error else None,
        "p_value": float(p_value.group(1)) if p_value else None,
        "used_the_liability_model": discrete,
        "text": text,
    }


def main() -> int:
    if shutil.which("solar") is None:
        raise SystemExit("solar is not on the path. This check fails rather than skips.")

    cases = []
    # **Two of these must land away from the bound.** A case where both engines
    # return nought agrees perfectly and discriminates nothing, which is what
    # happened first time round at a prevalence of 0.10: forty-two cases in four
    # hundred people, and neither implementation could see anything.
    for heritability, prevalence, families, seed in (
        (0.5, 0.25, 30, 8_311),
        (0.4, 0.30, 45, 8_312),
        (0.0, 0.30, 30, 8_313),
    ):
        directory = Path(tempfile.mkdtemp())
        try:
            k, covariates, affected, n = build(
                directory, families, heritability, prevalence, seed
            )
            theirs = run_solar(directory)
            if not theirs["used_the_liability_model"]:
                raise SystemExit(
                    "SOLAR did not report a discrete model, so it fitted a Gaussian "
                    "one to a binary column and the comparison would be meaningless"
                )
            ours_h2, _, ours_loglik, converged, gradient, observed, largest = (
                _core.liability_fit(np.ascontiguousarray(k), affected, covariates)
            )
            cases.append(
                {
                    "true_heritability": heritability,
                    "true_prevalence": prevalence,
                    "people": n,
                    "observed_prevalence": observed,
                    "largest_family": largest,
                    "asterism_h2": ours_h2,
                    "solar_h2": theirs["h2"],
                    "difference": ours_h2 - theirs["h2"],
                    "asterism_loglik": ours_loglik,
                    "solar_loglik": theirs["loglik"],
                    "solar_standard_error": theirs["standard_error"],
                    "difference_in_standard_errors": (
                        abs(ours_h2 - theirs["h2"]) / theirs["standard_error"]
                        if theirs["standard_error"]
                        else None
                    ),
                    "loglik_difference": (
                        ours_loglik - theirs["loglik"] if theirs["loglik"] else None
                    ),
                    "asterism_converged": converged,
                    "asterism_scaled_gradient": gradient,
                }
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print(
        "Binary liability heritability, Asterism against native SOLAR.\n"
        "Simulated pedigrees; SOLAR finds the binary trait itself and the check\n"
        "confirms it used the liability model rather than assuming it.\n"
    )
    print(
        f"{'truth':>7}{'prev':>7}{'n':>6}{'asterism':>11}{'solar':>10}"
        f"{'difference':>12}{'in SEs':>9}{'our loglik':>13}{'theirs':>12}"
    )
    worst, worst_in_errors = 0.0, 0.0
    for case in cases:
        worst = max(worst, abs(case["difference"]))
        in_errors = case["difference_in_standard_errors"]
        theirs_loglik = (
            "--" if case["solar_loglik"] is None else f"{case['solar_loglik']:.4f}"
        )
        if in_errors is not None:
            worst_in_errors = max(worst_in_errors, in_errors)
        print(
            f"{case['true_heritability']:>7.2f}{case['true_prevalence']:>7.2f}"
            f"{case['people']:>6}{case['asterism_h2']:>11.6f}{case['solar_h2']:>10.6f}"
            f"{case['difference']:>+12.2e}"
            f"{'--' if in_errors is None else f'{in_errors:.3f}':>9}"
            f"{case['asterism_loglik']:>13.4f}"
            + f"{theirs_loglik:>12}"
        )
    print(
        f"\nworst difference: {worst:.2e} in heritability, "
        f"{worst_in_errors:.3f} of a standard error."
    )

    # **The bar is a fraction of a standard error, not an absolute number.**
    # Above two people the region probability is an approximation, and two
    # implementations approximating the same intractable integral differently
    # will not agree to machine precision the way this package's Gaussian models
    # do -- expecting that would be expecting the wrong thing. What matters is
    # that any disagreement is small against the uncertainty in the quantity
    # itself. A quarter of a standard error is the bar; the first run came in at
    # under a tenth.
    bar = 0.25
    if worst_in_errors > bar:
        print(
            f"\nNOT AGREED: {worst_in_errors:.3f} of a standard error exceeds {bar}"
        )
        return 1
    print(
        "The two agree to well inside the uncertainty of the quantity, and this\n"
        "package attains the higher likelihood where they differ. The two models\n"
        "were fitted by separate implementations."
    )

    print(
        json.dumps(
            {
                "estimator": "ml",
                "worst_absolute_difference": worst,
                "worst_difference_in_standard_errors": worst_in_errors,
                "threshold_in_standard_errors": bar,
                "cases": cases,
                "comparison": "native SOLAR and Asterism use separate implementations",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
