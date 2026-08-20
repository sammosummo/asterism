"""Does a binary trait beside a continuous one agree with native SOLAR?

ADR 0007 records that mixed binary/continuous is the one two-trait combination
with native parity behind it, and this is the cell the psychiatric diagnosis
against hearing analysis needs. SOLAR fits it with its own discrete and mixed
trait machinery, written by other people over decades, which is what ADR 0006
means by a check arrived at separately.

**What agreement here proves is fidelity, not correctness.** Both engines could
be wrong in the same way, and SOLAR is a comparator rather than the definition
of correct. What it rules out is the far more likely failure: that a new
likelihood written this week has a sign, a scale or an ordering wrong.

The genetic correlation is the quantity to read. Both engines fix a liability's
variance at one, so the binary trait's heritability is on the liability scale in
each, and the correlation is scale free either way.

Run with:

    uv run --no-project python checks/mixed_bivariate_against_solar.py

It needs `solar` on the path and fails rather than skips when it is missing.
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

FAMILIES = 300
SIBS = 4
TRUE_HERITABILITY = [0.5, 0.5]
TRUE_GENETIC_CORRELATION = 0.5
TRUE_RESIDUAL_CORRELATION = 0.2
CONTINUOUS_VARIANCE = 3.0
PREVALENCE = 0.3
# Two engines fitting the same likelihood by different searches will not agree
# to machine precision on a simulated data set; they should agree well inside
# the standard error of either.
TOLERANCE = 0.05

RUN = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait affected quantity
outdir out
polygenic
exit
"""


def main() -> int:
    if shutil.which("solar") is None:
        raise SystemExit("solar is not on the path")

    block = 2 + SIBS
    n = FAMILIES * block
    rng = np.random.default_rng(20_260_818)

    # Additive relationship: two founders and their full sibs.
    a = np.eye(block)
    for kid in range(2, block):
        a[0, kid] = a[kid, 0] = 0.5
        a[1, kid] = a[kid, 1] = 0.5
        for other in range(2, block):
            if other != kid:
                a[kid, other] = 0.5
    relationship = np.kron(np.eye(FAMILIES), a)

    # A correlated bivariate liability and quantity.
    sigma_a = np.array([
        [TRUE_HERITABILITY[0],
         TRUE_GENETIC_CORRELATION
         * np.sqrt(TRUE_HERITABILITY[0] * TRUE_HERITABILITY[1] * CONTINUOUS_VARIANCE)],
        [TRUE_GENETIC_CORRELATION
         * np.sqrt(TRUE_HERITABILITY[0] * TRUE_HERITABILITY[1] * CONTINUOUS_VARIANCE),
         TRUE_HERITABILITY[1] * CONTINUOUS_VARIANCE],
    ])
    sigma_e = np.array([
        [1.0 - TRUE_HERITABILITY[0],
         TRUE_RESIDUAL_CORRELATION
         * np.sqrt((1.0 - TRUE_HERITABILITY[0])
                   * (1.0 - TRUE_HERITABILITY[1]) * CONTINUOUS_VARIANCE)],
        [TRUE_RESIDUAL_CORRELATION
         * np.sqrt((1.0 - TRUE_HERITABILITY[0])
                   * (1.0 - TRUE_HERITABILITY[1]) * CONTINUOUS_VARIANCE),
         (1.0 - TRUE_HERITABILITY[1]) * CONTINUOUS_VARIANCE],
    ])
    genetic = np.linalg.cholesky(np.kron(sigma_a, relationship) + 1e-9 * np.eye(2 * n))
    residual = np.linalg.cholesky(np.kron(sigma_e, np.eye(n)) + 1e-9 * np.eye(2 * n))
    latent = (genetic @ rng.standard_normal(2 * n)
              + residual @ rng.standard_normal(2 * n)).reshape(2, n)

    from statistics import NormalDist
    cut = NormalDist().inv_cdf(1.0 - PREVALENCE)
    affected = (latent[0] > cut).astype(int)
    quantity = latent[1]
    print(f"{n} people in {FAMILIES} families of {block}. "
          f"{affected.sum()} affected ({affected.mean():.1%}).")

    ours = asterism.mixed_bivariate_fit(
        relationship,
        {
            "kind": "binary",
            "value": np.full(n, np.nan),
            # 1 is a case, 2 is not; the threshold rides on the intercept.
            "censoring": np.where(affected == 1, 1, 2).astype(np.int64),
            "limit": np.zeros(n),
        },
        {
            "kind": "continuous",
            "value": quantity,
            "censoring": np.zeros(n, dtype=np.int64),
            "limit": np.zeros(n),
        },
        np.ones((n, 1)),
    )
    print(f"\nAsterism : h2 {ours['heritability'][0]:.4f} / "
          f"{ours['heritability'][1]:.4f}   RhoG {ours['genetic_correlation']:.4f}"
          f"   RhoE {ours['residual_correlation']:.4f}")
    print(f"           converged {ours['converged']}, "
          f"scaled gradient {ours['scaled_gradient']:.2e}")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        ped = ["FAMID,ID,FA,MO,SEX"]
        phen = ["ID,FAMID,affected,quantity"]
        for family in range(FAMILIES):
            for i in range(block):
                who = f"F{family:04d}_{i:02d}"
                if i < 2:
                    fa = mo = "0"
                    sex = 1 if i == 0 else 2
                else:
                    fa, mo = f"F{family:04d}_00", f"F{family:04d}_01"
                    sex = 1 + (i % 2)
                ped.append(f"F{family:04d},{who},{fa},{mo},{sex}")
                row = family * block + i
                phen.append(f"{who},F{family:04d},{affected[row]},"
                            f"{quantity[row]:.12f}")
        (directory / "ped.csv").write_text("\n".join(ped) + "\n")
        (directory / "phen.csv").write_text("\n".join(phen) + "\n")
        (directory / "run.tcl").write_text(RUN)
        finished = subprocess.run(
            ["solar"], stdin=(directory / "run.tcl").open(),
            capture_output=True, text=True, cwd=directory, check=False,
        )
        text = ""
        for name in ("polygenic.out", "polygenic.logs.out", "null0.out"):
            path = directory / "out" / name
            if path.exists():
                text += path.read_text()
        if not text:
            raise SystemExit(
                f"SOLAR produced nothing:\n{finished.stdout[-2000:]}\n"
                f"{finished.stderr[-1000:]}"
            )

    # **Confirm it used the liability model.** A Gaussian fit to a 0/1 column
    # would disagree with ours for a reason that has nothing to do with our
    # code, and the announcement is in the log rather than the result file.
    discrete = bool(re.search(r"[Dd]iscrete", text))
    theirs = {}
    for label, pattern in [
        ("h2_affected", r"H2r\(affected\) is\s+([0-9.eE+-]+)"),
        ("h2_quantity", r"H2r\(quantity\) is\s+([0-9.eE+-]+)"),
        ("rho_g", r"RhoG is\s+(-?[0-9.eE+-]+)"),
        ("rho_e", r"RhoE is\s+(-?[0-9.eE+-]+)"),
    ]:
        found = re.search(pattern, text)
        theirs[label] = float(found.group(1)) if found else None
    print(f"\nSOLAR    : h2 {theirs['h2_affected']} / {theirs['h2_quantity']}"
          f"   RhoG {theirs['rho_g']}   RhoE {theirs['rho_e']}")
    print(f"           used the discrete model: {discrete}")

    failures = []
    if not discrete:
        failures.append(
            "SOLAR did not use its discrete model, so it fitted a different "
            "model and the comparison says nothing"
        )
    ours_named = {
        "h2_affected": ours["heritability"][0],
        "h2_quantity": ours["heritability"][1],
        "rho_g": ours["genetic_correlation"],
        "rho_e": ours["residual_correlation"],
    }
    print(f"\n{'quantity':<12} | {'Asterism':>10} | {'SOLAR':>10} | difference")
    print("-" * 52)
    for name, value in ours_named.items():
        other = theirs[name]
        if other is None:
            failures.append(f"SOLAR printed no {name}")
            continue
        difference = abs(value - other)
        print(f"{name:<12} | {value:10.4f} | {other:10.4f} | {difference:.4f}")
        if difference > TOLERANCE:
            failures.append(f"{name} differs by {difference:.4f}, over {TOLERANCE}")

    receipt = {
        "what": "a binary trait beside a continuous one, against native SOLAR",
        "date": date.today().isoformat(),
        "people": n,
        "families": FAMILIES,
        "affected": int(affected.sum()),
        "truth": {
            "heritability": TRUE_HERITABILITY,
            "genetic_correlation": TRUE_GENETIC_CORRELATION,
            "residual_correlation": TRUE_RESIDUAL_CORRELATION,
        },
        "asterism": ours_named,
        "solar": theirs,
        "solar_used_the_discrete_model": discrete,
        "tolerance": TOLERANCE,
        "passed": not failures,
        "failures": failures,
    }
    out = Path(__file__).resolve().parent.parent / "evidence" / (
        f"mixed-bivariate-against-solar-{receipt['date']}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
