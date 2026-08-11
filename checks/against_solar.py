"""Compare Asterism's ML fit against native SOLAR.

`docs/adr/0006` puts correctness in external comparisons with a fixed division of
labour: SOLAR for ML, R `regress` for REML. This is the ML side. SOLAR's
`polygenic` maximises the likelihood, so it is the right comparator for the ML
estimator and the wrong one for the REML default — which is why the two checks
exist rather than one.

Run with:

    uv run --no-project python checks/against_solar.py

It needs `solar` on the path and fails rather than skips when it is missing.

**One trap, and it cost a wrong answer before it was found.** SOLAR knows the
pedigree, so it will silently overrule a sex that contradicts a parental role —
a founder recorded as male who appears as somebody's mother becomes female in
`pedindex.out`, with no error. If the design matrix handed to Asterism still
carries the original coding, the two fits then use different covariates and
disagree for a reason that has nothing to do with either estimator. Sex is
therefore derived from role here, once, and used for both sides.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import asterism
sys.path.insert(0, str(Path(__file__).parent))
from against_r import extended_family, roster  # noqa: E402

# SOLAR prints seven significant figures, so that is the most agreement it can
# demonstrate. `docs/adr/0006`'s 1e-6 relative tolerance is at the edge of what
# is observable here; the tolerance below is set to what SOLAR's own printing
# supports rather than to a tighter number the comparison cannot see.
H2_TOLERANCE = 5e-7
LOGLIK_TOLERANCE = 1e-6

# Sex by parental role. 0 is a mother and 1 a father; 2 to 4 are the mothers of
# the grandchildren and 5 to 7 their fathers; 8 to 13 are free.
SEX = [2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2]

RUN = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait y
covariate age^1,2#sex
outdir out
polygenic
exit
"""


def build(directory: Path, families: int, h2: float, seed: int):
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

    age = np.zeros(n)
    male = np.zeros(n)
    for f in range(families):
        for i in range(block):
            row = f * block + i
            decade = 7.0 if i < 2 else (4.5 if i < 8 else 2.0)
            age[row] = (decade - 4.5) / 2.0 + ((row % 7) - 3.0) / 10.0
            male[row] = 1.0 if SEX[i] == 1 else 0.0

    # The same span SOLAR's `age^1,2#sex` builds: age, age squared, sex, and the
    # two products. Whether SOLAR centres its columns does not matter — the
    # likelihood depends on the design only through its column span, and
    # centring does not change that.
    x = np.column_stack([np.ones(n), age, age**2, male, age * male, age**2 * male])
    beta = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(h2 * k + (1.0 - h2) * np.eye(n))
    y = x @ beta + factor @ rng.standard_normal(n)

    rows = ["ID,FAMID,age,sex,y"]
    for f in range(families):
        for i in range(block):
            row = f * block + i
            rows.append(
                f"F{f:03d}_{i:02d},F{f:03d},{age[row]:.12f},{SEX[i]},{y[row]:.12f}"
            )
    (directory / "phen.csv").write_text("\n".join(rows) + "\n")
    (directory / "run.tcl").write_text(RUN)
    return k, x, y, n


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
    logs = directory / "out" / "polygenic.logs.out"
    if not out.exists():
        raise SystemExit(f"SOLAR produced no result:\n{finished.stdout}\n{finished.stderr}")
    text = out.read_text() + logs.read_text()

    h2 = float(re.search(r"H2r is\s+([0-9.eE+-]+)", text).group(1))
    se = float(re.search(r"H2r Std\. Error:\s+([0-9.eE+-]+)", text).group(1))
    poly = float(
        re.search(r"Loglikelihood of polygenic model is\s+(-?[0-9.eE+-]+)", text).group(1)
    )
    sporadic = float(
        re.search(r"Loglikelihood of sporadic model is\s+(-?[0-9.eE+-]+)", text).group(1)
    )
    # SOLAR silently rewrites a sex that contradicts a parental role. If it did,
    # the covariates are not the ones Asterism was given and the comparison is
    # void, so check rather than trust.
    declared = {
        line.split(",")[1]: line.split(",")[4]
        for line in (directory / "ped.csv").read_text().splitlines()[1:]
    }
    for line in (directory / "pedindex.out").read_text().splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        person = fields[-1]
        for name, sex in declared.items():
            if person.endswith(name):
                if fields[3] != sex:
                    raise SystemExit(
                        f"SOLAR reassigned the sex of {name} from {sex} to {fields[3]}; "
                        "the two fits would not share a design"
                    )
                break
    return {"h2": h2, "se": se, "loglik": poly, "sporadic_loglik": sporadic}


def main() -> int:
    if shutil.which("solar") is None:
        raise SystemExit("solar is not on the path. This check fails rather than skips.")

    print("Asterism against native SOLAR, ML, 25 families, six fixed effects.\n")
    print(f"{'truth':>6} {'asterism h2':>12} {'solar h2':>12} {'rel':>10} "
          f"{'se ours':>9} {'se solar':>9} {'loglik offset':>15} {'unexplained':>12}")

    results = []
    failures = []
    for h2, seed in ((0.5, 202), (0.2, 203), (0.7, 204)):
        directory = Path(tempfile.mkdtemp())
        k, x, y, n = build(directory, 25, h2, seed)
        theirs = run_solar(directory)
        ours = asterism.prepare(x, k).fit(y, "ml")

        relative = abs(ours["h2"] - theirs["h2"]) / (1.0 + abs(theirs["h2"]))
        # ML is a density for n observations, so the constant SOLAR drops is
        # n/2 * log(2*pi). REML drops (n - p)/2 * log(2*pi) instead, which is
        # why `checks/against_r.py` expects a different number.
        expected = 0.5 * n * math.log(2.0 * math.pi)
        offset = ours["loglik"] - theirs["loglik"]
        unexplained = abs(offset + expected)

        results.append(
            {
                "true_h2": h2,
                "seed": seed,
                "n": n,
                "asterism": {
                    "h2": ours["h2"],
                    "se": (ours["standard_errors"] or {}).get("h2"),
                    "loglik": ours["loglik"],
                },
                "solar": theirs,
                "h2_relative": relative,
                "loglik_offset": offset,
                "loglik_offset_expected": -expected,
                "loglik_offset_unexplained": unexplained,
            }
        )
        print(
            f"{h2:>6.2f} {ours['h2']:>12.7f} {theirs['h2']:>12.7f} {relative:>10.2e} "
            f"{(ours['standard_errors'] or {}).get('h2', float('nan')):>9.5f} "
            f"{theirs['se']:>9.5f} {offset:>15.6f} {unexplained:>12.2e}"
        )
        if not relative < H2_TOLERANCE:
            failures.append(f"true h2 {h2}: heritability differs by {relative:.3e}")
        if not unexplained < LOGLIK_TOLERANCE:
            failures.append(
                f"true h2 {h2}: the log-likelihood offset is not n/2*log(2*pi); "
                f"{unexplained:.3e} unaccounted for"
            )

    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/against-solar-2026-08-11.json").write_text(
        json.dumps({"results": results}, indent=2) + "\n"
    )

    if failures:
        print("\nDISAGREEMENT:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nAgreement to every digit SOLAR prints.")
    print("This proves fidelity, not correctness (`docs/adr/0006`).")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main())
