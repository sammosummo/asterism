"""The two-trait ML fit against native SOLAR.

This is the ML comparison for the two-trait model; the separate R `regress`
check covers REML. SOLAR's `polygenic` maximises the likelihood, so it provides
the like-for-like comparison here rather than for the REML default.

This is the check that matters most for two traits, because SOLAR reports the
genetic and residual correlations directly rather than as covariances that have
to be turned into ratios. Nothing is being re-expressed on the way to the
comparison, so a disagreement would be a disagreement about the answer.

**The data are unbalanced on purpose.** Every seventh person lacks the second
trait. Eigen-simplification per family block survives extra traits through the
Kronecker structure but breaks when people are missing different traits, so a
balanced check would pass without touching the part most likely to be wrong.

Run with:

    uv run --no-project python checks/bivariate_against_solar.py

It needs `solar` on the path and fails rather than skips when it is missing.

**The sex trap applies here as it does to one trait.** SOLAR knows the pedigree
and will silently overrule a sex contradicting a parental role, with no error. If
Asterism is still handed the original coding the two fits use different
covariates and disagree for a reason belonging to neither estimator. Sex is
derived from role and the result is checked against `pedindex.out`.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from asterism import _core

sys.path.insert(0, str(Path(__file__).parent))
from against_r import extended_family, roster
from against_solar import SEX

# SOLAR prints seven significant figures, so that is the most agreement it can
# demonstrate. The tolerance is set to what its printing supports rather than to
# a tighter number the comparison cannot see.
TOLERANCE = 5e-6

RUN = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait y1 y2
covariate age^1,2#sex
outdir out
polygenic
exit
"""


def build(directory: Path, families: int, truth: dict, seed: int):
    """Write the pedigree and phenotypes, and return what Asterism needs."""
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

    covariates = np.column_stack(
        [np.ones(n), age, age**2, male, age * male, age**2 * male]
    )

    h1, h2 = truth["h1"], truth["h2"]
    rg, re_ = truth["rg"], truth["re"]
    genetic = np.array([[h1, rg * np.sqrt(h1 * h2)], [rg * np.sqrt(h1 * h2), h2]])
    residual = np.array(
        [
            [1 - h1, re_ * np.sqrt((1 - h1) * (1 - h2))],
            [re_ * np.sqrt((1 - h1) * (1 - h2)), 1 - h2],
        ]
    )
    full = np.kron(genetic, k) + np.kron(residual, np.eye(n))
    rng = np.random.default_rng(seed)
    draw = np.linalg.cholesky(full + 1e-10 * np.eye(2 * n)) @ rng.standard_normal(2 * n)

    beta1 = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])
    beta2 = np.array([-1.0, 0.15, 0.08, -0.30, 0.02, 0.01])
    y1 = covariates @ beta1 + draw[:n]
    y2 = covariates @ beta2 + draw[n:]

    # Unbalanced: every seventh person lacks the second trait. This is the case
    # eigen-simplification cannot take, so it is the case worth checking.
    observed = np.array([[True, row % 7 != 0] for row in range(n)])

    rows = ["ID,FAMID,age,sex,y1,y2"]
    for f in range(families):
        for i in range(block):
            row = f * block + i
            second = f"{y2[row]:.12f}" if observed[row, 1] else ""
            rows.append(
                f"F{f:03d}_{i:02d},F{f:03d},{age[row]:.12f},{SEX[i]},"
                f"{y1[row]:.12f},{second}"
            )
    (directory / "phen.csv").write_text("\n".join(rows) + "\n")
    (directory / "run.tcl").write_text(RUN)

    # Asterism takes one row per observed person-trait, in person order.
    values, design = [], []
    for row in range(n):
        if observed[row, 0]:
            values.append(y1[row])
            design.append(np.concatenate([covariates[row], np.zeros(6)]))
        if observed[row, 1]:
            values.append(y2[row])
            design.append(np.concatenate([np.zeros(6), covariates[row]]))
    return k, observed, np.array(design), np.array(values), n


def check_sex_survived(directory: Path) -> None:
    """SOLAR rewrites a sex contradicting a parental role. Check rather than trust."""
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
                        f"SOLAR reassigned the sex of {name} from {sex} to "
                        f"{fields[3]}; the two fits would not share a design"
                    )
                break


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
    logs = directory / "out" / "polygenic.logs.out"
    if logs.exists():
        text += logs.read_text()

    def one(pattern: str) -> float:
        found = re.search(pattern, text)
        if found is None:
            raise SystemExit(f"SOLAR printed nothing matching {pattern!r}")
        return float(found.group(1))

    check_sex_survived(directory)
    return {
        "h2_first": one(r"H2r\(y1\) is\s+([0-9.eE+-]+)"),
        "h2_second": one(r"H2r\(y2\) is\s+([0-9.eE+-]+)"),
        "rho_g": one(r"RhoG is\s+(-?[0-9.eE+-]+)"),
        "rho_e": one(r"RhoE is\s+(-?[0-9.eE+-]+)"),
    }


def main() -> int:
    if shutil.which("solar") is None:
        raise SystemExit("solar is not on the path. This check fails rather than skips.")

    truth = {"h1": 0.6, "h2": 0.35, "rg": 0.55, "re": 0.25}
    families, seed = 25, 411
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        k, observed, design, values, n = build(directory, families, truth, seed)
        theirs = run_solar(directory)

        theta, loglik, gradient, converged = _core.bivariate_fit(
            np.ascontiguousarray(k),
            observed.tolist(),
            np.ascontiguousarray(design),
            np.ascontiguousarray(values),
            False,  # ML, because SOLAR's polygenic is ML
        )
    ours = {
        "h2_first": theta[2],
        "h2_second": theta[3],
        "rho_g": theta[4],
        "rho_e": theta[5],
    }

    print(
        f"Two traits, ML, against native SOLAR. {families} families, n = {n}, "
        f"{observed[:, 0].sum()} with the first trait and "
        f"{observed[:, 1].sum()} with the second.\n"
    )
    print(f"{'':<12}{'asterism':>13}{'solar':>13}{'difference':>13}{'truth':>8}")

    failures = []
    if not converged:
        failures.append("Asterism did not declare convergence")
    if not gradient < 1e-7:
        failures.append(f"Asterism scaled projected gradient was {gradient:.3e}")

    differences = {}
    labels = {
        "h2_first": "h2 first",
        "h2_second": "h2 second",
        "rho_g": "rho_g",
        "rho_e": "rho_e",
    }
    true_values = {
        "h2_first": truth["h1"],
        "h2_second": truth["h2"],
        "rho_g": truth["rg"],
        "rho_e": truth["re"],
    }
    for key, label in labels.items():
        difference = abs(ours[key] - theirs[key])
        differences[key] = difference
        print(
            f"{label:<12}{ours[key]:>13.7f}{theirs[key]:>13.7f}"
            f"{difference:>13.2e}{true_values[key]:>8.2f}"
        )
        if not difference < TOLERANCE:
            failures.append(f"{label} differs by {difference:.3e}")

    if failures:
        print("\nDISAGREEMENT:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"\nAgreement on every reported quantity, unbalanced, to {TOLERANCE:.0e}.")
    print(f"Asterism scaled projected gradient: {gradient:.3e}.")
    print("SOLAR reports the correlations directly, so nothing was re-expressed")
    print("on the way to this comparison.")
    print("The independently implemented ML calculations agree.")

    print(
        json.dumps(
            {
                "seed": seed,
                "families": families,
                "people": n,
                "observations_by_trait": [
                    int(observed[:, 0].sum()),
                    int(observed[:, 1].sum()),
                ],
                "balanced": False,
                "estimator": "ml",
                "truth": truth,
                "asterism": {
                    **ours,
                    "loglik": loglik,
                    "scaled_gradient": gradient,
                    "converged": converged,
                },
                "solar": theirs,
                "absolute_differences": differences,
                "comparison_tolerance": TOLERANCE,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
