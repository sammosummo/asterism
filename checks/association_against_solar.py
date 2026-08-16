"""Does the association test agree with native SOLAR?

SOLAR performs measured-genotype association by putting the marker in as a
covariate and fitting the polygenic model around it. This checks that model
against Asterism's separate implementation.

**The comparison is against `refitted`, not `held`.** SOLAR fits the variance
components with the marker in the design, so comparing it with the held-variance
sweep would show a difference that is the assumption rather than a fault. The
held sweep is what a scan uses and is fast; this check is about whether the
model underneath it is right, so it uses the mode that matches what SOLAR does.

The gap between the two modes is worth seeing as well, so both are reported and
only the matching one is required to agree.

Everything here is simulated and written to a temporary directory outside the
workspace. No real genotype or phenotype is written anywhere.

Run with:

    uv run --no-project python checks/association_against_solar.py

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

import numpy as np

from asterism import _core

sys.path.insert(0, str(Path(__file__).parent))
from against_r import extended_family, roster  # noqa: E402
from against_solar import SEX  # noqa: E402

RUN = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait y
covariate age sex marker
outdir out
polygenic -screen
exit
"""


def build(directory: Path, families: int, heritability: float, effect: float, seed: int):
    """Write a pedigree and a phenotype with one marker of known effect."""
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
    # A marker shared within families, as a real one is: the case where
    # relatedness and genotype are correlated is the whole point of the model.
    marker = np.zeros(n)
    for f in range(families):
        founder = rng.normal()
        for i in range(block):
            marker[f * block + i] = np.clip(
                round(founder + 0.7 * rng.normal() + 1.0), 0.0, 2.0
            )
    covariance = heritability * k + (1.0 - heritability) * np.eye(n)
    noise = np.linalg.cholesky(covariance + 1e-10 * np.eye(n)) @ rng.standard_normal(n)
    y = 2.0 + 0.02 * age + 0.4 * male + effect * marker + noise

    rows = ["ID,FAMID,age,sex,marker,y"]
    for f in range(families):
        for i in range(block):
            row = f * block + i
            rows.append(
                f"F{f:03d}_{i:02d},F{f:03d},{age[row]:.12f},{SEX[i]},"
                f"{marker[row]:.12f},{y[row]:.12f}"
            )
    (directory / "phen.csv").write_text("\n".join(rows) + "\n")
    (directory / "run.tcl").write_text(RUN)

    design = np.column_stack([np.ones(n), age, male])
    return np.ascontiguousarray(k), design, y, marker


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

    # **`polygenic -screen` is what tests a covariate.** Plain `polygenic`
    # fits it and says nothing about whether it matters, and the p-value has to
    # come from the screen, which drops each covariate and refits -- a
    # likelihood ratio, and so the mode Asterism should be compared in.
    p_value = re.search(r"marker\s+p\s*=\s*([0-9.eE+-]+)", text)
    # The estimate and its standard error live in the model output.
    model = directory / "out" / "poly.out"
    beta = None
    if model.exists():
        found = re.search(
            r"bmarker\s+(-?[0-9.eE+-]+)\s+([0-9.eE+-]+)", model.read_text()
        )
        if found:
            beta = float(found.group(1))
    h2 = re.search(r"H2r is\s+([0-9.eE+-]+)", text)
    if p_value is None:
        raise SystemExit(
            "SOLAR printed no p-value for the marker covariate; the output was:\n"
            + "\n".join(line for line in text.splitlines() if "marker" in line)[:1500]
        )
    return {
        "p_value": float(p_value.group(1)),
        "beta": beta,
        "h2": float(h2.group(1)) if h2 else None,
        "text": text,
    }


def main() -> int:
    if shutil.which("solar") is None:
        raise SystemExit("solar is not on the path. This check fails rather than skips.")

    cases = []
    for heritability, effect, seed in ((0.5, 0.0, 5_101), (0.5, 0.30, 5_102), (0.3, 0.55, 5_103)):
        directory = Path(tempfile.mkdtemp())
        try:
            k, design, y, marker = build(directory, 40, heritability, effect, seed)
            theirs = run_solar(directory)
            markers = np.ascontiguousarray(marker.reshape(-1, 1))
            _, _, refitted, _ = _core.association_sweep(
                k, design, y, markers, "refitted", None
            )
            _, _, held, _ = _core.association_sweep(k, design, y, markers, "held", None)
            r_effect, r_error, _, r_ratio, r_p, _, r_code = refitted[0]
            h_effect, _, h_wald, _, h_p, _, h_code = held[0]
            if r_code or h_code:
                raise SystemExit(f"Asterism refused the marker: {r_code or h_code}")
            cases.append(
                {
                    "true_heritability": heritability,
                    "true_effect": effect,
                    "people": len(y),
                    "solar_p": theirs["p_value"],
                    "solar_beta": theirs["beta"],
                    "solar_h2": theirs["h2"],
                    "asterism_refitted_p": r_p,
                    "asterism_refitted_effect": r_effect,
                    "asterism_refitted_statistic": r_ratio,
                    "asterism_held_p": h_p,
                    "asterism_held_effect": h_effect,
                    "asterism_held_statistic": h_wald,
                }
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print(
        "Association, Asterism against native SOLAR.\n"
        "SOLAR refits the variance components with the marker in the design, so\n"
        "`refitted` is the like-for-like mode. `held` is what a scan uses and is\n"
        "shown beside it so the cost of the assumption is visible.\n"
    )
    print(
        f"{'h2':>5}{'effect':>8}{'solar p':>12}{'refitted p':>13}{'held p':>12}"
        f"{'solar beta':>12}{'refitted':>11}"
    )
    worst = 0.0
    for case in cases:
        solar_p = case["solar_p"]
        ours = case["asterism_refitted_p"]
        # Compare on the log scale: a p-value is read as an order of magnitude.
        gap = abs(np.log10(max(ours, 1e-300)) - np.log10(max(solar_p, 1e-300)))
        worst = max(worst, gap)
        beta = "--" if case["solar_beta"] is None else f"{case['solar_beta']:.4f}"
        print(
            f"{case['true_heritability']:>5.2f}{case['true_effect']:>8.2f}"
            f"{solar_p:>12.3e}{ours:>13.3e}{case['asterism_held_p']:>12.3e}"
            f"{beta:>12}{case['asterism_refitted_effect']:>11.4f}"
        )
    print(f"\nworst disagreement: {worst:.3f} orders of magnitude in the p-value")

    bar = 0.15
    if worst > bar:
        print(f"\nNOT AGREED: {worst:.3f} exceeds {bar} orders of magnitude")
        return 1
    print("The independently implemented refitted association results agree.")

    print(
        json.dumps(
            {
                "compared": "refitted, because SOLAR refits with the marker present",
                "worst_log10_p_disagreement": worst,
                "threshold": bar,
                "cases": cases,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
