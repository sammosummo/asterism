"""Compare Asterism's multi-component fit against native SOLAR, and pin down
exactly how far SOLAR can be used as a comparator for the spatial model.

Every other SOLAR comparison in this directory checks a model that needs no
matrix beyond the one SOLAR builds from the pedigree. The component and spatial
families had no external comparison at all: their evidence was simulation
against their own generator, which cannot catch a fault shared by the simulator
and the fitter. This check closes as much of that gap as SOLAR permits, and
measures the rest so that nobody later assumes it was closed.

**The finding that shapes this check.** SOLAR evaluates the likelihood one
pedigree at a time, so a loaded matrix contributes only its within-pedigree
blocks: every covariance between individuals in *different* pedigrees is
silently discarded. `matrix debug` still reports the whole file, minimum and
maximum included, so nothing warns you. Two consequences:

- With a block-diagonal matrix the two implementations are comparing the same
  model, and they agree to every digit SOLAR prints. That is a real external
  check of the component likelihood, its optimiser and its variances.
- With a matrix whose covariance crosses pedigrees — which is what a spatial
  kernel is for — SOLAR is fitting a *different, truncated* model. It cannot
  serve as a comparator there, and the disagreement is SOLAR's structure rather
  than either implementation being wrong.

The second part is asserted rather than described, so the limitation stays a
tested fact. What consequently remains simulation-only is the genuinely
cross-family spatial covariance, the estimated decay rate, and the bootstrap.

Run with:

    uv run --no-project python checks/spatial_against_solar.py

It needs `solar` on the path and fails rather than skips when it is missing.

One further trap, inherited from `against_solar.py`: SOLAR silently overrules a
sex that contradicts a parental role, which would leave the two fits using
different covariates. Sex is derived from role here and checked afterwards
against `pedindex.out`.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import asterism
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from against_r import extended_family, roster
from against_solar import SEX

# SOLAR prints seven significant figures for its parameters, and the variances
# here are order one. The tolerances follow that printed precision rather than a
# tighter number the comparison cannot observe.
VARIANCE_TOLERANCE = 5e-6
LOGLIK_TOLERANCE = 1e-4

# The square the population is scattered over, and the decay rate held fixed
# while both implementations estimate variances. Holding the rate fixed is what
# makes the comparison possible at all: `exp(-lambda * D)` is then an ordinary
# known matrix, and SOLAR cannot estimate a parameter that enters the covariance
# non-linearly.
SQUARE_KM = 10.0
LAMBDA_PER_KM = 0.6

FAMILIES = 25

# `model new` clears loaded matrices, so the spatial matrix is loaded after it.
# `polygsd` builds the standard model in SOLAR's standard-deviation
# parameterisation — mean, covariate betas, `esd` and `gsd` — and the omega is
# then overridden to add the third term, which is how SOLAR's own `linkqsd`
# adds a linkage element.
RUN = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait y
covariate age^1,2#sex
polygsd
matrix load {matrix} spa
parameter ssd = 0.4 lower 0 upper 100
omega = I*esd*esd + phi2*gsd*gsd + spa*ssd*ssd
outdir out
maximize
exit
"""


def spatial_kernel(place: np.ndarray) -> np.ndarray:
    difference = place[:, None, :] - place[None, :, :]
    distance = np.sqrt((difference**2).sum(axis=2))
    return np.exp(-LAMBDA_PER_KM * distance)


def within_families(kernel: np.ndarray, block: int) -> np.ndarray:
    """The kernel with every cross-pedigree entry removed.

    Each diagonal block is a principal submatrix of a positive-definite matrix
    and so is itself positive definite, which makes the block-diagonal result a
    valid covariance in its own right rather than a mutilated one.
    """
    restricted = np.zeros_like(kernel)
    for start in range(0, kernel.shape[0], block):
        stop = start + block
        restricted[start:stop, start:stop] = kernel[start:stop, start:stop]
    return restricted


def write_matrix(directory: Path, name: str, identifiers, families, matrix) -> None:
    """SOLAR wants a gzipped CSV holding the lower triangle, with every diagonal
    present. `exp(0)` is one, so the diagonal requirement is met by the kernel
    itself rather than by patching values in."""
    rows = ["id1,id2,famid1,famid2,matrix1"]
    for i in range(len(identifiers)):
        for j in range(i + 1):
            rows.append(
                f"{identifiers[i]},{identifiers[j]},"
                f"{families[i]},{families[j]},{matrix[i, j]:.12f}"
            )
    payload = ("\n".join(rows) + "\n").encode()
    (directory / name).write_bytes(gzip.compress(payload))


def build(directory: Path, variances, seed: int):
    """Write SOLAR's inputs and return the matching Asterism inputs."""
    family = extended_family()
    block = len(family)
    k = roster(FAMILIES)
    n = k.shape[0]

    identifiers, family_labels = [], []
    lines = ["FAMID,ID,FA,MO,SEX"]
    for f in range(FAMILIES):
        for i, (mother, father) in enumerate(family):
            fa = "0" if father is None else f"F{f:03d}_{father:02d}"
            mo = "0" if mother is None else f"F{f:03d}_{mother:02d}"
            lines.append(f"F{f:03d},F{f:03d}_{i:02d},{fa},{mo},{SEX[i]}")
            identifiers.append(f"F{f:03d}_{i:02d}")
            family_labels.append(f"F{f:03d}")
    (directory / "ped.csv").write_text("\n".join(lines) + "\n")

    # Positions are drawn independently of family, so the kernel is not a
    # disguised family indicator. Were people placed by family, the kernel would
    # be nearly collinear with the kinship matrix and the two variances would
    # trade off freely, which would hide a real disagreement.
    place = np.random.default_rng(seed + 1).uniform(0.0, SQUARE_KM, size=(n, 2))
    dense = spatial_kernel(place)
    restricted = within_families(dense, block)
    write_matrix(directory, "dense.csv.gz", identifiers, family_labels, dense)
    write_matrix(directory, "blocks.csv.gz", identifiers, family_labels, restricted)

    age = np.zeros(n)
    male = np.zeros(n)
    for f in range(FAMILIES):
        for i in range(block):
            row = f * block + i
            decade = 7.0 if i < 2 else (4.5 if i < 8 else 2.0)
            age[row] = (decade - 4.5) / 2.0 + ((row % 7) - 3.0) / 10.0
            male[row] = 1.0 if SEX[i] == 1 else 0.0

    x = np.column_stack([np.ones(n), age, age**2, male, age * male, age**2 * male])
    beta = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])

    genetic, spatial_variance, residual = variances
    covariance = genetic * k + spatial_variance * restricted + residual * np.eye(n)
    rng = np.random.default_rng(seed)
    y = x @ beta + np.linalg.cholesky(covariance) @ rng.standard_normal(n)

    rows = ["ID,FAMID,age,sex,y"]
    for f in range(FAMILIES):
        for i in range(block):
            row = f * block + i
            rows.append(
                f"F{f:03d}_{i:02d},F{f:03d},{age[row]:.12f},{SEX[i]},{y[row]:.12f}"
            )
    (directory / "phen.csv").write_text("\n".join(rows) + "\n")
    return k, dense, restricted, x, y, n


def run_solar(directory: Path, matrix_file: str) -> dict:
    (directory / "run.tcl").write_text(RUN.format(matrix=matrix_file))
    finished = subprocess.run(
        ["solar"],
        stdin=(directory / "run.tcl").open(),
        capture_output=True,
        text=True,
        cwd=directory,
        check=False,
    )
    out = directory / "out"
    text = finished.stdout + finished.stderr
    # `last.mod` records the model as it stood before maximising, so the fitted
    # values come from the optimisation report rather than from it.
    if not (out / "solar.out").exists():
        raise SystemExit(f"SOLAR produced no optimisation report:\n{text}")
    report = (out / "solar.out").read_text()

    def parameter(name: str) -> float:
        found = re.search(
            rf"^\s+{name}\s+(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)\s", report, re.M
        )
        if found is None:
            raise SystemExit(f"SOLAR did not report {name}:\n{report}")
        return float(found.group(1))

    loglik = re.search(r"Loglikelihood\s*=\s*(-?[0-9.eE+-]+)", report)
    iterations = re.search(r"Iterations\s*=\s*([0-9]+)", report)
    if loglik is None or iterations is None or int(iterations.group(1)) < 1:
        raise SystemExit(f"SOLAR did not complete a maximisation:\n{report}")
    analysed = re.search(r"sample size including probands used is\s+([0-9]+)", report)
    if analysed is None:
        raise SystemExit(f"SOLAR did not report its sample size:\n{report}")

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

    return {
        "genetic": parameter("gsd") ** 2,
        "spatial": parameter("ssd") ** 2,
        "residual": parameter("esd") ** 2,
        "loglik": float(loglik.group(1)),
        "iterations": int(iterations.group(1)),
        "people": int(analysed.group(1)),
    }


def compare(label, ours, theirs, n, failures, results, extra=None):
    """Report the two fits side by side. SOLAR drops the n/2*log(2*pi) that a
    density carries, exactly as in `against_solar.py`."""
    mine = {
        "genetic": ours["variances"][0],
        "spatial": ours["variances"][1],
        "residual": ours["variances"][2],
    }
    print(f"  {'component':>10} {'asterism':>12} {'solar':>12} {'abs':>10}")
    for component in ("genetic", "spatial", "residual"):
        difference = abs(mine[component] - theirs[component])
        print(
            f"  {component:>10} {mine[component]:>12.7f} "
            f"{theirs[component]:>12.7f} {difference:>10.2e}"
        )
        if not difference < VARIANCE_TOLERANCE:
            failures.append(f"{label}: {component} differs by {difference:.3e}")

    expected = 0.5 * n * math.log(2.0 * math.pi)
    unexplained = abs((ours["loglik"] - theirs["loglik"]) + expected)
    print(
        f"  {'loglik':>10} {ours['loglik']:>12.4f} "
        f"{theirs['loglik'] - expected:>12.4f} {unexplained:>10.2e}"
        "   (SOLAR shown with the density constant restored)\n"
    )
    if not unexplained < LOGLIK_TOLERANCE:
        failures.append(
            f"{label}: the log-likelihood offset is not n/2*log(2*pi); "
            f"{unexplained:.3e} unaccounted for"
        )
    record = {
        "case": label,
        "n": n,
        "asterism": mine | {"loglik": ours["loglik"]},
        "solar": theirs,
        "loglik_offset_unexplained": unexplained,
    }
    results.append(record | (extra or {}))


def main() -> int:
    if shutil.which("solar") is None:
        raise SystemExit("solar is not on the path. This check fails rather than skips.")

    print(
        "Asterism's component model against native SOLAR, ML, "
        f"{FAMILIES} families, six fixed effects,\n"
        f"kernel exp(-{LAMBDA_PER_KM} * km) held fixed so it is an ordinary known "
        "matrix.\n"
    )

    results: list = []
    failures: list = []

    for variances, seed in (((0.5, 0.3, 0.7), 301), ((0.2, 0.6, 0.9), 302)):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            k, dense, restricted, x, y, n = build(directory, variances, seed)
            block_solar = run_solar(directory, "blocks.csv.gz")
            dense_solar = run_solar(directory, "dense.csv.gz")

        truth = "/".join(f"{v:g}" for v in variances)

        # The comparison SOLAR can actually make: a matrix it can represent.
        print(f"Within-pedigree kernel, truth {truth} — the models are the same:")
        ours = asterism.ComponentModel([k, restricted], x).fit(y, reml=False)
        if block_solar["people"] != n:
            failures.append(
                f"truth {truth}: SOLAR analysed {block_solar['people']} of {n} people"
            )
        compare(f"truth {truth}, within-pedigree", ours, block_solar, n, failures, results,
                {"true_variances": list(variances), "seed": seed})

        # The comparison SOLAR silently cannot make. Given the dense kernel it
        # returns the within-pedigree answer, so this asserts the truncation
        # rather than merely warning about it.
        print(
            f"Dense kernel, truth {truth} — SOLAR is handed covariance that\n"
            "crosses pedigrees and quietly drops it:"
        )
        dense_ours = asterism.ComponentModel([k, dense], x).fit(y, reml=False)
        print(
            f"  {'component':>10} {'asterism':>12} {'solar':>12}\n"
            f"  {'genetic':>10} {dense_ours['variances'][0]:>12.7f} "
            f"{dense_solar['genetic']:>12.7f}\n"
            f"  {'spatial':>10} {dense_ours['variances'][1]:>12.7f} "
            f"{dense_solar['spatial']:>12.7f}\n"
            f"  {'residual':>10} {dense_ours['variances'][2]:>12.7f} "
            f"{dense_solar['residual']:>12.7f}\n"
        )
        truncated = max(
            abs(block_solar[c] - dense_solar[c])
            for c in ("genetic", "spatial", "residual")
        )
        if not truncated < VARIANCE_TOLERANCE:
            failures.append(
                f"truth {truth}: SOLAR given the dense kernel no longer matches its "
                f"own within-pedigree fit ({truncated:.3e}); the truncation this "
                "check documents has changed and the comment above is now wrong"
            )
        separation = max(
            abs(dense_ours["variances"][i] - dense_solar[c])
            for i, c in enumerate(("genetic", "spatial", "residual"))
        )
        if not separation > VARIANCE_TOLERANCE:
            failures.append(
                f"truth {truth}: the dense and truncated fits agree to "
                f"{separation:.3e}, so this design no longer demonstrates anything"
            )
        results.append(
            {
                "case": f"truth {truth}, dense kernel",
                "solar_equals_its_own_within_pedigree_fit_to": truncated,
                "asterism_dense_differs_from_solar_by": separation,
            }
        )

    print(json.dumps({"results": results}, indent=2))

    if failures:
        print("\nDISAGREEMENT:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        "\nWith a matrix SOLAR can represent, the component model agrees with it to\n"
        "every digit SOLAR prints. That is the external comparison the component\n"
        "and spatial families previously had none of.\n\n"
        "With covariance that crosses pedigrees, SOLAR returns its within-pedigree\n"
        "answer without warning, so it is not a comparator for the spatial kernel\n"
        "as such. Cross-family spatial covariance, the estimated decay rate and the\n"
        "bootstrap remain checked by simulation only."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
