"""How long Asterism takes, against native SOLAR, on the same analysis.

Two numbers matter and they are very different.

The first is one analysis from a standing start: pedigree in, heritability out.
Both sides do the same work — build the relationship matrix, read the
phenotype, fit.

The second is the marginal cost of another trait on the same pedigree. SOLAR
starts again every time. Asterism decomposes once and fits against the stored
form, so a second trait costs almost nothing. The lab has 1,246 recorded SOLAR
runs, and most of them share a pedigree, so this is the number that decides how
long an afternoon takes.

Run with:

    uv run --no-project python checks/speed.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import asterism
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from against_r import extended_family, roster
from against_solar import RUN, SEX


def build(
    directory: Path, families: int, seed: int
) -> tuple[
    list[str],
    list[str | None],
    list[str | None],
    np.ndarray,
    np.ndarray,
    int,
]:
    """Create identical pedigree and phenotype inputs for both implementations.

    Args:
        directory: Temporary directory receiving SOLAR's input files.
        families: Number of extended pedigrees to simulate.
        seed: Seed controlling the shared simulated phenotype.

    Returns:
        Identifiers, parental identifiers, design, phenotype and sample size.
    """
    family: list[tuple[int | None, int | None]] = extended_family()
    """Loaded the canonical extended-family parental structure."""

    block: int = len(family)
    """Counted people in one copy of the extended pedigree."""

    n: int = families * block
    """Calculated the total sample size across replicated families."""

    ids, father, mother = [], [], []
    """Initialised identifiers and parental links for the replicated pedigree."""

    for f in range(families):
        for i, (mum, dad) in enumerate(family):
            ids.append(f"F{f:05d}_{i:02d}")
            father.append(None if dad is None else f"F{f:05d}_{dad:02d}")
            mother.append(None if mum is None else f"F{f:05d}_{mum:02d}")

    lines: list[str] = ["FAMID,ID,FA,MO,SEX"]
    """Initialised SOLAR's pedigree table with its required header."""

    for index, person in enumerate(ids):
        fa: str = father[index] or "0"
        """Rendered a missing father with SOLAR's zero sentinel."""

        mo: str = mother[index] or "0"
        """Rendered a missing mother with SOLAR's zero sentinel."""

        lines.append(f"{person[:6]},{person},{fa},{mo},{SEX[index % block]}")
        """Added one pedigree row with the shared identifier and sex coding."""

    (directory / "ped.csv").write_text("\n".join(lines) + "\n")

    age: np.ndarray = np.zeros(n)
    """Allocated the standardised age covariate for every simulated person."""

    male: np.ndarray = np.zeros(n)
    """Allocated the binary sex covariate for every simulated person."""

    for index in range(n):
        within: int = index % block
        """Located the person's role inside the repeated family template."""

        decade: float = 7.0 if within < 2 else (4.5 if within < 8 else 2.0)
        """Assigned the role-specific age decade before within-role jitter."""

        age[index] = (decade - 4.5) / 2.0 + ((index % 7) - 3.0) / 10.0
        """Standardised age and added deterministic within-role variation."""

        male[index] = 1.0 if SEX[within] == 1 else 0.0
        """Converted the shared pedigree sex code to the fitted binary covariate."""

    x: np.ndarray = np.column_stack(
        [np.ones(n), age, age**2, male, age * male, age**2 * male]
    )
    """Constructed the six-column fixed-effect design used by both fits."""

    k: np.ndarray = roster(families)
    """Constructed the relationship matrix for the replicated pedigree."""

    rng: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic phenotype generator for this timing case."""

    factor: np.ndarray = np.linalg.cholesky(0.5 * k + 0.5 * np.eye(n))
    """Factorised the equal genetic and residual phenotype covariance."""

    y: np.ndarray = x @ np.array(
        [2.0, 0.3, -0.1, 0.5, 0.05, -0.02]
    ) + factor @ rng.standard_normal(n)
    """Simulated the shared quantitative phenotype with six fixed effects."""

    rows: list[str] = ["ID,FAMID,age,sex,y"]
    """Initialised SOLAR's phenotype table with its required header."""

    for index, person in enumerate(ids):
        rows.append(
            f"{person},{person[:6]},{age[index]:.12f},{SEX[index % block]},{y[index]:.12f}"
        )
        """Added one high-precision phenotype and covariate row for SOLAR."""
    (directory / "phen.csv").write_text("\n".join(rows) + "\n")
    (directory / "run.tcl").write_text(RUN)
    return ids, father, mother, x, y, n


def main() -> int:
    if shutil.which("solar") is None:
        raise SystemExit(
            "solar is not on the path. This check fails rather than skips."
        )

    print("One heritability analysis, pedigree to result, six fixed effects.\n")
    print(
        f"{'n':>7} {'SOLAR':>10} {'Asterism':>10} {'ratio':>8} "
        f"{'  of which build':>17} {'prepare':>9} {'fit':>9} {'next trait':>11}"
    )

    for families, seed in ((25, 301), (100, 302), (400, 303)):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory: Path = Path(temporary_directory)
            """Resolved the isolated directory used for this SOLAR timing run."""

            ids, father, mother, x, y, n = build(directory, families, seed)
            """Built identical in-memory and on-disk inputs for both implementations."""

            started: float = time.perf_counter()
            """Captured the start of the native SOLAR analysis."""

            subprocess.run(
                ["solar"],
                stdin=(directory / "run.tcl").open(),
                capture_output=True,
                cwd=directory,
                check=False,
            )
            solar_seconds: float = time.perf_counter() - started
            """Measured SOLAR's complete pedigree-to-result elapsed time."""

            if not (directory / "out" / "polygenic.out").exists():
                raise SystemExit(f"SOLAR produced no result at n = {n}")

            started = time.perf_counter()
            """Captured the start of Asterism's matching pedigree-to-result analysis."""

            k, _order = asterism.relationship_matrix(ids, father, mother, keep=ids)
            """Built Asterism's relationship matrix in the input identifier order."""

            built: float = time.perf_counter()
            """Captured completion of Asterism's relationship-matrix construction."""

            model: asterism.PreparedModel = asterism.prepare(x, k)
            """Prepared the decomposed model reused by every phenotype fit."""

            prepared: float = time.perf_counter()
            """Captured completion of the reusable model preparation."""

            model.fit(y)
            finished: float = time.perf_counter()
            """Captured completion of Asterism's first phenotype fit."""

            # A second trait on the same pedigree: the preparation is already done,
            # so only the fit runs again.
            repeats: int = 20
            """Set the repeat count used to stabilise marginal-fit timing."""

            again: float = time.perf_counter()
            """Captured the start of repeated fits on the prepared model."""

            for _ in range(repeats):
                model.fit(y)
            marginal: float = (time.perf_counter() - again) / repeats
            """Calculated the mean cost of another trait on the prepared pedigree."""

            total: float = finished - started
            """Calculated Asterism's complete first-analysis elapsed time."""
            print(
                f"{n:>7} {solar_seconds:>9.2f}s {total:>9.3f}s {solar_seconds / total:>7.0f}x "
                f"{built - started:>16.3f}s {prepared - built:>8.3f}s "
                f"{finished - prepared:>8.3f}s {marginal:>10.4f}s"
            )

    print(
        "\nSOLAR's column is everything it does: load the pedigree, compute its own"
        "\nrelationship matrix, load the phenotype, and fit. Asterism's is the same"
        "\nwork through the builder, prepare and fit."
        "\n\nThe last column is what a second trait on the same pedigree costs, which"
        "\nis where the prepared model earns its place. SOLAR has no equivalent: it"
        "\nrepeats its whole column every time."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
