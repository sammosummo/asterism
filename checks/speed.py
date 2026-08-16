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

import numpy as np

import asterism

sys.path.insert(0, str(Path(__file__).parent))
from against_r import extended_family, roster  # noqa: E402
from against_solar import RUN, SEX  # noqa: E402


def build(directory: Path, families: int, seed: int):
    """The same data on disk for SOLAR and in memory for Asterism."""
    family = extended_family()
    block = len(family)
    n = families * block

    ids, father, mother = [], [], []
    for f in range(families):
        for i, (mum, dad) in enumerate(family):
            ids.append(f"F{f:05d}_{i:02d}")
            father.append(None if dad is None else f"F{f:05d}_{dad:02d}")
            mother.append(None if mum is None else f"F{f:05d}_{mum:02d}")

    lines = ["FAMID,ID,FA,MO,SEX"]
    for index, person in enumerate(ids):
        fa = father[index] or "0"
        mo = mother[index] or "0"
        lines.append(f"{person[:6]},{person},{fa},{mo},{SEX[index % block]}")
    (directory / "ped.csv").write_text("\n".join(lines) + "\n")

    age = np.zeros(n)
    male = np.zeros(n)
    for index in range(n):
        within = index % block
        decade = 7.0 if within < 2 else (4.5 if within < 8 else 2.0)
        age[index] = (decade - 4.5) / 2.0 + ((index % 7) - 3.0) / 10.0
        male[index] = 1.0 if SEX[within] == 1 else 0.0
    x = np.column_stack([np.ones(n), age, age**2, male, age * male, age**2 * male])

    k = roster(families)
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(0.5 * k + 0.5 * np.eye(n))
    y = x @ np.array([2.0, 0.3, -0.1, 0.5, 0.05, -0.02]) + factor @ rng.standard_normal(n)

    rows = ["ID,FAMID,age,sex,y"]
    for index, person in enumerate(ids):
        rows.append(
            f"{person},{person[:6]},{age[index]:.12f},{SEX[index % block]},{y[index]:.12f}"
        )
    (directory / "phen.csv").write_text("\n".join(rows) + "\n")
    (directory / "run.tcl").write_text(RUN)
    return ids, father, mother, x, y, n


def main() -> int:
    if shutil.which("solar") is None:
        raise SystemExit("solar is not on the path. This check fails rather than skips.")

    print("One heritability analysis, pedigree to result, six fixed effects.\n")
    print(f"{'n':>7} {'SOLAR':>10} {'Asterism':>10} {'ratio':>8} "
          f"{'  of which build':>17} {'prepare':>9} {'fit':>9} {'next trait':>11}")

    for families, seed in ((25, 301), (100, 302), (400, 303)):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            ids, father, mother, x, y, n = build(directory, families, seed)

            started = time.perf_counter()
            subprocess.run(
                ["solar"],
                stdin=(directory / "run.tcl").open(),
                capture_output=True,
                cwd=directory,
                check=False,
            )
            solar_seconds = time.perf_counter() - started
            if not (directory / "out" / "polygenic.out").exists():
                raise SystemExit(f"SOLAR produced no result at n = {n}")

            started = time.perf_counter()
            k, order = asterism.relationship_matrix(ids, father, mother, keep=ids)
            built = time.perf_counter()
            model = asterism.prepare(x, k)
            prepared = time.perf_counter()
            model.fit(y)
            finished = time.perf_counter()

            # A second trait on the same pedigree: the preparation is already done,
            # so only the fit runs again.
            repeats = 20
            again = time.perf_counter()
            for _ in range(repeats):
                model.fit(y)
            marginal = (time.perf_counter() - again) / repeats

            total = finished - started
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
