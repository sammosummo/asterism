"""Do the kinship-class tests hold their level?

The class-weighted kinship model splits the direct parent-offspring cells of the
relationship matrix into four classes and asks whether they carry the same
variance. On the GOBS traits it says they do not for both amyloid biomarkers,
with mother-son unusually low and mother-daughter unusually high in each. That is
the most interesting thing the session found and it rests on two tests nobody
has checked.

**Two tests, and they are not the same question.** The omnibus pools every class
back together and asks whether splitting bought anything, on `k - 1` degrees of
freedom. Each contrast pools one class against everything else, on one. The
omnibus is what protects the contrasts from being read as four independent
findings, so if the omnibus does not hold its level nothing downstream does.

**This calibrates on the real pedigree and the real class split.** The classes
are badly unbalanced in GOBS -- far more measured mothers than fathers -- and a
balanced simulation would say nothing about the class that rests on the fewest
pairs and is most likely to mislead.

**The criterion is validity, not uniformity.** Every class variance is bounded
below at nought, so under the null a share of fits sit on a bound and the
statistic has an atom. What must hold is that the rejection rate does not exceed
its level.

Run with:

    ASTERISM_REPLICATES=200 uv run --no-project python checks/kinship_equality_calibration.py

It reads the GOBS pedigree, which is study material, and simulates every
response it uses. No observed phenotype is read.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import asterism

DATABASE = Path(
    "~/MathiasLab/staging/studies/existing/safs/data/SAFS.db"
).expanduser()
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "200"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "6"))
LEVELS = (0.01, 0.05, 0.10)
# The share of variance the additive component carries under the null, near
# what the real traits gave.
ADDITIVE = 0.5
# Under the alternative, one class carries this much more. Mother-daughter is
# the class the amyloids put high, so it is the one moved.
LIFTED = "mother_daughter"
LIFT = 0.5

STATE: dict = {}


def _start(payload: dict) -> None:
    STATE.update(payload)


def structure():
    """The real GOBS pedigree, split by class of parent-offspring tie."""
    db = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    pedigree = list(db.execute("select id, fa, mo, sex from pedigree"))
    ids = [str(p) for p, _, _, _ in pedigree]
    fathers = [None if str(f) in ("0", "", "None") else str(f) for _, f, _, _ in pedigree]
    mothers = [None if str(m) in ("0", "", "None") else str(m) for _, _, m, _ in pedigree]
    sexes = {str(p): str(s) for p, _, _, s in pedigree}
    ages = {}
    for person, age in db.execute(
        "select subject_id, gobs_age_at_consent_years from gobs_demographics"
    ):
        try:
            value = float(str(age).strip())
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and str(person) in sexes:
            ages[str(person)] = value

    keep = sorted(ages)
    split = asterism.kinship_classes(
        ids, fathers, mothers, [sexes.get(p) for p in ids], keep=keep
    )
    order = split["order"]
    matrices = [np.ascontiguousarray(m) for m in split["matrices"]]
    age = np.array([ages[p] for p in order])
    age = (age - age.mean()) / age.std()
    male = np.array([1.0 if sexes[p] == "1" else 0.0 for p in order])
    design = np.column_stack(
        [np.ones(len(order)), age, age**2, male, age * male, age**2 * male]
    )
    return matrices, split["class_names"], split["pairs"], design


def one(job):
    scenario, index = job
    matrices = STATE["matrices"]
    design = STATE["design"]
    names = STATE["names"]
    factor = STATE["factors"][scenario]
    n = design.shape[0]
    y = factor @ np.random.default_rng(770_000 + index).standard_normal(n)

    model = asterism.ComponentModel(matrices, design)
    out = {"scenario": scenario, "contrasts": {}}
    try:
        out["omnibus"] = model.equality_test(y)["p_value"]
    except Exception:
        out["omnibus"] = None
    for component, name in enumerate(names, start=1):
        others = [m for index_, m in enumerate(matrices) if index_ != component]
        pooled = np.ascontiguousarray(sum(others[1:], others[0]))
        try:
            pair = asterism.ComponentModel([matrices[component], pooled], design)
            out["contrasts"][name] = pair.equality_test(y, [0, 1])["p_value"]
        except Exception:
            out["contrasts"][name] = None
    return out


def main() -> int:
    matrices, names, pairs, design = structure()
    n = design.shape[0]
    whole = sum(matrices[1:], matrices[0])

    factors = {}
    # Null: one additive variance for everything, which is the ordinary model.
    covariance = ADDITIVE * whole + (1.0 - ADDITIVE) * np.eye(n)
    factors["null"] = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    # Alternative: one class carries more than the rest.
    lifted = ADDITIVE * whole + LIFT * matrices[1 + list(names).index(LIFTED)]
    covariance = lifted + (1.0 - ADDITIVE) * np.eye(n)
    factors["lifted"] = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))

    print(
        f"Kinship-class tests on the real GOBS pedigree, REML. {n} people.\n"
        f"Pairs per class: " + ", ".join(f"{n_} {c}" for c, n_ in pairs.items()) + ".\n"
        f"Null: one additive variance for every class. "
        f"Alternative: {LIFTED} carries {LIFT} more.\n"
        f"{REPLICATES} replicates; only the response is simulated.\n"
    )

    shared = {"matrices": matrices, "design": design, "names": names, "factors": factors}
    jobs = [(s, i) for s in ("null", "lifted") for i in range(REPLICATES)]
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS, initializer=_start, initargs=(shared,)) as pool:
        results = list(pool.map(one, jobs, chunksize=1))
    print(f"{len(results)} of {len(jobs)} in "
          f"{(time.perf_counter() - started) / 60:.0f} minutes.\n")

    failures: list[str] = []
    recorded: dict = {}
    for scenario in ("null", "lifted"):
        got = [r for r in results if r["scenario"] == scenario]
        is_null = scenario == "null"
        print(f"{scenario}  --  {'level' if is_null else 'power'}")
        print(f"  {'test':<20}" + "".join(f"{f'p<={l:g}':>10}" for l in LEVELS) + f"{'of':>6}")
        recorded[scenario] = {}
        rows = [("omnibus", [r["omnibus"] for r in got])] + [
            (name, [r["contrasts"][name] for r in got]) for name in names
        ]
        for label, values in rows:
            have = [v for v in values if v is not None]
            if not have:
                failures.append(f"{scenario}: nothing computed for {label}")
                continue
            rates = [float(np.mean([v <= level for v in have])) for level in LEVELS]
            recorded[scenario][label] = {
                "computed": len(have),
                "rejection": {str(l): r for l, r in zip(LEVELS, rates)},
            }
            marks = []
            for level, rate in zip(LEVELS, rates):
                if not is_null:
                    marks.append(f"{rate:>10.3f}")
                    continue
                error = np.sqrt(level * (1 - level) / len(have))
                ceiling = level + 1.96 * error
                marks.append(f"{rate:>9.3f}{'!' if rate > ceiling else ' '}")
                if rate > ceiling:
                    failures.append(
                        f"{label} rejects {rate:.3f} at {level:g}, above {ceiling:.3f}"
                    )
            print(f"  {label:<20}" + "".join(marks) + f"{len(have):>6}")
        print()

    print("A `!` marks a rejection rate above its level's one-sided binomial ceiling.")
    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nThe omnibus and every contrast hold their level on the real pedigree,")
    print("with its real and badly unbalanced classes.")

    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/kinship-equality-calibration-2026-08-14.json").write_text(
        json.dumps(
            {
                "what": "level and power of the kinship-class equality tests",
                "date": "2026-08-14",
                "estimator": "reml",
                "pedigree": "the real GOBS pedigree; only responses simulated",
                "people": n,
                "pairs": pairs,
                "additive_share": ADDITIVE,
                "lifted_class": LIFTED,
                "lift": LIFT,
                "replicates": REPLICATES,
                "levels": list(LEVELS),
                "results": recorded,
                "note": (
                    "the omnibus is what protects the contrasts; read a contrast only "
                    "where the omnibus rejects"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
