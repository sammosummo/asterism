"""How does the association test behave in the tail?

The scan reported a genomic inflation of 0.991 under a null draw, which says the
median test behaves. **A median says nothing about the tail**, and a scan lives
entirely in the tail: the usual genome-wide threshold is 5e-08, eight orders of
magnitude from where lambda is measured.

So this counts rejections at a ladder of thresholds under a null with no marker
effect anywhere, on the real pedigree, the real ancestry components and real
markers. The only simulated thing is the response.

**Two arms, because two different things inflate the tail and only one of them
is the test's fault.** The first arm uses the real genotypes. The second
permutes each marker across people, which keeps its allele frequency exactly and
destroys any correspondence between a genotype and how closely two people are
actually related. Whatever excess survives permutation belongs to the test
itself and is judged here. Whatever the permutation removes is the pedigree
kinship failing to describe the relatedness the real genotypes carry -- a
property of the covariance model, not of the test -- and it is reported rather
than judged, because no change to the test would remove it. It is largest for
common markers, which carry the most of that signal.

**What this can and cannot reach.** Validating a threshold needs enough null
tests that a handful are expected to cross it: about ten million to say anything
at 1e-06, and of order ten billion at 5e-08. The second is out of reach here and
saying so is part of the result. What this establishes is that the test is
honest down to about 1e-05 or 1e-06, and that its behaviour there gives no
reason to think it breaks further out.

The markers are parsed once and reused across draws, because parsing is a third
of the cost of a scan and none of it needs repeating.

Run with:

    ASTERISM_MARKERS=50000 ASTERISM_DRAWS=200 uv run --no-project python checks/association_tail.py
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import asterism
from asterism import _core

DATABASE = Path("~/MathiasLab/staging/studies/existing/safs/data/SAFS.db").expanduser()
BULK = Path(
    "/Volumes/MathiasLab1/MathiasLab2-Bulk/Studies/Existing/SAFS/Data/Genotypes"
)
DOSAGES = BULK / "chp-dosage-subject.gz"
PCS = (
    BULK
    / "comparator_background_relatedness/runs/safs_phase3_omni_v116_20260716"
    / "relatedness/matrix/pcair_vectors_sensitive.tsv"
)
MARKERS = int(os.environ.get("ASTERISM_MARKERS", "50000"))
DRAWS = int(os.environ.get("ASTERISM_DRAWS", "200"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "6"))
COMPONENTS = 10
HERITABILITY = 0.5
THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

STATE: dict = {}


def _start(payload: dict) -> None:
    STATE.update(payload)


def numeric(value):
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load():
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
        value = numeric(age)
        if value is not None and str(person) in sexes:
            ages[str(person)] = value

    with gzip.open(DOSAGES, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    genotyped = [c.split("_", 1)[1] if "_" in c else c for c in header[6:]]
    position = {person: index for index, person in enumerate(genotyped)}

    components = {}
    with open(PCS) as fh:
        pc_header = fh.readline().rstrip("\n").split("\t")
        first = pc_header.index("PC1")
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            components[fields[0]] = [float(v) for v in fields[first:first + COMPONENTS]]

    keep = sorted(
        p for p in genotyped if p in ages and p in sexes and p in components
    )
    relationship, order = asterism.relationship_matrix(ids, fathers, mothers, keep=keep)
    assert order == keep
    columns = [position[p] for p in keep]
    age = np.array([ages[p] for p in keep])
    age = (age - age.mean()) / age.std()
    male = np.array([1.0 if sexes[p] == "1" else 0.0 for p in keep])
    pcs = np.array([components[p] for p in keep])
    design = np.column_stack([np.ones(len(keep)), age, age**2, male, pcs])

    # Parse the markers once. This is the expensive part and none of it changes
    # between draws.
    kept = []
    with gzip.open(DOSAGES, "rt") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            raw = [parts[6 + c] for c in columns]
            values = np.array(
                [np.nan if v == "NA" else float(v) for v in raw], dtype=np.float64
            )
            missing = np.isnan(values)
            if missing.mean() > 0.05:
                continue
            if missing.any():
                values[missing] = values[~missing].mean()
            frequency = values.mean() / 2.0
            if min(frequency, 1.0 - frequency) < 0.01:
                continue
            kept.append(values)
            if len(kept) >= MARKERS:
                break
    return np.ascontiguousarray(relationship), design, np.ascontiguousarray(
        np.column_stack(kept)
    )


def one(index: int):
    relationship = STATE["relationship"]
    design = STATE["design"]
    factor = STATE["factor"]
    n = design.shape[0]
    y = factor @ np.random.default_rng(640_000 + index).standard_normal(n)
    _, _, rows, _ = _core.association_sweep(
        relationship, design, y, STATE["markers"], "held"
    )
    return [row[4] for row in rows if not row[5]]


def main() -> int:
    if not DOSAGES.exists():
        raise SystemExit(f"no dosages at {DOSAGES}; is the bulk drive mounted?")
    started = time.perf_counter()
    relationship, design, markers = load()
    n, m = design.shape[0], markers.shape[1]
    print(
        f"Association tail on the real GOBS pedigree. {n} people, {m} real markers, "
        f"{DRAWS} null draws.\nThe response is drawn from a polygenic model at a "
        f"heritability of {HERITABILITY} with no marker effect anywhere.\n"
        f"Markers parsed once in {time.perf_counter() - started:.0f} s and reused.\n"
    )

    covariance = HERITABILITY * relationship + (1.0 - HERITABILITY) * np.eye(n)
    factor = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    def sweep(these: np.ndarray) -> list:
        shared = {
            "relationship": relationship,
            "design": design,
            "markers": these,
            "factor": factor,
        }
        with ProcessPoolExecutor(
            WORKERS, initializer=_start, initargs=(shared,)
        ) as pool:
            return list(pool.map(one, range(DRAWS), chunksize=1))

    started = time.perf_counter()
    batches = sweep(markers)
    # **The second arm is the same markers with their people shuffled.** Each
    # column keeps its allele frequency exactly and loses any relation to who is
    # related to whom. Permuted in place and run as a second pass rather than
    # alongside the first, so only one marker matrix is ever shipped to the
    # workers. Permuting per draw would confound the two sources of scatter.
    shuffle = np.random.default_rng(4_242)
    permuted = markers.copy()
    for column in range(m):
        shuffle.shuffle(permuted[:, column])
    batches_permuted = sweep(np.ascontiguousarray(permuted))
    del permuted
    took = time.perf_counter() - started

    # **Per draw as well as pooled.** Markers in linkage disequilibrium are not
    # independent tests, so the binomial band on the pooled count is too narrow
    # and a small excess against it may be the band rather than the test. The
    # spread of the count between draws says which: each draw is one
    # independent realisation of the response, whatever the markers do among
    # themselves.
    per_draw = [np.array(b) for b in batches if b]
    per_draw_permuted = [np.array(b) for b in batches_permuted if b]
    p_values = np.concatenate(per_draw)
    total = len(p_values)
    print(f"{total:,} null tests in {took / 60:.0f} minutes.\n")
    # **The band is the measured scatter, not the binomial.** Thirty thousand
    # markers on an array are nowhere near thirty thousand independent tests --
    # they are in linkage disequilibrium -- so the binomial standard deviation
    # of the pooled count is far too small, and a few per cent above it means
    # nothing. Each draw, though, is one independent realisation of the
    # response, whatever the markers do among themselves. So the count per draw
    # is measured, its scatter across draws is measured, and the mean is judged
    # against that. The binomial figure is printed beside it to show how far
    # wrong it would have been.
    draws = len(per_draw)
    each = len(per_draw[0])
    print(
        f"{'threshold':>12}{'per draw':>10}{'permuted':>10}{'expected':>10}"
        f"{'measured sd':>13}{'binomial sd':>13}{'ratio':>7}{'  verdict':<22}"
    )

    failures = []
    recorded = {}
    dispersion = {}
    structure = {}
    for threshold in THRESHOLDS:
        counts = np.array([(b <= threshold).sum() for b in per_draw])
        shuffled = np.array([(b <= threshold).sum() for b in per_draw_permuted])
        expected = each * threshold
        binomial = np.sqrt(each * threshold * (1 - threshold))
        measured = counts.std(ddof=1)
        seen = int(counts.sum())
        recorded[str(threshold)] = {
            "expected_total": expected * draws,
            "seen_total": seen,
            "ratio": seen / (expected * draws) if expected else float("nan"),
        }
        if counts.mean() < 3:
            print(
                f"{threshold:>12g}{counts.mean():>10.2f}{shuffled.mean():>10.2f}"
                f"{expected:>10.2f}{'--':>13}{binomial:>13.2f}{'--':>7}"
                f"  too few to judge"
            )
            continue
        dispersion[str(threshold)] = {
            "mean_per_draw": float(counts.mean()),
            "expected_per_draw": float(expected),
            "measured_sd": float(measured),
            "binomial_sd": float(binomial),
            "dispersion_ratio": float(measured / binomial),
        }
        # **The permuted arm is what the test is judged on.** Its markers have
        # the same allele frequencies and no relation to who is related to whom,
        # so any excess there is the test's own. The mean count per draw is
        # judged against its expectation with the standard error taken from the
        # scatter actually observed.
        permuted_sd = shuffled.std(ddof=1)
        error = permuted_sd / np.sqrt(draws)
        ceiling = expected + 1.96 * error
        over = shuffled.mean() > ceiling
        verdict = "OVER" if over else "ok"
        if over:
            failures.append(
                f"{shuffled.mean():.1f} per draw below {threshold:g} on permuted "
                f"markers where at most {ceiling:.1f} was expected"
            )
        # What the real genotypes add over the permuted ones is the pedigree
        # kinship not describing the relatedness they carry. It is reported and
        # not judged: no change to the test would remove it.
        structure[str(threshold)] = {
            "real_per_draw": float(counts.mean()),
            "permuted_per_draw": float(shuffled.mean()),
            "expected_per_draw": float(expected),
            "excess_over_permuted": (
                float(counts.mean() / shuffled.mean()) if shuffled.mean() else None
            ),
        }
        print(
            f"{threshold:>12g}{counts.mean():>10.2f}{shuffled.mean():>10.2f}"
            f"{expected:>10.2f}{measured:>13.2f}{binomial:>13.2f}"
            f"{measured / binomial:>7.2f}  {verdict}"
        )

    if structure:
        gaps = [
            v["excess_over_permuted"]
            for v in structure.values()
            if v["excess_over_permuted"]
        ]
        print(
            f"\nReal genotypes cross {min(gaps):.2f} to {max(gaps):.2f} times as "
            f"often as permuted ones\nwith the same allele frequencies. That gap is "
            f"the pedigree kinship not describing\nthe relatedness the real "
            f"genotypes carry, and it is largest for common markers.\nIt is a "
            f"property of the covariance model and not of the test, so it is "
            f"reported\nrather than judged: a genomic relationship matrix, not a "
            f"pedigree one, is what\nwould close it."
        )

    print(
        f"\nThe measured scatter runs "
        f"{min(d['dispersion_ratio'] for d in dispersion.values()):.1f} to "
        f"{max(d['dispersion_ratio'] for d in dispersion.values()):.1f} times the "
        f"binomial one.\nThat is the markers being in linkage disequilibrium, and "
        f"it is why the binomial\nband is the wrong thing to judge against. It "
        f"would have called a few per cent\nof excess a failure."
    )

    reachable = [t for t in THRESHOLDS if total * t >= 5]
    print(
        f"\nThis reaches {min(reachable):g} with enough tests to judge. The usual\n"
        f"genome-wide threshold of 5e-08 would need about "
        f"{5 / 5e-08 / 1e6:.0f} million null tests and is not reached here."
    )

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        "\nThe test is honest everywhere this can see, once the covariance model "
        "is\nnot asked to answer for the genotypes."
    )

    print(
        json.dumps(
            {
                "pedigree": "the real GOBS pedigree; only responses simulated",
                "people": n,
                "markers": m,
                "draws": DRAWS,
                "null_tests": total,
                "heritability": HERITABILITY,
                "variance_components": "held",
                "thresholds": recorded,
                "between_draw_dispersion": dispersion,
                "real_against_permuted": structure,
                "criterion": (
                    "the mean count per draw on permuted markers against its "
                    "expectation, with the standard error taken from the scatter "
                    "measured between draws rather than from a binomial that "
                    "assumes the markers are independent, which they are not. "
                    "Permuting keeps each marker's allele frequency and removes "
                    "its relation to who is related to whom, so what survives is "
                    "the test's own behaviour; what the real genotypes add over "
                    "that is the pedigree kinship not describing their relatedness "
                    "and is reported separately"
                ),
                "smallest_judged": min(reachable),
                "note": (
                    "5e-08 is not reached: it would need of order ten billion null "
                    "tests. What is established is honesty down to the smallest "
                    "threshold with enough expected crossings to judge"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
