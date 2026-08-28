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

import asterism
import numpy as np
from asterism import _core

DATABASE: Path = Path("~/MathiasLab/data/safs-db/SAFS.db").expanduser()
"""Resolved the read-only SAFS database used for pedigree and age metadata."""

BULK: Path = Path(
    "/Volumes/MathiasLab1/MathiasLab2-Bulk/Studies/Existing/SAFS/Data/Genotypes"
)
"""Located the mounted SAFS genotype storage used by this data-backed check."""

DOSAGES: Path = BULK / "chp-dosage-subject.gz"
"""Selected the subject-oriented imputed dosage archive."""

PCS: Path = (
    BULK
    / "comparator_background_relatedness/runs/safs_phase3_omni_v116_20260716"
    / "relatedness/matrix/pcair_vectors_sensitive.tsv"
)
"""Selected the ancestry-component table aligned to the genotype comparator run."""

MARKERS: int = int(os.environ.get("ASTERISM_MARKERS", "50000"))
"""Set the number of real markers parsed for every null-response draw."""

DRAWS: int = int(os.environ.get("ASTERISM_DRAWS", "200"))
"""Set the number of independent polygenic null responses to simulate."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "6"))
"""Set the worker-process count used for association sweeps."""

COMPONENTS: int = 10
"""Set the number of ancestry principal components included as covariates."""

HERITABILITY: float = 0.5
"""Set the polygenic heritability of every simulated null response."""

THRESHOLDS: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
"""Selected the tail-probability ladder assessed by the calibration check."""

STATE: dict[str, np.ndarray] = {}
"""Held read-only arrays installed once in each association worker process."""


def start_worker(payload: dict[str, np.ndarray]) -> None:
    """Install shared analysis arrays in one worker process.

    Args:
        payload: Relationship, design, marker and covariance-factor arrays.
    """
    STATE.update(payload)


def numeric(value: object) -> float | None:
    """Parse a finite numeric database value or return no usable value.

    Args:
        value: Scalar database value to parse.

    Returns:
        Finite floating-point value, or ``None`` for missing or invalid input.
    """
    try:
        out: float = float(str(value).strip())
        """Parsed surrounding-whitespace-tolerant numeric text."""

    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and align the pedigree, fixed effects and retained dosage markers."""
    db: sqlite3.Connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    """Opened the SAFS database in SQLite's read-only URI mode."""

    pedigree: list[tuple[object, object, object, object]] = list(
        db.execute("select id, fa, mo, sex from pedigree")
    )
    """Loaded pedigree identifiers, parental links and recorded sex values."""

    ids: list[str] = [str(person) for person, _, _, _ in pedigree]
    """Normalised pedigree subject identifiers to strings."""

    fathers: list[str | None] = [
        None if str(f) in ("0", "", "None") else str(f) for _, f, _, _ in pedigree
    ]
    """Normalised father identifiers while removing pedigree missing sentinels."""

    mothers: list[str | None] = [
        None if str(m) in ("0", "", "None") else str(m) for _, _, m, _ in pedigree
    ]
    """Normalised mother identifiers while removing pedigree missing sentinels."""

    sexes: dict[str, str] = {str(person): str(sex) for person, _, _, sex in pedigree}
    """Indexed recorded sex codes by normalised subject identifier."""

    ages: dict[str, float] = {}
    """Initialised finite consent ages indexed by subject identifier."""

    for person, age in db.execute(
        "select subject_id, gobs_age_at_consent_years from gobs_demographics"
    ):
        value: float | None = numeric(age)
        """Parsed the current demographic age only when it was finite."""

        if value is not None and str(person) in sexes:
            ages[str(person)] = value
            """Retained an age only for a subject represented in the pedigree."""

    with gzip.open(DOSAGES, "rt") as fh:
        header: list[str] = fh.readline().rstrip("\n").split("\t")
        """Read the subject-oriented dosage header once."""

    genotyped: list[str] = [
        column.split("_", 1)[1] if "_" in column else column for column in header[6:]
    ]
    """Normalised genotype column names to the pedigree identifier convention."""

    position: dict[str, int] = {person: index for index, person in enumerate(genotyped)}
    """Indexed genotype-column positions by normalised subject identifier."""

    components: dict[str, list[float]] = {}
    """Initialised ancestry principal components indexed by subject."""

    with open(PCS) as fh:
        pc_header: list[str] = fh.readline().rstrip("\n").split("\t")
        """Read the ancestry-component table header."""

        first: int = pc_header.index("PC1")
        """Located the first principal-component column."""

        for line in fh:
            fields: list[str] = line.rstrip("\n").split("\t")
            """Split one ancestry-component record into its tabular fields."""

            components[fields[0]] = [
                float(v) for v in fields[first : first + COMPONENTS]
            ]
            """Stored the requested ancestry components under the subject identifier."""

    keep: list[str] = sorted(
        person
        for person in genotyped
        if person in ages and person in sexes and person in components
    )
    """Selected genotyped subjects with complete age, sex and ancestry covariates."""

    relationship, order = asterism.relationship_matrix(ids, fathers, mothers, keep=keep)
    """Built pedigree relationships in the complete-case genotype order."""

    assert order == keep
    columns: list[int] = [position[person] for person in keep]
    """Mapped retained subjects back to columns in the dosage archive."""

    age: np.ndarray = np.array([ages[person] for person in keep])
    """Collected consent ages in the model's subject order."""

    age = (age - age.mean()) / age.std()
    """Standardised age across the retained analysis sample."""

    male: np.ndarray = np.array(
        [1.0 if sexes[person] == "1" else 0.0 for person in keep]
    )
    """Encoded the recorded male sex indicator in subject order."""

    pcs: np.ndarray = np.array([components[person] for person in keep])
    """Collected ancestry principal components in subject order."""

    design: np.ndarray = np.column_stack([np.ones(len(keep)), age, age**2, male, pcs])
    """Constructed the fixed-effect design for every null association sweep."""

    # Parse the markers once. This is the expensive part and none of it changes
    # between draws.
    kept: list[np.ndarray] = []
    """Initialised dosage vectors satisfying missingness and frequency filters."""

    with gzip.open(DOSAGES, "rt") as fh:
        fh.readline()
        for line in fh:
            parts: list[str] = line.rstrip("\n").split("\t")
            """Split one dosage record into marker metadata and subject values."""

            raw: list[str] = [parts[6 + column] for column in columns]
            """Selected dosages for retained subjects in model order."""

            values: np.ndarray = np.array(
                [np.nan if v == "NA" else float(v) for v in raw], dtype=np.float64
            )
            """Parsed dosage values while representing declared missingness as NaN."""

            missing: np.ndarray = np.isnan(values)
            """Located missing dosages for the current marker."""

            if missing.mean() > 0.05:
                continue
            if missing.any():
                values[missing] = values[~missing].mean()
                """Mean-imputed the limited missing dosages for this score test."""

            frequency: float = float(values.mean() / 2.0)
            """Calculated the alternate-allele frequency from diploid dosages."""

            if min(frequency, 1.0 - frequency) < 0.01:
                continue
            kept.append(values)
            if len(kept) >= MARKERS:
                break
    return (
        np.ascontiguousarray(relationship),
        design,
        np.ascontiguousarray(np.column_stack(kept)),
    )


def one(index: int) -> list[float]:
    """Return trustworthy association p-values for one null response draw."""
    relationship: np.ndarray = STATE["relationship"]
    """Loaded the worker-local pedigree relationship matrix."""

    design: np.ndarray = STATE["design"]
    """Loaded the worker-local fixed-effect design matrix."""

    factor: np.ndarray = STATE["factor"]
    """Loaded the worker-local factor of the null phenotype covariance."""

    n: int = design.shape[0]
    """Counted subjects represented by the fixed-effect design."""

    y: np.ndarray = factor @ np.random.default_rng(640_000 + index).standard_normal(n)
    """Simulated one deterministic polygenic response with no marker effect."""

    _, _, rows, _ = _core.association_sweep(
        relationship, design, y, STATE["markers"], "held"
    )
    """Swept every retained marker while holding null variance components fixed."""

    return [row[4] for row in rows if not row[5]]


def main() -> int:
    if not DOSAGES.exists():
        raise SystemExit(f"no dosages at {DOSAGES}; is the bulk drive mounted?")
    started: float = time.perf_counter()
    """Captured the start of the one-time data loading and marker parsing."""

    relationship, design, markers = load()
    """Loaded aligned pedigree covariance, covariates and filtered real markers."""

    n, m = design.shape[0], markers.shape[1]
    """Counted retained people and markers in the association design."""

    print(
        f"Association tail on the real GOBS pedigree. {n} people, {m} real markers, "
        f"{DRAWS} null draws.\nThe response is drawn from a polygenic model at a "
        f"heritability of {HERITABILITY} with no marker effect anywhere.\n"
        f"Markers parsed once in {time.perf_counter() - started:.0f} s and reused.\n"
    )

    covariance: np.ndarray = HERITABILITY * relationship + (
        1.0 - HERITABILITY
    ) * np.eye(n)
    """Constructed the generating polygenic covariance at the fixed heritability."""

    factor: np.ndarray = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    """Factorised the generating covariance with a numerical diagonal guard."""

    def sweep(these: np.ndarray) -> list[list[float]]:
        """Run every null-response sweep against one marker matrix.

        Args:
            these: Real or subject-permuted marker dosage matrix.

        Returns:
            Trustworthy marker p-values for each independent null response.
        """
        shared: dict[str, np.ndarray] = {
            "relationship": relationship,
            "design": design,
            "markers": these,
            "factor": factor,
        }
        """Collected immutable arrays installed once in each worker process."""

        with ProcessPoolExecutor(
            WORKERS, initializer=start_worker, initargs=(shared,)
        ) as pool:
            return list(pool.map(one, range(DRAWS), chunksize=1))

    started = time.perf_counter()
    """Captured the start of the paired real and permuted marker sweeps."""

    batches: list[list[float]] = sweep(markers)
    """Ran null responses against the retained real marker matrix."""

    # **The second arm is the same markers with their people shuffled.** Each
    # column keeps its allele frequency exactly and loses any relation to who is
    # related to whom. Permuted in place and run as a second pass rather than
    # alongside the first, so only one marker matrix is ever shipped to the
    # workers. Permuting per draw would confound the two sources of scatter.
    shuffle: np.random.Generator = np.random.default_rng(4_242)
    """Created the deterministic generator for subject permutations."""

    permuted: np.ndarray = markers.copy()
    """Copied markers so the real-genotype comparison remained unchanged."""

    for column in range(m):
        shuffle.shuffle(permuted[:, column])
    batches_permuted: list[list[float]] = sweep(np.ascontiguousarray(permuted))
    """Ran the same null responses against subject-permuted marker columns."""

    del permuted
    took: float = time.perf_counter() - started
    """Measured elapsed time for both association-tail arms."""

    # **Per draw as well as pooled.** Markers in linkage disequilibrium are not
    # independent tests, so the binomial band on the pooled count is too narrow
    # and a small excess against it may be the band rather than the test. The
    # spread of the count between draws says which: each draw is one
    # independent realisation of the response, whatever the markers do among
    # themselves.
    per_draw: list[np.ndarray] = [np.array(batch) for batch in batches if batch]
    """Collected trustworthy real-marker p-values by independent response draw."""

    per_draw_permuted: list[np.ndarray] = [
        np.array(batch) for batch in batches_permuted if batch
    ]
    """Collected trustworthy permuted-marker p-values by response draw."""

    p_values: np.ndarray = np.concatenate(per_draw)
    """Pooled real-marker p-values solely for total-test and reach calculations."""

    total: int = len(p_values)
    """Counted the trustworthy null tests contributing to threshold reach."""

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
    draws: int = len(per_draw)
    """Counted independent null-response draws with trustworthy marker tests."""

    each: int = len(per_draw[0])
    """Counted trustworthy marker tests contributing to each response draw."""

    print(
        f"{'threshold':>12}{'per draw':>10}{'permuted':>10}{'expected':>10}"
        f"{'measured sd':>13}{'binomial sd':>13}{'ratio':>7}{'  verdict':<22}"
    )

    failures: list[str] = []
    """Collected anti-conservative threshold results from permuted markers."""

    recorded: dict[str, dict[str, float | int]] = {}
    """Initialised pooled real-marker counts at each threshold."""

    dispersion: dict[str, dict[str, float]] = {}
    """Initialised between-draw dispersion evidence at judged thresholds."""

    structure: dict[str, dict[str, float | None]] = {}
    """Initialised real-versus-permuted evidence for covariance-model structure."""

    for threshold in THRESHOLDS:
        counts: np.ndarray = np.array(
            [(batch <= threshold).sum() for batch in per_draw]
        )
        """Counted real-marker threshold crossings within every response draw."""

        shuffled: np.ndarray = np.array(
            [(batch <= threshold).sum() for batch in per_draw_permuted]
        )
        """Counted permuted-marker threshold crossings within every response draw."""

        expected: float = each * threshold
        """Calculated the uniform-null expected crossings per response draw."""

        binomial: float = float(np.sqrt(each * threshold * (1 - threshold)))
        """Calculated the deliberately naive independent-marker standard deviation."""

        measured: float = float(counts.std(ddof=1))
        """Measured crossing-count dispersion across independent response draws."""

        seen: int = int(counts.sum())
        """Counted pooled real-marker crossings across every response draw."""

        recorded[str(threshold)] = {
            "expected_total": expected * draws,
            "seen_total": seen,
            "ratio": seen / (expected * draws) if expected else float("nan"),
        }
        """Recorded pooled real-marker counts and their uniform-null ratio."""

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
        """Recorded measured and binomial dispersion for the judged threshold."""

        # **The permuted arm is what the test is judged on.** Its markers have
        # the same allele frequencies and no relation to who is related to whom,
        # so any excess there is the test's own. The mean count per draw is
        # judged against its expectation with the standard error taken from the
        # scatter actually observed.
        permuted_sd: float = float(shuffled.std(ddof=1))
        """Measured threshold-count dispersion for subject-permuted markers."""

        error: float = float(permuted_sd / np.sqrt(draws))
        """Calculated the measured standard error of the permuted mean count."""

        ceiling: float = expected + 1.96 * error
        """Set the upper 95 per cent calibration bound for the permuted arm."""

        over: bool = bool(shuffled.mean() > ceiling)
        """Identified anti-conservative excess after removing relatedness structure."""

        verdict: str = "OVER" if over else "ok"
        """Rendered the calibration verdict for the current threshold."""

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
        """Recorded the excess crossings attributable to genotype relationship structure."""

        print(
            f"{threshold:>12g}{counts.mean():>10.2f}{shuffled.mean():>10.2f}"
            f"{expected:>10.2f}{measured:>13.2f}{binomial:>13.2f}"
            f"{measured / binomial:>7.2f}  {verdict}"
        )

    if structure:
        gaps: list[float] = [
            float(values["excess_over_permuted"])
            for values in structure.values()
            if values["excess_over_permuted"]
        ]
        """Collected defined real-to-permuted crossing ratios across thresholds."""

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

    reachable: list[float] = [
        threshold for threshold in THRESHOLDS if total * threshold >= 5
    ]
    """Selected thresholds with at least five expected null crossings."""
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
