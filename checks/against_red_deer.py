"""Asterism's component model against a published analysis of wild red deer.

Stopher, Walling, Morris, Guinness, Clutton-Brock, Pemberton and Nussey (2012),
"Shared spatial effects on quantitative genetic parameters: accounting for
spatial autocorrelation and home range overlap reduces estimates of heritability
in wild red deer", Evolution 66(8):2411-2426, doi:10.1111/j.1558-5646.2012.01620.x.
The data are the paper's own deposit, Dryad doi:10.5061/dryad.jf04r362.

That paper is the reason this comparison is worth having. It fitted an animal
model to four traits of female red deer on the Isle of Rum, then added a matrix
of home range overlap between individuals and watched heritability collapse.
The overlap matrix is published with the data, so the models it reports are
ordinary variance-component models over supplied matrices, and Asterism can fit
exactly the same thing and be checked against a printed table.

What is compared are the paper's first and last columns of its Table 2: the
model with no spatial term, and the model with the home range overlap matrix.
The three spatial-autocorrelation columns are a different model — a separable
first-order autoregressive process on the row and column coordinates, fitted in
ASReml — and Asterism has no such model. Its own spatial family is an isotropic
kernel of distance, which is not the same object, so those columns are outside
what this check can claim.

Three of the four traits are repeated measures on the same female. That costs
nothing here: the paper's permanent-environment, birth-year and maternal terms
are incidence matrices, so the whole model is a sum of covariance components at
the level of the record, which is what ComponentModel already takes.

Run it with the package's own environment::

    .venv/bin/python checks/against_red_deer.py
    .venv/bin/python checks/against_red_deer.py --trait lbs --scale log

The data are not in the repository. Point RED_DEER_DATA at the deposit, or
leave it unset and let the script find `staging/data/rum-red-deer` by walking up.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import numpy as np

import asterism


# ----------------------------------------------------------------- the deposit

def data_directory() -> Path:
    """Where the Dryad deposit was unpacked."""
    named = os.environ.get("RED_DEER_DATA")
    if named:
        return Path(named).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "staging" / "data" / "rum-red-deer"
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        "The red deer deposit was not found. Unpack Dryad doi:10.5061/dryad.jf04r362 "
        "into staging/data/rum-red-deer, or set RED_DEER_DATA."
    )


def read_table(path: Path) -> list[dict[str, str]]:
    """A tab-separated file with a header, as a list of dictionaries."""
    lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    header = [name.strip() for name in lines[0].split("\t")]
    return [
        dict(zip(header, [cell.strip() for cell in line.split("\t")]))
        for line in lines[1:]
    ]


MISSING = {"", "NA", ".", "0"}


def read_pedigree(path: Path) -> tuple[list[str], list[str | None], list[str | None], list[str]]:
    """The deer pedigree, with an invented sire wherever the father is unknown.

    Asterism refuses an individual with one parent known and the other not,
    rather than guessing what the silence means. Every such deer here has a
    known mother and an unknown father, which is what wild paternity looks
    like. Giving each unknown father its own founder identity says exactly what
    the missing entry is meant to say — a sire unrelated to every other animal
    in the pedigree — and reproduces the usual convention exactly: a child of a
    known dam and an unknown sire gets half its dam's relationship to everybody
    else, and an inbreeding coefficient of nought.
    """
    rows = read_table(path)
    ids = [row["ID"] for row in rows]
    father: list[str | None] = []
    mother: list[str | None] = []
    invented = []
    for row in rows:
        sire = None if row["FATHER"] in MISSING else row["FATHER"]
        dam = None if row["MOTHER"] in MISSING else row["MOTHER"]
        if sire is None and dam is not None:
            sire = f"#sire{len(invented):04d}"
            invented.append(sire)
        father.append(sire)
        mother.append(dam)
    recorded = list(ids)  # the deer themselves, in the order the overlap matrix uses
    ids.extend(invented)
    father.extend([None] * len(invented))
    mother.extend([None] * len(invented))
    return ids, father, mother, recorded


def read_overlap(path: Path, order: list[str]) -> np.ndarray:
    """The home range overlap matrix, expanded from its lower triangle.

    The deposit stores it as row, column, value with no identifier file. Its
    indices run over the pedigree file in the order that file is written, which
    is checked rather than assumed: the paper says overlap was available for 948
    females in spring and 766 in the rut, and that is exactly how many indices
    carry a non-zero off-diagonal entry.
    """
    n = len(order)
    matrix = np.zeros((n, n))
    with path.open() as handle:
        for line in handle:
            i, j, value = line.split()
            matrix[int(i) - 1, int(j) - 1] = float(value)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    return matrix


# ------------------------------------------------------------------ the design

def design_matrix(rows: list[dict[str, str]], terms: list[tuple[str, str]]) -> np.ndarray:
    """The fixed effects, with an intercept, factors coded against their first level."""
    columns = [np.ones(len(rows))]
    for name, kind in terms:
        if kind == "numeric":
            values = np.array([float(row[name]) for row in rows])
            columns.append(values)
        elif kind == "square":
            values = np.array([float(row[name]) for row in rows])
            columns.append(values**2)
        elif kind == "factor":
            levels = sorted({row[name] for row in rows})
            for level in levels[1:]:
                columns.append(
                    np.array([1.0 if row[name] == level else 0.0 for row in rows])
                )
        else:
            raise ValueError(f"unknown term kind {kind}")
    return np.column_stack(columns)


def incidence(labels: list[str]) -> np.ndarray:
    """One where two records share a label, nought where they do not."""
    codes = {label: i for i, label in enumerate(sorted(set(labels)))}
    index = np.array([codes[label] for label in labels])
    return (index[:, None] == index[None, :]).astype(float)


def expand(matrix: np.ndarray, index: np.ndarray) -> np.ndarray:
    """A matrix over individuals, read at the level of the record."""
    return matrix[np.ix_(index, index)]


# ------------------------------------------------------------------ the traits

TRAITS = {
    "rhr": {
        "file": "Rut+home+range+data.DAT",
        "response": "HOMERANGE",
        "scale": "log",
        "animal": "ANIMAL",
        "mother": "MOTHER",
        "year": "Year",
        "overlap": "ruthroverlap.grm",
        "permanent": True,
        "terms": [
            ("AGE", "numeric"),
            ("REGION", "factor"),
            ("LOCALPOP", "numeric"),
            ("NoFixes", "numeric"),
        ],
        "published": {
            "none": {"pe": 0.000, "a": 0.168, "year": 0.017, "m": 0.084,
                     "residual": 0.265, "h2": 31.31, "sum": 0.534},
            "overlap": {"pe": 0.000, "a": 0.001, "year": 0.013, "m": 0.006,
                        "s": 0.598, "residual": 0.260, "h2": 0.114, "sum": 0.878},
            "chisq": 785.8,
        },
    },
    "shr": {
        "file": "Spring+home+range+data.dat",
        "response": "HOMERANGE",
        "scale": "log",
        "animal": "ANIMAL",
        "mother": "MOTHER",
        "year": "Year",
        "overlap": "sproverlap.grm",
        "permanent": True,
        "terms": [
            ("AGE", "numeric"),
            ("AGE", "square"),
            ("LOCALPOPD", "numeric"),
            ("REGION", "factor"),
            ("REPS", "factor"),
        ],
        "published": {
            "none": {"pe": 0.000, "a": 0.193, "year": 0.012, "m": 0.017,
                     "residual": 0.220, "h2": 43.666, "sum": 0.442},
            "overlap": {"pe": 0.000, "a": 0.002, "year": 0.008, "m": 0.000,
                        "s": 0.487, "residual": 0.206, "h2": 0.284, "sum": 0.703},
            "chisq": 1313.2,
        },
    },
    "bw": {
        "file": "Birth+weight+data.dat",
        "response": "BirthWt",
        "scale": "raw",
        "animal": "ANIMAL",
        "mother": "MOTHER",
        "year": "BirthYear",
        "overlap": "sproverlap.grm",
        "permanent": True,
        # Table 2 drops region: with it the autocorrelation models were singular,
        # so the paper reports every birth weight column without it.
        "terms": [
            ("MumAge", "numeric"),
            ("MumAge", "square"),
            ("MumReps", "factor"),
            ("Sex", "factor"),
        ],
        "published": {
            "none": {"pe": 0.049, "a": 0.530, "year": 0.081, "m": 0.046,
                     "residual": 0.783, "h2": 35.594, "sum": 1.489},
            "overlap": {"pe": 0.082, "a": 0.402, "year": 0.107, "m": 0.038,
                        "s": 0.088, "residual": 0.782, "h2": 26.818, "sum": 1.499},
            "chisq": None,  # the paper's 5.3 is for the model that keeps region
        },
    },
    "lbs": {
        "file": "LBS+data.dat",
        "response": "LBS",
        "scale": "log10p1",
        "animal": "ANIMAL",
        "mother": "MOTHER",
        "year": "BirthYear",
        "overlap": "sproverlap.grm",
        # Measured once per female, so there is no permanent environment term
        # to separate from the residual.
        "permanent": False,
        "terms": [("Region", "factor")],
        "published": {
            "a": None, "none": {"a": 0.009, "year": 0.077, "m": 0.007,
                                "residual": 0.103, "h2": 4.592, "sum": 0.196},
            "overlap": {"a": 0.000, "year": 0.040, "m": 0.003, "s": 0.045,
                        "residual": 0.076, "h2": 0.000, "sum": 0.164},
            "chisq": 185.6,
        },
    },
}

SCALES = {
    "raw": lambda v: v,
    "log": np.log,
    "logp1": lambda v: np.log(v + 1.0),
    "log10p1": lambda v: np.log10(v + 1.0),
}


def build(trait: str, scale: str | None, directory: Path):
    """Everything one trait needs: response, design, and the component matrices."""
    spec = TRAITS[trait]
    rows = read_table(directory / spec["file"])

    ped_ids, father, mother, recorded = read_pedigree(directory / "DeerPed.ped")
    known = set(recorded)

    # Records are dropped only where the model cannot be written down for them:
    # a missing response, a missing fixed effect, an animal with no pedigree
    # record, or an unknown mother when a maternal term is fitted.
    needed = [spec["response"], spec["year"], spec["mother"]] + [
        name for name, _ in spec["terms"]
    ]
    kept = []
    dropped = {"missing_field": 0, "not_in_pedigree": 0}
    for row in rows:
        if any(row.get(name, "") in MISSING for name in needed):
            dropped["missing_field"] += 1
            continue
        if row[spec["animal"]] not in known:
            dropped["not_in_pedigree"] += 1
            continue
        kept.append(row)

    animals = sorted({row[spec["animal"]] for row in kept})
    relationship, order = asterism.relationship_matrix(
        ped_ids, father, mother, keep=animals
    )
    relationship = np.asarray(relationship)
    position = {name: i for i, name in enumerate(order)}
    index = np.array([position[row[spec["animal"]]] for row in kept])

    overlap_full = read_overlap(directory / spec["overlap"], recorded)
    where = {name: i for i, name in enumerate(recorded)}
    overlap_rows = np.array([where[row[spec["animal"]]] for row in kept])

    response = SCALES[scale or spec["scale"]](
        np.array([float(row[spec["response"]]) for row in kept])
    )
    x = design_matrix(kept, spec["terms"])

    matrices = {
        "a": expand(relationship, index),
        "year": incidence([row[spec["year"]] for row in kept]),
        "m": incidence([row[spec["mother"]] for row in kept]),
    }
    if spec["permanent"]:
        matrices["pe"] = incidence([row[spec["animal"]] for row in kept])
    matrices["s"] = expand(overlap_full, overlap_rows)

    return spec, kept, response, x, matrices, dropped, len(animals)


ORDER_NONE = ["a", "pe", "year", "m"]
ORDER_OVERLAP = ["a", "pe", "year", "m", "s"]


def fit(matrices: dict[str, np.ndarray], names: list[str], x, y) -> dict:
    present = [name for name in names if name in matrices]
    model = asterism.ComponentModel([matrices[name] for name in present], x)
    record = model.fit(y, reml=True)
    variances = dict(zip(present + ["residual"], record["variances"]))
    total = sum(variances.values())
    return {
        "variances": variances,
        "sum": total,
        "h2_percent": 100.0 * variances["a"] / total,
        "loglik": record["loglik"],
        "converged": record["converged"],
        "scaled_gradient": record["scaled_gradient"],
        "components": present + ["residual"],
    }


def report(trait: str, scale: str | None, directory: Path) -> dict:
    spec, kept, y, x, matrices, dropped, n_animals = build(trait, scale, directory)
    none = fit(matrices, ORDER_NONE, x, y)
    overlap = fit(matrices, ORDER_OVERLAP, x, y)
    chisq = 2.0 * (overlap["loglik"] - none["loglik"])

    print(f"\n{'=' * 78}")
    print(f"{trait.upper()}  {len(kept)} records on {n_animals} females"
          f"   scale={scale or spec['scale']}   design={x.shape[1]} columns")
    if any(dropped.values()):
        print(f"  dropped: {dropped['missing_field']} incomplete, "
              f"{dropped['not_in_pedigree']} not in the pedigree")

    for label, fitted, published in (
        ("no spatial effect", none, spec["published"]["none"]),
        ("home range overlap", overlap, spec["published"]["overlap"]),
    ):
        print(f"\n  {label}")
        print(f"    {'':<10}{'asterism':>11}{'published':>11}{'difference':>12}")
        for name in fitted["components"]:
            got = fitted["variances"][name]
            want = published.get(name)
            if want is None:
                print(f"    {name:<10}{got:>11.3f}{'—':>11}{'':>12}")
            else:
                print(f"    {name:<10}{got:>11.3f}{want:>11.3f}{got - want:>12.3f}")
        print(f"    {'sum':<10}{fitted['sum']:>11.3f}{published['sum']:>11.3f}"
              f"{fitted['sum'] - published['sum']:>12.3f}")
        print(f"    {'h2 %':<10}{fitted['h2_percent']:>11.3f}{published['h2']:>11.3f}"
              f"{fitted['h2_percent'] - published['h2']:>12.3f}")
        if not fitted["converged"]:
            print("    DID NOT CONVERGE")

    want = spec["published"]["chisq"]
    print(f"\n  likelihood ratio for adding overlap: {chisq:.1f} on 1 df"
          + (f"   published {want}" if want else "   (published value is for another model)"))

    return {
        "trait": trait,
        "scale": scale or spec["scale"],
        "records": len(kept),
        "females": n_animals,
        "dropped": dropped,
        "design_columns": int(x.shape[1]),
        "no_spatial": none,
        "with_overlap": overlap,
        "likelihood_ratio": chisq,
        "published": spec["published"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", action="append", choices=sorted(TRAITS))
    parser.add_argument("--scale", choices=sorted(SCALES))
    parser.add_argument("--evidence", action="store_true",
                        help="write the record to evidence/")
    arguments = parser.parse_args()

    directory = data_directory()
    traits = arguments.trait or ["lbs", "bw", "rhr", "shr"]
    print(f"Red deer of Rum, against Stopher et al. (2012) Table 2.")
    print(f"Data: {directory}")

    records = [report(trait, arguments.scale, directory) for trait in traits]

    if arguments.evidence:
        out = (Path(__file__).resolve().parents[1] / "evidence"
               / f"against-red-deer-{date.today().isoformat()}.json")
        out.write_text(json.dumps({
            "source": "Stopher et al. 2012 Evolution 66(8):2411-2426",
            "data": "Dryad doi:10.5061/dryad.jf04r362",
            "records": records,
        }, indent=2, default=float) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
