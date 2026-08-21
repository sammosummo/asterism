"""PROTOTYPE: build the settled Aim 2 recruitment-count scenarios.

The output is deliberately identifier-free.  Source identifiers are used only
in memory to recover pedigree relationships and the positive family-history
route; the saved arrays contain a projected relationship matrix, ages, roles,
family-unit numbers, observation masks and row-number anchors.

This is not the canonical Aim 2 campaign.  It builds six structural designs:
one, one-and-a-half or two enrolled case relatives on average, with and without
the separately labelled 250-person family-history extension.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

PEDIGREE = Path(
    "/Users/samuelmathias/MathiasLab/promoted/studies/existing/safs/"
    "data/pedigrees/review-history/pedigree.csv"
)
REDCAP = Path(
    "/Users/samuelmathias/AcousticIngest/outputs/intermediate/redcap/raw/"
    "redcap_acoustic_20260811T150210.json"
)
FAMILY_HISTORY = Path(
    "/Users/samuelmathias/MathiasLab/staging/studies/existing/acoustic/"
    "projects/acoustic-ingest/outputs/phenotypes/family_history.csv"
)

SEED = 20_260_821
ACOUSTIC_TARGET = 600
INDEX_CASES = 250
EXTENSION_TARGET = 250
MAXIMUM_UNIT_SIZE = 8
PARENT_GAP = 24.0
INDEX_AGE = (68.0, 88.0)
EXTENSION_AGE_MEAN = 76.0
EXTENSION_AGE_SD = 8.0

INDEX_CASE = 0
FIRST_CASE_RELATIVE = 1
SECOND_CASE_RELATIVE = 2
ACOUSTIC = 3
FAMILY_HISTORY_EXTENSION = 4
ROLE_NAMES = {
    INDEX_CASE: "index_case",
    FIRST_CASE_RELATIVE: "first_case_relative",
    SECOND_CASE_RELATIVE: "second_case_relative",
    ACOUSTIC: "acoustic",
    FAMILY_HISTORY_EXTENSION: "family_history_extension",
}

TRUTHS = {
    "no_loading": {"a": 0.0, "b": 0.4},
    "no_path": {"a": 0.6, "b": 0.0},
    "vertical_0_24": {"a": 0.6, "b": 0.4},
}
FIXED = {"c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def pedigree(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    father = {row["id"]: (row["fa"] if row["fa"] != "0" else None) for row in rows}
    mother = {row["id"]: (row["mo"] if row["mo"] != "0" else None) for row in rows}
    order: list[str] = []
    placed: set[str] = set()
    pending = list(father)
    while pending:
        remaining: list[str] = []
        progressed = False
        for person in pending:
            parents = (father[person], mother[person])
            if all(parent is None or parent in placed for parent in parents):
                order.append(person)
                placed.add(person)
                progressed = True
            else:
                remaining.append(person)
        if not progressed:
            raise SystemExit("pedigree has a loop")
        pending = remaining

    index = {person: row for row, person in enumerate(order)}
    additive = np.zeros((len(order), len(order)), dtype=np.float64)
    for row, person in enumerate(order):
        fi = index[father[person]] if father[person] is not None else -1
        mi = index[mother[person]] if mother[person] is not None else -1
        left = additive[fi, :row] if fi >= 0 else 0.0
        right = additive[mi, :row] if mi >= 0 else 0.0
        additive[row, :row] = 0.5 * (left + right)
        additive[:row, row] = additive[row, :row]
        parental = additive[fi, mi] if fi >= 0 and mi >= 0 else 0.0
        additive[row, row] = 0.5 * (1.0 + parental)
    additive *= 2.0
    return order, index, additive


def acoustic_ages(path: Path, index: dict[str, int]) -> dict[str, float]:
    records = json.loads(path.read_text())
    ages: dict[str, float] = {}
    for record in records:
        person = str(record.get("record_id"))
        if person not in index:
            continue
        for field in ("age_visit_v2", "age_v2", "age_ssq"):
            try:
                age = float(record.get(field))
            except (TypeError, ValueError):
                continue
            if 0.0 < age < 120.0:
                ages[person] = age
                break
    return ages


def positive_reporters(path: Path) -> set[str]:
    reporters: set[str] = set()
    for row in csv.DictReader(path.open(newline="")):
        if str(row.get("alzheimers_fam", "")).strip() in {"1", "1.0"} or str(
            row.get("dementia_fam", "")
        ).strip() in {"1", "1.0"}:
            reporters.add(str(row["ID"]))
    return reporters


class Groups:
    def __init__(self, people: list[str]):
        self.parent = {person: person for person in people}
        self.members = {person: {person} for person in people}

    def find(self, person: str) -> str:
        parent = self.parent[person]
        if parent != person:
            self.parent[person] = self.find(parent)
        return self.parent[person]

    def add_to(self, person: str, anchor: str) -> None:
        root = self.find(anchor)
        self.parent[person] = root
        self.members[root].add(person)

    def join(self, left: str, right: str, maximum: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return True
        if len(self.members[a]) + len(self.members[b]) > maximum:
            return False
        if min(self.members[b]) < min(self.members[a]):
            a, b = b, a
        self.parent[b] = a
        for person in self.members[b]:
            self.parent[person] = a
        self.members[a].update(self.members.pop(b))
        return True

    def ordered(self, order_index: dict[str, int]) -> list[list[str]]:
        groups = [
            sorted(members, key=order_index.__getitem__)
            for members in self.members.values()
        ]
        return sorted(groups, key=lambda members: min(order_index[p] for p in members))


def choose_people(
    order: list[str],
    index: dict[str, int],
    additive: np.ndarray,
    observed_age: dict[str, float],
    reporters: set[str],
    rng: np.random.Generator,
):
    actual = sorted(observed_age, key=index.__getitem__)
    if len(actual) > ACOUSTIC_TARGET:
        chosen = rng.choice(actual, size=ACOUSTIC_TARGET, replace=False)
        actual = sorted((str(person) for person in chosen), key=index.__getitem__)

    reporter_rows = [index[p] for p in actual if p in reporters]
    if not reporter_rows:
        raise SystemExit(
            "no positive family-history reporter is in the Acoustic pedigree"
        )

    all_rows = np.arange(len(order))
    proximity_to_reporter = additive[np.ix_(all_rows, reporter_rows)].max(axis=1)
    reserved_route = {
        order[row]
        for row in np.flatnonzero(
            (proximity_to_reporter >= 0.20) & (proximity_to_reporter < 0.75)
        )
        if order[row] not in observed_age
    }

    need = ACOUSTIC_TARGET - len(actual)
    actual_rows = [index[p] for p in actual]
    proximity_to_acoustic = additive[np.ix_(all_rows, actual_rows)].max(axis=1)
    top_up = [
        order[row]
        for row in np.flatnonzero(proximity_to_acoustic >= 0.20)
        if order[row] not in observed_age and order[row] not in reserved_route
    ]
    rng.shuffle(top_up)
    if len(top_up) < need:
        relaxed = [
            person
            for person in sorted(reserved_route, key=index.__getitem__)
            if person not in top_up
        ]
        rng.shuffle(relaxed)
        top_up.extend(relaxed)
    if len(top_up) < need:
        raise SystemExit(
            f"only {len(top_up)} eligible Acoustic projections for {need} needed"
        )
    top_up = top_up[:need]

    seen_ages = np.array([observed_age[p] for p in actual], dtype=float)
    age = dict(observed_age)
    for person in top_up:
        age[person] = float(rng.choice(seen_ages))
    acoustic = actual + top_up

    anchor_people = [p for p in actual if p in reporters]
    anchor_rows = [index[p] for p in anchor_people]
    candidate_records: list[tuple[str, list[tuple[str, float]]]] = []
    for person in order:
        if person in age or person in observed_age:
            continue
        row = index[person]
        eligible = [
            (anchor, float(additive[row, anchor_row]))
            for anchor, anchor_row in zip(anchor_people, anchor_rows, strict=True)
            if 0.20 <= additive[row, anchor_row] < 0.75
        ]
        if eligible:
            candidate_records.append((person, eligible))
    rng.shuffle(candidate_records)

    groups = Groups(acoustic)
    assigned: dict[str, str] = {}
    anchor_load = collections.Counter()
    for person, eligible in candidate_records:
        eligible.sort(key=lambda item: (anchor_load[item[0]], -item[1], index[item[0]]))
        anchor = next(
            (
                candidate
                for candidate, _ in eligible
                if len(groups.members[groups.find(candidate)]) < MAXIMUM_UNIT_SIZE
            ),
            None,
        )
        if anchor is None:
            continue
        groups.add_to(person, anchor)
        assigned[person] = anchor
        anchor_load[anchor] += 1
        if len(assigned) == EXTENSION_TARGET:
            break
    if len(assigned) != EXTENSION_TARGET:
        raise SystemExit(
            f"the flag-routed structural pool placed {len(assigned)} of {EXTENSION_TARGET} people"
        )

    acoustic_edges: list[tuple[float, str, str]] = []
    for left_pos, left in enumerate(acoustic):
        left_row = index[left]
        for right in acoustic[left_pos + 1 :]:
            value = float(additive[left_row, index[right]])
            if value > 0.0:
                acoustic_edges.append((value, left, right))
    acoustic_edges.sort(key=lambda item: (-item[0], index[item[1]], index[item[2]]))
    for _, left, right in acoustic_edges:
        groups.join(left, right, MAXIMUM_UNIT_SIZE)

    units = groups.ordered(index)
    extension = sorted(assigned, key=index.__getitem__)
    extension_age = {
        person: float(
            np.clip(rng.normal(EXTENSION_AGE_MEAN, EXTENSION_AGE_SD), 40.0, 95.0)
        )
        for person in extension
    }
    return {
        "actual_acoustic": actual,
        "projected_acoustic": top_up,
        "acoustic": acoustic,
        "acoustic_age": {person: age[person] for person in acoustic},
        "reporters": anchor_people,
        "route_candidate_count": len(candidate_records),
        "extension": extension,
        "extension_anchor": assigned,
        "extension_age": extension_age,
        "units": units,
    }


def build_case_families(rng: np.random.Generator):
    families = []
    for family in range(INDEX_CASES):
        proband_age = float(rng.uniform(*INDEX_AGE))
        child_ages = [
            float(np.clip(proband_age - rng.normal(PARENT_GAP, 6.0), 25.0, 74.0))
            for _ in range(2)
        ]
        families.append((family, proband_age, child_ages))
    return families


def connected(matrix: np.ndarray) -> bool:
    if matrix.shape[0] <= 1:
        return True
    seen = {0}
    stack = [0]
    while stack:
        here = stack.pop()
        for other in np.flatnonzero(matrix[here] > 0.0):
            other = int(other)
            if other != here and other not in seen:
                seen.add(other)
                stack.append(other)
    return len(seen) == matrix.shape[0]


def scenario_id(relative_yield: float, extension_n: int) -> str:
    yield_name = str(relative_yield).replace(".", "_")
    prefix = "core_plus_structural_fh250" if extension_n else "confirmed_core"
    return f"{prefix}_yield_{yield_name}"


def assemble(
    *,
    relative_yield: float,
    extension_n: int,
    structural: dict,
    case_families: list,
    index: dict[str, int],
    additive: np.ndarray,
    out_dir: Path,
    source_commit: str,
):
    include_extension = extension_n == EXTENSION_TARGET
    extension_set = set(structural["extension"] if include_extension else [])
    acoustic_set = set(structural["acoustic"])

    units: list[list[str]] = []
    for maximal in structural["units"]:
        kept = [
            person
            for person in maximal
            if person in acoustic_set or person in extension_set
        ]
        if kept:
            units.append(kept)

    people: list[str] = []
    ages: list[float] = []
    roles: list[int] = []
    families: list[int] = []
    observe_outcome: list[bool] = []
    claimed_person_anchor: list[tuple[str, str]] = []

    for family_id, unit in enumerate(units):
        for person in unit:
            people.append(person)
            families.append(family_id)
            if person in acoustic_set:
                roles.append(ACOUSTIC)
                ages.append(structural["acoustic_age"][person])
            else:
                roles.append(FAMILY_HISTORY_EXTENSION)
                ages.append(structural["extension_age"][person])
                claimed_person_anchor.append(
                    (person, structural["extension_anchor"][person])
                )
            observe_outcome.append(True)

    pedigree_n = len(people)
    rows = [index[person] for person in people]
    relationship = np.zeros((pedigree_n, pedigree_n), dtype=np.float64)
    for family_id in range(len(units)):
        local = np.flatnonzero(np.asarray(families) == family_id)
        source_rows = [rows[int(row)] for row in local]
        relationship[np.ix_(local, local)] = additive[np.ix_(source_rows, source_rows)]

    second_count = {1.0: 0, 1.5: INDEX_CASES // 2, 2.0: INDEX_CASES}[relative_yield]
    claimed_row_pairs: list[tuple[int, int, str]] = []
    person_to_row = {person: row for row, person in enumerate(people)}
    for person, anchor in claimed_person_anchor:
        claimed_row_pairs.append(
            (
                person_to_row[person],
                person_to_row[anchor],
                "family_history_route_anchor",
            )
        )

    for case_number, proband_age, child_ages in case_families:
        family_id = len(units) + case_number
        size = 3 if case_number < second_count else 2
        start = len(ages)
        ages.extend([proband_age, child_ages[0]])
        roles.extend([INDEX_CASE, FIRST_CASE_RELATIVE])
        families.extend([family_id, family_id])
        observe_outcome.extend([True, True])
        if size == 3:
            ages.append(child_ages[1])
            roles.append(SECOND_CASE_RELATIVE)
            families.append(family_id)
            observe_outcome.append(False)
        claimed_row_pairs.append((start + 1, start, "case_relative_to_index"))
        if size == 3:
            claimed_row_pairs.append((start + 2, start, "case_relative_to_index"))
            claimed_row_pairs.append((start + 2, start + 1, "case_full_sibling"))

    total = len(ages)
    grown = np.zeros((total, total), dtype=np.float64)
    grown[:pedigree_n, :pedigree_n] = relationship
    relationship = grown
    for case_number in range(INDEX_CASES):
        family_id = len(units) + case_number
        local = np.flatnonzero(np.asarray(families) == family_id)
        relationship[np.ix_(local, local)] = 0.5
        relationship[local, local] = 1.0

    inbreeding = np.diag(relationship[:pedigree_n, :pedigree_n]) - 1.0
    inbred_count = int((inbreeding > 1.0e-9).sum())
    largest_inbreeding = float(inbreeding.max(initial=0.0))
    np.fill_diagonal(relationship, 1.0)

    family_array = np.asarray(families, dtype=np.int32)
    role_array = np.asarray(roles, dtype=np.int8)
    age_array = np.asarray(ages, dtype=np.float64)
    observe_array = np.asarray(observe_outcome, dtype=bool)
    claimed_anchor = np.full(total, -1, dtype=np.int32)
    claimed_type = np.zeros(total, dtype=np.int8)
    for person_row, anchor_row, kind in claimed_row_pairs:
        if claimed_anchor[person_row] < 0:
            claimed_anchor[person_row] = anchor_row
            claimed_type[person_row] = (
                1 if kind in {"case_relative_to_index", "case_full_sibling"} else 2
            )

    claimed_failures = []
    for person_row, anchor_row, kind in claimed_row_pairs:
        if family_array[person_row] != family_array[anchor_row]:
            claimed_failures.append(f"{kind}:cross_unit")
        if relationship[person_row, anchor_row] <= 0.0:
            claimed_failures.append(f"{kind}:missing_relationship")
    if claimed_failures:
        raise SystemExit(
            f"claimed relationship audit failed: {collections.Counter(claimed_failures)}"
        )

    cross = family_array[:, None] != family_array[None, :]
    if np.any(np.abs(relationship[cross]) > 0.0):
        raise SystemExit("projected design contains a non-zero cross-unit relationship")

    minimum_eigenvalue = 1.0
    size_histogram: collections.Counter[int] = collections.Counter()
    disconnected = 0
    for family_id in np.unique(family_array):
        local = np.flatnonzero(family_array == family_id)
        block = relationship[np.ix_(local, local)]
        size_histogram[len(local)] += 1
        minimum_eigenvalue = min(
            minimum_eigenvalue, float(np.linalg.eigvalsh(block).min())
        )
        if not connected(block):
            disconnected += 1
    if minimum_eigenvalue < -1.0e-9 or disconnected:
        raise SystemExit(
            f"family validation failed: minimum eigenvalue {minimum_eigenvalue}, "
            f"disconnected {disconnected}"
        )

    source_cross_positive = 0
    source_cross_first_degree = 0
    if pedigree_n:
        source = additive[np.ix_(rows, rows)]
        upper = np.triu(np.ones_like(source, dtype=bool), 1)
        base_cross = family_array[:pedigree_n, None] != family_array[None, :pedigree_n]
        source_cross_positive = int((upper & base_cross & (source > 0.0)).sum())
        source_cross_first_degree = int((upper & base_cross & (source >= 0.45)).sum())

    design_name = scenario_id(relative_yield, extension_n)
    design_path = out_dir / f"{design_name}.npz"
    np.savez_compressed(
        design_path,
        relationship=relationship,
        age=age_array,
        role=role_array,
        family=family_array,
        observe_outcome=observe_array,
        claimed_anchor=claimed_anchor,
        claimed_type=claimed_type,
    )

    role_counts = {
        ROLE_NAMES[code]: int((role_array == code).sum()) for code in sorted(ROLE_NAMES)
    }
    observed_by_role = {
        ROLE_NAMES[code]: int(((role_array == code) & observe_array).sum())
        for code in sorted(ROLE_NAMES)
    }
    relationship_type_histogram = collections.Counter(
        kind for _, _, kind in claimed_row_pairs
    )
    extension_degree_histogram = collections.Counter()
    for person_row, anchor_row, kind in claimed_row_pairs:
        if kind != "family_history_route_anchor":
            continue
        value = float(relationship[person_row, anchor_row])
        extension_degree_histogram[
            "first_degree" if value >= 0.45 else "second_degree"
        ] += 1
    manifest = {
        "prototype_warning": (
            "THROWAWAY STRUCTURAL DESIGN. This is not a realised roster, grant text, "
            "or an analysis-ready inferential release."
        ),
        "question": (
            "What level and power does the settled recruitment-count design have "
            "under ordered CDR, across relative yield, with and without a separately "
            "labelled 250-person family-history structural extension?"
        ),
        "scenario_id": design_name,
        "source_commit": source_commit,
        "builder_sha256": sha256(Path(__file__)),
        "design_file": design_path.name,
        "design_sha256": sha256(design_path),
        "counts": {
            "total": total,
            "roles": role_counts,
            "outcomes_observed_by_role": observed_by_role,
            "family_units": len(np.unique(family_array)),
            "family_size_histogram": {
                str(size): count for size, count in sorted(size_histogram.items())
            },
            "relative_yield": relative_yield,
            "family_history_extension": extension_n,
        },
        "recruitment": {
            "acoustic_target": ACOUSTIC_TARGET,
            "acoustic_target_status": "projected up-to target",
            "acoustic_observed_ages": len(structural["actual_acoustic"]),
            "acoustic_projected_ages": len(structural["projected_acoustic"]),
            "index_cases": INDEX_CASES,
            "case_relative_policy": (
                "attempt two; these fixed-yield cells retain every family with one"
            ),
            "case_relative_structure": (
                "structural all-adult-offspring scenario; two offspring are treated "
                "as full siblings and remain in the index trio"
            ),
            "extension_status": (
                "structural scenario, not demonstrated recruitment feasibility"
                if extension_n
                else "not included"
            ),
            "extension_route_rule": (
                "identifier-free selection from unexamined first/second-degree pedigree "
                "members of an Acoustic participant with a positive authoritative "
                "family-history flag; the reported free-text relative is not normalised"
            ),
            "extension_route_candidates": structural["route_candidate_count"],
            "positive_reporter_anchors": len(structural["reporters"]),
            "extension_age_scenario": (
                f"Normal({EXTENSION_AGE_MEAN}, {EXTENSION_AGE_SD}^2), clipped to 40-95; "
                "not an observed recruit age distribution"
            ),
        },
        "relationship_audit": {
            "scale": "twice pedigree kinship with unit diagonal",
            "claimed_ties": len(claimed_row_pairs),
            "claimed_ties_within_fitted_unit": len(claimed_row_pairs),
            "claimed_relationship_type_histogram": dict(relationship_type_histogram),
            "extension_degree_histogram": dict(extension_degree_histogram),
            "claimed_tie_failures": 0,
            "disconnected_units": disconnected,
            "nonzero_projected_cross_unit_ties": 0,
            "source_positive_ties_projected_away": source_cross_positive,
            "source_first_degree_ties_projected_away": source_cross_first_degree,
            "minimum_family_eigenvalue": minimum_eigenvalue,
            "inbred_people_with_diagonal_above_one_before_projection": inbred_count,
            "largest_dropped_inbreeding_coefficient": largest_inbreeding,
        },
        "outcome_observation": {
            "index_case": True,
            "first_case_relative": True,
            "second_case_relative": False,
            "acoustic": True,
            "family_history_extension": True,
            "note": (
                "The second case relative is enrolled for hearing/genotype/covariates; "
                "the current Strategy does not require their dementia outcome."
            ),
        },
        "ascertainment_boundary": (
            "The available model conditions on the named affected index case. Fixed "
            "relative yield additionally assumes invitation and enrolment are ignorable "
            "after diagnosis, stage, source, referral route and covariates. The structural "
            "extension is fitted population-unconditioned; its report-based selection "
            "denominator is not yet implementable from a normalised roster."
        ),
        "source_files": {
            "pedigree": str(PEDIGREE),
            "acoustic_redcap": str(REDCAP),
            "family_history_flags": str(FAMILY_HISTORY),
        },
    }
    manifest_path = out_dir / f"{design_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_sha256"] = sha256(manifest_path)
    return manifest, manifest_path


def write_cells(manifests: list[dict], cells_path: Path, source_commit: str) -> None:
    cells = []
    for manifest in sorted(manifests, key=lambda item: item["scenario_id"]):
        relative_yield = float(manifest["counts"]["relative_yield"])
        extension_n = int(manifest["counts"]["family_history_extension"])
        for outcome in ("ordered_cdr", "binary_probable_ad"):
            for arm in ("all_audiogram_bound", "index_proband_abr_bound"):
                for truth_name, truth_parameters in TRUTHS.items():
                    cell_id = len(cells)
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "design_id": manifest["scenario_id"],
                            "design_file": manifest["design_file"],
                            "design_sha256": manifest["design_sha256"],
                            "manifest_file": f"{manifest['scenario_id']}.manifest.json",
                            "design_manifest_sha256": manifest["manifest_sha256"],
                            "relative_yield": relative_yield,
                            "family_history_extension": extension_n,
                            "outcome": outcome,
                            "hearing_arm": arm,
                            "truth": truth_name,
                            "truth_parameters": truth_parameters,
                            "fixed_parameters": FIXED,
                            "alpha": 0.025,
                            "replicates": 200,
                            "bootstrap_replicates": 50,
                            "qmc_points": 0,
                            "seed_base": 2_608_210,
                            "result_file": f"cell_{cell_id:03d}.json",
                        }
                    )
    if len(cells) != 72:
        raise SystemExit(f"expected 72 cells, built {len(cells)}")
    payload = {
        "prototype": "mediation grant recruitment design, 21 August 2026",
        "source_commit": source_commit,
        "cells": cells,
    }
    cells_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    global PEDIGREE, REDCAP, FAMILY_HISTORY

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--pedigree", type=Path, default=PEDIGREE)
    parser.add_argument("--redcap", type=Path, default=REDCAP)
    parser.add_argument("--family-history", type=Path, default=FAMILY_HISTORY)
    args = parser.parse_args()

    PEDIGREE, REDCAP, FAMILY_HISTORY = args.pedigree, args.redcap, args.family_history
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cells.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    order, index, additive = pedigree(PEDIGREE)
    observed_age = acoustic_ages(REDCAP, index)
    reporters = positive_reporters(FAMILY_HISTORY)
    structural = choose_people(order, index, additive, observed_age, reporters, rng)
    case_families = build_case_families(rng)
    source_commit = git_commit()

    manifests = []
    for relative_yield in (1.0, 1.5, 2.0):
        for extension_n in (0, EXTENSION_TARGET):
            manifest, manifest_path = assemble(
                relative_yield=relative_yield,
                extension_n=extension_n,
                structural=structural,
                case_families=case_families,
                index=index,
                additive=additive,
                out_dir=args.out_dir,
                source_commit=source_commit,
            )
            manifests.append(manifest)
            counts = manifest["counts"]
            print(
                f"{manifest['scenario_id']}: {counts['total']} people, "
                f"{counts['family_units']} units; {manifest_path.name}"
            )
    write_cells(manifests, args.cells, source_commit)
    print(f"{len(manifests)} designs and 72 explicit cells written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
