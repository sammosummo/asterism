"""PROTOTYPE: run one explicit grant-recruitment level or power cell."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import asterism
import numpy as np
from asterism.latent_mediation import simulate

INDEX_CASE = 0
AUDIOMETRY_ERROR = 0.15
ABR_SENSITIVITY = 0.80
ABR_SPECIFICITY = 0.85

_CELL: dict[str, Any] = {}
_DESIGN: dict[str, Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def prevalence_for(age: float) -> float:
    if age < 60:
        return 0.001
    if age < 65:
        return 0.005
    if age < 70:
        return 0.02
    if age < 75:
        return 0.04
    if age < 80:
        return 0.08
    if age < 85:
        return 0.15
    return 0.25


def cdr_shares(rate: float) -> list[float]:
    return [1.0 - 2.0 * rate, rate, 0.56 * rate, 0.31 * rate, 0.13 * rate]


def initialise_worker(cell: dict[str, Any], design_path: str) -> None:
    global _CELL, _DESIGN
    _CELL = cell
    loaded = np.load(design_path)
    family = loaded["family"]
    unit_rows = [np.flatnonzero(family == unit) for unit in np.unique(family)]
    _DESIGN = {
        "relationship": loaded["relationship"],
        "age": loaded["age"],
        "role": loaded["role"],
        "family": family,
        "observe_outcome": loaded["observe_outcome"],
        "units": unit_rows,
    }


def run_replicate(replicate: int) -> dict[str, Any]:
    started = time.perf_counter()
    cell = _CELL
    d = _DESIGN
    families = []
    for unit_number, rows in enumerate(d["units"]):
        relationship = [
            [float(value) for value in row]
            for row in d["relationship"][np.ix_(rows, rows)]
        ]
        roles = d["role"][rows]
        ages = d["age"][rows]
        prevalence = [prevalence_for(float(age)) for age in ages]
        has_proband = bool((roles == INDEX_CASE).any())
        proband = int(np.argmax(roles == INDEX_CASE)) if has_proband else None

        if cell["hearing_arm"] == "index_proband_abr_bound" and has_proband:
            measurement_error = [
                None if row == proband else AUDIOMETRY_ERROR for row in range(len(rows))
            ]
            proxy = [row == proband for row in range(len(rows))]
        else:
            measurement_error = [AUDIOMETRY_ERROR] * len(rows)
            proxy = [False] * len(rows)

        if cell["outcome"] == "ordered_cdr":
            outcome_prevalence = None
            outcome_categories = [cdr_shares(rate) for rate in prevalence]
            ascertainment_category = 2
        else:
            outcome_prevalence = prevalence
            outcome_categories = None
            ascertainment_category = None

        families.extend(
            simulate(
                relationship=relationship,
                **cell["truth_parameters"],
                **cell["fixed_parameters"],
                families=1,
                seed=int(cell["seed_base"]) + 7_919 * replicate + 13 * unit_number,
                outcome_prevalence=outcome_prevalence,
                outcome_category_prevalence=outcome_categories,
                ascertainment_category=ascertainment_category,
                observe_outcome=[bool(value) for value in d["observe_outcome"][rows]],
                measurement_error_variance=measurement_error,
                observe_mediator_proxy=proxy,
                sensitivity=ABR_SENSITIVITY,
                specificity=ABR_SPECIFICITY,
                ascertainment=(
                    "condition_on_named_proband_case"
                    if has_proband
                    else "population_unconditioned"
                ),
                proband_index=proband,
            )
        )

    case_from = 2 if cell["outcome"] == "ordered_cdr" else 1
    cases = sum(
        1
        for family in families
        for value in family["outcome_status"]
        if value is not None and value >= case_from
    )
    observed_outcomes = sum(
        1
        for family in families
        for value in family["outcome_status"]
        if value is not None
    )
    record: dict[str, Any] = {
        "replicate": replicate,
        "p_value": None,
        "refusal": None,
        "cases": cases,
        "people": sum(len(family["outcome_status"]) for family in families),
        "outcomes_observed": observed_outcomes,
        "bootstrap_replicates": 0,
        "loading_reference": None,
        "test": None,
    }
    model = asterism.LatentMediationModel(families, qmc_points=int(cell["qmc_points"]))
    try:
        tested = json_value(
            model.test_vertical(
                bootstrap_replicates=int(cell["bootstrap_replicates_actual"])
            )
        )
        record["test"] = tested
        record["p_value"] = float(tested["p_value"])
        record["bootstrap_replicates"] = int(tested.get("bootstrap_replicates", 0))
        record["loading_reference"] = tested.get("loading_reference")
    except ValueError as refusal:
        record["refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
    record["seconds"] = time.perf_counter() - started
    return record


def wheel_path() -> Path | None:
    try:
        from asterism import _core

        return Path(_core.__file__).resolve()
    except (ImportError, TypeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    cell_manifest = json.loads(args.cells.read_text())
    matching = [
        cell for cell in cell_manifest["cells"] if int(cell["cell_id"]) == args.cell
    ]
    if len(matching) != 1:
        raise SystemExit(f"cell {args.cell} is not present exactly once")
    cell = matching[0]
    cell["replicates_actual"] = int(args.replicates or cell["replicates"])
    cell["bootstrap_replicates_actual"] = int(
        cell["bootstrap_replicates"] if args.bootstrap is None else args.bootstrap
    )

    design_path = args.design_dir / cell["design_file"]
    manifest_path = args.design_dir / cell["manifest_file"]
    if sha256(design_path) != cell["design_sha256"]:
        raise SystemExit("design SHA-256 does not match cells.json")
    if sha256(manifest_path) != cell["design_manifest_sha256"]:
        raise SystemExit("design manifest SHA-256 does not match cells.json")
    design_manifest = json.loads(manifest_path.read_text())
    if design_manifest["design_sha256"] != cell["design_sha256"]:
        raise SystemExit("design manifest and cell disagree about the design")

    print(
        f"cell {cell['cell_id']:03d}: {cell['design_id']} / {cell['outcome']} / "
        f"{cell['hearing_arm']} / {cell['truth']}\n"
        f"{cell['replicates_actual']} outer replicates, "
        f"{cell['bootstrap_replicates_actual']} requested bootstrap draws, "
        f"{args.workers} workers",
        flush=True,
    )
    started = time.perf_counter()
    rows = []
    if args.workers == 1:
        initialise_worker(cell, str(design_path))
        answers = map(run_replicate, range(cell["replicates_actual"]))
        for row in answers:
            rows.append(row)
            if len(rows) % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  {len(rows)}/{cell['replicates_actual']} outer replicates; "
                    f"{elapsed / len(rows):.1f}s per replicate wall average",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialise_worker,
            initargs=(cell, str(design_path)),
        ) as pool:
            for row in pool.map(
                run_replicate, range(cell["replicates_actual"]), chunksize=1
            ):
                rows.append(row)
                if len(rows) % 25 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"  {len(rows)}/{cell['replicates_actual']} outer replicates; "
                        f"{elapsed / len(rows):.1f}s per replicate wall average",
                        flush=True,
                    )

    alpha = float(cell["alpha"])
    rejections = sum(
        row["p_value"] is not None and float(row["p_value"]) < alpha for row in rows
    )
    refusal_codes = collections.Counter(
        row["refusal"] for row in rows if row["refusal"] is not None
    )
    n = cell["replicates_actual"]
    rate = rejections / n
    summary = {
        "kind": "power" if cell["truth"] == "vertical_0_24" else "level",
        "rejections": rejections,
        "rejection_rate": rate,
        "power_or_level": rate,
        "monte_carlo_se": math.sqrt(rate * (1.0 - rate) / n) if n else None,
        "successful_tests": n - sum(refusal_codes.values()),
        "refusals": sum(refusal_codes.values()),
        "refusal_codes": dict(refusal_codes),
        "mean_cases": float(np.mean([row["cases"] for row in rows])),
        "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
        "wall_seconds": time.perf_counter() - started,
        "refusals_stay_in_denominator": True,
    }
    wheel = wheel_path()
    payload = {
        "prototype_warning": design_manifest["prototype_warning"],
        "cell": cell,
        "source_commit": cell_manifest["source_commit"],
        "design_sha256": cell["design_sha256"],
        "design_manifest_sha256": cell["design_manifest_sha256"],
        "runner_sha256": sha256(Path(__file__)),
        "linux_or_local_extension": str(wheel) if wheel else None,
        "extension_sha256": sha256(wheel) if wheel and wheel.exists() else None,
        "measurement_placeholders": {
            "audiometry_error_variance": AUDIOMETRY_ERROR,
            "abr_sensitivity": ABR_SENSITIVITY,
            "abr_specificity": ABR_SPECIFICITY,
        },
        "outer_replicates_requested": n,
        "bootstrap_draws_requested": cell["bootstrap_replicates_actual"],
        "rows": rows,
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / cell["result_file"]
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"written {out_path}: {rejections}/{n} reject; {summary['refusals']} refusals",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
