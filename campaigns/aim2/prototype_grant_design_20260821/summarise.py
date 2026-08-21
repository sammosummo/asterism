#!/usr/bin/env python3
"""PROTOTYPE — WIPE ME: summarise the grant-design simulation cells.

This script is deliberately local to the throwaway prototype.  It refuses to
state a design verdict until every manifest cell has one complete, hash-matched
result.  A refused fit is a non-rejection and remains in every denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist, fmean
from typing import Any

HERE = Path(__file__).resolve().parent
ARTEFACTS = HERE / "artifacts" / "PROTOTYPE_WIPE_ME"
DEFAULT_CELLS = HERE / "cells.json"
DEFAULT_RESULTS = ARTEFACTS / "results"
DEFAULT_SUMMARY = ARTEFACTS / "summary.json"
DEFAULT_VERDICT = HERE / "PROTOTYPE_VERDICT.md"

ALPHA = 0.025
EXPECTED_REPLICATES = 200
EXPECTED_BOOTSTRAP_REPLICATES = 50
WILSON_Z = NormalDist().inv_cdf(0.975)
HASH_RE = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
COMMIT_RE = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)

YIELDS = (1.0, 1.5, 2.0)
EXTENSIONS = (0, 250)
OUTCOMES = ("ordered_cdr", "binary_probable_ad")
HEARING_ARMS = ("all_audiogram_bound", "index_proband_abr_bound")
TRUTHS = ("no_loading", "no_path", "vertical_0_24")
NULL_TRUTHS = ("no_loading", "no_path")
ALTERNATIVE_TRUTH = "vertical_0_24"

REQUIRED_CELL_KEYS = {
    "cell_id",
    "design_id",
    "design_sha256",
    "design_manifest_sha256",
    "relative_yield",
    "family_history_extension",
    "outcome",
    "hearing_arm",
    "truth",
    "truth_parameters",
    "fixed_parameters",
    "alpha",
    "replicates",
    "bootstrap_replicates",
    "qmc_points",
    "seed_base",
    "result_file",
}
REQUIRED_ROW_KEYS = {
    "replicate",
    "p_value",
    "refusal",
    "cases",
    "bootstrap_replicates",
    "loading_reference",
    "seconds",
}


class SummaryError(ValueError):
    """A failure that makes a prototype verdict unsafe."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise SummaryError(f"{label} is not a 64-character SHA-256 digest")
    return value.lower()


def _read_json(path: Path) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SummaryError(f"cannot read {path}: {error}") from error
    try:
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    except json.JSONDecodeError as error:
        raise SummaryError(f"invalid JSON in {path}: {error}") from error


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _combination(cell: dict[str, Any]) -> tuple[float, int, str, str, str]:
    return (
        float(cell["relative_yield"]),
        int(cell["family_history_extension"]),
        str(cell["outcome"]),
        str(cell["hearing_arm"]),
        str(cell["truth"]),
    )


def _factor_record(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_yield": float(cell["relative_yield"]),
        "family_history_extension": int(cell["family_history_extension"]),
        "outcome": cell["outcome"],
        "hearing_arm": cell["hearing_arm"],
        "truth": cell["truth"],
    }


def _validate_manifest(manifest: object) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise SummaryError("cells.json must contain one JSON object")
    for key in ("prototype", "source_commit", "cells"):
        if key not in manifest:
            raise SummaryError(f"cells.json is missing top-level field {key!r}")
    if not isinstance(manifest["prototype"], str) or not manifest["prototype"]:
        raise SummaryError("cells.json prototype must be a non-empty string")
    if (
        not isinstance(manifest["source_commit"], str)
        or COMMIT_RE.fullmatch(manifest["source_commit"]) is None
    ):
        raise SummaryError("cells.json source_commit must be a 40-character commit")
    cells = manifest["cells"]
    if not isinstance(cells, list):
        raise SummaryError("cells.json cells must be a list")

    expected_combinations = {
        (relative_yield, extension, outcome, hearing_arm, truth)
        for relative_yield in YIELDS
        for extension in EXTENSIONS
        for outcome in OUTCOMES
        for hearing_arm in HEARING_ARMS
        for truth in TRUTHS
    }
    seen_ids: set[int] = set()
    seen_files: set[str] = set()
    seen_combinations: set[tuple[float, int, str, str, str]] = set()
    hashes_by_design: dict[str, set[str]] = defaultdict(set)
    manifest_hashes_by_design: dict[str, set[str]] = defaultdict(set)

    for position, cell in enumerate(cells):
        label = f"cells[{position}]"
        if not isinstance(cell, dict):
            raise SummaryError(f"{label} must be an object")
        missing_keys = sorted(REQUIRED_CELL_KEYS - cell.keys())
        if missing_keys:
            raise SummaryError(f"{label} is missing fields: {', '.join(missing_keys)}")

        cell_id = cell["cell_id"]
        if not _is_int(cell_id) or cell_id < 0:
            raise SummaryError(f"{label}.cell_id must be a non-negative integer")
        if cell_id in seen_ids:
            raise SummaryError(f"cell_id {cell_id} occurs more than once")
        seen_ids.add(cell_id)

        design_id = cell["design_id"]
        if not isinstance(design_id, str) or not design_id:
            raise SummaryError(f"cell {cell_id} has no usable design_id")
        design_sha256 = _require_sha256(
            cell["design_sha256"], f"cell {cell_id} design_sha256"
        )
        design_manifest_sha256 = _require_sha256(
            cell["design_manifest_sha256"],
            f"cell {cell_id} design_manifest_sha256",
        )
        hashes_by_design[design_id].add(design_sha256)
        manifest_hashes_by_design[design_id].add(design_manifest_sha256)

        result_file = cell["result_file"]
        if (
            not isinstance(result_file, str)
            or Path(result_file).name != result_file
            or not result_file.startswith("cell_")
            or not result_file.endswith(".json")
        ):
            raise SummaryError(
                f"cell {cell_id} result_file must be a cell_*.json basename"
            )
        if result_file in seen_files:
            raise SummaryError(f"result_file {result_file!r} occurs more than once")
        seen_files.add(result_file)

        if not _is_number(cell["relative_yield"]):
            raise SummaryError(f"cell {cell_id} relative_yield must be numeric")
        if float(cell["relative_yield"]) not in YIELDS:
            raise SummaryError(f"cell {cell_id} has an unexpected relative_yield")
        if cell["family_history_extension"] not in EXTENSIONS:
            raise SummaryError(
                f"cell {cell_id} family_history_extension must be 0 or 250"
            )
        if cell["outcome"] not in OUTCOMES:
            raise SummaryError(f"cell {cell_id} has an unexpected outcome")
        if cell["hearing_arm"] not in HEARING_ARMS:
            raise SummaryError(f"cell {cell_id} has an unexpected hearing_arm")
        if cell["truth"] not in TRUTHS:
            raise SummaryError(f"cell {cell_id} has an unexpected truth")
        if not isinstance(cell["truth_parameters"], dict):
            raise SummaryError(f"cell {cell_id} truth_parameters must be an object")
        if not isinstance(cell["fixed_parameters"], dict):
            raise SummaryError(f"cell {cell_id} fixed_parameters must be an object")
        if not _is_number(cell["alpha"]) or not math.isclose(
            float(cell["alpha"]), ALPHA, rel_tol=0.0, abs_tol=1e-15
        ):
            raise SummaryError(f"cell {cell_id} alpha must be {ALPHA}")
        for field in ("replicates", "bootstrap_replicates"):
            if not _is_int(cell[field]) or cell[field] <= 0:
                raise SummaryError(f"cell {cell_id} {field} must be positive")
        if cell["replicates"] != EXPECTED_REPLICATES:
            raise SummaryError(
                f"cell {cell_id} must request {EXPECTED_REPLICATES} replicates"
            )
        if cell["bootstrap_replicates"] != EXPECTED_BOOTSTRAP_REPLICATES:
            raise SummaryError(
                f"cell {cell_id} must request "
                f"{EXPECTED_BOOTSTRAP_REPLICATES} bootstrap replicates"
            )
        if not _is_int(cell["qmc_points"]) or cell["qmc_points"] < 0:
            raise SummaryError(f"cell {cell_id} qmc_points must be non-negative")
        if not _is_int(cell["seed_base"]):
            raise SummaryError(f"cell {cell_id} seed_base must be an integer")

        combination = _combination(cell)
        if combination in seen_combinations:
            raise SummaryError(
                f"design combination occurs more than once: {combination!r}"
            )
        seen_combinations.add(combination)

    if seen_combinations != expected_combinations:
        missing = sorted(expected_combinations - seen_combinations)
        extra = sorted(seen_combinations - expected_combinations)
        detail = []
        if missing:
            detail.append(f"missing combinations: {missing!r}")
        if extra:
            detail.append(f"unexpected combinations: {extra!r}")
        raise SummaryError("cells.json grid is incomplete; " + "; ".join(detail))
    for design_id, hashes in hashes_by_design.items():
        if len(hashes) != 1:
            raise SummaryError(
                f"design_id {design_id!r} has more than one design_sha256"
            )
    for design_id, hashes in manifest_hashes_by_design.items():
        if len(hashes) != 1:
            raise SummaryError(
                f"design_id {design_id!r} has more than one design_manifest_sha256"
            )
    return sorted(cells, key=lambda cell: int(cell["cell_id"]))


def _wilson_interval(rejections: int, replicates: int) -> dict[str, float]:
    rate = rejections / replicates
    z2 = WILSON_Z * WILSON_Z
    denominator = 1.0 + z2 / replicates
    centre = (rate + z2 / (2.0 * replicates)) / denominator
    half_width = (
        WILSON_Z
        * math.sqrt(
            rate * (1.0 - rate) / replicates + z2 / (4.0 * replicates * replicates)
        )
        / denominator
    )
    return {
        "confidence": 0.95,
        "lower": max(0.0, centre - half_width),
        "upper": min(1.0, centre + half_width),
    }


def _level_assessment(interval: dict[str, float]) -> str:
    if interval["lower"] > ALPHA:
        return "above_target"
    if interval["upper"] < ALPHA:
        return "below_target"
    return "compatible_with_target"


def _validate_runner_summary(
    runner_summary: object,
    *,
    cell_id: int,
    rejections: int,
    refusals: int,
    refusal_codes: Counter[str],
    rejection_rate: float,
    monte_carlo_se: float,
    replicates: int,
    metric: str,
    mean_cases: float,
) -> None:
    if not isinstance(runner_summary, dict):
        raise SummaryError(f"cell {cell_id} result summary must be an object")
    required = {
        "kind",
        "rejections",
        "rejection_rate",
        "refusals",
        "refusal_codes",
        "monte_carlo_se",
        "successful_tests",
        "refusals_stay_in_denominator",
        "mean_cases",
    }
    missing = sorted(required - runner_summary.keys())
    if missing:
        raise SummaryError(
            f"cell {cell_id} result summary is missing: {', '.join(missing)}"
        )
    if runner_summary["rejections"] != rejections:
        raise SummaryError(f"cell {cell_id} runner rejection count disagrees")
    if runner_summary["refusals"] != refusals:
        raise SummaryError(f"cell {cell_id} runner refusal count disagrees")
    if runner_summary["refusal_codes"] != dict(refusal_codes):
        raise SummaryError(f"cell {cell_id} runner refusal codes disagree")
    if runner_summary["kind"] != metric:
        raise SummaryError(f"cell {cell_id} runner kind disagrees")
    if runner_summary["successful_tests"] != replicates - refusals:
        raise SummaryError(f"cell {cell_id} runner successful_tests disagrees")
    if runner_summary["refusals_stay_in_denominator"] is not True:
        raise SummaryError(f"cell {cell_id} runner dropped refusals")
    for field, observed, expected in (
        ("rejection_rate", runner_summary["rejection_rate"], rejection_rate),
        ("monte_carlo_se", runner_summary["monte_carlo_se"], monte_carlo_se),
        ("mean_cases", runner_summary["mean_cases"], mean_cases),
    ):
        if not _is_number(observed) or not math.isclose(
            float(observed), expected, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise SummaryError(f"cell {cell_id} runner {field} disagrees")


def _validate_result(
    path: Path, cell: dict[str, Any], source_commit: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    result, result_sha256 = _read_json(path)
    cell_id = int(cell["cell_id"])
    if not isinstance(result, dict):
        raise SummaryError(f"cell {cell_id} result must be a JSON object")
    required = {
        "cell",
        "design_manifest_sha256",
        "design_sha256",
        "outer_replicates_requested",
        "bootstrap_draws_requested",
        "source_commit",
        "rows",
        "summary",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise SummaryError(
            f"cell {cell_id} result is missing fields: {', '.join(missing)}"
        )
    embedded_cell = result["cell"]
    if not isinstance(embedded_cell, dict):
        raise SummaryError(f"cell {cell_id} embedded cell must be an object")
    allowed_runtime_fields = {"replicates_actual", "bootstrap_replicates_actual"}
    if set(embedded_cell) - set(cell) != allowed_runtime_fields:
        raise SummaryError(
            f"cell {cell_id} embedded cell has unexpected runtime fields"
        )
    embedded_manifest_cell = {
        key: value
        for key, value in embedded_cell.items()
        if key not in allowed_runtime_fields
    }
    if embedded_manifest_cell != cell:
        raise SummaryError(f"cell {cell_id} embedded cell does not match cells.json")
    result_design_sha256 = _require_sha256(
        result["design_sha256"], f"cell {cell_id} result design_sha256"
    )
    if result_design_sha256 != str(cell["design_sha256"]).lower():
        raise SummaryError(f"cell {cell_id} design_sha256 does not match cells.json")
    design_manifest_sha256 = _require_sha256(
        result["design_manifest_sha256"],
        f"cell {cell_id} result design_manifest_sha256",
    )
    expected_manifest_sha256 = cell.get("design_manifest_sha256")
    if expected_manifest_sha256 is not None and design_manifest_sha256 != (
        _require_sha256(
            expected_manifest_sha256,
            f"cell {cell_id} cells.json design_manifest_sha256",
        )
    ):
        raise SummaryError(
            f"cell {cell_id} design_manifest_sha256 does not match cells.json"
        )
    expected_replicates = int(cell["replicates"])
    if embedded_cell["replicates_actual"] != expected_replicates:
        raise SummaryError(
            f"cell {cell_id} embedded replicate override differs from cells.json"
        )
    expected_bootstrap = int(cell["bootstrap_replicates"])
    if embedded_cell["bootstrap_replicates_actual"] != expected_bootstrap:
        raise SummaryError(
            f"cell {cell_id} embedded bootstrap override differs from cells.json"
        )
    if result["outer_replicates_requested"] != expected_replicates:
        raise SummaryError(
            f"cell {cell_id} requested {result['outer_replicates_requested']!r} "
            f"rather than {expected_replicates} replicates"
        )
    if result["bootstrap_draws_requested"] != expected_bootstrap:
        raise SummaryError(
            f"cell {cell_id} requested {result['bootstrap_draws_requested']!r} "
            f"rather than {expected_bootstrap} bootstrap draws"
        )
    if result["source_commit"] != source_commit:
        raise SummaryError(f"cell {cell_id} source commit disagrees")
    rows = result["rows"]
    if not isinstance(rows, list):
        raise SummaryError(f"cell {cell_id} rows must be a list")
    if len(rows) != expected_replicates:
        raise SummaryError(
            f"cell {cell_id} has {len(rows)} of {expected_replicates} rows"
        )

    decisions: dict[int, dict[str, Any]] = {}
    refusal_codes: Counter[str] = Counter()
    rejections = 0
    cases: list[float] = []
    bootstrap_draws = 0
    elapsed_seconds = 0.0
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SummaryError(f"cell {cell_id} row {position} must be an object")
        missing_row_keys = sorted(REQUIRED_ROW_KEYS - row.keys())
        if missing_row_keys:
            raise SummaryError(
                f"cell {cell_id} row {position} is missing: "
                f"{', '.join(missing_row_keys)}"
            )
        replicate = row["replicate"]
        if not _is_int(replicate) or replicate < 0:
            raise SummaryError(
                f"cell {cell_id} row {position} has an invalid replicate"
            )
        if replicate in decisions:
            raise SummaryError(f"cell {cell_id} repeats replicate {replicate}")

        p_value = row["p_value"]
        refusal = row["refusal"]
        if refusal is None:
            if not _is_number(p_value) or not 0.0 <= float(p_value) <= 1.0:
                raise SummaryError(
                    f"cell {cell_id} replicate {replicate} has no p-value or refusal"
                )
            rejected = float(p_value) < ALPHA
        else:
            if not isinstance(refusal, str) or not refusal:
                raise SummaryError(
                    f"cell {cell_id} replicate {replicate} has an invalid refusal"
                )
            if p_value is not None:
                raise SummaryError(
                    f"cell {cell_id} replicate {replicate} has both p-value and refusal"
                )
            refusal_codes[refusal] += 1
            rejected = False
        rejections += int(rejected)

        if not _is_number(row["cases"]) or float(row["cases"]) < 0.0:
            raise SummaryError(
                f"cell {cell_id} replicate {replicate} has invalid cases"
            )
        cases.append(float(row["cases"]))
        if not _is_int(row["bootstrap_replicates"]) or row["bootstrap_replicates"] < 0:
            raise SummaryError(
                f"cell {cell_id} replicate {replicate} has invalid bootstrap count"
            )
        bootstrap_draws += int(row["bootstrap_replicates"])
        if not _is_number(row["seconds"]) or float(row["seconds"]) < 0.0:
            raise SummaryError(
                f"cell {cell_id} replicate {replicate} has invalid elapsed time"
            )
        elapsed_seconds += float(row["seconds"])
        decisions[replicate] = {
            "rejected": rejected,
            "refused": refusal is not None,
        }

    expected_ids = set(range(expected_replicates))
    observed_ids = set(decisions)
    if observed_ids != expected_ids:
        missing_ids = sorted(expected_ids - observed_ids)
        unexpected_ids = sorted(observed_ids - expected_ids)
        raise SummaryError(
            f"cell {cell_id} replicate IDs are incomplete; "
            f"missing={missing_ids!r}, unexpected={unexpected_ids!r}"
        )

    refusals = sum(refusal_codes.values())
    rejection_rate = rejections / expected_replicates
    monte_carlo_se = math.sqrt(
        rejection_rate * (1.0 - rejection_rate) / expected_replicates
    )
    mean_cases = fmean(cases)
    metric = "level" if cell["truth"] in NULL_TRUTHS else "power"
    _validate_runner_summary(
        result["summary"],
        cell_id=cell_id,
        rejections=rejections,
        refusals=refusals,
        refusal_codes=refusal_codes,
        rejection_rate=rejection_rate,
        monte_carlo_se=monte_carlo_se,
        replicates=expected_replicates,
        metric=metric,
        mean_cases=mean_cases,
    )

    interval = _wilson_interval(rejections, expected_replicates)
    cell_summary: dict[str, Any] = {
        "cell_id": cell_id,
        "design_id": cell["design_id"],
        "design_sha256": result_design_sha256,
        "design_manifest_sha256": design_manifest_sha256,
        "result_file": path.name,
        "result_file_sha256": result_sha256,
        **_factor_record(cell),
        "metric": metric,
        "alpha": ALPHA,
        "replicates_requested": expected_replicates,
        "replicates_observed": expected_replicates,
        "successful_fits": expected_replicates - refusals,
        "refusals": refusals,
        "refusal_codes": dict(sorted(refusal_codes.items())),
        "rejections": rejections,
        "non_rejections_including_refusals": expected_replicates - rejections,
        "refusals_in_denominator": True,
        "power_or_level": rejection_rate,
        "binomial_monte_carlo_se": monte_carlo_se,
        "wilson_interval": interval,
        "mean_cases": mean_cases,
        "bootstrap_replicates_requested_per_fit": int(cell["bootstrap_replicates"]),
        "bootstrap_replicates_reported_total": bootstrap_draws,
        "elapsed_seconds_reported_total": elapsed_seconds,
    }
    if metric == "level":
        cell_summary["level_assessment"] = _level_assessment(interval)
    return cell_summary, decisions


def _matching_nulls(
    alternative: dict[str, Any],
    summaries: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    matches = []
    for summary in summaries.values():
        if summary["truth"] not in NULL_TRUTHS:
            continue
        if all(
            summary[field] == alternative[field]
            for field in (
                "relative_yield",
                "family_history_extension",
                "outcome",
                "hearing_arm",
            )
        ):
            matches.append(summary)
    return sorted(matches, key=lambda summary: summary["truth"])


def _paired_extension_differences(
    cells: list[dict[str, Any]],
    summaries: dict[int, dict[str, Any]],
    decisions: dict[int, dict[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[tuple[float, str, str, str], dict[int, dict[str, Any]]] = defaultdict(
        dict
    )
    for cell in cells:
        if cell["truth"] != ALTERNATIVE_TRUTH:
            continue
        key = (
            float(cell["relative_yield"]),
            str(cell["outcome"]),
            str(cell["hearing_arm"]),
            str(cell["truth"]),
        )
        groups[key][int(cell["family_history_extension"])] = cell

    paired: list[dict[str, Any]] = []
    errors: list[str] = []
    for key in sorted(groups):
        by_extension = groups[key]
        if set(by_extension) != set(EXTENSIONS):
            errors.append(
                f"alternative pair {key!r} has extension arms {sorted(by_extension)!r}"
            )
            continue
        core = by_extension[0]
        extension = by_extension[250]
        core_id = int(core["cell_id"])
        extension_id = int(extension["cell_id"])
        if core_id not in summaries or extension_id not in summaries:
            continue
        pair_has_error = False
        for field in (
            "truth_parameters",
            "fixed_parameters",
            "alpha",
            "replicates",
            "bootstrap_replicates",
            "qmc_points",
            "seed_base",
        ):
            if core[field] != extension[field]:
                errors.append(
                    f"cells {core_id} and {extension_id} cannot be paired: "
                    f"{field} differs"
                )
                pair_has_error = True
        if pair_has_error:
            continue

        core_rows = decisions[core_id]
        extension_rows = decisions[extension_id]
        if set(core_rows) != set(extension_rows):
            errors.append(
                f"cells {core_id} and {extension_id} have different replicate IDs"
            )
            continue
        differences = [
            int(extension_rows[replicate]["rejected"])
            - int(core_rows[replicate]["rejected"])
            for replicate in sorted(core_rows)
        ]
        replicates = len(differences)
        difference = fmean(differences)
        if replicates > 1:
            variance = sum((value - difference) ** 2 for value in differences) / (
                replicates - 1
            )
            monte_carlo_se = math.sqrt(variance / replicates)
        else:
            monte_carlo_se = 0.0
        normal_interval = {
            "confidence": 0.95,
            "lower": max(-1.0, difference - WILSON_Z * monte_carlo_se),
            "upper": min(1.0, difference + WILSON_Z * monte_carlo_se),
        }
        extension_only = sum(value == 1 for value in differences)
        core_only = sum(value == -1 for value in differences)
        both = sum(
            core_rows[replicate]["rejected"] and extension_rows[replicate]["rejected"]
            for replicate in core_rows
        )
        neither = replicates - extension_only - core_only - both
        core_summary = summaries[core_id]
        extension_summary = summaries[extension_id]
        core_nulls = _matching_nulls(core_summary, summaries)
        extension_nulls = _matching_nulls(extension_summary, summaries)
        if len(core_nulls) != len(NULL_TRUTHS):
            errors.append(f"cell {core_id} does not have both matching null cells")
            continue
        if len(extension_nulls) != len(NULL_TRUTHS):
            errors.append(f"cell {extension_id} does not have both matching null cells")
            continue
        core_level_high = any(
            summary["level_assessment"] == "above_target" for summary in core_nulls
        )
        extension_level_high = any(
            summary["level_assessment"] == "above_target" for summary in extension_nulls
        )
        paired.append(
            {
                "relative_yield": key[0],
                "outcome": key[1],
                "hearing_arm": key[2],
                "truth": key[3],
                "core_cell_id": core_id,
                "extension_cell_id": extension_id,
                "replicates": replicates,
                "refusals_in_denominator": True,
                "core_refusals": core_summary["refusals"],
                "extension_refusals": extension_summary["refusals"],
                "both_refused": sum(
                    core_rows[replicate]["refused"]
                    and extension_rows[replicate]["refused"]
                    for replicate in core_rows
                ),
                "core_power_or_rejection_rate": core_summary["power_or_level"],
                "extension_power_or_rejection_rate": extension_summary[
                    "power_or_level"
                ],
                "core_matching_level_cells": [
                    summary["cell_id"] for summary in core_nulls
                ],
                "extension_matching_level_cells": [
                    summary["cell_id"] for summary in extension_nulls
                ],
                "core_level_high": core_level_high,
                "extension_level_high": extension_level_high,
                "reported_as_power": not core_level_high and not extension_level_high,
                "extension_minus_core_rejection_difference": difference,
                "paired_monte_carlo_se": monte_carlo_se,
                "paired_normal_interval": normal_interval,
                "paired_rejection_counts": {
                    "both": both,
                    "extension_only": extension_only,
                    "core_only": core_only,
                    "neither": neither,
                },
            }
        )
    return paired, errors


def _format_rate(rate: float) -> str:
    return f"{rate:.3f}"


def _format_interval(interval: dict[str, float]) -> str:
    return f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"


def _plain_label(value: str) -> str:
    return value.replace("_", " ")


def _incomplete_markdown(
    *,
    expected_cells: int,
    valid_cells: int,
    missing_cells: list[dict[str, Any]],
    unexpected_files: list[str],
    analysis_errors: list[str],
) -> str:
    lines = [
        "# Prototype verdict",
        "",
        "> **PROTOTYPE — WIPE ME.** This is a disposable design experiment, not "
        "grant evidence until the run is complete.",
        "",
        "## No verdict: incomplete run",
        "",
        f"Only {valid_cells} of {expected_cells} manifest cells have complete, "
        "hash-matched results. No power or recruitment-design verdict is made.",
        "",
        "Refused fits remain non-rejections in the denominator of any available "
        "cell summary.",
        "",
        "## Missing or unusable cells",
        "",
    ]
    if missing_cells:
        for missing in missing_cells:
            lines.append(
                f"- Cell `{missing['cell_id']}` ({missing['result_file']}): "
                f"{missing['reason']}"
            )
    else:
        lines.append("- None.")
    if unexpected_files:
        lines.extend(["", "## Unrecognised result files", ""])
        lines.extend(f"- `{name}`" for name in unexpected_files)
    if analysis_errors:
        lines.extend(["", "## Analysis errors", ""])
        lines.extend(f"- {error}" for error in analysis_errors)
    return "\n".join(lines)


def _complete_markdown(
    summaries: list[dict[str, Any]], paired: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    level_rows = [summary for summary in summaries if summary["metric"] == "level"]
    high_level_rows = [
        summary
        for summary in level_rows
        if summary["level_assessment"] == "above_target"
    ]
    positive = [row for row in paired if row["paired_normal_interval"]["lower"] > 0.0]
    negative = [row for row in paired if row["paired_normal_interval"]["upper"] < 0.0]
    uncertain = len(paired) - len(positive) - len(negative)
    differences = [row["extension_minus_core_rejection_difference"] for row in paired]
    verdict = {
        "null_cells": len(level_rows),
        "null_cells_with_wilson_interval_above_alpha": len(high_level_rows),
        "paired_extension_comparisons": len(paired),
        "paired_intervals_above_zero": len(positive),
        "paired_intervals_below_zero": len(negative),
        "paired_intervals_including_zero": uncertain,
        "minimum_paired_difference": min(differences),
        "maximum_paired_difference": max(differences),
        "burden_decision": (
            "not classified because no minimum worthwhile power increment or "
            "burden trade-off was prespecified"
        ),
    }

    lines = [
        "# Prototype verdict",
        "",
        "> **PROTOTYPE — WIPE ME.** This is a disposable design experiment.",
        "",
        "## Verdict",
        "",
        f"All {len(summaries)} manifest cells are complete and hash-matched. "
        f"At alpha {ALPHA:.3f}, {len(high_level_rows)} of {len(level_rows)} "
        "null cells have a 95% Wilson interval entirely above the target level.",
        "",
        f"Across {len(paired)} replicate-paired alternative comparisons, the "
        "250-person family-history extension changed the rejection rate by "
        f"{min(differences):+.3f} to {max(differences):+.3f}. The 95% paired "
        f"Monte Carlo interval is entirely above zero in {len(positive)} "
        f"comparisons, entirely below zero in {len(negative)}, and includes "
        f"zero in {uncertain}.",
        "",
        "No minimum worthwhile power increment or burden trade-off was "
        "prespecified. These simulations therefore measure the increment; they "
        "do not by themselves label the recruitment, staging, adjudication and "
        "budget burden justified. Refusals are non-rejections in every rate and "
        "paired comparison.",
        "",
        "A null cell whose Wilson interval is entirely above .025 makes the "
        "matching alternative result an alternative rejection rate, not a "
        "defensible power estimate.",
        "",
        "## Level",
        "",
        "| Yield | Extension | Outcome | Hearing bound | Null | Level | MC SE | "
        "95% Wilson interval | Refused | Assessment |",
        "|---:|---:|---|---|---|---:|---:|---|---:|---|",
    ]
    for row in sorted(
        level_rows,
        key=lambda item: (
            item["relative_yield"],
            item["family_history_extension"],
            item["outcome"],
            item["hearing_arm"],
            item["truth"],
        ),
    ):
        lines.append(
            f"| {row['relative_yield']:.1f} | "
            f"{row['family_history_extension']} | {_plain_label(row['outcome'])} "
            f"| {_plain_label(row['hearing_arm'])} | "
            f"{_plain_label(row['truth'])} | "
            f"{_format_rate(row['power_or_level'])} | "
            f"{row['binomial_monte_carlo_se']:.3f} | "
            f"{_format_interval(row['wilson_interval'])} | "
            f"{row['refusals']} | {_plain_label(row['level_assessment'])} |"
        )

    lines.extend(
        [
            "",
            "## Power and incremental extension information",
            "",
            "| Yield | Outcome | Hearing bound | Core rate | Extension rate | "
            "Paired difference | Paired MC SE | 95% paired MC interval | "
            "Core/extension refused | Reported as power |",
            "|---:|---|---|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['relative_yield']:.1f} | {_plain_label(row['outcome'])} | "
            f"{_plain_label(row['hearing_arm'])} | "
            f"{row['core_power_or_rejection_rate']:.3f} | "
            f"{row['extension_power_or_rejection_rate']:.3f} | "
            f"{row['extension_minus_core_rejection_difference']:+.3f} | "
            f"{row['paired_monte_carlo_se']:.3f} | "
            f"{_format_interval(row['paired_normal_interval'])} | "
            f"{row['core_refusals']}/{row['extension_refusals']} | "
            f"{'yes' if row['reported_as_power'] else 'no — level high'} |"
        )
    return "\n".join(lines), verdict


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise the disposable Aim 2 grant-design prototype."
    )
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--verdict", type=Path, default=DEFAULT_VERDICT)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        manifest, cells_file_sha256 = _read_json(arguments.cells)
        cells = _validate_manifest(manifest)
    except SummaryError as error:
        failure = {
            "prototype": "PROTOTYPE — WIPE ME",
            "complete": False,
            "alpha": ALPHA,
            "manifest_errors": [str(error)],
            "missing_cells": [],
            "verdict": None,
        }
        _write_json(arguments.summary, failure)
        _write_text(
            arguments.verdict,
            "\n".join(
                [
                    "# Prototype verdict",
                    "",
                    "> **PROTOTYPE — WIPE ME.**",
                    "",
                    "## No verdict: invalid cell manifest",
                    "",
                    str(error),
                ]
            ),
        )
        print(f"No verdict: {error}", file=sys.stderr)
        return 1

    expected_files = {str(cell["result_file"]) for cell in cells}
    observed_files = {path.name for path in arguments.results_dir.glob("cell_*.json")}
    unexpected_files = sorted(observed_files - expected_files)
    summaries: dict[int, dict[str, Any]] = {}
    decisions: dict[int, dict[int, dict[str, Any]]] = {}
    missing_cells: list[dict[str, Any]] = []
    manifest_hashes_by_design: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for cell in cells:
        cell_id = int(cell["cell_id"])
        result_file = str(cell["result_file"])
        path = arguments.results_dir / result_file
        if not path.is_file():
            missing_cells.append(
                {
                    "cell_id": cell_id,
                    "result_file": result_file,
                    "reason": "result file is absent",
                }
            )
            continue
        try:
            summary, replicate_decisions = _validate_result(
                path, cell, str(manifest["source_commit"])
            )
        except SummaryError as error:
            missing_cells.append(
                {
                    "cell_id": cell_id,
                    "result_file": result_file,
                    "reason": str(error),
                }
            )
            continue
        summaries[cell_id] = summary
        decisions[cell_id] = replicate_decisions
        manifest_hashes_by_design[str(cell["design_id"])][
            summary["design_manifest_sha256"]
        ].append(cell_id)

    for design_id, hashes in manifest_hashes_by_design.items():
        if len(hashes) <= 1:
            continue
        affected = sorted(
            cell_id for cell_ids in hashes.values() for cell_id in cell_ids
        )
        reason = (
            f"design_id {design_id!r} has inconsistent design_manifest_sha256 "
            "values across result files"
        )
        for cell_id in affected:
            summary = summaries.pop(cell_id)
            decisions.pop(cell_id)
            missing_cells.append(
                {
                    "cell_id": cell_id,
                    "result_file": summary["result_file"],
                    "reason": reason,
                }
            )

    paired, analysis_errors = _paired_extension_differences(cells, summaries, decisions)
    missing_cells.sort(key=lambda item: item["cell_id"])
    complete = (
        not missing_cells
        and not unexpected_files
        and not analysis_errors
        and len(summaries) == len(cells)
    )
    ordered_summaries = [summaries[cell_id] for cell_id in sorted(summaries)]
    output: dict[str, Any] = {
        "schema_version": 1,
        "prototype": manifest["prototype"],
        "source_commit": manifest["source_commit"],
        "generated_at": datetime.now(UTC).isoformat(),
        "complete": complete,
        "alpha": ALPHA,
        "cells_file": str(arguments.cells),
        "cells_file_sha256": cells_file_sha256,
        "results_directory": str(arguments.results_dir),
        "expected_cells": len(cells),
        "valid_cells": len(summaries),
        "missing_cells": missing_cells,
        "unexpected_result_files": unexpected_files,
        "analysis_errors": analysis_errors,
        "refusals_in_denominator": True,
        "cell_summaries": ordered_summaries,
        "paired_extension_differences": paired,
        "verdict": None,
    }

    if complete:
        markdown, verdict = _complete_markdown(ordered_summaries, paired)
        output["verdict"] = verdict
    else:
        markdown = _incomplete_markdown(
            expected_cells=len(cells),
            valid_cells=len(summaries),
            missing_cells=missing_cells,
            unexpected_files=unexpected_files,
            analysis_errors=analysis_errors,
        )
    _write_json(arguments.summary, output)
    _write_text(arguments.verdict, markdown)

    if complete:
        print(
            f"Complete: {len(cells)} cells summarised in {arguments.summary}; "
            f"verdict written to {arguments.verdict}."
        )
        return 0
    print(
        f"No verdict: {len(summaries)} of {len(cells)} cells are valid. "
        f"See {arguments.verdict}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
