"""Produce participant-free standard receipts through the installed wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import asterism
import numpy as np

if __package__:
    from .cross_platform_probe import PAIRS, SEED, mixed_traits, synthetic_problem
else:
    from cross_platform_probe import PAIRS, SEED, mixed_traits, synthetic_problem
"""Imported the shared public fixture in module and direct-script execution."""


class SyntheticReceiptError(RuntimeError):
    """A stable refusal to manufacture incomplete release evidence."""


@dataclass(frozen=True, slots=True)
class ReceiptJob:
    """One public synthetic analysis and its non-identifying receipt metadata."""

    analysis_id: str
    """Named the stable supported analysis from the release manifest."""
    design: dict[str, Any]
    """Held only non-identifying facts checked by release preflight."""
    model: dict[str, Any]
    """Recorded estimator and model choices required to reproduce the fit."""
    fit: Callable[[], dict[str, Any]]
    """Deferred every numerical operation until public preflight passed."""


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 of one saved artifact.

    Args:
        path: Present regular file whose exact bytes are identified.

    Returns:
        Lowercase hexadecimal SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_sha256(problem: dict[str, Any]) -> str:
    """Commit to every deterministic synthetic input without retaining its values.

    Args:
        problem: Participant-free array fixture shared by the supported examples.

    Returns:
        Lowercase hexadecimal SHA-256 over names, types, shapes and exact bytes.
    """
    digest: Any = hashlib.sha256()
    """Created one incremental commitment independent of JSON float rendering."""

    for name in sorted(problem):
        array: np.ndarray = np.ascontiguousarray(problem[name])
        """Normalised one synthetic input to an unambiguous byte representation."""

        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    """Committed to the complete named fixture in deterministic key order."""

    return str(digest.hexdigest())


def subject_order() -> list[str]:
    """Return ephemeral identifiers for the deterministic synthetic row order."""
    return [f"synthetic-{index:03d}" for index in range(2 * PAIRS)]


def fit_one_trait_receipt(problem: dict[str, Any], commitment: str) -> dict[str, Any]:
    """Fit the Gaussian model and return all three supported quantities."""
    prepared: Any = asterism.prepare(
        problem["design"],
        problem["relationship"],
        subject_order_sha256=commitment,
    )
    """Prepared the documented one-trait model with immutable row identity."""

    return dict(prepared.fit(problem["response"]))


def fit_component_receipt(problem: dict[str, Any], commitment: str) -> dict[str, Any]:
    """Fit the scale-invariant mean-diagonal component-reporting branch."""
    model: asterism.ComponentModel = asterism.ComponentModel(
        [problem["relationship"]],
        problem["design"],
        subject_order_sha256=commitment,
    )
    """Built one supplied relationship component plus the residual."""

    record: dict[str, Any] = model.fit(problem["response"])
    """Estimated the mean-diagonal contributions and proportions."""

    record["mean_diagonal_proportion_interval"] = model.interval(
        problem["response"], component=0
    )
    """Attached the profile interval for the same scale-invariant proportion."""

    return record


def fit_bivariate_receipt(problem: dict[str, Any], commitment: str) -> dict[str, Any]:
    """Fit continuous genetic correlation with its interval and test."""
    people: int = 2 * PAIRS
    """Read the fixed synthetic roster size."""

    observed: np.ndarray = np.ones((people, 2), dtype=bool)
    """Marked both synthetic traits observed for every person."""

    pair_design: np.ndarray = np.tile(np.eye(2), (people, 1))
    """Gave each trait its own intercept in person-within-trait order."""

    response: np.ndarray = np.column_stack(
        [problem["response"], problem["second_response"]]
    ).ravel()
    """Interleaved the two traits in the public bivariate row order."""

    model: asterism.BivariateModel = asterism.BivariateModel(
        problem["relationship"],
        observed,
        pair_design,
        subject_order_sha256=commitment,
    )
    """Built the documented unbalanced-capable bivariate model."""

    record: dict[str, Any] = model.fit(response)
    """Estimated the genetic correlation."""

    record["interval"] = model.interval(response, "rho_g")
    """Attached the profile interval for genetic correlation."""

    record["test"] = model.test(response, "rho_g", 0.0)
    """Attached the interior zero-correlation likelihood-ratio test."""

    return record


def fit_continuous_gxe_receipt(
    problem: dict[str, Any], commitment: str
) -> dict[str, Any]:
    """Fit the supported random-regression GxE report set."""
    model: asterism.GxeModel = asterism.GxeModel(
        problem["relationship"],
        problem["environment"],
        problem["design"],
        surface="random_regression",
        subject_order_sha256=commitment,
    )
    """Selected the crossover-capable supported surface before fitting."""

    record: dict[str, Any] = model.fit(problem["response"])
    """Estimated heritability and genetic correlation over the fixed grid."""

    record["heritability_interval"] = model.interval(
        problem["response"], quantity="heritability", first=0.0
    )
    """Profiled heritability at the centre of the supported environment."""

    record["genetic_correlation_interval"] = model.interval(
        problem["response"], quantity="correlation", first=-1.0, second=1.0
    )
    """Profiled the end-to-end genetic correlation on the fixed grid."""

    record["correlation_test"] = model.test(problem["response"], "correlation")
    """Tested whether the same genetic effects act across environments."""

    record["interaction_test"] = model.test(problem["response"], "interaction")
    """Tested the broader constant-genetic-covariance null."""

    return record


def fit_discrete_gxe_receipt(
    problem: dict[str, Any], commitment: str
) -> dict[str, Any]:
    """Fit both-environment genetic quantities and calibrated tests."""
    model: asterism.DiscreteGxeModel = asterism.DiscreteGxeModel(
        problem["relationship"],
        problem["discrete_environment"],
        problem["design"],
        levels=(0.0, 1.0),
        subject_order_sha256=commitment,
    )
    """Named both synthetic environment levels at model construction."""

    record: dict[str, Any] = model.fit(problem["response"])
    """Estimated group-specific variances and their genetic correlation."""

    record["correlation_interval"] = model.correlation_interval(problem["response"])
    """Attached the direct genetic-correlation profile interval."""

    record["gene_by_environment_test"] = model.test(
        problem["response"], "gene_by_environment"
    )
    """Attached the headline joint genetic-difference test."""

    record["correlation_test"] = model.test(problem["response"], "correlation")
    """Attached the same-genes boundary test."""

    record["genetic_variance_test"] = model.test(problem["response"], "genetic")
    """Attached the equal-genetic-variance interior test."""

    return record


def fit_liability_receipt(problem: dict[str, Any], commitment: str) -> dict[str, Any]:
    """Fit liability heritability with its profile interval and boundary test."""
    model: asterism.LiabilityModel = asterism.LiabilityModel(
        problem["relationship"],
        problem["binary_status"],
        problem["design"],
        subject_order_sha256=commitment,
    )
    """Built the public binary-liability model with exact row identity."""

    record: dict[str, Any] = model.fit()
    """Estimated liability-scale heritability by maximum likelihood."""

    record["interval"] = model.interval()
    """Attached the liability heritability profile interval."""

    record["test"] = model.test()
    """Attached the calibrated boundary-null test."""

    return record


def fit_tobit_receipt(problem: dict[str, Any], commitment: str) -> dict[str, Any]:
    """Fit censored hearing heritability with its interval and test."""
    arguments: tuple[Any, ...] = (
        problem["relationship"],
        problem["censored_values"],
        problem["censoring"],
        problem["limits"],
        problem["design"],
    )
    """Collected the exact shared public inputs for all three Tobit operations."""

    record: dict[str, Any] = asterism.tobit_fit(
        *arguments, subject_order_sha256=commitment
    )
    """Estimated complete-trait heritability under right censoring."""

    record["interval"] = asterism.tobit_interval(*arguments)
    """Attached the censoring-aware heritability profile interval."""

    record["test"] = asterism.tobit_test(*arguments)
    """Attached the boundary-null heritability test."""

    return record


def fit_mixed_bivariate_receipt(
    problem: dict[str, Any], commitment: str
) -> dict[str, Any]:
    """Fit the exact binary-diagnosis by censored-hearing pairing."""
    first, second = mixed_traits(problem)
    """Built the two documented public trait specifications."""

    arguments: tuple[Any, ...] = (
        problem["relationship"],
        first,
        second,
        problem["design"],
    )
    """Collected the joint inputs shared by fit, interval and test."""

    record: dict[str, Any] = asterism.mixed_bivariate_fit(
        *arguments, subject_order_sha256=commitment
    )
    """Estimated the cross-trait genetic correlation by maximum likelihood."""

    record["interval"] = asterism.mixed_bivariate_interval(
        *arguments, coordinate="genetic_correlation"
    )
    """Attached the supported genetic-correlation profile interval."""

    record["test"] = asterism.mixed_bivariate_test(
        *arguments, coordinate="genetic_correlation"
    )
    """Attached the interior zero-genetic-correlation test."""

    return record


def fit_spatial_receipt(problem: dict[str, Any], commitment: str) -> dict[str, Any]:
    """Fit the spatial component and its deterministic presence test."""
    model: asterism.SpatialModel = asterism.SpatialModel(
        [problem["relationship"]],
        problem["distance"],
        problem["design"],
        subject_order_sha256=commitment,
    )
    """Built one fixed relationship component plus the spatial component."""

    record: dict[str, Any] = model.fit(problem["response"])
    """Estimated the covariance contributions and descriptive spatial range."""

    record["spatial_presence_test"] = model.bootstrap(
        problem["response"], replicates=7, seed=SEED
    )
    """Attached a small deterministic public bootstrap as the end-to-end smoke."""

    return record


def receipt_jobs(problem: dict[str, Any], commitment: str) -> list[ReceiptJob]:
    """Return all supported jobs in release-manifest order.

    Args:
        problem: Shared deterministic participant-free fixture.
        commitment: SHA-256 of the ephemeral synthetic subject order.

    Returns:
        Nine lazy fit jobs carrying only non-identifying receipt metadata.
    """
    sample_size: int = 2 * PAIRS
    """Named the common synthetic roster size once for every design summary."""

    censoring_share: float = float(np.mean(problem["censoring"] != 0))
    """Measured the deterministic synthetic hearing censoring fraction."""

    common: dict[str, Any] = {"sample_size": sample_size, "largest_family": 2}
    """Collected design facts shared by the sibling-pair examples."""

    return [
        ReceiptJob(
            "one_trait_gaussian_heritability",
            {
                **common,
                "trait_type": "continuous",
                "components": "additive_relationship",
            },
            {"estimator": "reml", "transform": "none"},
            partial(fit_one_trait_receipt, problem, commitment),
        ),
        ReceiptJob(
            "several_covariance_components",
            {
                **common,
                "trait_type": "continuous",
                "components": "relationship_matrices",
                "component_reporting": "mean_diagonal",
            },
            {"estimator": "reml", "component_reporting": "mean_diagonal"},
            partial(fit_component_receipt, problem, commitment),
        ),
        ReceiptJob(
            "bivariate_genetic_correlation",
            {
                **common,
                "trait_type": "continuous_pair",
                "components": "additive_relationship",
            },
            {"estimator": "reml", "target": "rho_g"},
            partial(fit_bivariate_receipt, problem, commitment),
        ),
        ReceiptJob(
            "continuous_gene_by_environment",
            {
                **common,
                "trait_type": "continuous",
                "components": "random_regression_genetic_surface",
            },
            {
                "estimator": "reml",
                "surface": "random_regression",
                "reporting_grid": [-1.0, 0.0, 1.0],
            },
            partial(fit_continuous_gxe_receipt, problem, commitment),
        ),
        ReceiptJob(
            "discrete_gene_by_environment",
            {
                **common,
                "trait_type": "continuous_with_binary_environment",
                "components": "group_specific_additive_relationship",
            },
            {"estimator": "reml", "levels": [0.0, 1.0]},
            partial(fit_discrete_gxe_receipt, problem, commitment),
        ),
        ReceiptJob(
            "binary_liability_heritability",
            {
                **common,
                "trait_type": "binary",
                "components": "additive_relationship",
            },
            {"estimator": "ml", "scale": "liability"},
            partial(fit_liability_receipt, problem, commitment),
        ),
        ReceiptJob(
            "one_trait_censored",
            {
                **common,
                "trait_type": "right_censored_continuous",
                "components": "additive_relationship",
                "censoring_share": censoring_share,
            },
            {"estimator": "ml", "censoring": "right"},
            partial(fit_tobit_receipt, problem, commitment),
        ),
        ReceiptJob(
            "mixed_binary_censored_genetic_correlation",
            {
                **common,
                "trait_type": "mixed_pairings_with_continuous",
                "components": "additive_relationship",
                "censoring_share": censoring_share,
            },
            {
                "estimator": "ml",
                "kinds": ["censored", "continuous"],
                "target": "genetic_correlation",
            },
            partial(fit_mixed_bivariate_receipt, problem, commitment),
        ),
    ]


def runtime_wheel(wheel_paths: list[Path]) -> Path:
    """Select the saved wheel matching the interpreter's current platform.

    Args:
        wheel_paths: Complete release artifact set.

    Returns:
        The sole present wheel matching Linux or macOS.

    Raises:
        SyntheticReceiptError: If no unambiguous runtime artifact exists.
    """
    resolved: list[Path] = [path.resolve() for path in wheel_paths]
    """Normalised the caller's artifact paths before platform selection."""

    if not resolved or any(
        path.suffix != ".whl" or not path.is_file() for path in resolved
    ):
        raise SyntheticReceiptError("SYNTHETIC_RECEIPT_WHEEL_SET_INVALID")
    marker: str = "manylinux" if sys.platform.startswith("linux") else "macosx"
    """Selected the standard wheel filename marker for this interpreter."""

    matches: list[Path] = [path for path in resolved if marker in path.name]
    """Found artifacts capable of having supplied this platform's extension."""

    if len(matches) != 1:
        raise SyntheticReceiptError("SYNTHETIC_RECEIPT_RUNTIME_WHEEL_AMBIGUOUS")
    return matches[0]


def json_bytes(value: Any) -> bytes:
    """Serialize one receipt strictly, rejecting every non-finite number."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_synthetic_analysis_receipts(
    output_directory: Path,
    wheel_paths: list[Path],
) -> list[dict[str, Any]]:
    """Write one complete standard receipt for every supported analysis.

    Args:
        output_directory: New directory that will contain receipt JSON files.
        wheel_paths: Saved release wheels whose exact bytes identify the artifacts.

    Returns:
        Receipt index entries in release-manifest order.

    Raises:
        SyntheticReceiptError: If the installed build is not a fixed release.
    """
    manifest: dict[str, Any] = asterism.release_manifest()
    """Read the support contract embedded in the installed extension."""

    if manifest.get("release") is not True:
        raise SyntheticReceiptError("SYNTHETIC_RECEIPT_BUILD_NOT_RELEASED")
    build: dict[str, Any] = asterism.build_identity()
    """Read immutable source and dependency identity from the installed wheel."""

    if build.get("source_dirty") is not False:
        raise SyntheticReceiptError("SYNTHETIC_RECEIPT_BUILD_DIRTY")
    if output_directory.exists():
        raise SyntheticReceiptError("SYNTHETIC_RECEIPT_OUTPUT_EXISTS")
    wheel: Path = runtime_wheel(wheel_paths)
    """Selected the exact saved artifact installed for this runtime platform."""

    problem: dict[str, Any] = synthetic_problem()
    """Created the deterministic participant-free inputs used by every job."""

    order: list[str] = subject_order()
    """Created ephemeral synthetic identifiers used only to bind row order."""

    commitment: str = asterism.subject_order_commitment(order)
    """Committed to the shared fitted order without retaining identifiers."""

    jobs: list[ReceiptJob] = receipt_jobs(problem, commitment)
    """Prepared all numerical callbacks lazily behind their preflight checks."""

    manifest_ids: list[str] = [
        str(analysis["id"])
        for analysis in manifest.get("analyses", [])
        if isinstance(analysis, dict) and analysis.get("support") == "supported_in_0_1"
    ]
    """Read the complete ordered supported-analysis inventory from the wheel."""

    if [job.analysis_id for job in jobs] != manifest_ids:
        raise SyntheticReceiptError("SYNTHETIC_RECEIPT_ANALYSIS_INVENTORY_MISMATCH")

    artifact_sha256: str = sha256(wheel)
    """Committed every receipt to the exact saved runtime wheel bytes."""

    inputs_sha256: str = fixture_sha256(problem)
    """Committed every receipt to the complete participant-free fixture."""

    receipts: list[dict[str, Any]] = []
    """Accumulated receipts in memory so a failed set leaves no partial evidence."""

    for job in jobs:
        receipt: dict[str, Any] = asterism.run_analysis(
            job.analysis_id,
            job.design,
            job.fit,
            model=job.model,
            provenance={
                "wheel_sha256": artifact_sha256,
                "dependency_lock_sha256": build["uv_lock_sha256"],
                "consumer_commit": build["source_commit"],
                "input_commitments": {"synthetic_fixture_sha256": inputs_sha256},
            },
            subject_order=order,
        )
        """Ran the checks and the fit through the standard public seam."""

        receipts.append(receipt)
    """Produced every receipt before allowing any evidence directory to exist."""

    failures: list[str] = [
        f"{receipt['analysis']}:{receipt['outcome']}"
        for receipt in receipts
        if receipt.get("outcome") != "fitted"
        or not all(dict(receipt.get("facts") or {}).values())
    ]
    """Named every analysis that did not fit cleanly."""

    if failures:
        raise SyntheticReceiptError(
            "SYNTHETIC_RECEIPT_INCOMPLETE:" + ",".join(failures)
        )

    output_directory.mkdir(parents=True)
    """Created the new evidence directory only after all nine receipts passed."""

    index: list[dict[str, Any]] = []
    """Accumulated durable receipt identities without copying their numerical fields."""

    for receipt in receipts:
        analysis_id: str = str(receipt["analysis"])
        """Read the stable identifier used as this receipt's filename."""

        path: Path = output_directory / f"{analysis_id}.json"
        """Selected one deterministic collision-free receipt path."""

        payload: bytes = json_bytes(receipt)
        """Serialized strictly before any bytes were written."""

        path.write_bytes(payload)
        """Persisted the complete standard receipt beside release evidence."""

        index.append(
            {
                "analysis_id": analysis_id,
                "outcome": "fitted",
                "path": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "wheel_sha256": artifact_sha256,
            }
        )
    """Identified every receipt by stable analysis, relative path and exact bytes."""

    index_path: Path = output_directory / "index.json"
    """Selected the canonical machine-readable receipt inventory."""

    index_path.write_bytes(json_bytes({"schema_version": 1, "receipts": index}))
    """Persisted the complete ordered inventory after all receipt files existed."""

    return index


def main() -> int:
    """Run the fail-closed installed-wheel receipt command."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Write all supported participant-free standard receipts."
    )
    """Defined the shell-free interface used by release automation."""

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, nargs="+", required=True)
    arguments: argparse.Namespace = parser.parse_args()
    """Parsed the new evidence directory and saved artifact set."""

    try:
        write_synthetic_analysis_receipts(arguments.output, arguments.wheel)
    except SyntheticReceiptError as error:
        print(f"synthetic analysis receipts refused: {error}", file=sys.stderr)
        return 1
    print("Every supported analysis produced a complete synthetic receipt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
