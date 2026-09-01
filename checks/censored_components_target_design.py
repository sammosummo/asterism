"""Does the several-component censored model work at the design it will be used on?

The existing censored target-design rule, `tobit_target_design`, qualifies one
structured component and a residual. Every other censored rule declares the same
pair, so **nothing had been measured beyond two components at the target
design**: the shares reported for a person-level or household term at a
hundred-and-eighty-person family were an extrapolation from sibling pairs.

The structure is the reviewed, participant-free aggregate the censored rule
already uses: 1,909 analysed people, 202 relationship components with the
largest at 180, and six fixed-effect columns. Nothing here is a participant
value or a reconstructed pedigree.

Two changes make it the design this model actually meets. Each person
contributes **two records**, which is what wants a person-level component --
without one the resemblance between a person's own two ears has nowhere to go
but the heritability. And each family is cut into **households of three**,
giving a shared-environment kernel beside the genetic one. Both are the
construction `checks/sequential_against_ghk.py` climbs its component ladder on,
so that ladder's verdict on censored dimension applies to this design rather
than to a different one.

Run with:

    uv run --no-project python checks/censored_components_target_design.py \
        --workers 8 \
        --checkpoint .scientific-checkpoints/censored-components-target.json

One scientific grid can be split safely across independent scheduler tasks. A
ten-task Slurm array passes its zero-based task number to ``--shard-index`` and
the common value 10 to ``--shard-count``. Every task may receive the same base
``--checkpoint`` path: the command derives a distinct, labelled file for it.
After all tasks finish, pass those ten files to ``--merge-checkpoints``. Shards
only produce complete resumable rows; only the merge scores and writes evidence.

The ordinary release gate uses the explicitly asymptotic fifty-fifty boundary
reference. A constrained-null parametric bootstrap remains available with
``--test-reference bootstrap --bootstrap-replicates 999`` when finite-null
calibration is specifically wanted. Its inner thread count is allocated from
the CPUs visible to the job; ``--bootstrap-threads`` can select a smaller
allocation within the same outer-by-inner CPU bound. The instrument limits are
fixed before outcomes in either mode, and boundary mass remains a diagnostic
only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy
from scipy.stats import beta, chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asterism
from censoring_design import right_censoring_limit
from sequential_against_ghk import HOUSEHOLD_SIZE, household_kernel
from tobit_target_design import coverage_decision, level_decision, load_target_design

FIXTURE: Path = (
    Path(__file__).resolve().parent
    / "design_fixtures"
    / "one_trait_gaussian_heritability.json"
)
"""Named the reviewed aggregate this design is matched to."""

FIXTURE_SHA256: str = "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e"
"""Pinned the exact reviewed fixture bytes outside the file itself."""

RECORDS: int = 2
"""Records each person contributes, which is two ears at one frequency."""

CENSORING_SHARES: tuple[float, float] = (0.52, 0.75)
"""Fixed the two observed high-frequency audiogram censoring levels."""

HERITABLE_SHARES: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
"""Genetic, person-level, household and residual shares in the interior cell."""

NULL_SHARES: tuple[float, float, float, float] = (0.0, 0.3846, 0.3077, 0.3077)
"""The same design with no genetic variance, the other three sharing it out.

The person-level and household terms stay present and keep their ratio to one
another. A null that removed them as well would test the boundary rule on a
one-component problem, which is the released compatibility case measured by
the separate fixed-instrument reruns.
"""

TOTAL_VARIANCE: float = 4.0
"""Latent complete-trait variance, matching the existing censored rule."""

TESTED: int = 0
"""Component scored: the genetic one, whose share is the heritability."""

LEVEL: float = 0.05
"""Boundary-test type-I-error level."""

DEVIANCE_SETTLING_THRESHOLD: float = 1.0e-6
"""Mirror the public likelihood-ratio test's operational point mass.

``src/deviance.rs`` treats smaller positive gaps as rounding between two
independent likelihood searches.  The analytic record retains that raw gap,
while its p-value already honours the point mass, so this campaign must use the
same threshold when it records and validates the atom diagnostic.
"""

DEFAULT_TEST_REFERENCE: str = "asymptotic"
"""Use the ordinary analytic reference unless finite-null calibration is asked."""

ASYMPTOTIC_TEST_RULE: str = "asymptotic_mixture_50_50"
"""Name the explicit several-component asymptotic boundary reference."""

BOOTSTRAP_TEST_RULE: str = "parametric_bootstrap_add_one"
"""Name the optional constrained-null parametric-bootstrap reference."""

BOOTSTRAP_REPLICATES: int = 999
"""Null draws for the level gate, giving fifty exact ranks at alpha 0.05.

The add-one denominator is 1,000, so ``p <= 0.05`` is exactly attainable and
means at most 49 simulated exceedances. The campaign is configured to assess
the same 999-draw rule exposed for reported analyses.
"""

BOOTSTRAP_BASE_SEED: int = 1_981_000
"""Separated bootstrap streams from every outer target-design response stream."""

NOMINAL_COVERAGE: float = 0.975
"""Nominal coverage of the interval at this design, and it is not 0.95.

**The upper end of a component proportion cannot be reached when a person
contributes more than one record.** A proportion of one puts every other
component, the residual included, at nought, leaving a covariance of the
additive matrix alone -- and spread over records that matrix gives a person's
own records a correlation of exactly one, so it is singular. The profile cannot
be evaluated there, and by the rule in `src/interval.rs` an end that could not
be evaluated is covered rather than placed on evidence never gathered. So the
upper end is the bound, every time, for structural reasons and not because a
fit went wrong. Measured here at one, two and three components alike.

What is left to measure is the lower end, and a two-sided 95 per cent profile
interval puts 0.025 in each tail. Forcing the upper end to the bound hands the
upper tail back, so the quantity this campaign can score is 0.975. Scoring it
against 0.95 would ask the interval to fail a quarter of the time in a
direction the design has closed off.
"""

MONTE_CARLO_CONFIDENCE: float = 0.95
"""Confidence used by the one-sided exact anti-conservatism decisions."""

REPLICATES: int = 200
"""Replicates per scenario per censoring share; four cells make 800."""

WORKERS: int = 8
"""Processes used by the ordinary analytic campaign."""

BASE_SEED: int = 981_000
"""Base seed separating this rule's replicates from every other check."""

MEAN_COEFFICIENTS: tuple[float, float, float, float, float, float] = (
    1.0,
    0.2,
    -0.1,
    0.05,
    0.3,
    -0.2,
)
"""Fixed latent-mean coefficients used to set limits and generate outcomes."""

TARGET_ROWS: int = 3_818
"""Rows in the pinned design: 1,909 people, with two records each."""

MAXIMUM_BLOCK_ROWS: int = 360
"""Rows in the pinned design's largest 180-person family."""

LADDER_QUALIFIED_DIMENSION: int = 600
"""Largest censored dimension `sequential_against_ghk.py` has climbed.

Its own record says 600 is the largest rung climbed and not a limit that was
found: the ladder ran out before the approximation did. This check reports the
largest censored block it actually produced against that number, so a design
that outgrew the evidence says so rather than passing quietly.
"""

STATE: dict[str, Any] = {}
"""Held each worker's copy of the design, built once rather than per replicate."""


def in_operational_lrt_atom(statistic: float) -> bool:
    """Apply the public test's rounding threshold to a non-negative deviance."""
    return statistic < DEVIANCE_SETTLING_THRESHOLD


def shard_allocation(shard_index: int, shard_count: int) -> dict[str, int]:
    """Validate and describe one operational partition of the outer grid."""
    if (
        type(shard_index) is not int
        or type(shard_count) is not int
        or shard_count < 2
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_SHARD_ALLOCATION_INVALID")
    return {"shard_index": shard_index, "shard_count": shard_count}


def select_shard_jobs(
    jobs: list[tuple[str, float, int]],
    shard_index: int,
    shard_count: int,
) -> list[tuple[str, float, int]]:
    """Select one deterministic, disjoint stride through the declared grid."""
    shard_allocation(shard_index, shard_count)
    """Refused an invalid allocation before selecting any coordinate."""

    return [
        job
        for position, job in enumerate(jobs)
        if position % shard_count == shard_index
    ]


def shard_checkpoint_path(
    path: Path,
    shard_index: int,
    shard_count: int,
) -> Path:
    """Derive a unique checkpoint path from a common scheduler-array base."""
    shard_allocation(shard_index, shard_count)
    width: int = max(3, len(str(shard_count - 1)))
    """Kept lexical order equal to numeric order for ordinary array sizes."""

    label: str = f".shard-{shard_index:0{width}d}-of-{shard_count:0{width}d}"
    """Named both the selected partition and the full declared partition count."""

    return path.with_name(f"{path.stem}{label}{path.suffix}")


def allocated_cpu_count() -> int:
    """Return the logical CPUs available to this process or its scheduler job."""
    process_count: Any = getattr(os, "process_cpu_count", None)
    """Preferred Python's affinity-aware count where this runtime provides it."""

    count: int | None = process_count() if callable(process_count) else os.cpu_count()
    """Read an affinity-aware count where possible, otherwise the host count."""

    return max(1, count or 1)


def clopper_pearson(hits: int, trials: int) -> tuple[float, float]:
    """The exact interval on a binomial proportion.

    Args:
        hits: Successes observed.
        trials: Attempts made.

    Returns:
        The lower and upper limits of the 95 per cent exact interval.
    """
    low: float = beta.ppf(0.025, hits, trials - hits + 1) if hits else 0.0
    """Took the lower limit, or nought where nothing hit."""

    high: float = beta.ppf(0.975, hits + 1, trials - hits) if hits < trials else 1.0
    """Took the upper limit, or one where everything hit."""

    return float(low), float(high)


def require_test_rule(
    test: dict[str, Any],
    test_reference: str,
    expected_replicates: int | None,
) -> None:
    """Refuse a null result that did not use the predeclared reference."""
    expected_rule: str = (
        ASYMPTOTIC_TEST_RULE if test_reference == "asymptotic" else BOOTSTRAP_TEST_RULE
    )
    """Mapped the command mode to the public result rule it permits."""

    if test_reference not in {"asymptotic", "bootstrap"}:
        raise ValueError("CENSORED_COMPONENT_TARGET_TEST_REFERENCE_UNKNOWN")
    if test.get("rule") != expected_rule:
        raise ValueError("CENSORED_COMPONENT_TARGET_TEST_RULE_MISMATCH")
    if test_reference == "bootstrap" and (
        expected_replicates is None
        or expected_replicates < 1
        or test.get("replicates") != expected_replicates
        or test.get("requested") != expected_replicates
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_BOOTSTRAP_INCOMPLETE")


def campaign_source_paths(root: Path) -> list[Path]:
    """List scientific source, dependency and toolchain inputs to the campaign."""
    paths: set[Path] = {
        root / relative
        for relative in (
            "Cargo.lock",
            "Cargo.toml",
            "build.rs",
            "pyproject.toml",
            "rust-toolchain.toml",
            "uv.lock",
        )
    }
    """Included build inputs that can change numerics or the executable toolchain."""

    paths.update((root / "src").rglob("*.rs"))
    paths.update((root / "python" / "asterism").rglob("*.py"))
    paths.update(
        root / relative
        for relative in (
            "checks/censored_components_target_design.py",
            "checks/censoring_design.py",
            "checks/one_trait_coverage.py",
            "checks/sequential_against_ghk.py",
            "checks/tobit_target_design.py",
            "checks/design_fixtures/one_trait_gaussian_heritability.json",
        )
    )
    """Included the target check's exact local import and fixture closure.

    Unrelated checks and external fixtures are deliberately absent. Changing
    one of them cannot alter this campaign's generated values or inference, so
    binding it here would discard expensive valid rows for no scientific
    reason.
    """

    return sorted(path for path in paths if path.is_file())


def file_sha256(path: Path) -> str:
    """Hash one file without relying on its metadata or installed location."""
    digest: Any = hashlib.sha256()
    """Started the file's content digest."""

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_code_sha256(root: Path) -> str:
    """Bind resumable rows to every scientific and build input that made them."""
    digest: Any = hashlib.sha256()
    """Started one ordered digest over the complete implementation surface."""

    for path in campaign_source_paths(root):
        relative: str = path.relative_to(root).as_posix()
        """Named the input independently of the checkout's absolute location."""

        content: bytes = path.read_bytes()
        """Read the exact bytes that determine this campaign."""

        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def runtime_identity() -> dict[str, str]:
    """Describe the numerical runtime and platform that execute the campaign."""
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "scipy": scipy.__version__,
        "system": platform.system(),
    }


def reusable_checkpoint_signature(
    root: Path,
    replicates: int,
    test_reference: str,
    bootstrap_replicates: int | None = None,
    *,
    runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Describe scientific work reusable across metadata-only release rebuilds."""
    if test_reference not in {"asymptotic", "bootstrap"}:
        raise ValueError("CENSORED_COMPONENT_TARGET_TEST_REFERENCE_UNKNOWN")
    if test_reference == "asymptotic" and bootstrap_replicates is not None:
        raise ValueError(
            "CENSORED_COMPONENT_TARGET_BOOTSTRAP_REPLICATES_NOT_APPLICABLE"
        )
    if test_reference == "bootstrap" and (
        bootstrap_replicates is None or bootstrap_replicates < 1
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_BOOTSTRAP_REPLICATES_REQUIRED")
    """Required parameters belonging only to the selected reference mode."""

    fixture_sha256: str = file_sha256(
        root / "checks" / "design_fixtures" / "one_trait_gaussian_heritability.json"
    )
    """Recomputed the pinned fixture commitment from the candidate checkout."""

    if fixture_sha256 != FIXTURE_SHA256:
        raise ValueError("CENSORED_COMPONENT_TARGET_FIXTURE_MISMATCH")
    signature: dict[str, Any] = {
        "campaign": "fixed_instrument_component_test_v2",
        "replicates_per_cell": replicates,
        "test_reference": test_reference,
        "censoring_shares": list(CENSORING_SHARES),
        "fixture_sha256": fixture_sha256,
        "scientific_fingerprint_sha256": campaign_code_sha256(root),
        "runtime": runtime_identity() if runtime is None else runtime,
    }
    """Bound resumable rows to the selected scientific reference explicitly."""

    if bootstrap_replicates is not None:
        signature["bootstrap_replicates"] = bootstrap_replicates
        """Bound only bootstrap rows to their predeclared inner denominator."""
    return signature


def git_text(root: Path, arguments: list[str]) -> str:
    """Read one exact Git fact needed to verify the loaded extension."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    """Ran Git without a shell or any repository mutation."""

    if completed.returncode != 0:
        raise ValueError("CENSORED_COMPONENT_TARGET_GIT_IDENTITY_UNAVAILABLE")
    return completed.stdout.strip()


def validate_loaded_extension(root: Path) -> None:
    """Require the loaded native module to match this exact clean checkout now."""
    if git_text(root, ["status", "--porcelain", "--untracked-files=normal"]):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKOUT_NOT_CLEAN")
    current_commit: str = git_text(root, ["rev-parse", "HEAD"])
    """Read the clean checkout's exact source commit."""

    build: dict[str, Any] = asterism.build_identity()
    """Read the public identity, which verifies its digests against embedded bytes."""

    expected_commitments: dict[str, str] = {
        "cargo_lock_sha256": file_sha256(root / "Cargo.lock"),
        "release_manifest_sha256": file_sha256(root / "release.toml"),
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
    }
    """Recomputed every checkout commitment independently of the extension."""

    if (
        build.get("source_commit") != current_commit
        or build.get("source_dirty") is not False
        or any(
            build.get(field) != expected
            for field, expected in expected_commitments.items()
        )
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_EXTENSION_CHECKOUT_MISMATCH")


def producer_provenance(root: Path) -> dict[str, Any]:
    """Record the exact validated build used by one checkpoint invocation."""
    validate_loaded_extension(root)
    """Required provenance to describe the current clean checkout exactly."""

    build: dict[str, Any] = asterism.build_identity()
    """Read the native module's independently checked embedded build identity."""

    extension_sha256: str = asterism.installed_extension_sha256()
    """Hashed the exact native binary imported by this process."""

    return {
        "source_commit": build["source_commit"],
        "release_manifest_sha256": build["release_manifest_sha256"],
        "extension_sha256": extension_sha256,
        "build_identity": build,
    }


def checkpoint_signature(
    replicates: int,
    test_reference: str,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Validate this build, then describe scientifically reusable campaign work."""
    root: Path = Path(__file__).resolve().parent.parent
    """Located the checkout owning both Python and the loaded native module."""

    validate_loaded_extension(root)
    return reusable_checkpoint_signature(
        root,
        replicates,
        test_reference,
        bootstrap_replicates,
    )


def require_signature_reference(
    signature: dict[str, Any],
    test_reference: str,
    bootstrap_replicates: int | None,
) -> None:
    """Require orchestration parameters to match the persisted campaign mode."""
    if signature.get("test_reference") != test_reference:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_STALE")
    if test_reference == "bootstrap":
        if signature.get("bootstrap_replicates") != bootstrap_replicates:
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_STALE")
    elif bootstrap_replicates is not None or "bootstrap_replicates" in signature:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_STALE")


@lru_cache(maxsize=1)
def expected_censoring_limits() -> dict[float, float]:
    """Recompute both fixed instrument limits from the pinned design facts."""
    target: Any = load_target_design(FIXTURE, expected_sha256=FIXTURE_SHA256)
    """Loaded the reviewed person-level design independently of checkpoint rows."""

    design: npt.NDArray[np.float64] = np.repeat(target.design, RECORDS, axis=0)
    """Expanded each person's fixed effects to the two intended records."""

    coefficients: npt.NDArray[np.float64] = np.asarray(MEAN_COEFFICIENTS)
    """Recreated the mean model that determines the pre-outcome instrument limits."""

    means: npt.NDArray[np.float64] = design @ coefficients
    """Computed every marginal latent mean without reading a simulated response."""

    return {
        share: right_censoring_limit(means, TOTAL_VARIANCE, share)
        for share in CENSORING_SHARES
    }


def require_exact_field(
    row: dict[str, Any],
    field: str,
    expected_type: type[Any],
) -> Any:
    """Read a JSON primitive without accepting Python's bool-as-int coercion."""
    if field not in row or type(row[field]) is not expected_type:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    return row[field]


def validate_common_row(row: dict[str, Any]) -> tuple[str, float, int]:
    """Validate the primitive fields shared by complete and refused attempts."""
    scenario: str = require_exact_field(row, "scenario", str)
    """Read the exact scenario name."""

    share: float = require_exact_field(row, "censoring_share", float)
    """Read the exact intended censoring share."""

    replicate: int = require_exact_field(row, "replicate", int)
    """Read the exact outer replicate index."""

    worst: int = require_exact_field(row, "worst_censored_block", int)
    """Read the largest censored block produced by this outcome."""

    achieved: float = require_exact_field(row, "achieved_censoring_share", float)
    """Read the realised censoring share."""

    limit: float = require_exact_field(row, "censoring_limit", float)
    """Read the predeclared instrument limit applied to this outcome."""

    if (
        scenario not in {"null", "heritable"}
        or share not in CENSORING_SHARES
        or replicate < 0
        or not 0 <= worst <= MAXIMUM_BLOCK_ROWS
        or not math.isfinite(achieved)
        or not 0.0 <= achieved <= 1.0
        or not math.isfinite(limit)
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    censored_rows: float = achieved * TARGET_ROWS
    """Recovered the integer count represented by the stored achieved share."""

    if not math.isclose(
        censored_rows,
        round(censored_rows),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    if worst > round(censored_rows):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    return scenario, share, replicate


def validate_null_row(
    row: dict[str, Any],
    test_reference: str,
    bootstrap_replicates: int | None,
) -> None:
    """Validate a complete mode-specific null row and every derived decision."""
    expected_keys: set[str] = {
        "scenario",
        "censoring_share",
        "replicate",
        "worst_censored_block",
        "achieved_censoring_share",
        "censoring_limit",
        "p_value",
        "test_statistic",
        "test_rule",
        "rejected",
        "nuisance_at_bound",
        "in_lrt_atom",
    }
    """Named the fields measured by either null reference."""

    if test_reference == "bootstrap":
        expected_keys.update(
            {
                "bootstrap_exceedances",
                "bootstrap_replicates",
                "bootstrap_requested",
                "bootstrap_seed",
            }
        )
        """Permitted inner-draw counters only where that bootstrap actually ran."""

    if set(row) != expected_keys:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    p_value: float = require_exact_field(row, "p_value", float)
    """Read the selected reference's p-value."""

    statistic: float = require_exact_field(row, "test_statistic", float)
    """Read the observed likelihood-ratio statistic returned by the public test."""

    rule: str = require_exact_field(row, "test_rule", str)
    """Read the reported calibration rule."""

    rejected: bool = require_exact_field(row, "rejected", bool)
    """Read the stored level decision for consistency checking only."""

    nuisance: bool = require_exact_field(row, "nuisance_at_bound", bool)
    """Read whether the fitted constrained null encountered a nuisance boundary."""

    atom: bool = require_exact_field(row, "in_lrt_atom", bool)
    """Read the stored zero-statistic diagnostic for consistency checking only."""

    if (
        not math.isfinite(p_value)
        or not 0.0 <= p_value <= 1.0
        or not math.isfinite(statistic)
        or statistic < 0.0
        or nuisance
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    if test_reference == "asymptotic":
        require_test_rule({"rule": rule}, test_reference, bootstrap_replicates)
        operational_atom: bool = in_operational_lrt_atom(statistic)
        """Applied the same point-mass threshold that produced the public p-value."""

        expected_p: float = (
            1.0 if operational_atom else 0.5 * float(chi2.sf(statistic, 1))
        )
        """Recomputed the declared fifty-fifty asymptotic reference."""

        if (
            not math.isclose(p_value, expected_p, rel_tol=0.0, abs_tol=1e-14)
            or rejected is not (p_value <= LEVEL)
            or atom is not operational_atom
        ):
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
        return

    exceedances: int = require_exact_field(row, "bootstrap_exceedances", int)
    """Read the integer simulated exceedance count."""

    completed: int = require_exact_field(row, "bootstrap_replicates", int)
    """Read the number of successfully completed inner fits."""

    requested: int = require_exact_field(row, "bootstrap_requested", int)
    """Read the predeclared number of requested inner fits."""

    seed: int = require_exact_field(row, "bootstrap_seed", int)
    """Read the deterministic inner-stream seed."""

    require_test_rule(
        {"rule": rule, "replicates": completed, "requested": requested},
        test_reference,
        bootstrap_replicates,
    )
    if (
        bootstrap_replicates is None
        or not 0 <= exceedances <= bootstrap_replicates
        or (in_operational_lrt_atom(statistic) and exceedances != bootstrap_replicates)
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    expected_seed: int = (
        BOOTSTRAP_BASE_SEED
        + 7_919 * require_exact_field(row, "replicate", int)
        + 104_729 * int(require_exact_field(row, "censoring_share", float) * 100)
    )
    """Reconstructed the inner stream from the outer coordinate alone."""

    expected_p: float = (1.0 + exceedances) / (bootstrap_replicates + 1.0)
    """Reconstructed the add-one p-value from its integer numerator."""

    if (
        seed != expected_seed
        or not math.isclose(p_value, expected_p, rel_tol=0.0, abs_tol=1e-15)
        or rejected is not (p_value <= LEVEL)
        or atom is not in_operational_lrt_atom(statistic)
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")


def validate_heritable_row(row: dict[str, Any]) -> None:
    """Validate a complete interval row and recompute all derived decisions."""
    expected_keys: set[str] = {
        "scenario",
        "censoring_share",
        "replicate",
        "worst_censored_block",
        "achieved_censoring_share",
        "censoring_limit",
        "estimate",
        "lower",
        "upper",
        "upper_limited",
        "contains_upper_bound",
        "profile_failures",
        "bound_unreachable",
        "covered",
        "extra_failures",
    }
    """Named the exact persisted schema for a completed interval attempt."""

    if set(row) != expected_keys:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    estimate: float = require_exact_field(row, "estimate", float)
    """Read the fitted genetic proportion."""

    lower: float = require_exact_field(row, "lower", float)
    """Read the profile interval's lower endpoint."""

    upper: float = require_exact_field(row, "upper", float)
    """Read the profile interval's upper endpoint."""

    upper_limited: bool = require_exact_field(row, "upper_limited", bool)
    """Read whether the upper endpoint reached its structural limit."""

    contains: Any = row.get("contains_upper_bound")
    """Read the deliberately absent several-component boundary verdict."""

    if contains is not None or (upper_limited and upper != 1.0):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    profile_failures: int = require_exact_field(row, "profile_failures", int)
    """Read the number of unsuccessful constrained profile evaluations."""

    bound_unreachable: bool = require_exact_field(row, "bound_unreachable", bool)
    """Read the stored structural-bound state for consistency checking only."""

    covered: bool = require_exact_field(row, "covered", bool)
    """Read the stored coverage decision for consistency checking only."""

    extra_failures: int = require_exact_field(row, "extra_failures", int)
    """Read the stored non-structural profile-failure count."""

    if (
        not all(math.isfinite(value) for value in (estimate, lower, upper))
        or not 0.0 <= lower <= estimate <= upper <= 1.0
        or profile_failures < 0
        or extra_failures < 0
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
    expected_unreachable: bool = upper_limited
    """Recomputed whether the structural upper bound was unreachable."""

    expected_covered: bool = lower <= HERITABLE_SHARES[TESTED] <= upper
    """Recomputed coverage from the raw interval endpoints."""

    expected_extra: int = profile_failures - (1 if expected_unreachable else 0)
    """Removed only the one expected structural upper-bound failure."""

    if (
        expected_extra < 0
        or bound_unreachable is not expected_unreachable
        or covered is not expected_covered
        or extra_failures != expected_extra
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")


def validate_campaign_rows(
    rows: list[dict[str, Any]],
    jobs: list[tuple[str, float, int]],
    test_reference: str,
    bootstrap_replicates: int | None,
    *,
    require_complete: bool,
) -> None:
    """Refuse stale, malformed, duplicate or incomplete resumable results."""
    expected: set[tuple[str, float, int]] = set(jobs)
    """Named the complete predeclared outer coordinate grid."""

    seen: set[tuple[str, float, int]] = set()
    """Collected each coordinate at most once."""

    limits: dict[float, float] = {}
    """Required one fixed instrument limit for each intended censoring share."""

    expected_limits: dict[float, float] = expected_censoring_limits()
    """Independently recomputed both limits from the pinned pre-outcome design."""

    for row in rows:
        try:
            coordinate: tuple[str, float, int] = validate_common_row(row)
            """Read this retained row's complete scientific coordinate."""
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID"
            ) from error
        if coordinate not in expected:
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_COORDINATE_UNKNOWN")
        if coordinate in seen:
            raise ValueError(
                "CENSORED_COMPONENT_TARGET_CHECKPOINT_COORDINATE_DUPLICATE"
            )
        seen.add(coordinate)
        """Recorded this unique expected coordinate."""

        limit: float = require_exact_field(row, "censoring_limit", float)
        """Read the fixed instrument limit attached to this row."""

        previous_limit: float | None = limits.setdefault(coordinate[1], limit)
        """Remembered the first limit for this intended censoring share."""

        if previous_limit != limit or not math.isclose(
            limit,
            expected_limits[coordinate[1]],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_LIMIT_MISMATCH")

        if "refusal" in row:
            expected_refusal_keys: set[str] = {
                "scenario",
                "censoring_share",
                "replicate",
                "worst_censored_block",
                "achieved_censoring_share",
                "censoring_limit",
                "refusal",
            }
            """Named the complete schema of an explicitly refused attempt."""

            refusal: str = require_exact_field(row, "refusal", str)
            """Read the actual refusal rather than accepting an empty marker."""

            if set(row) != expected_refusal_keys or not refusal.strip():
                raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_ROW_INVALID")
        elif coordinate[0] == "null":
            validate_null_row(row, test_reference, bootstrap_replicates)
            """Revalidated the raw null result and all values derived from it."""
        else:
            validate_heritable_row(row)
            """Revalidated the raw interval and all values derived from it."""

    if require_complete and seen != expected:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_COORDINATES_MISSING")


def row_bound_unreachable(row: dict[str, Any]) -> bool:
    """Return the several-component upper-bound state without reading refusals."""
    return (
        "refusal" not in row
        and row["scenario"] == "heritable"
        and bool(row["upper_limited"])
    )


def row_failed(row: dict[str, Any]) -> bool:
    """Return whether an outer attempt refused or had an extra profile failure."""
    if "refusal" in row:
        return True
    if row["scenario"] != "heritable":
        return False
    return int(row["profile_failures"]) - int(row_bound_unreachable(row)) > 0


def validate_producer_history(
    history: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    """Require exact non-gating provenance for every retained outer coordinate."""
    owned: set[tuple[str, float, int]] = set()
    """Collected each completed coordinate under exactly one producing invocation."""

    for invocation in history:
        if set(invocation) != {"producer", "coordinates"}:
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_INVALID")
        producer: Any = invocation["producer"]
        """Selected this invocation's exact validated build record."""

        coordinates: Any = invocation["coordinates"]
        """Selected the outer coordinates first completed by that build."""

        if not isinstance(producer, dict) or not isinstance(coordinates, list):
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_INVALID")
        required: dict[str, type[Any]] = {
            "source_commit": str,
            "release_manifest_sha256": str,
            "extension_sha256": str,
            "build_identity": dict,
        }
        """Named the exact non-gating producer fields retained for audit."""

        if any(
            key not in producer or type(producer[key]) is not expected_type
            for key, expected_type in required.items()
        ):
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_INVALID")
        execution: Any = producer.get("execution")
        """Selected optional operational allocation metadata for this invocation."""

        if execution is not None:
            base_execution_keys: set[str] = {
                "allocated_cpu_count",
                "outer_workers",
                "bootstrap_threads_per_outer_worker",
            }
            """Named allocation facts common to unsharded and sharded workers."""

            shard_execution_keys: set[str] = {"shard_index", "shard_count"}
            """Named the optional scheduler partition as an indivisible pair."""

            if (
                not isinstance(execution, dict)
                or frozenset(execution)
                not in {
                    frozenset(base_execution_keys),
                    frozenset(base_execution_keys | shard_execution_keys),
                }
                or type(execution["allocated_cpu_count"]) is not int
                or execution["allocated_cpu_count"] < 1
                or type(execution["outer_workers"]) is not int
                or execution["outer_workers"] < 1
                or (
                    execution["bootstrap_threads_per_outer_worker"] is not None
                    and (
                        type(execution["bootstrap_threads_per_outer_worker"]) is not int
                        or execution["bootstrap_threads_per_outer_worker"] < 1
                    )
                )
            ):
                raise ValueError(
                    "CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_INVALID"
                )
            if shard_execution_keys <= set(execution):
                try:
                    shard_allocation(
                        execution["shard_index"],
                        execution["shard_count"],
                    )
                except ValueError as error:
                    raise ValueError(
                        "CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_INVALID"
                    ) from error
        build: dict[str, Any] = producer["build_identity"]
        """Selected the extension's complete public build identity."""

        if (
            len(producer["source_commit"]) != 40
            or len(producer["release_manifest_sha256"]) != 64
            or len(producer["extension_sha256"]) != 64
            or build.get("source_commit") != producer["source_commit"]
            or build.get("source_dirty") is not False
            or build.get("release_manifest_sha256")
            != producer["release_manifest_sha256"]
        ):
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_INVALID")
        for candidate in coordinates:
            if (
                not isinstance(candidate, list)
                or len(candidate) != 3
                or type(candidate[0]) is not str
                or type(candidate[1]) is not float
                or type(candidate[2]) is not int
            ):
                raise ValueError(
                    "CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_INVALID"
                )
            coordinate: tuple[str, float, int] = (
                candidate[0],
                candidate[1],
                candidate[2],
            )
            """Narrowed this JSON coordinate after exact primitive checks."""

            if coordinate in owned:
                raise ValueError(
                    "CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_DUPLICATE"
                )
            owned.add(coordinate)

    expected: set[tuple[str, float, int]] = {
        (
            require_exact_field(row, "scenario", str),
            require_exact_field(row, "censoring_share", float),
            require_exact_field(row, "replicate", int),
        )
        for row in rows
    }
    """Recovered the exact coordinates present in the scientific row collection."""

    if owned != expected:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_MISSING")


def require_shard_history_allocation(
    history: list[dict[str, Any]],
    allocation: dict[str, int],
) -> None:
    """Require every shard invocation to name the checkpoint partition it used."""
    for invocation in history:
        producer: dict[str, Any] = invocation["producer"]
        """Read the already structurally validated producer record."""

        execution: Any = producer.get("execution")
        """Selected the operational allocation recorded by this producer."""

        if not isinstance(execution, dict) or any(
            execution.get(field) != value for field, value in allocation.items()
        ):
            raise ValueError(
                "CENSORED_COMPONENT_TARGET_CHECKPOINT_SHARD_PROVENANCE_MISMATCH"
            )


def producer_identity(producer: dict[str, Any]) -> dict[str, Any]:
    """Return exact build provenance without its operational CPU allocation."""
    return {key: value for key, value in producer.items() if key != "execution"}


def read_checkpoint(
    path: Path,
    signature: dict[str, Any],
    jobs: list[tuple[str, float, int]],
    test_reference: str,
    bootstrap_replicates: int | None,
    *,
    shard: dict[str, int] | None = None,
    require_complete: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read compatible partial rows, or start empty when no checkpoint exists."""
    require_signature_reference(signature, test_reference, bootstrap_replicates)
    """Bound the selected row schema to the checkpoint signature itself."""

    if not path.exists():
        return [], []
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        """Parsed the operational checkpoint without treating it as evidence."""
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_INVALID") from error
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_STALE")
    stored_shard: Any = payload.get("shard")
    """Read operational partition metadata outside the scientific signature."""

    if stored_shard != shard:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_SHARD_MISMATCH")
    rows: Any = payload.get("rows")
    """Selected the retained completed outer attempts."""

    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_INVALID")
    validated: list[dict[str, Any]] = rows
    """Narrowed the JSON rows after their container and members were checked."""

    history: Any = payload.get("producer_history")
    """Selected non-gating exact-build provenance for every retained row."""

    if not isinstance(history, list) or not all(
        isinstance(invocation, dict) for invocation in history
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_PROVENANCE_INVALID")
    validated_history: list[dict[str, Any]] = history
    """Narrowed the producer history after checking its outer JSON shape."""

    validate_campaign_rows(
        validated,
        jobs,
        test_reference,
        bootstrap_replicates,
        require_complete=require_complete,
    )
    validate_producer_history(validated_history, validated)
    if shard is not None:
        require_shard_history_allocation(validated_history, shard)
    return validated, validated_history


def write_checkpoint(
    path: Path,
    signature: dict[str, Any],
    rows: list[dict[str, Any]],
    producer_history: list[dict[str, Any]],
    *,
    shard: dict[str, int] | None = None,
    writer_locked: bool = False,
) -> None:
    """Durably publish every completed attempt through a unique temporary file."""
    if not writer_locked:
        with checkpoint_writer(path):
            write_checkpoint(
                path,
                signature,
                rows,
                producer_history,
                shard=shard,
                writer_locked=True,
            )
        return
    """Acquired the campaign lock for callers not already holding it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    """Created only the explicitly requested operational checkpoint directory."""

    ordered: list[dict[str, Any]] = sorted(
        rows,
        key=lambda row: (
            str(row["scenario"]),
            float(row["censoring_share"]),
            int(row["replicate"]),
        ),
    )
    """Made checkpoint bytes independent of worker completion order."""

    ordered_history: list[dict[str, Any]] = [
        {
            "producer": invocation["producer"],
            "coordinates": sorted(
                invocation["coordinates"],
                key=lambda coordinate: (
                    str(coordinate[0]),
                    float(coordinate[1]),
                    int(coordinate[2]),
                ),
            ),
        }
        for invocation in producer_history
    ]
    """Made each producing invocation's coordinates independent of worker order."""

    checkpoint_payload: dict[str, Any] = {
        "signature": signature,
        "producer_history": ordered_history,
        "rows": ordered,
    }
    """Kept operational partition metadata outside scientific identity."""

    if shard is not None:
        checkpoint_payload["shard"] = shard
        """Stored the partition without making it part of scientific identity."""
    payload: str = json.dumps(
        checkpoint_payload,
        indent=1,
        sort_keys=True,
    )
    """Serialised the complete replacement before touching the visible path."""

    temporary_name: str | None = None
    """Tracked the unique unpublished path for cleanup after any interruption."""

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            """Retained the unique path for atomic publication or cleanup."""

            temporary.write(payload)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            """Made the replacement bytes durable before publication."""

        os.replace(temporary_name, path)
        """Published the replacement atomically on the same filesystem."""

        directory: int = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        """Opened the parent so the directory entry itself can be made durable."""

        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
            """Removed only an unpublished temporary file, if one remained."""


@contextmanager
def checkpoint_writer(path: Path | None) -> Iterator[None]:
    """Hold one advisory writer lock across checkpoint read, run and publication."""
    if path is None:
        with nullcontext():
            yield
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    """Made the operational checkpoint directory before opening its lock."""

    lock_path: Path = path.with_name(f".{path.name}.lock")
    """Named a stable inode whose advisory lock survives process crashes safely."""

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_LOCKED") from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        os.fsync(lock.fileno())
        """Recorded the active writer for a human inspecting a locked campaign."""

        try:
            yield
        finally:
            lock.seek(0)
            lock.truncate()
            lock.flush()
            os.fsync(lock.fileno())
            """Cleared the active-writer record after a clean or handled release."""

            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            """Released the lock even when a worker or validation failed."""


def checkpoint_shard_allocation(path: Path) -> dict[str, int]:
    """Read and validate one checkpoint's operational shard header."""
    if not path.exists():
        raise ValueError("CENSORED_COMPONENT_TARGET_MERGE_CHECKPOINT_MISSING")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        """Parsed only enough checkpoint structure to discover its partition."""
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_INVALID") from error
    if not isinstance(payload, dict):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_INVALID")
    candidate: Any = payload.get("shard")
    """Selected metadata deliberately kept outside the scientific signature."""

    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"shard_index", "shard_count"}
        or type(candidate.get("shard_index")) is not int
        or type(candidate.get("shard_count")) is not int
    ):
        raise ValueError("CENSORED_COMPONENT_TARGET_CHECKPOINT_SHARD_INVALID")
    try:
        return shard_allocation(candidate["shard_index"], candidate["shard_count"])
    except ValueError as error:
        raise ValueError(
            "CENSORED_COMPONENT_TARGET_CHECKPOINT_SHARD_INVALID"
        ) from error


def merge_shard_checkpoints(
    paths: list[Path],
    signature: dict[str, Any],
    jobs: list[tuple[str, float, int]],
    test_reference: str,
    bootstrap_replicates: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Merge a complete compatible shard set into one full campaign grid."""
    if len(paths) < 2:
        raise ValueError("CENSORED_COMPONENT_TARGET_MERGE_REQUIRES_MULTIPLE_SHARDS")
    require_signature_reference(signature, test_reference, bootstrap_replicates)
    """Bound every input to the selected row schema before reading any rows."""

    rows: list[dict[str, Any]] = []
    """Collected validated disjoint rows from every complete partition."""

    producer_history: list[dict[str, Any]] = []
    """Retained the exact row-producing provenance of every input shard."""

    seen_indices: set[int] = set()
    """Refused the same partition even if it arrives under two path names."""

    declared_count: int | None = None
    """Required one partition count across the complete merge set."""

    common_producer: dict[str, Any] | None = None
    """Required every shard to have run the same exact validated native build."""

    for path in paths:
        allocation: dict[str, int] = checkpoint_shard_allocation(path)
        """Read this input's validated operational partition."""

        shard_index: int = allocation["shard_index"]
        """Selected the zero-based partition represented by this input."""

        shard_count: int = allocation["shard_count"]
        """Selected the complete partition count represented by this input."""

        if declared_count is None:
            declared_count = shard_count
            """Established the partition count from the first validated input."""
        elif declared_count != shard_count:
            raise ValueError("CENSORED_COMPONENT_TARGET_MERGE_SHARD_COUNT_INCOMPATIBLE")
        if shard_index in seen_indices:
            raise ValueError("CENSORED_COMPONENT_TARGET_MERGE_SHARD_DUPLICATE")
        seen_indices.add(shard_index)
        """Accepted this declared partition exactly once."""

        shard_jobs: list[tuple[str, float, int]] = select_shard_jobs(
            jobs,
            shard_index,
            shard_count,
        )
        """Recomputed the exact subset from the full predeclared grid."""

        shard_rows, shard_history = read_checkpoint(
            path,
            signature,
            shard_jobs,
            test_reference,
            bootstrap_replicates,
            shard=allocation,
            require_complete=True,
        )
        """Required this shard to be complete before it could enter the merge."""

        for invocation in shard_history:
            identity: dict[str, Any] = producer_identity(invocation["producer"])
            """Compared the native producer while allowing CPU allocation to vary."""

            if common_producer is None:
                common_producer = identity
                """Established the exact producer required of later shards."""
            elif identity != common_producer:
                raise ValueError(
                    "CENSORED_COMPONENT_TARGET_MERGE_PROVENANCE_INCOMPATIBLE"
                )
        rows.extend(shard_rows)
        producer_history.extend(shard_history)

    assert declared_count is not None
    if seen_indices != set(range(declared_count)):
        raise ValueError("CENSORED_COMPONENT_TARGET_MERGE_SHARDS_MISSING")
    """Required every declared partition, not merely every row supplied."""

    validate_campaign_rows(
        rows,
        jobs,
        test_reference,
        bootstrap_replicates,
        require_complete=True,
    )
    validate_producer_history(producer_history, rows)
    """Revalidated uniqueness and provenance across partition boundaries."""

    rows.sort(
        key=lambda row: (
            require_exact_field(row, "scenario", str),
            require_exact_field(row, "censoring_share", float),
            require_exact_field(row, "replicate", int),
        )
    )
    return rows, producer_history, declared_count


def target_components() -> dict[str, Any]:
    """Build the three structured components and the design over records.

    Returns:
        The components, the fixed-effect design, the family row blocks and the
        largest family.
    """
    target: Any = load_target_design(FIXTURE, expected_sha256=FIXTURE_SHA256)
    """Read the reviewed aggregate and its structure-matched pedigree."""

    people: int = target.relationship.shape[0]
    """Counted the people the aggregate carries."""

    sharing: npt.NDArray[np.float64] = np.ones((RECORDS, RECORDS))
    """Named the pattern that makes a person's own records share everything."""

    additive: npt.NDArray[np.float64] = np.kron(target.relationship, sharing)
    """Spread the additive relationship over each person's records."""

    person: npt.NDArray[np.float64] = np.kron(np.eye(people), sharing)
    """Built the person-level component: one within a person, nought between."""

    household: npt.NDArray[np.float64] = np.kron(
        household_kernel(target.relationship, HOUSEHOLD_SIZE), sharing
    )
    """Built the household component by cutting each family into homes of three."""

    design: npt.NDArray[np.float64] = np.repeat(target.design, RECORDS, axis=0)
    """Gave a person's records the same six fixed-effect columns."""

    blocks: list[npt.NDArray[np.int64]] = []
    """Collected the row ranges of each independent family."""

    offset: int = 0
    """Tracked contiguous family slices in generated row order."""

    for size in target.component_sizes:
        blocks.append(np.arange(offset, offset + size * RECORDS))
        offset += size * RECORDS
        """Took this family's rows and advanced to the next."""
    """Collected the row ranges of each independent family."""

    return {
        "components": [additive, person, household],
        "design": design,
        "blocks": blocks,
        "largest_family": max(target.component_sizes),
        "people": people,
        "fixture_sha256": target.fixture_sha256,
    }


def target_metadata() -> dict[str, Any]:
    """Read target facts without expanding three record-level dense matrices."""
    target: Any = load_target_design(FIXTURE, expected_sha256=FIXTURE_SHA256)
    """Loaded only the reviewed person-level target structure."""

    people: int = target.relationship.shape[0]
    """Counted people without expanding records or covariance components."""

    return {
        "people": people,
        "rows": people * RECORDS,
        "largest_family": max(target.component_sizes),
        "fixture_sha256": target.fixture_sha256,
    }


def start_worker(
    test_reference: str,
    bootstrap_replicates: int | None,
    bootstrap_threads: int | None,
    expected_extension_sha256: str,
) -> None:
    """Build this worker's copy of the design and both generating factors."""
    if test_reference == "bootstrap":
        if bootstrap_threads is None or bootstrap_threads < 1:
            raise ValueError("CENSORED_COMPONENT_TARGET_BOOTSTRAP_THREADS_REQUIRED")
        os.environ["RAYON_NUM_THREADS"] = str(bootstrap_threads)
        """Bound inner Rust parallelism before this worker first enters the core."""
    elif test_reference != "asymptotic":
        raise ValueError("CENSORED_COMPONENT_TARGET_TEST_REFERENCE_UNKNOWN")
    elif bootstrap_threads is not None:
        raise ValueError("CENSORED_COMPONENT_TARGET_BOOTSTRAP_THREADS_NOT_APPLICABLE")

    root: Path = Path(__file__).resolve().parent.parent
    """Located the checkout independently inside this spawned process."""

    validate_loaded_extension(root)
    """Proved this worker imported the native module built from that clean checkout."""

    if asterism.installed_extension_sha256() != expected_extension_sha256:
        raise ValueError("CENSORED_COMPONENT_TARGET_WORKER_EXTENSION_MISMATCH")
    """Bound this worker to the exact native binary recorded by its parent."""

    STATE.clear()
    """Removed state retained by an earlier in-process smoke test."""

    STATE.update(target_components())
    """Built the components once for every replicate this worker runs."""

    STATE["test_reference"] = test_reference
    """Retained the command's selected null reference in every spawned worker."""

    if test_reference == "bootstrap":
        if bootstrap_replicates is None or bootstrap_replicates < 1:
            raise ValueError("CENSORED_COMPONENT_TARGET_BOOTSTRAP_REPLICATES_REQUIRED")
        STATE["bootstrap_replicates"] = bootstrap_replicates
        """Retained the optional mode's complete predeclared inner denominator."""
    elif bootstrap_replicates is not None:
        raise ValueError(
            "CENSORED_COMPONENT_TARGET_BOOTSTRAP_REPLICATES_NOT_APPLICABLE"
        )

    coefficients: npt.NDArray[np.float64] = np.asarray(MEAN_COEFFICIENTS)
    """Fixed the same six latent-mean coefficients used by every outcome."""

    STATE["mean_coefficients"] = coefficients
    """Retained the coefficients beside the limits they determine."""

    STATE["limits"] = {
        share: right_censoring_limit(
            STATE["design"] @ coefficients,
            TOTAL_VARIANCE,
            share,
        )
        for share in CENSORING_SHARES
    }
    """Solved both instrument limits once, before any outcome was drawn."""

    for scenario, shares in (("null", NULL_SHARES), ("heritable", HERITABLE_SHARES)):
        factors: list[npt.NDArray[np.float64]] = []
        """Collected independent family factors without a whole-target matrix."""

        for block in STATE["blocks"]:
            indices: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]] = np.ix_(
                block,
                block,
            )
            """Selected this family from each whole-target component."""

            covariance: npt.NDArray[np.float64] = TOTAL_VARIANCE * (
                shares[0] * STATE["components"][0][indices]
                + shares[1] * STATE["components"][1][indices]
                + shares[2] * STATE["components"][2][indices]
                + shares[3] * np.eye(block.size)
            )
            """Assembled only this independent family's covariance."""

            factors.append(np.linalg.cholesky(covariance + 1e-9 * np.eye(block.size)))
        STATE[f"factors_{scenario}"] = factors
        """Retained small family factors rather than another dense target copy."""
    """Built both generating factor lists once rather than per replicate."""

    STATE["model"] = asterism.CensoredComponentModel(
        STATE["components"],
        STATE["design"],
    )
    """Prepared one model for all attempts in this worker."""

    del STATE["components"]
    """Released the input matrices after the model took its owned copy."""


def one(job: tuple[str, float, int]) -> dict[str, Any]:
    """Fit and score one replicate at the target design.

    Args:
        job: The scenario, the censoring share and the replicate number.

    Returns:
        What this replicate measured, or the refusal that stopped it.
    """
    scenario, share, replicate = job
    """Read what this attempt is."""

    design: npt.NDArray[np.float64] = STATE["design"]
    """Read the fixed-effect design."""

    rows: int = design.shape[0]
    """Counted the rows once."""

    generator: np.random.Generator = np.random.default_rng(
        BASE_SEED + 7919 * replicate + 104_729 * int(share * 100) + len(scenario)
    )
    """Created the generator this replicate owns alone."""

    coefficients: npt.NDArray[np.float64] = STATE["mean_coefficients"]
    """Read the six coefficients fixed before limits and outcomes alike."""

    complete: npt.NDArray[np.float64] = design @ coefficients
    """Started the latent trait at its predeclared fixed mean."""

    for block, factor in zip(
        STATE["blocks"],
        STATE[f"factors_{scenario}"],
        strict=True,
    ):
        complete[block] += factor @ generator.standard_normal(block.size)
        """Drew this independent family's deviation into its target rows."""

    ceiling: float = float(STATE["limits"][share])
    """Applied the fixed instrument limit chosen before this outcome was drawn."""

    censoring: npt.NDArray[np.int64] = (complete >= ceiling).astype(np.int64)
    """Marked 1 at or above the predeclared limit, 0 where measured."""

    value: npt.NDArray[np.float64] = np.where(censoring == 0, complete, np.nan)
    """Kept measured values; censored ones are never read."""

    limit: npt.NDArray[np.float64] = np.full(rows, ceiling)
    """Gave every row the same instrument limit."""

    worst_block: int = max(int(censoring[block].sum()) for block in STATE["blocks"])
    """Recorded the largest censored dimension the likelihood had to resolve."""

    model: Any = STATE["model"]
    """Reused the worker-owned model instead of copying dense inputs per attempt."""

    result: dict[str, Any] = {
        "scenario": scenario,
        "censoring_share": share,
        "replicate": replicate,
        "worst_censored_block": worst_block,
        "achieved_censoring_share": float(censoring.sum()) / rows,
        "censoring_limit": ceiling,
    }
    """Started this replicate's record with what it was asked to do."""

    try:
        if scenario == "null":
            test_reference: str = str(STATE["test_reference"])
            """Selected the predeclared ordinary or optional null reference."""

            bootstrap_seed: int | None = None
            """Remained absent unless this attempt actually simulated a null grid."""

            expected_replicates: int | None = None
            """Remained absent for the analytic test, which has no inner grid."""

            if test_reference == "asymptotic":
                test: dict[str, Any] = model.test(
                    value,
                    censoring,
                    limit,
                    component=TESTED,
                )
                """Applied the explicit analytic boundary reference once."""
            else:
                direction: npt.NDArray[np.int64] = np.ones(rows, dtype=np.int64)
                """Declared the fixed right-censoring mechanism for every row."""

                bootstrap_seed = (
                    BOOTSTRAP_BASE_SEED + 7_919 * replicate + 104_729 * int(share * 100)
                )
                """Owned a reproducible inner null stream for this outer replicate."""

                expected_replicates = int(STATE["bootstrap_replicates"])
                """Read the predeclared complete inner denominator for this cell."""

                test = model.bootstrap(
                    value,
                    censoring,
                    limit,
                    direction,
                    component=TESTED,
                    replicates=expected_replicates,
                    seed=bootstrap_seed,
                )
                """Calibrated the statistic under its fitted constrained null."""

            require_test_rule(test, test_reference, expected_replicates)
            """Kept a wrong or incomplete reference out of the level numerator."""

            if test.get("nuisance_at_bound") is not False:
                raise ValueError("TOBIT_COMPONENT_TEST_NUISANCE_AT_BOUND")
            """Required the regular nuisance interior assumed by either reference."""

            result.update(
                p_value=float(test["p_value"]),
                test_statistic=float(test["statistic"]),
                test_rule=str(test["rule"]),
                rejected=bool(test["p_value"] <= LEVEL),
                nuisance_at_bound=bool(test["nuisance_at_bound"]),
                in_lrt_atom=in_operational_lrt_atom(float(test["statistic"])),
            )
            if test_reference == "bootstrap":
                result.update(
                    bootstrap_exceedances=int(test["exceedances"]),
                    bootstrap_replicates=int(test["replicates"]),
                    bootstrap_requested=int(test["requested"]),
                    bootstrap_seed=bootstrap_seed,
                )
                """Recorded inner work only for the mode that actually performed it."""
        else:
            interval: dict[str, Any] = model.interval(
                value, censoring, limit, component=TESTED
            )
            """Profiled the genetic component's mean-diagonal proportion."""

            bound_unreachable: bool = bool(
                interval["upper_limited"] and interval["contains_upper_bound"] is None
            )
            """Read whether the upper end is the bound because it could not be scored."""

            result.update(
                estimate=float(interval["estimate"]),
                lower=float(interval["lower"]),
                upper=float(interval["upper"]),
                upper_limited=bool(interval["upper_limited"]),
                contains_upper_bound=interval["contains_upper_bound"],
                profile_failures=int(interval["profile_failures"]),
                bound_unreachable=bound_unreachable,
                covered=bool(
                    interval["lower"] <= HERITABLE_SHARES[TESTED] <= interval["upper"]
                ),
                extra_failures=int(interval["profile_failures"])
                - (1 if bound_unreachable else 0),
            )
    except ValueError as refusal:
        result.update(refusal=str(refusal))
        """Recorded the refusal rather than dropping the replicate."""
    return result


def abort_process_pool(pool: ProcessPoolExecutor) -> None:
    """Stop active workers promptly after an exception or interrupt."""
    terminate_workers: Any = getattr(pool, "terminate_workers", None)
    """Used Python 3.14's public abrupt-shutdown operation when available."""

    if callable(terminate_workers):
        terminate_workers()
        return

    process_map: Any = getattr(pool, "_processes", None) or {}
    """Read Python 3.13's active worker mapping before executor shutdown."""

    processes: list[Any] = list(process_map.values())
    """Captured Python 3.13's worker handles before shutdown releases them."""

    pool.shutdown(wait=False, cancel_futures=True)
    """Cancelled work that had not yet reached a process."""

    for process in processes:
        if process.is_alive():
            process.terminate()
    """Asked every active 3.13 worker to exit without finishing a long bootstrap."""

    deadline: float = time.monotonic() + 2.0
    """Allowed one short common grace period, rather than one period per worker."""

    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join()
    """Escalated and reaped only workers that did not terminate during the grace."""

    pool.shutdown(wait=True, cancel_futures=True)
    """Reaped the executor's now-idle management resources before propagating."""


def worker_multiprocessing_context() -> Any:
    """Use spawn so children never inherit the parent's checkpoint-lock handle."""
    return multiprocessing.get_context("spawn")


def collect_campaign_rows(
    jobs: list[tuple[str, float, int]],
    workers: int,
    test_reference: str,
    bootstrap_replicates: int | None,
    bootstrap_threads: int | None,
    checkpoint: Path | None,
    signature: dict[str, Any],
    producer: dict[str, Any],
    *,
    shard: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resume and run a bounded queue while retaining each completed outer row."""
    require_signature_reference(signature, test_reference, bootstrap_replicates)
    """Refused a worker mode that disagrees with its persisted campaign identity."""

    with checkpoint_writer(checkpoint):
        recovered: tuple[list[dict[str, Any]], list[dict[str, Any]]] = (
            read_checkpoint(
                checkpoint,
                signature,
                jobs,
                test_reference,
                bootstrap_replicates,
                shard=shard,
            )
            if checkpoint is not None
            else ([], [])
        )
        """Recovered only compatible, unique completed outer attempts."""

        rows, producer_history = recovered
        """Separated scientific results from their non-gating producer provenance."""

        execution: dict[str, Any] = {
            "allocated_cpu_count": allocated_cpu_count(),
            "outer_workers": workers,
            "bootstrap_threads_per_outer_worker": bootstrap_threads,
        }
        """Recorded the actual outer-by-inner CPU allocation for this invocation."""

        if shard is not None:
            execution.update(shard)
            """Attributed this producer to its disjoint scheduler partition."""

        invocation: dict[str, Any] = {
            "producer": {
                **producer,
                "execution": execution,
            },
            "coordinates": [],
        }
        """Started this exact build's checkpoint and scoring invocation record."""

        producer_history.append(invocation)
        validate_producer_history(producer_history, rows)
        if checkpoint is not None:
            write_checkpoint(
                checkpoint,
                signature,
                rows,
                producer_history,
                shard=shard,
                writer_locked=True,
            )
            """Recorded this scoring build before reusing or producing any row."""

        completed: set[tuple[str, float, int]] = {
            (
                require_exact_field(row, "scenario", str),
                require_exact_field(row, "censoring_share", float),
                require_exact_field(row, "replicate", int),
            )
            for row in rows
        }
        """Located the exact coordinates already retained."""

        pending: Iterator[tuple[str, float, int]] = iter(
            job for job in jobs if job not in completed
        )
        """Kept every missing coordinate in the original deterministic order."""

        if rows:
            print(f"resumed {len(rows)} completed outer attempts from {checkpoint}")

        pool: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=start_worker,
            initargs=(
                test_reference,
                bootstrap_replicates,
                bootstrap_threads,
                producer["extension_sha256"],
            ),
            mp_context=worker_multiprocessing_context(),
        )
        """Started a bounded non-fork pool without inheriting the writer lock."""

        active: dict[Future[dict[str, Any]], tuple[str, float, int]] = {}
        """Bound the submitted queue to one outer attempt per worker."""

        def submit_next() -> bool:
            """Submit one missing coordinate, or report that the grid is exhausted."""
            try:
                job: tuple[str, float, int] = next(pending)
                """Selected the next deterministic outer coordinate."""
            except StopIteration:
                return False
            active[pool.submit(one, job)] = job
            """Bound the submitted future to its scientific coordinate."""

            return True

        def retain(row: dict[str, Any]) -> None:
            """Validate and durably retain one completed or explicitly refused row."""
            candidate: list[dict[str, Any]] = [*rows, row]
            """Kept malformed worker output out of the in-memory valid collection."""

            validate_campaign_rows(
                candidate,
                jobs,
                test_reference,
                bootstrap_replicates,
                require_complete=False,
            )
            rows.append(row)
            invocation["coordinates"].append(
                [
                    require_exact_field(row, "scenario", str),
                    require_exact_field(row, "censoring_share", float),
                    require_exact_field(row, "replicate", int),
                ]
            )
            """Attributed the newly completed coordinate to this exact build."""

            validate_producer_history(producer_history, rows)
            if checkpoint is not None:
                write_checkpoint(
                    checkpoint,
                    signature,
                    rows,
                    producer_history,
                    shard=shard,
                    writer_locked=True,
                )

        failed_job: tuple[str, float, int] | None = None
        """Held the first outer coordinate whose future or retention failed."""

        failure: BaseException | None = None
        """Deferred a worker exception until every simultaneously completed row saved."""

        aborted: bool = False
        """Distinguished ordinary completion from exceptional prompt shutdown."""

        try:
            for _ in range(workers):
                if not submit_next():
                    break
            """Started no more expensive attempts than there are measured workers."""

            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                """Waited only until at least one bounded outer attempt completed."""

                done.update(future for future in active if future.done())
                """Included every result already available at this checkpoint moment."""

                for future in done:
                    job: tuple[str, float, int] = active.pop(future)
                    """Recovered this completed future's exact outer coordinate."""

                    try:
                        retain(future.result())
                    except BaseException as error:
                        if failure is None:
                            failed_job = job
                            """Remembered the first failed coordinate."""

                            failure = error
                            """Retained its exception until simultaneous successes saved."""
                if failure is not None:
                    for future in active:
                        future.cancel()
                    raise RuntimeError(
                        f"CENSORED_COMPONENT_TARGET_WORKER_FAILED {failed_job}"
                    ) from failure
                for _ in range(len(done)):
                    if not submit_next():
                        break
        except BaseException:
            aborted = True
            """Marked the executor for abrupt rather than ordinary shutdown."""

            abort_process_pool(pool)
            """Stopped costly active bootstraps before propagating any interruption."""

            raise
        finally:
            if not aborted:
                pool.shutdown(wait=True, cancel_futures=True)
                """Reaped every child after the ordinary complete campaign."""

        validate_campaign_rows(
            rows,
            jobs,
            test_reference,
            bootstrap_replicates,
            require_complete=True,
        )
        """Required the exact complete coordinate grid before scientific scoring."""

    rows.sort(
        key=lambda row: (
            require_exact_field(row, "scenario", str),
            require_exact_field(row, "censoring_share", float),
            require_exact_field(row, "replicate", int),
        )
    )
    """Made reporting and final evidence independent of completion order."""

    return rows, producer_history


def main() -> int:
    """Score the several-component censored model at its intended design.

    Returns:
        Zero when every cell met its rule and nothing failed.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )
    """Built the command line."""

    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--test-reference",
        choices=("asymptotic", "bootstrap"),
        default=DEFAULT_TEST_REFERENCE,
        help=(
            "null reference: ordinary asymptotic mixture (default), or the "
            "optional constrained-null bootstrap"
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        help=(
            "inner draws for --test-reference bootstrap; defaults to "
            f"{BOOTSTRAP_REPLICATES} in that mode and is invalid otherwise"
        ),
    )
    parser.add_argument(
        "--bootstrap-threads",
        type=int,
        help=(
            "Rayon threads per outer worker for --test-reference bootstrap; "
            "defaults to the visible CPU allocation divided across workers, "
            "and outer workers times inner threads may not exceed that allocation"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--shard-index",
        type=int,
        help=(
            "zero-based scheduler-array index; requires --shard-count and "
            "--checkpoint, whose path is labelled uniquely for this shard"
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="number of disjoint scheduler-array partitions; requires --shard-index",
    )
    parser.add_argument(
        "--merge-checkpoints",
        type=Path,
        nargs="+",
        metavar="SHARD_CHECKPOINT",
        help=(
            "merge a complete set of compatible shard checkpoints, score the "
            "full grid and write evidence; incompatible with --checkpoint or "
            "shard options"
        ),
    )
    parser.add_argument("--no-write", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()
    """Read the campaign size, resume path and evidence-write choice."""

    if arguments.replicates < 1 or arguments.workers < 1:
        parser.error("replicates and workers must be positive")
    sharding: bool = (
        arguments.shard_index is not None or arguments.shard_count is not None
    )
    """Detected either half of an explicitly requested scheduler partition."""

    merging: bool = arguments.merge_checkpoints is not None
    """Detected the read-only-input mode that performs the one final score."""

    if (arguments.shard_index is None) != (arguments.shard_count is None):
        parser.error("--shard-index and --shard-count must be supplied together")
    if sharding and merging:
        parser.error("shard production and --merge-checkpoints are incompatible")
    if sharding and arguments.checkpoint is None:
        parser.error("shard production requires --checkpoint")
    if merging and arguments.checkpoint is not None:
        parser.error("--merge-checkpoints is incompatible with --checkpoint")
    if merging and len(arguments.merge_checkpoints) < 2:
        parser.error("--merge-checkpoints requires at least two shard checkpoints")
    if sharding:
        try:
            shard_allocation(arguments.shard_index, arguments.shard_count)
        except ValueError as error:
            parser.error(str(error))
    if (
        arguments.test_reference == "asymptotic"
        and arguments.bootstrap_replicates is not None
    ):
        parser.error("--bootstrap-replicates requires --test-reference bootstrap")
    if (
        arguments.test_reference == "asymptotic"
        and arguments.bootstrap_threads is not None
    ):
        parser.error("--bootstrap-threads requires --test-reference bootstrap")
    if merging and arguments.bootstrap_threads is not None:
        parser.error("--bootstrap-threads does not apply while merging checkpoints")

    bootstrap_replicates: int | None = arguments.bootstrap_replicates
    """Held an inner denominator only in the optional bootstrap mode."""

    bootstrap_threads: int | None = arguments.bootstrap_threads
    """Held an inner Rayon allocation only in the optional bootstrap mode."""

    available_cpus: int = allocated_cpu_count()
    """Read the local or scheduler-constrained logical CPU allocation once."""

    if arguments.test_reference == "bootstrap":
        if bootstrap_replicates is None:
            bootstrap_replicates = BOOTSTRAP_REPLICATES
            """Applied the documented complete inner grid for optional bootstrap."""
        if bootstrap_replicates < 1:
            parser.error("bootstrap replicates must be positive")
        if not merging:
            if bootstrap_threads is None:
                bootstrap_threads = max(1, available_cpus // arguments.workers)
                """Shared the visible allocation across concurrent outer fits."""
            elif bootstrap_threads < 1:
                parser.error("bootstrap threads must be positive")
            if arguments.workers * bootstrap_threads > available_cpus:
                parser.error(
                    "outer workers times bootstrap threads exceed the visible CPU "
                    "allocation; reduce --workers or --bootstrap-threads"
                )

    structure: dict[str, Any] = target_metadata()
    """Read reporting facts without retaining another expanded component set."""

    print(
        f"{structure['people']} people, {RECORDS} records each, "
        f"largest family {structure['largest_family']}, "
        f"{arguments.replicates} replicates per cell"
    )
    print(
        f"components: additive, person-level, households of {HOUSEHOLD_SIZE}, residual"
    )
    print(f"null reference: {arguments.test_reference}")
    if bootstrap_replicates is not None:
        if merging:
            print(
                f"bootstrap null replicates per outer null fit: {bootstrap_replicates}; "
                "worker allocations will be read from the shard checkpoints"
            )
        else:
            print(
                f"bootstrap null replicates per outer null fit: "
                f"{bootstrap_replicates}; Rayon threads per outer worker: "
                f"{bootstrap_threads}"
            )

    jobs: list[tuple[str, float, int]] = [
        (scenario, share, replicate)
        for scenario in ("null", "heritable")
        for share in CENSORING_SHARES
        for replicate in range(arguments.replicates)
    ]
    """Listed every attempt this campaign makes."""

    signature: dict[str, Any] = checkpoint_signature(
        arguments.replicates,
        arguments.test_reference,
        bootstrap_replicates,
    )
    """Bound any resumed rows to this exact design, build and implementation."""

    root: Path = Path(__file__).resolve().parent.parent
    """Located the checkout whose validated producer provenance is retained."""

    producer: dict[str, Any] = producer_provenance(root)
    """Recorded the exact build running or scoring this metadata-equivalent work."""

    print(
        "producer: "
        f"commit={producer['source_commit']} "
        f"manifest={producer['release_manifest_sha256']} "
        f"extension={producer['extension_sha256']}"
    )

    merged_shard_count: int | None = None
    """Recorded how many independent checkpoints fed a final merge, if any."""

    try:
        if sharding:
            assert arguments.shard_index is not None
            assert arguments.shard_count is not None
            assert arguments.checkpoint is not None
            allocation: dict[str, int] = shard_allocation(
                arguments.shard_index,
                arguments.shard_count,
            )
            """Narrowed the already parsed and validated scheduler partition."""

            selected_jobs: list[tuple[str, float, int]] = select_shard_jobs(
                jobs,
                arguments.shard_index,
                arguments.shard_count,
            )
            """Selected this task's deterministic stride through the full grid."""

            shard_checkpoint: Path = shard_checkpoint_path(
                arguments.checkpoint,
                arguments.shard_index,
                arguments.shard_count,
            )
            """Derived a distinct path even when every array task gets one base."""

            print(
                f"shard {arguments.shard_index}/{arguments.shard_count}: "
                f"{len(selected_jobs)} of {len(jobs)} outer attempts; "
                f"checkpoint={shard_checkpoint}"
            )
            shard_rows, _ = collect_campaign_rows(
                selected_jobs,
                arguments.workers,
                arguments.test_reference,
                bootstrap_replicates,
                bootstrap_threads,
                shard_checkpoint,
                signature,
                producer,
                shard=allocation,
            )
            """Completed only this shard under its own checkpoint writer lock."""

            if (
                checkpoint_signature(
                    arguments.replicates,
                    arguments.test_reference,
                    bootstrap_replicates,
                )
                != signature
            ):
                raise ValueError(
                    "CENSORED_COMPONENT_TARGET_CAMPAIGN_CHANGED_DURING_RUN"
                )
            print(
                f"SHARD COMPLETE: {len(shard_rows)} validated outer attempts; "
                "no scientific score or evidence was written"
            )
            return 0
        if merging:
            rows, producer_history, merged_shard_count = merge_shard_checkpoints(
                arguments.merge_checkpoints,
                signature,
                jobs,
                arguments.test_reference,
                bootstrap_replicates,
            )
            """Read a complete disjoint set before entering final scoring."""

            producer_history.append({"producer": producer, "coordinates": []})
            validate_producer_history(producer_history, rows)
            """Recorded the exact clean build that performed the final score."""
        else:
            rows, producer_history = collect_campaign_rows(
                jobs,
                arguments.workers,
                arguments.test_reference,
                bootstrap_replicates,
                bootstrap_threads,
                arguments.checkpoint,
                signature,
                producer,
            )
            """Recovered and completed the ordinary unsharded outer grid."""

        if (
            checkpoint_signature(
                arguments.replicates,
                arguments.test_reference,
                bootstrap_replicates,
            )
            != signature
        ):
            raise ValueError("CENSORED_COMPONENT_TARGET_CAMPAIGN_CHANGED_DURING_RUN")
        """Revalidated code, runtime and extension after the multiweek computation."""
    except ValueError as error:
        parser.error(str(error))

    failed: list[dict[str, Any]] = [row for row in rows if row_failed(row)]
    """Collected every attempt that did not produce the inference asked of it.

    The one profile failure at an unreachable upper bound is not counted here.
    Nothing went wrong in it: the bound is a covariance the design cannot form,
    and the interval says so through `upper_limited`. A failure anywhere else in
    the bracket is counted, because that is a fit that should have been made and
    was not.
    """

    limited: int = sum(1 for row in rows if row_bound_unreachable(row))
    """Counted the intervals whose upper end is the bound it could not evaluate."""

    worst_block: int = max(int(row["worst_censored_block"]) for row in rows)
    """Took the largest censored dimension anywhere in the campaign."""

    cells: list[dict[str, Any]] = []
    """Collected what each cell measured."""

    failures: list[str] = []
    """Collected every rule this campaign broke."""

    print()
    print(
        f"{'scenario':>10} {'censored':>9} {'attempts':>9} {'measured':>9} "
        f"{'rate':>8} {'exact interval':>22} {'nominal':>8} {'LRT atom':>10}"
    )
    print("-" * 93)
    for scenario in ("null", "heritable"):
        for share in CENSORING_SHARES:
            cell: list[dict[str, Any]] = [
                row
                for row in rows
                if row["scenario"] == scenario and row["censoring_share"] == share
            ]
            """Selected this cell's attempts."""

            usable: list[dict[str, Any]] = [row for row in cell if "refusal" not in row]
            """Selected the attempts that produced a number."""

            if not usable:
                failures.append(f"{scenario} at {share:.2f}: nothing was measured")
                continue
            nominal: float = LEVEL if scenario == "null" else NOMINAL_COVERAGE
            """Named what that count is scored against."""

            hits: int = sum(
                1
                for row in usable
                if (
                    float(row["p_value"]) <= LEVEL
                    if scenario == "null"
                    else float(row["lower"])
                    <= HERITABLE_SHARES[TESTED]
                    <= float(row["upper"])
                )
            )
            """Recomputed the cell decision from raw p-values or endpoints."""

            atom_share: float | None = (
                sum(1 for row in usable if bool(row["in_lrt_atom"])) / len(usable)
                if scenario == "null"
                else None
            )
            """Described the finite null statistic's operational point mass."""

            attempted: int = len(cell)
            """Retained the complete predeclared outer denominator."""

            low, high = clopper_pearson(hits, attempted)
            """Described the rate without filtering failed attempts."""

            decision: dict[str, object] = (
                level_decision(
                    rejections=hits,
                    attempted=attempted,
                    level=LEVEL,
                    confidence=MONTE_CARLO_CONFIDENCE,
                )
                if scenario == "null"
                else coverage_decision(
                    covered=hits,
                    attempted=attempted,
                    nominal=NOMINAL_COVERAGE,
                    confidence=MONTE_CARLO_CONFIDENCE,
                )
            )
            """Applied the matching one-sided exact anti-conservatism rule."""

            print(
                f"{scenario:>10} {share:>9.2f} {len(cell):>9d} {len(usable):>9d} "
                f"{hits / attempted:>8.4f} "
                f"{f'[{low:.4f}, {high:.4f}]':>22} {nominal:>8.2f}"
                f"{'' if atom_share is None else f'{atom_share:>11.3f}'}"
            )
            cells.append(
                {
                    "scenario": scenario,
                    "censoring_share": share,
                    "attempted": len(cell),
                    "measured": len(usable),
                    "rate": hits / attempted,
                    "exact_interval": [low, high],
                    "nominal": nominal,
                    "one_sided_decision": decision,
                    "share_in_lrt_atom": atom_share,
                    "lrt_atom_is_diagnostic_only": scenario == "null",
                }
            )
            if not bool(decision["passed"]):
                failures.append(
                    f"{scenario} at {share:.2f}: {hits / attempted:.4f} against a "
                    f"nominal {nominal:.3f} is {decision['verdict']} under the "
                    "predeclared one-sided exact rule"
                )
    print()
    print(
        f"Largest censored block anywhere in the campaign: {worst_block}, against the "
        f"{LADDER_QUALIFIED_DIMENSION} the region ladder has climbed."
    )
    intervals: int = sum(1 for row in rows if "lower" in row)
    """Counted the attempts that produced an interval at all."""

    if intervals:
        print(
            f"Intervals whose upper end is the unreachable bound: {limited}/{intervals}."
            " A proportion of one leaves a person's own records perfectly"
            " correlated, so the coverage above is a lower-end statement scored"
            f" against {NOMINAL_COVERAGE}."
        )
    if worst_block > LADDER_QUALIFIED_DIMENSION:
        failures.append(
            f"the campaign produced a censored block of {worst_block}, past the "
            f"{LADDER_QUALIFIED_DIMENSION} the ladder has qualified, so its region "
            "probabilities are outside the evidence"
        )
    if failed:
        failures.append(
            f"{len(failed)} of {len(rows)} attempts failed, and none is permitted"
        )
    if not arguments.no_write:
        record: Path = (
            Path(__file__).resolve().parent.parent
            / "evidence"
            / f"censored-components-target-design-{date.today().isoformat()}.json"
        )
        """Named the dated evidence file."""

        record.write_text(
            json.dumps(
                {
                    "check": "censored_components_target_design",
                    "participant_free": True,
                    "people": structure["people"],
                    "records_per_person": RECORDS,
                    "rows": structure["rows"],
                    "largest_family": structure["largest_family"],
                    "household_size": HOUSEHOLD_SIZE,
                    "components": [
                        "additive_relationship",
                        "person_level",
                        "household",
                        "residual",
                    ],
                    "heritable_shares": list(HERITABLE_SHARES),
                    "null_shares": list(NULL_SHARES),
                    "censoring_limits_fixed_before_outcomes": True,
                    "expected_censoring_shares": list(CENSORING_SHARES),
                    "achieved_censoring_share_ranges": {
                        str(share): [
                            min(
                                float(row["achieved_censoring_share"])
                                for row in rows
                                if row["censoring_share"] == share
                            ),
                            max(
                                float(row["achieved_censoring_share"])
                                for row in rows
                                if row["censoring_share"] == share
                            ),
                        ]
                        for share in CENSORING_SHARES
                    },
                    "test_reference": arguments.test_reference,
                    "required_test_rule": (
                        ASYMPTOTIC_TEST_RULE
                        if arguments.test_reference == "asymptotic"
                        else BOOTSTRAP_TEST_RULE
                    ),
                    "monte_carlo_confidence": MONTE_CARLO_CONFIDENCE,
                    **(
                        {
                            "bootstrap_replicates_per_outer_null_fit": (
                                bootstrap_replicates
                            ),
                            **(
                                {"merged_shard_count": merged_shard_count}
                                if merged_shard_count is not None
                                else {
                                    "bootstrap_threads_per_outer_worker": (
                                        bootstrap_threads
                                    ),
                                    "allocated_cpu_count": available_cpus,
                                }
                            ),
                        }
                        if bootstrap_replicates is not None
                        else {}
                    ),
                    **(
                        {"merged_shard_count": merged_shard_count}
                        if merged_shard_count is not None
                        and bootstrap_replicates is None
                        else {}
                    ),
                    "attempted": len(rows),
                    "failed": len(failed),
                    "largest_censored_block": worst_block,
                    "intervals_limited_at_unreachable_bound": limited,
                    "ladder_qualified_dimension": LADDER_QUALIFIED_DIMENSION,
                    "fixture_sha256": structure["fixture_sha256"],
                    "producer_history": producer_history,
                    "cells": cells,
                    "passed": not failures,
                },
                indent=1,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten to {record}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        "\nPASSED: the boundary test held its level and the interval held its "
        "coverage at both censoring shares, with no failed attempt, at a "
        "several-component set."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
