"""Calibrate the five discrete-GxE tests on a values-free target envelope.

The exact reviewed fixture retains only component-size, group-count,
cross-group related-pair, and fixed-design aggregates.  It deterministically
generates a participant-free pedigree, group assignment, and synthetic age
design; it does not reconstruct target rows, ages, phenotypes, relationship
coefficients, or family-specific group composition.  Every response is
simulated, and every refusal, nonconvergence, or failed public test remains in
the prespecified denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asterism
import numpy as np
import numpy.typing as npt
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from one_trait_coverage import pedigree_relationship, structure_matched_pedigree

NULLS: tuple[str, ...] = (
    "gene_by_environment",
    "any_difference",
    "correlation",
    "genetic",
    "residual",
)
"""Named the five documented public test nulls."""

SCENARIO_CHOICES: tuple[str, ...] = (
    "null",
    "different_genes",
    "different_scale",
    "noisier",
)
"""Named every fixed scientific calibration scenario accepted by the command."""

TARGET_FIXTURE_SHA256: str = (
    "2cceae6f2c5ea3f0c3de5946daeaa55cf2f35edd6d365fd1f45499f463d4c1c6"
)
"""Pinned the exact reviewed aggregate-only target fixture bytes."""

# The null: one genetic standard deviation, one residual, same genes in both
# sexes. A heritability of 0.5, near what the real GOBS traits give.
GENETIC: tuple[float, float] = (0.707, 0.707)
"""Set equal group genetic standard deviations near target heritability."""

RESIDUAL: tuple[float, float] = (0.707, 0.707)
"""Set equal group residual standard deviations under the complete null."""

SCENARIOS: dict[
    str,
    tuple[tuple[float, float], tuple[float, float], float],
] = {
    # Nothing to find. Every test must hold its level.
    "null": (GENETIC, RESIDUAL, 1.0),
    # The genes differ across the sexes: the genetic finding proper.
    "different_genes": (GENETIC, RESIDUAL, 0.4),
    # The genetic variance differs: a difference of scale, not of genes.
    "different_scale": ((0.95, 0.45), RESIDUAL, 1.0),
    # One sex measured twice as noisily, identical genetics. The genetic tests
    # must not reject. This is the scenario the two free residual variances
    # exist for.
    "noisier": (GENETIC, (0.5, 1.0), 1.0),
}
"""Fixed the null, distinct-genes, distinct-scale, and noise scenarios."""

EXPECTED_RULES: dict[str, str] = {
    "gene_by_environment": "mixture_chi2_1_chi2_2",
    "any_difference": "mixture_chi2_2_chi2_3",
    "correlation": "mixture_50_50",
    "genetic": "chi2_1",
    "residual": "chi2_1",
}
"""Pinned the documented reference rule for every public test."""

TRUE_EFFECTS: npt.NDArray[np.float64] = np.asarray([0.5, 0.2, -0.1, 0.25])
"""Selected fixed nonzero effects for every synthetic calibration response."""

BASE_SEED: int = 910_000
"""Owned the deterministic response stream independently of worker scheduling."""

STATE: dict[str, Any] = {}
"""Held immutable participant-free arrays inside campaign worker processes."""


@dataclass(frozen=True)
class TargetEnvelope:
    """Hold one participant-free structure-matched discrete-GxE design."""

    relationship: npt.NDArray[np.float64]
    """Synthetic block-diagonal additive relationship matrix."""

    group: npt.NDArray[np.float64]
    """Synthetic two-level environment matching only reviewed group counts."""

    design: npt.NDArray[np.float64]
    """Synthetic full-rank four-column fixed-effect design."""

    component_sizes: tuple[int, ...]
    """Reviewed target component-size histogram in deterministic order."""

    group_counts: tuple[int, int]
    """Exact synthetic group counts matched to retained target aggregates."""

    nonzero_relationship_pairs: int
    """Number of related synthetic pairs below the matrix diagonal."""

    observed_nonzero_relationship_pairs: int
    """Retained aggregate target related-pair count, never pair identities."""

    cross_group_nonzero_relationship_pairs: int
    """Number of synthetic related pairs crossing environment groups."""

    observed_cross_group_nonzero_relationship_pairs: int
    """Retained aggregate target cross-group related-pair count."""

    mixed_group_components: int
    """Number of synthetic relationship components containing both groups."""

    fixture_sha256: str
    """Exact reviewed fixture digest used to select the design."""


def fixture_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 of one values-free fixture.

    Args:
        path: Regular fixture file whose bytes select the target envelope.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        ValueError: If the selected path is not a regular file.
    """
    if not path.is_file():
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_NOT_FILE")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_envelope(path: Path, *, expected_sha256: str) -> TargetEnvelope:
    """Build a synthetic target design from reviewed aggregate facts only.

    Args:
        path: Values-free aggregate fixture selected by the release command.
        expected_sha256: Exact reviewed fixture digest pinned outside the file.

    Returns:
        Deterministic participant-free pedigree, group, and covariate envelope.

    Raises:
        ValueError: If identity, aggregates, generator output, or design facts
            disagree with the reviewed contract.
    """
    selected_sha256: str = fixture_sha256(path)
    """Bound the generated design to exact reviewed fixture bytes."""

    if selected_sha256 != expected_sha256:
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_SHA256_MISMATCH")

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    """Parsed the aggregate-only fixture after exact-byte verification."""

    if not isinstance(loaded, dict):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_ROOT_INVALID")
    fixture: dict[str, object] = loaded
    """Narrowed the fixture to the required object shape."""

    identity: tuple[object, ...] = (
        fixture.get("schema_version"),
        fixture.get("fixture_id"),
        fixture.get("reviewed"),
        fixture.get("participant_free"),
        fixture.get("provenance"),
    )
    """Collected the complete values-free review identity."""

    if identity != (
        1,
        "discrete_gene_by_environment_target_design",
        True,
        True,
        "accepted historical receipts and aggregate-only read-only target review",
    ):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_IDENTITY_INVALID")

    observed_value: object = fixture.get("observed_structure")
    """Selected the retained non-identifying target aggregates."""

    if not isinstance(observed_value, dict):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_STRUCTURE_INVALID")
    observed: dict[str, object] = observed_value
    """Narrowed the observed structural envelope to an object."""

    histogram_value: object = observed.get("component_size_histogram")
    """Read the reviewed component-size histogram without family labels."""

    if not isinstance(histogram_value, dict):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_HISTOGRAM_INVALID")
    try:
        histogram: dict[int, int] = {
            int(size): int(count) for size, count in histogram_value.items()
        }
        """Converted JSON keys to positive component-size integers."""
    except (TypeError, ValueError) as error:
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_HISTOGRAM_INVALID") from error

    if any(size < 1 or count < 1 for size, count in histogram.items()):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_HISTOGRAM_INVALID")

    component_sizes: tuple[int, ...] = tuple(
        size for size, count in sorted(histogram.items()) for _ in range(count)
    )
    """Expanded the reviewed values-free histogram deterministically."""

    observed_identity: tuple[object, ...] = (
        observed.get("analysis_n"),
        observed.get("component_count"),
        observed.get("largest_component"),
        observed.get("group_counts"),
        observed.get("nonzero_relationship_pairs"),
        observed.get("cross_group_nonzero_relationship_pairs"),
        observed.get("mixed_group_components"),
        observed.get("fixed_effect_columns"),
        observed.get("fixed_effect_rank"),
    )
    """Collected every target aggregate required by the structural claim."""

    if observed_identity != (
        1_910,
        203,
        180,
        [758, 1_152],
        27_821,
        13_322,
        77,
        ["intercept", "standardised_age", "standardised_age_squared", "first_group"],
        4,
    ):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_STRUCTURE_INVALID")
    if (
        sum(component_sizes) != observed["analysis_n"]
        or len(component_sizes) != observed["component_count"]
        or max(component_sizes) != observed["largest_component"]
    ):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_COMPONENTS_INVALID")

    acceptance_value: object = fixture.get("calibration_acceptance")
    """Selected the prewritten release-sized Monte Carlo decision contract."""

    expected_acceptance: dict[str, object] = {
        "release_replicates": 500,
        "levels": [0.01, 0.05, 0.1],
        "fail_when": ("one_sided_95_percent_clopper_pearson_lower_above_level"),
        "maximum_refused_or_nonconverged": 0,
        "scenarios": list(SCENARIO_CHOICES),
        "tests": list(NULLS),
    }
    """Restated every acceptance coordinate independently of fixture edits."""

    if acceptance_value != expected_acceptance:
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_ACCEPTANCE_INVALID")

    pedigree_value: object = fixture.get("synthetic_pedigree")
    """Selected the participant-free pedigree generator contract."""

    if not isinstance(pedigree_value, dict):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_PEDIGREE_INVALID")
    pedigree: dict[str, object] = pedigree_value
    """Narrowed the synthetic pedigree settings to an object."""

    pedigree_identity: tuple[object, ...] = (
        pedigree.get("generator"),
        pedigree.get("seed"),
        pedigree.get("sibling_mean"),
        pedigree.get("marry_in_probability"),
        pedigree.get("maximum_generations"),
        pedigree.get("generated_nonzero_relationship_pairs"),
        pedigree.get("relative_nonzero_pair_discrepancy"),
        pedigree.get("maximum_relative_nonzero_pair_discrepancy"),
    )
    """Collected every fixed pedigree generator and acceptance coordinate."""

    if pedigree_identity != (
        "founder_order_structure_matched",
        7,
        3.0,
        0.35,
        5,
        27_691,
        0.004672729233312965,
        0.005,
    ):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_PEDIGREE_INVALID")

    generator: np.random.Generator = np.random.default_rng(int(pedigree["seed"]))
    """Recreated the reviewed participant-free pedigree stream."""

    blocks: list[npt.NDArray[np.float64]] = []
    """Collected independently generated synthetic relationship components."""

    for size in component_sizes:
        parents: tuple[tuple[int | None, int | None], ...] = structure_matched_pedigree(
            size,
            generator,
            sibling_mean=float(pedigree["sibling_mean"]),
            marry_in_probability=float(pedigree["marry_in_probability"]),
            maximum_generations=int(pedigree["maximum_generations"]),
        )
        """Generated one exact-sized founder-ordered synthetic pedigree."""

        blocks.append(pedigree_relationship(parents))
    """Converted every synthetic pedigree to a twice-kinship block."""

    people: int = sum(component_sizes)
    """Confirmed the complete participant-free design dimension."""

    relationship: npt.NDArray[np.float64] = np.zeros((people, people), dtype=float)
    """Allocated the synthetic block-diagonal relationship matrix."""

    component_index: npt.NDArray[np.int64] = np.empty(people, dtype=np.int64)
    """Tracked only synthetic component membership for aggregate validation."""

    offset: int = 0
    """Located the next empty relationship block."""

    for index, block in enumerate(blocks):
        stop: int = offset + block.shape[0]
        """Located the exclusive end of the current synthetic component."""

        relationship[offset:stop, offset:stop] = block
        """Placed one synthetic pedigree on the block diagonal."""

        component_index[offset:stop] = index
        """Recorded synthetic component membership without identifiers."""

        offset = stop
        """Advanced to the next independent component."""

    pair_rows, pair_columns = np.nonzero(np.tril(relationship, k=-1))
    """Selected related synthetic pairs only for aggregate counting."""

    nonzero_pairs: int = pair_rows.size
    """Counted nonzero synthetic relationships below the diagonal."""

    if nonzero_pairs != pedigree["generated_nonzero_relationship_pairs"]:
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_PEDIGREE_OUTPUT_INVALID")
    pair_discrepancy: float = abs(
        nonzero_pairs - int(observed["nonzero_relationship_pairs"])
    ) / int(observed["nonzero_relationship_pairs"])
    """Measured the declared total relationship-pair structural discrepancy."""

    if not np.isclose(
        pair_discrepancy,
        float(pedigree["relative_nonzero_pair_discrepancy"]),
        rtol=0.0,
        atol=1e-15,
    ) or pair_discrepancy > float(
        pedigree["maximum_relative_nonzero_pair_discrepancy"]
    ):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_PEDIGREE_OUTPUT_INVALID")

    assignment_value: object = fixture.get("synthetic_group_assignment")
    """Selected the exact-count participant-free group generator."""

    if not isinstance(assignment_value, dict):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_GROUP_INVALID")
    assignment: dict[str, object] = assignment_value
    """Narrowed the synthetic group settings to an object."""

    assignment_identity: tuple[object, ...] = (
        assignment.get("generator"),
        assignment.get("seed"),
        assignment.get("generated_group_counts"),
        assignment.get("generated_cross_group_nonzero_relationship_pairs"),
        assignment.get("generated_mixed_group_components"),
        assignment.get("maximum_relative_cross_pair_discrepancy"),
    )
    """Collected every fixed group generator and structural result."""

    if assignment_identity != (
        "exact_count_permutation",
        974,
        [758, 1_152],
        13_322,
        77,
        0.005,
    ):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_GROUP_INVALID")

    group_index: npt.NDArray[np.int64] = np.ones(people, dtype=np.int64)
    """Started every synthetic row in the larger environment group."""

    first_group: npt.NDArray[np.int64] = np.random.default_rng(
        int(assignment["seed"])
    ).choice(people, 758, replace=False)
    """Selected the exact smaller-group count without participant information."""

    group_index[first_group] = 0
    """Assigned selected synthetic rows to the first environment group."""

    group_counts: tuple[int, int] = (
        int(np.sum(group_index == 0)),
        int(np.sum(group_index == 1)),
    )
    """Recomputed exact group counts from generated labels."""

    cross_pairs: int = int(np.sum(group_index[pair_rows] != group_index[pair_columns]))
    """Counted cross-environment related pairs in the synthetic design."""

    mixed_components: int = sum(
        np.unique(group_index[component_index == index]).size == 2
        for index in range(len(component_sizes))
    )
    """Counted synthetic components contributing cross-environment information."""

    if (
        list(group_counts) != assignment["generated_group_counts"]
        or cross_pairs != assignment["generated_cross_group_nonzero_relationship_pairs"]
        or mixed_components != assignment["generated_mixed_group_components"]
    ):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_GROUP_OUTPUT_INVALID")
    cross_discrepancy: float = abs(
        cross_pairs - int(observed["cross_group_nonzero_relationship_pairs"])
    ) / int(observed["cross_group_nonzero_relationship_pairs"])
    """Measured cross-group structural disagreement from aggregate counts only."""

    if cross_discrepancy > float(assignment["maximum_relative_cross_pair_discrepancy"]):
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_GROUP_OUTPUT_INVALID")

    fixed_value: object = fixture.get("fixed_effects")
    """Selected the participant-free fixed-design generator contract."""

    expected_fixed: dict[str, object] = {
        "age_generator": "normal_then_standardise",
        "age_seed": 20_260_821,
        "generating_mean": 45.0,
        "generating_standard_deviation": 15.0,
        "matches_participant_age_moments": False,
        "require_full_rank": True,
    }
    """Restated every pre-reviewed fixed-design setting outside the fixture."""

    if fixed_value != expected_fixed:
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_DESIGN_INVALID")

    age_years: npt.NDArray[np.float64] = np.random.default_rng(
        int(expected_fixed["age_seed"])
    ).normal(
        float(expected_fixed["generating_mean"]),
        float(expected_fixed["generating_standard_deviation"]),
        people,
    )
    """Generated synthetic ages without retaining target age moments."""

    age: npt.NDArray[np.float64] = (age_years - age_years.mean()) / age_years.std()
    """Standardized age within the participant-free synthetic design."""

    design: npt.NDArray[np.float64] = np.column_stack(
        [np.ones(people), age, age * age, (group_index == 0).astype(float)]
    )
    """Reproduced only the reviewed four-column design identity."""

    if np.linalg.matrix_rank(design) != observed["fixed_effect_rank"]:
        raise ValueError("DISCRETE_GXE_TARGET_FIXTURE_DESIGN_INVALID")

    return TargetEnvelope(
        relationship=np.ascontiguousarray(relationship),
        group=np.ascontiguousarray(group_index.astype(float) + 1.0),
        design=np.ascontiguousarray(design),
        component_sizes=component_sizes,
        group_counts=group_counts,
        nonzero_relationship_pairs=nonzero_pairs,
        observed_nonzero_relationship_pairs=int(observed["nonzero_relationship_pairs"]),
        cross_group_nonzero_relationship_pairs=cross_pairs,
        observed_cross_group_nonzero_relationship_pairs=int(
            observed["cross_group_nonzero_relationship_pairs"]
        ),
        mixed_group_components=mixed_components,
        fixture_sha256=selected_sha256,
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fully explicit participant-free calibration coordinates.

    Args:
        argv: Optional argument sequence used by tests and programmatic callers.

    Returns:
        Validated command-line coordinates with no scientific defaults.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built one fail-closed command interface for release automation."""

    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--levels", type=float, nargs="+", required=True)
    parser.add_argument(
        "--scenarios",
        choices=SCENARIO_CHOICES,
        nargs="+",
        required=True,
    )
    parser.add_argument("--tests", choices=NULLS, nargs="+", required=True)
    parser.add_argument("--no-write", action="store_true", required=True)
    arguments: argparse.Namespace = parser.parse_args(argv)
    """Read every fixed design, simulation, and evidence coordinate explicitly."""

    if arguments.replicates < 1 or arguments.workers < 1:
        parser.error("--replicates and --workers must be positive")
    if len(set(arguments.levels)) != len(arguments.levels) or any(
        not 0.0 < level < 1.0 for level in arguments.levels
    ):
        parser.error(
            "--levels must be unique probabilities strictly between zero and one"
        )
    if len(set(arguments.scenarios)) != len(arguments.scenarios):
        parser.error("--scenarios must be unique")
    if len(set(arguments.tests)) != len(arguments.tests):
        parser.error("--tests must be unique")
    return arguments


def calibration_decision(
    *,
    rejections: int,
    computed: int,
    attempted: int,
    level: float,
    level_check: bool,
) -> dict[str, object]:
    """Score one fixed calibration cell without dropping failed attempts.

    Args:
        rejections: Successful tests at or below the nominal threshold.
        computed: Attempts yielding a valid public test result.
        attempted: Prespecified total replicate count.
        level: Nominal rejection probability under the null.
        level_check: Whether this scenario gives the selected test nothing to find.

    Returns:
        Exact one-sided Monte Carlo decision and complete denominator facts.

    Raises:
        ValueError: If counts or nominal level are internally inconsistent.
    """
    if (
        attempted < 1
        or computed < 0
        or rejections < 0
        or rejections > computed
        or computed > attempted
        or not 0.0 < level < 1.0
    ):
        raise ValueError("DISCRETE_GXE_CALIBRATION_COUNTS_INVALID")

    rate: float | None = rejections / computed if computed else None
    """Computed the empirical rate only where a public test was available."""

    lower: float | None = (
        None
        if computed == 0
        else (
            0.0
            if rejections == 0
            else float(beta.ppf(0.05, rejections, computed - rejections + 1))
        )
    )
    """Calculated the exact one-sided 95% Clopper-Pearson lower bound."""

    if computed != attempted:
        verdict: str = "incomplete"
        """Made any refusal or unsuccessful fit a campaign failure."""

        passed: bool = False
        """Prevented missing results from vanishing from the decision."""
    elif not level_check:
        verdict = "power_not_gated"
        """Kept alternative-scenario power descriptive without a post hoc floor."""

        passed = True
        """Required completeness but did not invent a power threshold."""
    elif lower is not None and lower > level:
        verdict = "anti_conservative"
        """Found the rejection probability credibly above its nominal level."""

        passed = False
        """Failed the prewritten one-sided Monte Carlo decision."""
    else:
        verdict = "compatible"
        """Found no one-sided exact evidence of anti-conservatism."""

        passed = True
        """Accepted only a complete statistically compatible level cell."""

    return {
        "attempted": attempted,
        "computed": computed,
        "rejections": rejections,
        "rate": rate,
        "one_sided_95_percent_lower": lower,
        "level": level,
        "judged_as": "level" if level_check else "power",
        "verdict": verdict,
        "passed": passed,
    }


def start_worker(payload: dict[str, Any]) -> None:
    """Install immutable participant-free arrays in one worker process.

    Args:
        payload: Relationship, groups, design, factors, and requested tests.
    """
    STATE.clear()
    STATE.update(payload)


def simulating_covariance(
    relationship: npt.NDArray[np.float64],
    group: npt.NDArray[np.float64],
    genetic: tuple[float, float],
    residual: tuple[float, float],
    correlation: float,
) -> npt.NDArray[np.float64]:
    """Build the prespecified discrete-GxE simulation covariance.

    Args:
        relationship: Participant-free additive relationship matrix.
        group: Two-level synthetic environment labels.
        genetic: Group-specific genetic standard deviations.
        residual: Group-specific residual standard deviations.
        correlation: Cross-environment genetic correlation.

    Returns:
        Dense Gaussian covariance used only to simulate known-truth responses.
    """
    first: npt.NDArray[np.bool_] = group == 1.0
    """Located the first synthetic environment group."""

    standard_deviation: npt.NDArray[np.float64] = np.where(
        first,
        genetic[0],
        genetic[1],
    )
    """Assigned each row its generating genetic standard deviation."""

    across: npt.NDArray[np.float64] = np.where(
        first[:, None] == first[None, :],
        1.0,
        correlation,
    )
    """Applied the cross-group correlation only between environments."""

    covariance: npt.NDArray[np.float64] = (
        relationship * np.outer(standard_deviation, standard_deviation) * across
    )
    """Combined additive relationships with group genetic covariance."""

    covariance[np.diag_indices(group.size)] += (
        np.where(
            first,
            residual[0],
            residual[1],
        )
        ** 2
    )
    """Added group-specific independent residual variance."""

    return covariance


def run_replicate(job: tuple[str, int]) -> dict[str, Any]:
    """Run and classify one public discrete-GxE fit and requested test set.

    Args:
        job: Fixed scenario name and zero-based replicate index.

    Returns:
        One complete, refused, nonconverged, or test-refused attempt record.
    """
    scenario, index = job
    """Read deterministic simulation coordinates from the work item."""

    relationship: npt.NDArray[np.float64] = STATE["relationship"]
    """Selected the immutable participant-free relationship matrix."""

    group: npt.NDArray[np.float64] = STATE["group"]
    """Selected the immutable synthetic environment labels."""

    design: npt.NDArray[np.float64] = STATE["design"]
    """Selected the immutable participant-free fixed-effect design."""

    factor: npt.NDArray[np.float64] = STATE["factors"][scenario]
    """Selected the known generating covariance factor for this scenario."""

    requested_tests: tuple[str, ...] = STATE["tests"]
    """Selected the exact public tests fixed by the command."""

    scenario_index: int = SCENARIO_CHOICES.index(scenario)
    """Separated deterministic random streams across scientific scenarios."""

    seed: int = BASE_SEED + scenario_index * 1_000_003 + index
    """Owned a response stream independent of process scheduling."""

    response: npt.NDArray[np.float64] = (
        design @ TRUE_EFFECTS
        + factor @ np.random.default_rng(seed).standard_normal(group.size)
    )
    """Drew the sole simulated participant-free response for this attempt."""

    model: asterism.DiscreteGxeModel = asterism.DiscreteGxeModel(
        relationship,
        group,
        design,
        levels=(1.0, 2.0),
    )
    """Constructed the model through the documented public Python interface."""

    try:
        fit: dict[str, Any] = model.fit(response)
        """Requested the free REML fit through the public method."""
    except ValueError as error:
        return {
            "scenario": scenario,
            "index": index,
            "seed": seed,
            "status": "refused",
            "failure": str(error),
            "correlation": None,
            "p_values": {},
        }

    if fit.get("converged") is not True:
        return {
            "scenario": scenario,
            "index": index,
            "seed": seed,
            "status": "nonconverged",
            "failure": "public free fit returned converged=false",
            "correlation": fit.get("genetic_correlation"),
            "p_values": {},
        }

    p_values: dict[str, float] = {}
    """Collected successful public test p-values by documented null name."""

    for null in requested_tests:
        try:
            outcome: dict[str, Any] = model.test(response, null)
            """Ran one requested comparison through the public test method."""
        except ValueError as error:
            return {
                "scenario": scenario,
                "index": index,
                "seed": seed,
                "status": "test_refused",
                "failure": f"{null}: {error}",
                "correlation": fit.get("genetic_correlation"),
                "p_values": p_values,
            }
        if outcome.get("rule") != EXPECTED_RULES[null]:
            return {
                "scenario": scenario,
                "index": index,
                "seed": seed,
                "status": "test_refused",
                "failure": (
                    f"{null}: expected rule {EXPECTED_RULES[null]!r}, "
                    f"got {outcome.get('rule')!r}"
                ),
                "correlation": fit.get("genetic_correlation"),
                "p_values": p_values,
            }
        p_values[null] = float(outcome["p_value"])
        """Retained only a successful finite public test result."""

    return {
        "scenario": scenario,
        "index": index,
        "seed": seed,
        "status": "complete",
        "failure": None,
        "correlation": float(fit["genetic_correlation"]),
        "p_values": p_values,
    }


def level_status(
    scenario: str,
    null: str,
) -> bool:
    """Return whether one scenario gives a selected test nothing to find.

    Args:
        scenario: Fixed generating scenario name.
        null: Public test null being judged.

    Returns:
        True for level calibration and False for descriptive power.
    """
    genetic, residual, correlation = SCENARIOS[scenario]
    """Read the fixed generating covariance coordinates."""

    is_complete_null: bool = scenario == "null"
    """Identified the one scenario satisfying all five nulls."""

    nothing_to_find: dict[str, bool] = {
        "gene_by_environment": genetic[0] == genetic[1] and correlation >= 1.0,
        "any_difference": is_complete_null,
        "correlation": correlation >= 1.0,
        "genetic": genetic[0] == genetic[1],
        "residual": residual[0] == residual[1],
    }
    """Encoded each null separately, especially the noisier-group sentinel."""

    return nothing_to_find[null]


def score_campaign(
    results: list[dict[str, Any]],
    *,
    scenarios: tuple[str, ...],
    tests: tuple[str, ...],
    levels: tuple[float, ...],
    replicates: int,
) -> tuple[dict[str, object], list[str]]:
    """Aggregate every attempt under prewritten exact Monte Carlo rules.

    Args:
        results: Classified public fit/test attempt records.
        scenarios: Prespecified scenario order.
        tests: Prespecified public null order.
        levels: Nominal levels judged in every selected cell.
        replicates: Fixed attempted denominator per scenario.

    Returns:
        Scenario-indexed evidence and all campaign failures.
    """
    recorded: dict[str, object] = {}
    """Collected complete per-scenario denominators and decisions."""

    failures: list[str] = []
    """Collected every unsuccessful attempt and anti-conservative level cell."""

    for scenario in scenarios:
        selected: list[dict[str, Any]] = [
            result for result in results if result["scenario"] == scenario
        ]
        """Selected the fixed replicate denominator for this scenario."""

        status_counts: dict[str, int] = {
            status: sum(result["status"] == status for result in selected)
            for status in ("complete", "refused", "nonconverged", "test_refused")
        }
        """Counted every mutually exclusive outcome classification."""

        if len(selected) != replicates or sum(status_counts.values()) != replicates:
            failures.append(f"{scenario}: not every prespecified replicate was scored")
        for status in ("refused", "nonconverged", "test_refused"):
            if status_counts[status]:
                failures.append(
                    f"{scenario}: {status_counts[status]} {status} attempts"
                )

        test_records: dict[str, object] = {}
        """Collected one exact decision grid per public test."""

        for null in tests:
            p_values: list[float] = [
                float(result["p_values"][null])
                for result in selected
                if result["status"] == "complete" and null in result["p_values"]
            ]
            """Retained only complete public results without changing the denominator."""

            decisions: dict[str, object] = {}
            """Collected exact decisions across requested nominal levels."""

            for level in levels:
                rejections: int = sum(value <= level for value in p_values)
                """Counted successful public p-values crossing this threshold."""

                decision: dict[str, object] = calibration_decision(
                    rejections=rejections,
                    computed=len(p_values),
                    attempted=replicates,
                    level=level,
                    level_check=level_status(scenario, null),
                )
                """Applied the exact one-sided fail-closed Monte Carlo rule."""

                decisions[str(level)] = decision
                """Indexed this exact decision by its explicit nominal level."""

                if not decision["passed"]:
                    failures.append(
                        f"{scenario}/{null}/{level:g}: {decision['verdict']}"
                    )

            test_records[null] = {
                "expected_rule": EXPECTED_RULES[null],
                "judged_as": "level" if level_status(scenario, null) else "power",
                "decisions": decisions,
            }
            """Recorded the rule, interpretation, and exact level decisions."""

        correlations: list[float] = [
            float(result["correlation"])
            for result in selected
            if result["status"] == "complete" and result["correlation"] is not None
        ]
        """Selected complete free-fit correlations for boundary diagnostics."""

        recorded[scenario] = {
            "attempted": replicates,
            **status_counts,
            "correlation_at_upper_bound": (
                float(np.mean(np.asarray(correlations) > 1.0 - 1e-6))
                if correlations
                else None
            ),
            "median_correlation": (
                float(np.median(correlations)) if correlations else None
            ),
            "tests": test_records,
        }
        """Recorded complete denominators and decisions for this scenario."""
    return recorded, failures


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit values-free public-API discrete-GxE campaign.

    Args:
        argv: Optional explicit command coordinates for tests and callers.

    Returns:
        Zero only when every selected attempt and Monte Carlo decision passes.
    """
    arguments: argparse.Namespace = parse_arguments(argv)
    """Read all scientific campaign coordinates without environment defaults."""

    envelope: TargetEnvelope = target_envelope(
        arguments.fixture,
        expected_sha256=arguments.fixture_sha256,
    )
    """Built the deterministic target structure from reviewed aggregates only."""

    scenarios: tuple[str, ...] = tuple(arguments.scenarios)
    """Fixed selected scenario order for streams and evidence."""

    tests: tuple[str, ...] = tuple(arguments.tests)
    """Fixed selected public test order for every replicate."""

    levels: tuple[float, ...] = tuple(arguments.levels)
    """Fixed selected nominal levels before simulation."""

    factors: dict[str, npt.NDArray[np.float64]] = {}
    """Collected one reusable known-covariance factor per scenario."""

    for scenario in scenarios:
        genetic, residual, correlation = SCENARIOS[scenario]
        """Read this scenario's fixed generating covariance coordinates."""

        covariance: npt.NDArray[np.float64] = simulating_covariance(
            envelope.relationship,
            envelope.group,
            genetic,
            residual,
            correlation,
        )
        """Built the participant-free known-truth response covariance."""

        factors[scenario] = np.linalg.cholesky(
            covariance + 1e-10 * np.eye(covariance.shape[0])
        )
        """Factored the generating covariance once outside the replicate loop."""

    payload: dict[str, Any] = {
        "relationship": envelope.relationship,
        "group": envelope.group,
        "design": envelope.design,
        "factors": factors,
        "tests": tests,
    }
    """Prepared immutable participant-free worker state."""

    jobs: list[tuple[str, int]] = [
        (scenario, index)
        for scenario in scenarios
        for index in range(arguments.replicates)
    ]
    """Enumerated the complete prespecified scenario-by-replicate grid."""

    started: float = time.perf_counter()
    """Started campaign timing after fixed covariance preparation."""

    if arguments.workers == 1:
        start_worker(payload)
        results: list[dict[str, Any]] = [run_replicate(job) for job in jobs]
        """Ran the deterministic serial path used by bounded integration tests."""
    else:
        with ProcessPoolExecutor(
            arguments.workers,
            initializer=start_worker,
            initargs=(payload,),
        ) as pool:
            results = list(pool.map(run_replicate, jobs, chunksize=1))
            """Ran attempts without scheduling-dependent random streams."""

    recorded, failures = score_campaign(
        results,
        scenarios=scenarios,
        tests=tests,
        levels=levels,
        replicates=arguments.replicates,
    )
    """Applied fixed denominators and exact prewritten Monte Carlo decisions."""

    release_sized: bool = (
        arguments.replicates == 500
        and scenarios == SCENARIO_CHOICES
        and tests == NULLS
        and levels == (0.01, 0.05, 0.1)
    )
    """Distinguished bounded smoke evidence from the exact release campaign."""

    record: dict[str, object] = {
        "check": "discrete_gxe_calibration",
        "participant_free": True,
        "fixture_sha256": envelope.fixture_sha256,
        "synthetic_structure_matching": True,
        "participant_structure_reconstructed": False,
        "public_interfaces": [
            "asterism.DiscreteGxeModel.fit",
            "asterism.DiscreteGxeModel.test",
        ],
        "estimator": "reml",
        "people": envelope.group.size,
        "group_counts": list(envelope.group_counts),
        "component_count": len(envelope.component_sizes),
        "largest_component": max(envelope.component_sizes),
        "nonzero_relationship_pairs": envelope.nonzero_relationship_pairs,
        "observed_nonzero_relationship_pairs": (
            envelope.observed_nonzero_relationship_pairs
        ),
        "cross_group_nonzero_relationship_pairs": (
            envelope.cross_group_nonzero_relationship_pairs
        ),
        "observed_cross_group_nonzero_relationship_pairs": (
            envelope.observed_cross_group_nonzero_relationship_pairs
        ),
        "mixed_group_components": envelope.mixed_group_components,
        "replicates_per_scenario": arguments.replicates,
        "levels": list(levels),
        "scenarios": list(scenarios),
        "tests": list(tests),
        "workers": arguments.workers,
        "attempted": len(jobs),
        "elapsed_seconds": time.perf_counter() - started,
        "every_replicate_scored": sum(
            int(scenario_record[status])
            for scenario_record in recorded.values()
            for status in ("complete", "refused", "nonconverged", "test_refused")
        )
        == len(jobs),
        "release_sized": release_sized,
        "acceptance": (
            "one-sided 95% Clopper-Pearson lower bound must not exceed level; "
            "zero refused, nonconverged, or test-refused attempts"
        ),
        "results": recorded,
        "passed": not failures,
        "failures": failures,
    }
    """Built one auditable values-free public-API calibration receipt."""

    print(json.dumps(record, indent=2))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
