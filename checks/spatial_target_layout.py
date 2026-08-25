"""Qualify spatial-component presence on a values-free target layout.

The fixture retains only aggregate relationship-component sizes, shared-location
counts, trait-level missingness counts, and pairwise-distance quantiles. The
coordinates used here are newly generated points in a synthetic planar system;
they are not perturbed participant locations and cannot be transformed back to
one. The fit includes both a shared-location component and the spatial kernel so
that an estimated short range cannot silently stand in for a household effect.

The target result is fail-closed. Every requested parametric-bootstrap refit must
produce a usable statistic, the free fit must converge, and the deterministic
spatial alternative must be detected at the prewritten five-per-cent level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import asterism
import numpy as np
import numpy.typing as npt
from one_trait_coverage import (
    historical_covariate_design,
    pedigree_relationship,
    structure_matched_pedigree,
)

EXPECTED_FIXTURE_IDENTITY: tuple[object, ...] = (
    1,
    "spatial_component_presence_target_layout",
    True,
    True,
    "target-scale synthetic spatial-layout stress envelope",
)
"""Pinned the complete review and scientific-scope identity."""

EXPECTED_DISTANCE_QUANTILES: tuple[float, ...] = (
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99,
    1.00,
)
"""Selected aggregate geometry coordinates before generating the layout."""

EXPECTED_BOOTSTRAP_RULE: str = "parametric_bootstrap_add_one"
"""Pinned the only valid reference for unidentified spatial range under the null."""

TRUE_BETA: npt.NDArray[np.float64] = np.asarray(
    [2.0, 0.30, -0.10, 0.50, 0.05, -0.02], dtype=np.float64
)
"""Fixed the synthetic mean independently of every covariance fit."""


@dataclass(frozen=True)
class SpatialTargetEnvelope:
    """Hold one deterministic participant-free target spatial design.

    Attributes:
        relationship: Structure-matched additive relationship matrix.
        household: Shared-synthetic-location indicator matrix.
        distance: Synthetic planar pairwise distances in kilometres.
        fixed_effects: Six-column age-by-sex mean design.
        component_sizes: Reviewed relationship-component sizes.
        location_sizes: Reviewed shared-location cluster sizes.
        eligible_people: Aggregate count before trait/covariate selection.
        missing_people: Aggregate rows excluded before model construction.
        fixture_sha256: Exact values-free fixture commitment.
        relative_quantile_errors: Synthetic-versus-aggregate distance errors.
    """

    relationship: npt.NDArray[np.float64]
    """Stored the complete synthetic additive relationship matrix."""

    household: npt.NDArray[np.float64]
    """Stored the complete shared-synthetic-location covariance matrix."""

    distance: npt.NDArray[np.float64]
    """Stored the synthetic planar pairwise distances in kilometres."""

    fixed_effects: npt.NDArray[np.float64]
    """Stored the tracked six-column fixed-effect design identity."""

    component_sizes: tuple[int, ...]
    """Retained aggregate relationship-component sizes in generated row order."""

    location_sizes: tuple[int, ...]
    """Retained aggregate location-cluster sizes before row permutation."""

    eligible_people: int
    """Retained the values-free usable-geocode count before analysis selection."""

    missing_people: int
    """Retained the values-free trait-or-covariate exclusion count."""

    fixture_sha256: str
    """Bound the generated target design to exact reviewed fixture bytes."""

    relative_quantile_errors: dict[str, float]
    """Measured synthetic distance geometry against frozen aggregate quantiles."""


@dataclass(frozen=True)
class SpatialProblem:
    """Hold arrays needed by one public spatial fit and presence bootstrap.

    Attributes:
        relationship: Additive relationship matrix.
        household: Shared-location covariance matrix.
        distance: Pairwise distances in kilometres.
        fixed_effects: Fixed mean design.
        response: Complete continuous response.
    """

    relationship: npt.NDArray[np.float64]
    """Stored the first fixed covariance component."""

    household: npt.NDArray[np.float64]
    """Stored the second fixed covariance component."""

    distance: npt.NDArray[np.float64]
    """Stored the spatial-kernel distance argument."""

    fixed_effects: npt.NDArray[np.float64]
    """Stored the public model's fixed-effect design."""

    response: npt.NDArray[np.float64]
    """Stored the complete simulated continuous outcome."""


def fixture_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 of exact fixture bytes.

    Args:
        path: Values-free reviewed fixture.

    Returns:
        Lowercase hexadecimal SHA-256 commitment.
    """
    payload: bytes = path.read_bytes()
    """Read the fixture without normalising its JSON representation."""

    return hashlib.sha256(payload).hexdigest()


def load_fixture(path: Path) -> dict[str, object]:
    """Load one JSON object from a values-free fixture.

    Args:
        path: Reviewed target-layout fixture.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the fixture root is not an object.
    """
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    """Parsed only the committed local values-free record."""

    if not isinstance(loaded, dict):
        raise ValueError("SPATIAL_TARGET_FIXTURE_ROOT_INVALID")
    return loaded


def nested_keys(value: object) -> set[str]:
    """Collect mapping keys recursively without exposing stored values.

    Args:
        value: Arbitrarily nested JSON-shaped value.

    Returns:
        Set of every mapping key below the supplied root.
    """
    keys: set[str] = set()
    """Started the values-free key inventory."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(nested_keys(child))
        """Visited mapping children without retaining their scalar values."""
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            keys.update(nested_keys(child))
        """Visited JSON array children recursively."""
    return keys


def mapping_field(
    record: Mapping[str, object], key: str, code: str
) -> dict[str, object]:
    """Return one required nested mapping with a stable refusal code.

    Args:
        record: Parent JSON object.
        key: Required child key.
        code: Stable error when the child is not a mapping.

    Returns:
        Child mapping narrowed to string keys.

    Raises:
        ValueError: If the requested child is not a mapping.
    """
    value: object = record.get(key)
    """Read the requested fixture field without a permissive default."""

    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def integer_histogram(value: object, code: str) -> dict[int, int]:
    """Parse one positive integer histogram from a JSON object.

    Args:
        value: Candidate JSON histogram.
        code: Stable refusal for malformed sizes or counts.

    Returns:
        Positive integer sizes mapped to positive integer counts.

    Raises:
        ValueError: If the object is malformed or contains nonpositive entries.
    """
    if not isinstance(value, dict):
        raise ValueError(code)

    try:
        histogram: dict[int, int] = {
            int(size): int(count) for size, count in value.items()
        }
        """Converted JSON string keys and numeric values to integers."""
    except (TypeError, ValueError) as error:
        raise ValueError(code) from error

    if any(size < 1 or count < 1 for size, count in histogram.items()):
        raise ValueError(code)
    return histogram


def expanded_histogram(histogram: Mapping[int, int]) -> tuple[int, ...]:
    """Expand a positive size histogram in deterministic ascending order.

    Args:
        histogram: Positive sizes mapped to their multiplicities.

    Returns:
        Repeated sizes ordered first by size.
    """
    return tuple(
        size for size, count in sorted(histogram.items()) for _ in range(count)
    )


def synthetic_relationship(
    component_sizes: Sequence[int], settings: Mapping[str, object]
) -> tuple[npt.NDArray[np.float64], int]:
    """Generate a block-diagonal structure-matched pedigree relationship.

    Args:
        component_sizes: Exact aggregate independent-component sizes.
        settings: Frozen predecessor generator and acceptance settings.

    Returns:
        Complete relationship matrix and its nonzero lower-triangle pair count.

    Raises:
        ValueError: If generator settings or generated aggregates drift.
    """
    identity: tuple[object, ...] = (
        settings.get("name"),
        settings.get("seed"),
        settings.get("sibling_mean"),
        settings.get("marry_in_probability"),
        settings.get("maximum_generations"),
        settings.get("generated_nonzero_relationship_pairs"),
        settings.get("maximum_relative_nonzero_pair_discrepancy"),
    )
    """Collected every generator coordinate and prewritten structural bound."""

    if identity != (
        "founder_order_structure_matched",
        7,
        3.0,
        0.35,
        5,
        23_498,
        0.05,
    ):
        raise ValueError("SPATIAL_TARGET_PEDIGREE_GENERATOR_INVALID")

    people: int = sum(component_sizes)
    """Counted target analysis rows across independent components."""

    relationship: npt.NDArray[np.float64] = np.zeros((people, people), dtype=np.float64)
    """Allocated the complete block-diagonal relationship matrix."""

    generator: np.random.Generator = np.random.default_rng(7)
    """Recreated the fixed participant-free pedigree stream."""

    offset: int = 0
    """Tracked the first global row of each relationship component."""

    for size in component_sizes:
        parents: tuple[tuple[int | None, int | None], ...] = structure_matched_pedigree(
            size,
            generator,
            sibling_mean=3.0,
            marry_in_probability=0.35,
            maximum_generations=5,
        )
        """Generated one founder-ordered pedigree of the exact aggregate size."""

        block: npt.NDArray[np.float64] = pedigree_relationship(parents)
        """Converted the synthetic pedigree to its additive relationship block."""

        stop: int = offset + size
        """Located the exclusive end of the component's square block."""

        relationship[offset:stop, offset:stop] = block
        """Placed the independent relationship block on the complete diagonal."""

        offset = stop
        """Advanced to the next independent component."""
    """Generated all structure-matched relationship components."""

    nonzero_pairs: int = int(np.count_nonzero(np.tril(relationship, k=-1)))
    """Counted nonzero off-diagonal relationships for structural evidence."""

    if nonzero_pairs != 23_498:
        raise ValueError("SPATIAL_TARGET_PEDIGREE_GENERATION_DRIFTED")
    return relationship, nonzero_pairs


def synthetic_location_sizes(histogram: Mapping[int, int]) -> tuple[int, ...]:
    """Order exact location sizes so every synthetic outlier is a singleton.

    Args:
        histogram: Reviewed aggregate shared-location size histogram.

    Returns:
        Location sizes aligned with the synthetic unique-location generator.

    Raises:
        ValueError: If the histogram cannot supply the fifty frozen outliers.
    """
    singleton_count: int = int(histogram.get(1, 0))
    """Counted aggregate locations occupied by one analysis row."""

    if singleton_count < 50:
        raise ValueError("SPATIAL_TARGET_OUTLIER_SINGLETONS_UNAVAILABLE")

    clustered: npt.NDArray[np.int64] = np.asarray(
        [
            size
            for size, count in sorted(histogram.items())
            if size > 1
            for _ in range(count)
        ],
        dtype=np.int64,
    )
    """Expanded nonsingleton aggregate locations before deterministic shuffling."""

    generator: np.random.Generator = np.random.default_rng(20_260_812)
    """Recreated the fixed synthetic-layout stream."""

    generator.shuffle(clustered)
    """Separated location size from relationship-component ordering."""

    core_singletons: npt.NDArray[np.int64] = np.ones(
        singleton_count - 50, dtype=np.int64
    )
    """Reserved all but fifty singleton locations for the core layout."""

    outlier_singletons: npt.NDArray[np.int64] = np.ones(50, dtype=np.int64)
    """Reserved fifty singletons across regional, far, and extreme locations."""

    sizes: npt.NDArray[np.int64] = np.concatenate(
        [core_singletons, clustered, outlier_singletons]
    )
    """Placed every multiresident location in the core by construction."""

    return tuple(int(size) for size in sizes)


def synthetic_location_coordinates(
    location_sizes: Sequence[int], settings: Mapping[str, object]
) -> npt.NDArray[np.float64]:
    """Generate synthetic planar location coordinates from aggregate geometry.

    Args:
        location_sizes: Exact aggregate people per unique synthetic location.
        settings: Frozen mixture coordinates and geometric tolerance.

    Returns:
        One two-dimensional coordinate per unique synthetic location.

    Raises:
        ValueError: If any generator coordinate differs from the reviewed fixture.
    """
    core: dict[str, object] = mapping_field(
        settings, "core", "SPATIAL_TARGET_LAYOUT_CORE_INVALID"
    )
    """Selected the synthetic core-neighbourhood settings."""

    regional: dict[str, object] = mapping_field(
        settings, "regional", "SPATIAL_TARGET_LAYOUT_REGIONAL_INVALID"
    )
    """Selected the synthetic regional-outlier settings."""

    far: dict[str, object] = mapping_field(
        settings, "far", "SPATIAL_TARGET_LAYOUT_FAR_INVALID"
    )
    """Selected the synthetic far-outlier settings."""

    extreme: dict[str, object] = mapping_field(
        settings, "extreme", "SPATIAL_TARGET_LAYOUT_EXTREME_INVALID"
    )
    """Selected the synthetic extreme-outlier settings."""

    identity: tuple[object, ...] = (
        settings.get("coordinate_system"),
        settings.get("matches_observed_locations"),
        settings.get("seed"),
        settings.get("row_permutation_seed"),
        settings.get("outlying_locations_are_singletons"),
        settings.get("maximum_relative_quantile_error"),
        tuple(core.values()),
        tuple(regional.values()),
        tuple(far.values()),
        tuple(extreme.values()),
    )
    """Collected the complete synthetic geometry identity in fixture order."""

    if identity != (
        "synthetic planar kilometres",
        False,
        20_260_812,
        20_260_813,
        True,
        0.10,
        (1_090, 40, "independent Student t", 2.0, 4.3, 0.55),
        (40, [250.0, 0.0], 30.0),
        (5, [570.0, 0.0], 30.0),
        (5, [2980.0, 0.0], 20.0),
    ):
        raise ValueError("SPATIAL_TARGET_LAYOUT_GENERATOR_INVALID")

    locations: int = len(location_sizes)
    """Counted unique synthetic locations from the aggregate histogram."""

    if locations != 1_140:
        raise ValueError("SPATIAL_TARGET_LOCATION_COUNT_INVALID")

    generator: np.random.Generator = np.random.default_rng(20_260_812)
    """Recreated the same stream used to shuffle nonsingleton sizes."""

    clustered: npt.NDArray[np.int64] = np.asarray(
        [size for size in location_sizes if size > 1], dtype=np.int64
    )
    """Recreated the nonsingleton vector solely to advance the shared stream."""

    generator.shuffle(clustered)
    """Advanced the generator exactly as location-size construction did."""

    coordinates: npt.NDArray[np.float64] = np.empty((locations, 2), dtype=np.float64)
    """Allocated one nonobserved planar point per synthetic location."""

    core_locations: int = 1_090
    """Pinned the count whose pairwise geometry determines ordinary distances."""

    centres: npt.NDArray[np.float64] = generator.standard_t(2.0, size=(40, 2)) * 4.3
    """Generated forty heavy-tailed synthetic neighbourhood centres."""

    coordinates[:core_locations] = centres[
        np.arange(core_locations) % 40
    ] + generator.normal(0.0, 0.55, size=(core_locations, 2))
    """Placed core locations around the frozen synthetic neighbourhoods."""

    regional_stop: int = core_locations + 40
    """Located the exclusive end of regional singleton locations."""

    coordinates[core_locations:regional_stop] = np.asarray([250.0, 0.0]) + (
        generator.normal(0.0, 30.0, size=(40, 2))
    )
    """Generated the aggregate-matched regional distance shoulder."""

    far_stop: int = regional_stop + 5
    """Located the exclusive end of five far singleton locations."""

    coordinates[regional_stop:far_stop] = np.asarray([570.0, 0.0]) + (
        generator.normal(0.0, 30.0, size=(5, 2))
    )
    """Generated the aggregate-matched upper distance tail."""

    coordinates[far_stop:] = np.asarray([2_980.0, 0.0]) + generator.normal(
        0.0, 20.0, size=(5, 2)
    )
    """Generated five extreme singletons to match the aggregate maximum scale."""

    return coordinates


def pairwise_euclidean(
    coordinates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return pairwise Euclidean distance for synthetic planar kilometres.

    Args:
        coordinates: One two-dimensional synthetic point per model row.

    Returns:
        Symmetric dense distance matrix with an exact-zero diagonal.
    """
    displacement: npt.NDArray[np.float64] = (
        coordinates[:, None, :] - coordinates[None, :, :]
    )
    """Computed every synthetic planar displacement."""

    distance: npt.NDArray[np.float64] = np.sqrt(
        np.sum(displacement * displacement, axis=2)
    )
    """Reduced two-coordinate displacements to distances in kilometres."""

    np.fill_diagonal(distance, 0.0)
    """Pinned the exact distance self-comparison required by the public model."""

    return distance


def distance_quantiles(
    distance: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Summarise unique off-diagonal distances at prespecified quantiles.

    Args:
        distance: Symmetric person-level distance matrix.

    Returns:
        String-formatted quantile coordinates mapped to distances.
    """
    upper: npt.NDArray[np.float64] = distance[np.triu_indices(distance.shape[0], k=1)]
    """Selected each unordered person pair exactly once."""

    values: npt.NDArray[np.float64] = np.quantile(upper, EXPECTED_DISTANCE_QUANTILES)
    """Evaluated only the aggregate coordinates frozen before generation."""

    return {
        f"{quantile:.2f}": float(value)
        for quantile, value in zip(EXPECTED_DISTANCE_QUANTILES, values, strict=True)
    }


def quantile_errors(
    generated: Mapping[str, float], observed: Mapping[str, object]
) -> dict[str, float]:
    """Calculate relative error against nonzero reviewed distance quantiles.

    Args:
        generated: Synthetic values at the frozen quantile coordinates.
        observed: Reviewed aggregate values from the controlled source.

    Returns:
        Absolute relative errors by formatted quantile coordinate.

    Raises:
        ValueError: If the reviewed quantile coordinates or values are invalid.
    """
    expected_keys: set[str] = {f"{value:.2f}" for value in EXPECTED_DISTANCE_QUANTILES}
    """Formatted the exact prewritten quantile inventory."""

    if set(observed) != expected_keys:
        raise ValueError("SPATIAL_TARGET_DISTANCE_QUANTILES_INVALID")

    errors: dict[str, float] = {}
    """Allocated the complete generated-versus-observed error record."""

    for key in sorted(expected_keys, key=float):
        raw: object = observed[key]
        """Read one reviewed positive aggregate distance."""

        if not isinstance(raw, (int, float)) or not np.isfinite(raw) or raw <= 0.0:
            raise ValueError("SPATIAL_TARGET_DISTANCE_QUANTILES_INVALID")

        errors[key] = abs(generated[key] - float(raw)) / float(raw)
        """Measured scale-free discrepancy without retaining any pair distance."""
    return errors


def spatial_target_envelope(
    path: Path, *, expected_sha256: str
) -> SpatialTargetEnvelope:
    """Verify one aggregate fixture and build its synthetic target layout.

    Args:
        path: Reviewed values-free target fixture.
        expected_sha256: Exact digest pinned independently by the release command.

    Returns:
        Participant-free matrices and auditable aggregate design facts.

    Raises:
        ValueError: If fixture identity, aggregates, or generation drift.
    """
    selected_sha256: str = fixture_sha256(path)
    """Bound all generated arrays to exact reviewed fixture bytes."""

    if selected_sha256 != expected_sha256:
        raise ValueError("SPATIAL_TARGET_FIXTURE_SHA256_MISMATCH")

    fixture: dict[str, object] = load_fixture(path)
    """Loaded the values-free aggregate after exact-byte verification."""

    identity: tuple[object, ...] = (
        fixture.get("schema_version"),
        fixture.get("fixture_id"),
        fixture.get("reviewed"),
        fixture.get("participant_free"),
        fixture.get("purpose"),
    )
    """Collected the complete fixture review and purpose identity."""

    if identity != EXPECTED_FIXTURE_IDENTITY:
        raise ValueError("SPATIAL_TARGET_FIXTURE_IDENTITY_INVALID")

    forbidden: set[str] = {
        "subject_id",
        "latitude",
        "longitude",
        "coordinates",
        "distance_matrix",
    }
    """Named participant-level fields forbidden from the durable fixture."""

    if nested_keys(fixture) & forbidden:
        raise ValueError("SPATIAL_TARGET_FIXTURE_CONTAINS_PARTICIPANT_FIELDS")

    missingness: dict[str, object] = mapping_field(
        fixture, "missingness", "SPATIAL_TARGET_MISSINGNESS_INVALID"
    )
    """Selected the target selection counts."""

    missingness_identity: tuple[object, ...] = (
        missingness.get("eligible_geocoded_people"),
        missingness.get("analysed_people"),
        missingness.get("missing_trait_or_covariate_rows"),
        missingness.get("selection"),
        missingness.get("synthetic_excluded_rows"),
    )
    """Collected every reviewed target missingness fact and non-claim."""

    if missingness_identity != (
        1_883,
        1_792,
        91,
        "complete trait, age, sex, pedigree, and usable geocode",
        "count only; excluded rows never enter a fitted matrix",
    ):
        raise ValueError("SPATIAL_TARGET_MISSINGNESS_INVALID")

    relationship_record: dict[str, object] = mapping_field(
        fixture, "relationship", "SPATIAL_TARGET_RELATIONSHIP_INVALID"
    )
    """Selected reviewed relationship aggregates and generator settings."""

    component_histogram: dict[int, int] = integer_histogram(
        relationship_record.get("component_size_histogram"),
        "SPATIAL_TARGET_RELATIONSHIP_HISTOGRAM_INVALID",
    )
    """Parsed the exact target relationship-component histogram."""

    component_sizes: tuple[int, ...] = expanded_histogram(component_histogram)
    """Expanded aggregate relationship component sizes deterministically."""

    relationship_identity: tuple[object, ...] = (
        relationship_record.get("component_count"),
        relationship_record.get("largest_component"),
        relationship_record.get("observed_nonzero_relationship_pairs"),
        len(component_sizes),
        max(component_sizes),
        sum(component_sizes),
    )
    """Collected both stated and derived target pedigree aggregates."""

    if relationship_identity != (190, 165, 24_513, 190, 165, 1_792):
        raise ValueError("SPATIAL_TARGET_RELATIONSHIP_AGGREGATES_INVALID")

    generator_settings: dict[str, object] = mapping_field(
        relationship_record,
        "synthetic_generator",
        "SPATIAL_TARGET_PEDIGREE_GENERATOR_INVALID",
    )
    """Selected fixed pedigree generation and discrepancy settings."""

    relationship, generated_pairs = synthetic_relationship(
        component_sizes, generator_settings
    )
    """Generated the structure-matched additive relationship matrix."""

    observed_pairs: int = 24_513
    """Named the reviewed nonzero relationship-pair aggregate."""

    relative_pair_error: float = abs(generated_pairs - observed_pairs) / observed_pairs
    """Measured synthetic relationship density against the aggregate target."""

    if relative_pair_error != generator_settings.get(
        "relative_nonzero_pair_discrepancy"
    ):
        raise ValueError("SPATIAL_TARGET_PEDIGREE_DISCREPANCY_DRIFTED")
    if relative_pair_error > 0.05:
        raise ValueError("SPATIAL_TARGET_PEDIGREE_DISCREPANCY_EXCEEDED")

    shared: dict[str, object] = mapping_field(
        fixture, "shared_locations", "SPATIAL_TARGET_SHARED_LOCATIONS_INVALID"
    )
    """Selected reviewed shared-location aggregates."""

    location_histogram: dict[int, int] = integer_histogram(
        shared.get("location_size_histogram"),
        "SPATIAL_TARGET_LOCATION_HISTOGRAM_INVALID",
    )
    """Parsed the exact people-per-location histogram."""

    location_sizes: tuple[int, ...] = synthetic_location_sizes(location_histogram)
    """Ordered location sizes for the frozen synthetic geometry."""

    if (
        shared.get("unique_locations"),
        shared.get("zero_distance_pairs"),
        len(location_sizes),
        sum(location_sizes),
        max(location_sizes),
    ) != (1_140, 1_014, 1_140, 1_792, 7):
        raise ValueError("SPATIAL_TARGET_SHARED_LOCATION_AGGREGATES_INVALID")

    layout_settings: dict[str, object] = mapping_field(
        fixture, "synthetic_layout", "SPATIAL_TARGET_LAYOUT_GENERATOR_INVALID"
    )
    """Selected the participant-free aggregate-matched layout settings."""

    unique_coordinates: npt.NDArray[np.float64] = synthetic_location_coordinates(
        location_sizes, layout_settings
    )
    """Generated one explicitly nonobserved point per synthetic location."""

    location_labels: npt.NDArray[np.int64] = np.repeat(
        np.arange(len(location_sizes), dtype=np.int64),
        np.asarray(location_sizes, dtype=np.int64),
    )
    """Expanded unique locations to exact person-level aggregate multiplicities."""

    person_coordinates: npt.NDArray[np.float64] = np.repeat(
        unique_coordinates,
        np.asarray(location_sizes, dtype=np.int64),
        axis=0,
    )
    """Expanded synthetic points without exposing any observed location."""

    permutation: npt.NDArray[np.int64] = np.random.default_rng(20_260_813).permutation(
        person_coordinates.shape[0]
    )
    """Broke location ordering away from pedigree and covariate generation."""

    person_coordinates = person_coordinates[permutation]
    """Applied only the participant-free row permutation to synthetic points."""

    location_labels = location_labels[permutation]
    """Kept shared-location membership aligned with permuted synthetic points."""

    distance: npt.NDArray[np.float64] = pairwise_euclidean(person_coordinates)
    """Built the public model's synthetic pairwise-distance argument."""

    household: npt.NDArray[np.float64] = (
        location_labels[:, None] == location_labels[None, :]
    ).astype(np.float64)
    """Built shared-location covariance separately from smooth spatial decay."""

    if int(np.count_nonzero(np.triu(household, k=1))) != 1_014:
        raise ValueError("SPATIAL_TARGET_ZERO_DISTANCE_PAIR_COUNT_DRIFTED")

    summary_record: dict[str, object] = mapping_field(
        fixture, "distance_summary", "SPATIAL_TARGET_DISTANCE_SUMMARY_INVALID"
    )
    """Selected aggregate pair count and quantiles."""

    observed_quantiles: dict[str, object] = mapping_field(
        summary_record,
        "quantiles",
        "SPATIAL_TARGET_DISTANCE_QUANTILES_INVALID",
    )
    """Selected only the prewritten aggregate quantile values."""

    if (
        summary_record.get("units"),
        summary_record.get("pair_count"),
    ) != ("kilometres", 1_604_736):
        raise ValueError("SPATIAL_TARGET_DISTANCE_SUMMARY_INVALID")

    generated_quantiles: dict[str, float] = distance_quantiles(distance)
    """Reduced the synthetic layout to the identical aggregate coordinates."""

    relative_quantile_errors: dict[str, float] = quantile_errors(
        generated_quantiles, observed_quantiles
    )
    """Measured only aggregate geometry agreement."""

    if max(relative_quantile_errors.values()) > 0.10:
        raise ValueError("SPATIAL_TARGET_DISTANCE_GEOMETRY_MISMATCH")

    fixed_record: dict[str, object] = mapping_field(
        fixture, "fixed_effects", "SPATIAL_TARGET_FIXED_EFFECTS_INVALID"
    )
    """Selected the synthetic fixed-effect identity and explicit non-claim."""

    expected_columns: list[str] = [
        "intercept",
        "synthetic_standardised_age",
        "synthetic_standardised_age_squared",
        "synthetic_sex",
        "synthetic_standardised_age_by_sex",
        "synthetic_standardised_age_squared_by_sex",
    ]
    """Restated the tracked six-column formula independently of fixture edits."""

    if (
        fixed_record.get("columns"),
        fixed_record.get("age_and_sex_seed"),
        fixed_record.get("require_full_rank"),
        fixed_record.get("matches_observed_covariate_moments"),
    ) != (expected_columns, 1, True, False):
        raise ValueError("SPATIAL_TARGET_FIXED_EFFECTS_INVALID")

    fixed_effects: npt.NDArray[np.float64] = historical_covariate_design(1_792, 1)
    """Generated a full synthetic age-by-sex mean design at target scale."""

    if np.linalg.matrix_rank(fixed_effects) != 6:
        raise ValueError("SPATIAL_TARGET_FIXED_EFFECTS_RANK_DEFICIENT")

    return SpatialTargetEnvelope(
        relationship=relationship,
        household=household,
        distance=distance,
        fixed_effects=fixed_effects,
        component_sizes=component_sizes,
        location_sizes=location_sizes,
        eligible_people=1_883,
        missing_people=91,
        fixture_sha256=selected_sha256,
        relative_quantile_errors=relative_quantile_errors,
    )


def simulation_problem(
    envelope: SpatialTargetEnvelope, fixture: Mapping[str, object]
) -> SpatialProblem:
    """Simulate the fixed target spatial alternative from reviewed truth.

    Args:
        envelope: Verified participant-free target design.
        fixture: Parsed values-free fixture carrying the prewritten truth.

    Returns:
        Complete target-scale public spatial fit inputs.

    Raises:
        ValueError: If the truth or response seed differs from the reviewed fixture.
    """
    truth: dict[str, object] = mapping_field(
        fixture, "simulation_truth", "SPATIAL_TARGET_TRUTH_INVALID"
    )
    """Selected the fixed covariance alternative and outcome seed."""

    identity: tuple[object, ...] = (
        truth.get("additive"),
        truth.get("shared_location"),
        truth.get("spatial"),
        truth.get("residual"),
        truth.get("half_distance_km"),
        truth.get("response_seed"),
    )
    """Collected every target simulation coordinate before generating outcomes."""

    if identity != (0.35, 0.10, 0.25, 0.30, 35.0, 20_260_812):
        raise ValueError("SPATIAL_TARGET_TRUTH_INVALID")

    decay: float = float(np.log(2.0) / 35.0)
    """Converted the interpretable half-distance to exponential decay per kilometre."""

    covariance: npt.NDArray[np.float64] = (
        0.35 * envelope.relationship
        + 0.10 * envelope.household
        + 0.25 * np.exp(-decay * envelope.distance)
        + 0.30 * np.eye(envelope.relationship.shape[0])
    )
    """Assembled the exact four-component participant-free generating covariance."""

    factor: npt.NDArray[np.float64] = np.linalg.cholesky(
        covariance + 1e-10 * np.eye(covariance.shape[0])
    )
    """Factored the target covariance with a numerical-only diagonal guard."""

    generator: np.random.Generator = np.random.default_rng(20_260_812)
    """Created the prewritten deterministic target response stream."""

    response: npt.NDArray[np.float64] = (
        envelope.fixed_effects @ TRUE_BETA
        + factor @ generator.standard_normal(covariance.shape[0])
    )
    """Generated one complete target spatial alternative independently of fitting."""

    return SpatialProblem(
        relationship=envelope.relationship,
        household=envelope.household,
        distance=envelope.distance,
        fixed_effects=envelope.fixed_effects,
        response=response,
    )


def target_decision(
    fit: Mapping[str, object],
    bootstrap: Mapping[str, object],
    *,
    requested: int,
    alpha: float,
    refusal: str | None,
) -> dict[str, object]:
    """Apply the prewritten fail-closed target detection policy.

    Args:
        fit: Public ``SpatialModel.fit`` record or an empty mapping.
        bootstrap: Public ``SpatialModel.bootstrap`` record or an empty mapping.
        requested: Exact prespecified null-replicate denominator.
        alpha: Prespecified target detection level.
        refusal: Stable estimator exception string, when one occurred.

    Returns:
        Boolean release-rule verdict and one stable reason.
    """
    if refusal is not None:
        return {"passed": False, "reason": "estimator_refused"}
    if fit.get("converged") is not True:
        return {"passed": False, "reason": "free_fit_not_converged"}

    finite_fit_fields: tuple[object, ...] = (
        fit.get("loglik"),
        fit.get("scaled_gradient"),
    )
    """Selected free-fit diagnostics required for numerical reporting."""

    if not all(
        isinstance(value, (int, float)) and np.isfinite(value)
        for value in finite_fit_fields
    ):
        return {"passed": False, "reason": "free_fit_nonfinite"}

    if (
        bootstrap.get("requested") != requested
        or bootstrap.get("replicates") != requested
    ):
        return {"passed": False, "reason": "bootstrap_incomplete"}
    if bootstrap.get("rule") != EXPECTED_BOOTSTRAP_RULE:
        return {"passed": False, "reason": "bootstrap_rule_invalid"}

    statistic: object = bootstrap.get("statistic")
    """Selected the observed likelihood-ratio statistic."""

    p_value: object = bootstrap.get("p_value")
    """Selected the add-one bootstrap probability."""

    smallest: object = bootstrap.get("smallest_p_value")
    """Selected the public record's finite-resolution limit."""

    if not all(
        isinstance(value, (int, float)) and np.isfinite(value)
        for value in (statistic, p_value, smallest)
    ):
        return {"passed": False, "reason": "bootstrap_nonfinite"}

    expected_smallest: float = 1.0 / (requested + 1)
    """Calculated the exact finite resolution independently of the result."""

    if (
        float(statistic) < 0.0
        or not expected_smallest <= float(p_value) <= 1.0
        or not np.isclose(float(smallest), expected_smallest, atol=0.0, rtol=1e-14)
    ):
        return {"passed": False, "reason": "bootstrap_invalid"}
    if float(p_value) > alpha:
        return {"passed": False, "reason": "target_presence_not_detected"}
    return {"passed": True, "reason": "target_presence_detected"}


def run_spatial_problem(
    problem: SpatialProblem,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Run one public spatial fit and complete null bootstrap.

    Args:
        problem: Complete aligned public model inputs.
        bootstrap_replicates: Exact nonzero null-replicate denominator.
        bootstrap_seed: Deterministic compiled bootstrap stream.
        alpha: Prewritten target detection threshold.

    Returns:
        Public records, refusal code, and fail-closed decision.
    """
    if bootstrap_replicates < 1:
        raise ValueError("SPATIAL_TARGET_BOOTSTRAP_REPLICATES_INVALID")
    if bootstrap_seed < 0:
        raise ValueError("SPATIAL_TARGET_BOOTSTRAP_SEED_INVALID")

    synthetic_order: list[str] = [
        f"spatial-target-row-{row}" for row in range(problem.response.shape[0])
    ]
    """Named only synthetic row coordinates for an identifier-free order commitment."""

    model: asterism.SpatialModel = asterism.SpatialModel(
        [problem.relationship, problem.household],
        problem.distance,
        problem.fixed_effects,
        subject_order_sha256=asterism.subject_order_commitment(synthetic_order),
    )
    """Built the documented additive, shared-location, and spatial model."""

    try:
        fit: dict[str, object] = model.fit(
            problem.response, reml=True, integrated=False
        )
        """Fitted the free spatial model through the supported public API."""

        if fit.get("converged") is True:
            bootstrap: dict[str, object] = model.bootstrap(
                problem.response,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
                reml=True,
                integrated=False,
            )
            """Ran every null refit inside the compiled public bootstrap."""

            refusal: str | None = None
            """Recorded successful completion without an estimator exception."""
        else:
            bootstrap = {}
            """Suppressed inferential work after a nonconverged free fit."""

            refusal = None
            """Distinguished an inspectable nonconvergence from an exception."""
    except ValueError as error:
        fit = locals().get("fit", {})
        """Retained any inspectable free fit produced before bootstrap refusal."""

        bootstrap = {}
        """Refused to fabricate a partial bootstrap record."""

        refusal = str(error)
        """Preserved the stable estimator code in release evidence."""

    decision: dict[str, object] = target_decision(
        fit,
        bootstrap,
        requested=bootstrap_replicates,
        alpha=alpha,
        refusal=refusal,
    )
    """Applied the prewritten zero-failure and target-detection policy."""

    return {
        "fit": fit,
        "bootstrap": bootstrap,
        "refusal": refusal,
        "decision": decision,
    }


def small_spatial_problem() -> SpatialProblem:
    """Build a compact real public-API sentinel with distinct fixed components.

    Returns:
        Forty sibling pairs whose shared-location pairs cross family boundaries.
    """
    pairs: int = 40
    """Selected enough rows for stable fixed and spatial component fitting."""

    people: int = 2 * pairs
    """Counted complete synthetic sibling-pair rows."""

    relationship: npt.NDArray[np.float64] = np.eye(people)
    """Started the unit-diagonal additive relationship matrix."""

    for pair in range(pairs):
        first: int = 2 * pair
        """Located the first sibling row."""

        second: int = first + 1
        """Located the second sibling row."""

        relationship[first, second] = relationship[second, first] = 0.5
        """Added one symmetric first-degree relationship."""
    """Built forty unrelated additive sibling blocks."""

    location_labels: npt.NDArray[np.int64] = (
        (np.arange(people, dtype=np.int64) + 1) % people
    ) // 2
    """Paired rows across sibling boundaries into distinct shared locations."""

    household: npt.NDArray[np.float64] = (
        location_labels[:, None] == location_labels[None, :]
    ).astype(np.float64)
    """Built a shared-location component not collinear with relationship."""

    angle: npt.NDArray[np.float64] = 2.0 * np.pi * location_labels / pairs
    """Placed shared locations around a synthetic circle."""

    coordinates: npt.NDArray[np.float64] = np.column_stack(
        [20.0 * np.cos(angle), 20.0 * np.sin(angle)]
    )
    """Generated compact planar positions with repeated shared locations."""

    distance: npt.NDArray[np.float64] = pairwise_euclidean(coordinates)
    """Converted positions to the public model's distance-only interface."""

    fixed_effects: npt.NDArray[np.float64] = np.column_stack(
        [np.ones(people), np.linspace(-1.0, 1.0, people)]
    )
    """Built a full-rank intercept-and-linear mean design."""

    decay: float = float(np.log(2.0) / 10.0)
    """Selected a ten-kilometre spatial half-distance."""

    covariance: npt.NDArray[np.float64] = (
        0.25 * relationship
        + 0.10 * household
        + 0.45 * np.exp(-decay * distance)
        + 0.20 * np.eye(people)
    )
    """Generated a strong but identifiable spatial alternative."""

    generator: np.random.Generator = np.random.default_rng(20_260_812)
    """Created a deterministic sentinel response stream."""

    response: npt.NDArray[np.float64] = fixed_effects @ np.asarray(
        [1.0, 0.2]
    ) + np.linalg.cholesky(covariance) @ generator.standard_normal(people)
    """Generated the complete sentinel response independently of fitting."""

    return SpatialProblem(
        relationship=relationship,
        household=household,
        distance=distance,
        fixed_effects=fixed_effects,
        response=response,
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse every scientific coordinate explicitly from the command line.

    Args:
        argv: Optional argument sequence for tests; defaults to process arguments.

    Returns:
        Parsed required target fixture, bootstrap, and output settings.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Created the standalone target-layout command parser."""

    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    """Required every scientific input rather than accepting ambient defaults."""

    destination: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(
        required=True
    )
    """Required either captured standard output or one explicit result path."""

    destination.add_argument("--no-write", action="store_true")
    destination.add_argument("--output", type=Path)
    """Prevented silent writes or default output destinations."""

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete target design and emit one strict evidence record.

    Args:
        argv: Optional explicit command arguments.

    Returns:
        Zero only when every target-layout and bootstrap rule passes.
    """
    arguments: argparse.Namespace = parse_arguments(argv)
    """Read the exact fixture, bootstrap, and output coordinates."""

    started: float = time.perf_counter()
    """Started target wall-time evidence before any matrix construction."""

    try:
        envelope: SpatialTargetEnvelope = spatial_target_envelope(
            arguments.fixture,
            expected_sha256=arguments.fixture_sha256,
        )
        """Verified and generated the complete participant-free target layout."""

        fixture: dict[str, object] = load_fixture(arguments.fixture)
        """Reloaded only values-free truth and acceptance settings."""

        acceptance: dict[str, object] = mapping_field(
            fixture,
            "bootstrap_acceptance",
            "SPATIAL_TARGET_BOOTSTRAP_ACCEPTANCE_INVALID",
        )
        """Selected the prewritten inferential decision before execution."""

        acceptance_identity: tuple[object, ...] = (
            acceptance.get("replicates"),
            acceptance.get("seed"),
            acceptance.get("alpha"),
            acceptance.get("rule"),
            acceptance.get("require_all_requested_replicates"),
            acceptance.get("any_refit_failure"),
            acceptance.get("require_target_presence_detection"),
        )
        """Collected every frozen bootstrap coordinate and refusal condition."""

        if acceptance_identity != (
            199,
            20_260_812,
            0.05,
            EXPECTED_BOOTSTRAP_RULE,
            True,
            "refuse the complete inferential result",
            True,
        ):
            raise ValueError("SPATIAL_TARGET_BOOTSTRAP_ACCEPTANCE_INVALID")
        if arguments.bootstrap_replicates != acceptance.get("replicates"):
            raise ValueError("SPATIAL_TARGET_BOOTSTRAP_REPLICATES_MISMATCH")
        if arguments.bootstrap_seed != acceptance.get("seed"):
            raise ValueError("SPATIAL_TARGET_BOOTSTRAP_SEED_MISMATCH")

        problem: SpatialProblem = simulation_problem(envelope, fixture)
        """Generated the prewritten target spatial alternative."""

        result: dict[str, object] = run_spatial_problem(
            problem,
            bootstrap_replicates=arguments.bootstrap_replicates,
            bootstrap_seed=arguments.bootstrap_seed,
            alpha=float(acceptance["alpha"]),
        )
        """Ran the supported public fit and complete null bootstrap."""

        evidence: dict[str, object] = {
            "schema_version": 1,
            "analysis": "spatial_component_presence",
            "check": "spatial_target_layout",
            "participant_free": True,
            "fixture_sha256": envelope.fixture_sha256,
            "design": {
                "eligible_geocoded_people": envelope.eligible_people,
                "analysed_people": envelope.relationship.shape[0],
                "missing_trait_or_covariate_rows": envelope.missing_people,
                "relationship_components": len(envelope.component_sizes),
                "largest_relationship_component": max(envelope.component_sizes),
                "unique_synthetic_locations": len(envelope.location_sizes),
                "zero_distance_pairs": int(
                    np.count_nonzero(np.triu(envelope.household, k=1))
                ),
                "relative_distance_quantile_errors": (
                    envelope.relative_quantile_errors
                ),
                "observed_locations_retained": False,
            },
            "bootstrap_replicates": arguments.bootstrap_replicates,
            "bootstrap_seed": arguments.bootstrap_seed,
            "fit": result["fit"],
            "bootstrap": result["bootstrap"],
            "refusal": result["refusal"],
            "decision": result["decision"],
            "elapsed_seconds": time.perf_counter() - started,
            "passed": result["decision"]["passed"],
        }
        """Reduced target arrays to values-free auditable release evidence."""
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        evidence = {
            "schema_version": 1,
            "analysis": "spatial_component_presence",
            "check": "spatial_target_layout",
            "participant_free": True,
            "fixture_sha256": arguments.fixture_sha256,
            "refusal": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "passed": False,
        }
        """Emitted a strict failed receipt instead of losing the requested attempt."""

    payload: str = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    """Serialized strict JSON that cannot hide nonfinite numerical output."""

    if arguments.output is None:
        print(payload, end="")
        """Wrote captured standard output only when explicitly requested."""
    else:
        if arguments.output.exists():
            raise SystemExit("SPATIAL_TARGET_OUTPUT_EXISTS")
        arguments.output.write_text(payload, encoding="utf-8")
        """Wrote once to an explicit nonexisting evidence destination."""

    return 0 if evidence["passed"] is True else 1


if __name__ == "__main__":
    sys.exit(main())
