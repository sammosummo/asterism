"""Measure one-trait Gaussian profile coverage through the public Python API.

Every simulated replicate is scored. A refused or nonconverged free fit and a
profile carrying any failed constrained evaluation are coverage misses. At
heritability bounds, membership comes from the public mixture-calibrated
``contains_lower_bound`` and ``contains_upper_bound`` fields rather than from
an endpoint merely landing on the bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import asterism
import numpy as np
import numpy.typing as npt
from scipy.stats import beta

ACCEPTABLE_COVERAGE_BAND: tuple[float, float] = (0.940, 0.960)
"""Historical exact-binomial compatibility band from ADR 0004."""

HISTORICALLY_CONSERVATIVE_TRUTHS: frozenset[float] = frozenset({0.05, 0.07})
"""Small heritabilities known to over-cover at the 350-person design."""

BASE_SEED: int = 20_260_811
"""Base of disjoint deterministic streams ported from the Rust campaign."""

TRUE_BETA: npt.NDArray[np.float64] = np.array(
    [2.0, 0.30, -0.10, 0.50, 0.05, -0.02], dtype=np.float64
)
"""Generating coefficients for the common six-column fixed-effect design."""

EXTENDED_FAMILY_PARENTS: tuple[tuple[int | None, int | None], ...] = (
    (None, None),
    (None, None),
    (0, 1),
    (0, 1),
    (0, 1),
    (None, None),
    (None, None),
    (None, None),
    (2, 5),
    (2, 5),
    (3, 6),
    (3, 6),
    (4, 7),
    (4, 7),
)
"""Three-generation family ported exactly from the Rust coverage campaign."""


@dataclass(frozen=True)
class SimulationEnvelope:
    """Hold one participant-free relationship and fixed-effect design.

    Attributes:
        relationship: Unit-diagonal additive relationship matrix.
        fixed_effects: Six-column intercept and age-by-sex design.
        component_sizes: Independent relationship-component sizes in row order.
        nonzero_relationship_pairs: Nonzero off-diagonal lower-triangle count.
        fixture_sha256: Exact target-fixture identity, absent for standard designs.
    """

    relationship: npt.NDArray[np.float64]
    """Stored the complete block-diagonal relationship matrix."""

    fixed_effects: npt.NDArray[np.float64]
    """Stored the deterministic six-column fixed-effect design."""

    component_sizes: tuple[int, ...]
    """Retained independent component sizes in generated row order."""

    nonzero_relationship_pairs: int
    """Counted nonzero off-diagonal lower-triangle relationships."""

    fixture_sha256: str | None = None
    """Bound target evidence to a reviewed fixture when one selected the design."""

    observed_nonzero_relationship_pairs: int | None = None
    """Retained the predecessor aggregate pair count for target comparison."""

    relative_pair_count_discrepancy: float | None = None
    """Measured synthetic-versus-observed structural pair-count discrepancy."""


@dataclass(frozen=True)
class CellJob:
    """Describe one independently reproducible heritability coverage cell.

    Attributes:
        truth: Generating heritability.
        truth_index: Position in the explicit command truth grid.
        replicates: Number of fits, all scored.
        families: Explicit family/component count.
        fixture_path: Optional reviewed target-design fixture.
        fixture_sha256: Expected exact fixture digest when a target is selected.
    """

    truth: float
    """Selected one generating heritability on the closed unit interval."""

    truth_index: int
    """Selected the deterministic seed offset independent of worker scheduling."""

    replicates: int
    """Fixed the unconditional denominator for this cell."""

    families: int
    """Fixed the repeated-family or target component count."""

    fixture_path: Path | None
    """Selected the target design only when exact reviewed bytes were supplied."""

    fixture_sha256: str | None
    """Required the exact target-fixture identity independently of its contents."""


def pedigree_relationship(
    parents: Sequence[tuple[int | None, int | None]],
) -> npt.NDArray[np.float64]:
    """Construct twice-kinship for an ordered, non-inbred synthetic pedigree.

    Args:
        parents: Maternal and paternal row indices, with parents preceding children.

    Returns:
        Unit-diagonal additive relationship matrix in pedigree order.
    """
    people: int = len(parents)
    """Counted synthetic pedigree rows."""

    kinship: npt.NDArray[np.float64] = np.zeros((people, people), dtype=np.float64)
    """Allocated the symmetric kinship matrix before ordered recursion."""

    for person in range(people):
        mother: int | None = parents[person][0]
        """Read the founder-ordered maternal row when one was recorded."""

        father: int | None = parents[person][1]
        """Read the founder-ordered paternal row when one was recorded."""

        for earlier in range(person + 1):
            if person == earlier:
                value: float = (
                    0.5 * (1.0 + kinship[mother, father])
                    if mother is not None and father is not None
                    else 0.5
                )
                """Computed self-kinship, including parental inbreeding if present."""
            elif mother is not None and father is not None:
                value = 0.5 * (kinship[mother, earlier] + kinship[father, earlier])
                """Recursed from the later person's two recorded parents."""
            else:
                value = 0.0
                """Kept unrelated synthetic founders independent of earlier rows."""
            kinship[person, earlier] = value
            """Filled the lower-triangle relationship from the recursion."""

            kinship[earlier, person] = value
            """Mirrored the relationship into the symmetric upper triangle."""
        """Filled one symmetric row and column from the ordered parent records."""
    return 2.0 * kinship


def six_column_design(component_sizes: Sequence[int]) -> npt.NDArray[np.float64]:
    """Build intercept, age, age squared, sex, and both age-by-sex products.

    Args:
        component_sizes: Positive independent-family sizes in row order.

    Returns:
        Deterministic full fixed-effect design matching the common SOLAR formula.
    """
    people: int = sum(component_sizes)
    """Counted design rows across independent synthetic components."""

    design: npt.NDArray[np.float64] = np.ones((people, 6), dtype=np.float64)
    """Allocated the intercept and five covariate columns."""

    offset: int = 0
    """Tracked the first global row of each synthetic component."""

    for size in component_sizes:
        if size < 1:
            raise ValueError("COVERAGE_COMPONENT_SIZE_INVALID")
        oldest_end: int = max(1, round(size / 7))
        """Reserved approximately one seventh of a family for the oldest generation."""

        middle_end: int = max(oldest_end + 1, round(4 * size / 7))
        """Reserved the next generation while leaving younger rows where possible."""

        middle_end = min(middle_end, size)
        """Kept the generation split inside very small components."""

        for within in range(size):
            decade: float = (
                7.0 if within < oldest_end else 4.5 if within < middle_end else 2.0
            )
            """Assigned an age generation from synthetic within-component position."""

            row: int = offset + within
            """Located the synthetic person in the complete design."""

            age: float = (decade - 4.5) / 2.0 + ((row % 7) - 3.0) / 10.0
            """Standardised age and added the Rust campaign's deterministic jitter."""

            sex: float = float(row % 2 == 0)
            """Alternated the binary sex covariate independently of outcomes."""

            design[row, 1:] = (age, age * age, sex, age * sex, age * age * sex)
            """Filled age, sex, and interaction values for this synthetic row."""
        """Filled every fixed-effect row for one synthetic component."""

        offset += size
        """Advanced to the next independent component's row block."""
    return design


def standard_envelope(families: int) -> SimulationEnvelope:
    """Build repeated copies of the Rust campaign's extended family.

    Args:
        families: Positive number of unrelated 14-person families.

    Returns:
        Public-Python coverage design with a block-diagonal relationship matrix.
    """
    if families < 1:
        raise ValueError("COVERAGE_FAMILIES_INVALID")

    family_relationship: npt.NDArray[np.float64] = pedigree_relationship(
        EXTENDED_FAMILY_PARENTS
    )
    """Built the textbook three-generation relationship block once."""

    family_size: int = family_relationship.shape[0]
    """Read the fixed number of people in the extended-family template."""

    component_sizes: tuple[int, ...] = (family_size,) * families
    """Repeated the independently generated family size exactly as requested."""

    people: int = families * family_size
    """Counted complete rows in the repeated-family design."""

    relationship: npt.NDArray[np.float64] = np.zeros((people, people), dtype=np.float64)
    """Allocated the complete block-diagonal relationship matrix."""

    for family in range(families):
        start: int = family * family_size
        """Located one unrelated family's first row."""

        stop: int = start + family_size
        """Located the exclusive end of that family's square block."""

        relationship[start:stop, start:stop] = family_relationship
        """Placed one extended-family relationship block on the diagonal."""
    """Copied the same reviewed synthetic pedigree into unrelated components."""

    nonzero_pairs: int = int(np.count_nonzero(np.tril(relationship, k=-1)))
    """Counted nonzero off-diagonal relationships for auditable design facts."""

    return SimulationEnvelope(
        relationship=relationship,
        fixed_effects=six_column_design(component_sizes),
        component_sizes=component_sizes,
        nonzero_relationship_pairs=nonzero_pairs,
    )


def fixture_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 of one exact design-fixture file.

    Args:
        path: Reviewed values-free JSON fixture.

    Returns:
        Digest of the exact bytes used to select the target design.
    """
    payload: bytes = path.read_bytes()
    """Read the exact fixture bytes without normalising their JSON form."""

    return hashlib.sha256(payload).hexdigest()


def structure_matched_pedigree(
    size: int,
    generator: np.random.Generator,
    *,
    sibling_mean: float,
    marry_in_probability: float,
    maximum_generations: int,
) -> tuple[tuple[int | None, int | None], ...]:
    """Grow one founder-ordered pedigree to an exact component size.

    Args:
        size: Exact number of synthetic people in the component.
        generator: Shared deterministic stream across all histogram components.
        sibling_mean: Mean sibship size in the predecessor generator.
        marry_in_probability: Probability that a child forms the next couple.
        maximum_generations: Maximum descendant depth before sideways growth.

    Returns:
        Parent-index pairs with every parent preceding its child.

    Notes:
        This is a local identifier-free adaptation of Astrarium's historical
        participant-free ``gobsped`` generator. It matches aggregate structure;
        it does not reconstruct any observed pedigree.
    """
    if size < 1:
        raise ValueError("TARGET_FIXTURE_COMPONENT_SIZE_INVALID")

    parents: list[tuple[int | None, int | None]] = [(None, None)]
    """Started the component with one unrelated synthetic founder."""

    if size == 1:
        return tuple(parents)

    parents.append((None, None))
    """Completed the first-generation founder couple."""

    generations: list[int] = [1, 1]
    """Tracked generation depth parallel to the identifier-free parent records."""

    couples: list[tuple[int, int, int]] = [(0, 1, 1)]
    """Queued founder-ordered sire, dam, and generation coordinates."""

    unmated: list[tuple[int, int]] = []
    """Retained descendants available for later sideways growth."""

    while len(parents) < size:
        if not couples:
            generation: int
            """Reserved the branch depth selected from either growth pool."""

            if not unmated:
                candidates: list[int] = [
                    person
                    for person, generation in enumerate(generations)
                    if generation < maximum_generations
                ]
                """Found existing people able to form a shallower new branch."""

                if not candidates:
                    raise ValueError("TARGET_FIXTURE_PEDIGREE_GROWTH_EXHAUSTED")
                person: int = candidates[int(generator.integers(len(candidates)))]
                """Selected a deterministic eligible person from the shared stream."""

                generation = generations[person]
                """Retained that person's generation for the new couple."""
            else:
                selection: int = int(generator.integers(len(unmated)))
                """Selected one queued unmated descendant deterministically."""

                person, generation = unmated.pop(selection)
                """Removed the selected descendant from the available pool."""

            spouse: int = len(parents)
            """Assigned the next synthetic row to an unrelated married-in founder."""

            parents.append((None, None))
            generations.append(generation)
            """Added the married-in founder at the selected branch depth."""

            couples.append(
                (person, spouse, generation)
                if generator.random() < 0.5
                else (spouse, person, generation)
            )
            """Randomised parental role without changing kinship."""
            continue

        sire, dam, generation = couples.pop(0)
        """Took the oldest queued couple to preserve historical generator order."""

        child_count: int = 1 + int(generator.poisson(max(0.0, sibling_mean - 1.0)))
        """Drew the historical shifted-Poisson sibship size."""

        children: list[int] = []
        """Collected children eligible to form the following generation."""

        for _ in range(child_count):
            if len(parents) >= size:
                break
            child: int = len(parents)
            """Assigned the next founder-ordered row to this couple's child."""

            parents.append((sire, dam))
            generations.append(generation + 1)
            children.append(child)
        """Added the complete or size-truncated sibship."""

        for child in children:
            child_generation: int = generation + 1
            """Named the child's depth for marriage and stopping decisions."""

            if child_generation >= maximum_generations:
                continue
            if generator.random() < marry_in_probability:
                if len(parents) >= size:
                    unmated.append((child, child_generation))
                    continue
                spouse = len(parents)
                """Assigned the next row to this child's unrelated spouse."""

                parents.append((None, None))
                generations.append(child_generation)
                """Added the married-in founder at the child's generation."""

                couples.append(
                    (child, spouse, child_generation)
                    if generator.random() < 0.5
                    else (spouse, child, child_generation)
                )
                """Queued the new couple with a deterministic parental orientation."""
            else:
                unmated.append((child, child_generation))
                """Reserved the descendant for possible later sideways growth."""
        """Routed every eligible child to a couple or the unmated pool."""
    return tuple(parents)


def historical_covariate_design(people: int, seed: int) -> npt.NDArray[np.float64]:
    """Build the six-column fixed design used by the historical target gate.

    Args:
        people: Number of synthetic rows.
        seed: Dedicated covariate stream, independent of simulated outcomes.

    Returns:
        Intercept, standardised age, age squared, sex, and both interactions.
    """
    generator: np.random.Generator = np.random.Generator(np.random.PCG64(seed))
    """Recreated the historical fixed-design random stream exactly."""

    age_years: npt.NDArray[np.float64] = generator.normal(45.0, 15.0, people)
    """Drew participant-free age values solely to establish design geometry."""

    age: npt.NDArray[np.float64] = (age_years - age_years.mean()) / age_years.std()
    """Standardised age within the complete synthetic target design."""

    sex: npt.NDArray[np.float64] = generator.integers(0, 2, people).astype(float)
    """Drew the fixed binary covariate from the same dedicated stream."""

    return np.column_stack(
        [np.ones(people), age, age * age, sex, age * sex, age * age * sex]
    )


def target_envelope(
    path: Path,
    *,
    expected_sha256: str,
    families: int,
) -> SimulationEnvelope:
    """Verify a reviewed aggregate fixture and build its synthetic pedigree.

    Args:
        path: Values-free aggregate fixture selected by the release command.
        expected_sha256: Exact reviewed fixture digest pinned outside the file.
        families: Explicit command-line component count, required to match the fixture.

    Returns:
        Structure-matched participant-free relationship and fixed-effect design.

    Raises:
        ValueError: If fixture identity, aggregates, generator output, or fixed
            acceptance facts disagree with the reviewed contract.
    """
    selected_sha256: str = fixture_sha256(path)
    """Bound the generated design to the exact selected fixture bytes."""

    if selected_sha256 != expected_sha256:
        raise ValueError("TARGET_FIXTURE_SHA256_MISMATCH")

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    """Parsed the values-free aggregate only after exact-byte verification."""

    if not isinstance(loaded, dict):
        raise ValueError("TARGET_FIXTURE_ROOT_INVALID")
    fixture: dict[str, object] = loaded
    """Narrowed the parsed fixture to its required object shape."""

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
        "one_trait_gaussian_heritability_target_design",
        True,
        True,
        "reviewed predecessor aggregate receipt",
    ):
        raise ValueError("TARGET_FIXTURE_IDENTITY_INVALID")

    histogram_value: object = fixture.get("component_size_histogram")
    """Selected the exact reviewed relationship-component histogram."""

    if not isinstance(histogram_value, dict):
        raise ValueError("TARGET_FIXTURE_HISTOGRAM_INVALID")

    try:
        histogram: dict[int, int] = {
            int(size): int(count) for size, count in histogram_value.items()
        }
        """Converted JSON object keys to positive component-size integers."""
    except (TypeError, ValueError) as error:
        raise ValueError("TARGET_FIXTURE_HISTOGRAM_INVALID") from error

    if any(size < 1 or count < 1 for size, count in histogram.items()):
        raise ValueError("TARGET_FIXTURE_HISTOGRAM_INVALID")

    component_sizes: tuple[int, ...] = tuple(
        size for size, count in sorted(histogram.items()) for _ in range(count)
    )
    """Expanded the reviewed histogram in deterministic ascending-size order."""

    analysis_n: object = fixture.get("analysis_n")
    """Read the reviewed one-trait sample-size aggregate."""

    component_count: object = fixture.get("component_count")
    """Read the reviewed relationship-component count."""

    largest_component: object = fixture.get("largest_component")
    """Read the reviewed maximum component size."""

    if (
        component_count != len(component_sizes)
        or families != component_count
        or analysis_n != sum(component_sizes)
        or largest_component != max(component_sizes)
    ):
        raise ValueError("TARGET_FIXTURE_COMPONENT_AGGREGATES_INVALID")

    synthetic_value: object = fixture.get("synthetic_pedigree")
    """Selected the fixed historical generator settings and structural receipt."""

    if not isinstance(synthetic_value, dict):
        raise ValueError("TARGET_FIXTURE_GENERATOR_INVALID")
    synthetic: dict[str, object] = synthetic_value
    """Narrowed the synthetic-generator configuration to an object."""

    generator_identity: tuple[object, ...] = (
        synthetic.get("generator"),
        synthetic.get("seed"),
        synthetic.get("sibling_mean"),
        synthetic.get("marry_in_probability"),
        synthetic.get("maximum_generations"),
        synthetic.get("generated_nonzero_relationship_pairs"),
        synthetic.get("maximum_relative_nonzero_pair_discrepancy"),
    )
    """Collected every pre-reviewed generator and structural-acceptance setting."""

    if generator_identity != (
        "founder_order_structure_matched",
        7,
        3.0,
        0.35,
        5,
        27_691,
        0.005,
    ):
        raise ValueError("TARGET_FIXTURE_GENERATOR_INVALID")

    acceptance_value: object = fixture.get("coverage_acceptance")
    """Selected the pre-written asymmetric coverage decision from the fixture."""

    expected_acceptance: dict[str, object] = {
        "nominal": 0.95,
        "compatible_band": [0.94, 0.96],
        "fail_when": "clopper_pearson_upper_below_0.94",
        "conservative_when": "clopper_pearson_lower_above_0.96",
        "historically_conservative_truths": [0.05, 0.07],
    }
    """Restated the executable acceptance constants independently of fixture edits."""

    if acceptance_value != expected_acceptance:
        raise ValueError("TARGET_FIXTURE_ACCEPTANCE_INVALID")

    generator: np.random.Generator = np.random.default_rng(int(synthetic["seed"]))
    """Created the historical stream shared across components in fixed order."""

    blocks: list[npt.NDArray[np.float64]] = []
    """Collected independently generated pedigree relationship components."""

    for size in component_sizes:
        parents: tuple[tuple[int | None, int | None], ...] = structure_matched_pedigree(
            size,
            generator,
            sibling_mean=float(synthetic["sibling_mean"]),
            marry_in_probability=float(synthetic["marry_in_probability"]),
            maximum_generations=int(synthetic["maximum_generations"]),
        )
        """Generated one exact-sized founder-ordered synthetic pedigree."""

        blocks.append(pedigree_relationship(parents))
    """Converted every synthetic pedigree to its standard twice-kinship block."""

    people: int = sum(component_sizes)
    """Confirmed the dense matrix dimension from reviewed component sizes."""

    relationship: npt.NDArray[np.float64] = np.zeros((people, people), dtype=np.float64)
    """Allocated the participant-free target relationship matrix."""

    offset: int = 0
    """Tracked block placement without retaining synthetic identifiers."""

    for block in blocks:
        stop: int = offset + block.shape[0]
        """Located the exclusive end of the current relationship component."""

        relationship[offset:stop, offset:stop] = block
        """Placed one generated pedigree block on the relationship diagonal."""

        offset = stop
        """Advanced the row cursor to the next independent pedigree block."""
    """Materialised the complete structure-matched block-diagonal relationship."""

    generated_pairs: int = int(np.count_nonzero(np.tril(relationship, k=-1)))
    """Measured synthetic nonzero relationships independently of fixture claims."""

    if generated_pairs != synthetic["generated_nonzero_relationship_pairs"]:
        raise ValueError("TARGET_FIXTURE_GENERATED_PAIR_COUNT_MISMATCH")

    observed_pairs_value: object = fixture.get("observed_nonzero_relationship_pairs")
    """Read the values-free predecessor aggregate used only for structure matching."""

    if not isinstance(observed_pairs_value, int) or observed_pairs_value < 1:
        raise ValueError("TARGET_FIXTURE_OBSERVED_PAIR_COUNT_INVALID")
    observed_pairs: int = observed_pairs_value
    """Narrowed the reviewed aggregate to a positive pair count."""

    discrepancy: float = abs(generated_pairs - observed_pairs) / observed_pairs
    """Computed the pre-specified relative structural mismatch."""

    recorded_discrepancy: object = synthetic.get("relative_nonzero_pair_discrepancy")
    """Read the independently reviewable discrepancy recorded in the fixture."""

    if not isinstance(recorded_discrepancy, int | float) or not np.isclose(
        discrepancy, float(recorded_discrepancy), rtol=0.0, atol=1e-15
    ):
        raise ValueError("TARGET_FIXTURE_PAIR_DISCREPANCY_INVALID")
    if discrepancy > float(synthetic["maximum_relative_nonzero_pair_discrepancy"]):
        raise ValueError("TARGET_FIXTURE_PAIR_DISCREPANCY_EXCEEDED")

    fixed_value: object = fixture.get("fixed_effects")
    """Selected the participant-free historical fixed-design settings."""

    if not isinstance(fixed_value, dict) or fixed_value.get("covariate_seed") != 1:
        raise ValueError("TARGET_FIXTURE_FIXED_EFFECTS_INVALID")

    return SimulationEnvelope(
        relationship=relationship,
        fixed_effects=historical_covariate_design(people, 1),
        component_sizes=component_sizes,
        nonzero_relationship_pairs=generated_pairs,
        fixture_sha256=selected_sha256,
        observed_nonzero_relationship_pairs=observed_pairs,
        relative_pair_count_discrepancy=discrepancy,
    )


def covariance_factors(
    envelope: SimulationEnvelope, truth: float
) -> tuple[npt.NDArray[np.float64], ...]:
    """Factor generating covariance separately inside each family component.

    Args:
        envelope: Participant-free relationship and family-block design.
        truth: Generating heritability.

    Returns:
        Lower Cholesky factors in component row order.
    """
    if not 0.0 <= truth <= 1.0:
        raise ValueError("COVERAGE_TRUTH_INVALID")

    factors: list[npt.NDArray[np.float64]] = []
    """Collected reusable covariance factors for one scientific cell."""

    offset: int = 0
    """Tracked the first row of each independent relationship component."""

    for size in envelope.component_sizes:
        stop: int = offset + size
        """Located the exclusive end of the component's square matrix block."""

        relationship: npt.NDArray[np.float64] = envelope.relationship[
            offset:stop, offset:stop
        ]
        """Selected one synthetic pedigree's additive relationship block."""

        covariance: npt.NDArray[np.float64] = truth * relationship
        """Scaled the additive covariance by the generating heritability."""

        covariance = covariance + (1.0 - truth) * np.eye(size)
        """Added independent residual variance to preserve unit marginal scale."""

        factors.append(np.linalg.cholesky(covariance))
        offset = stop
        """Factored the block once and advanced to the next family."""
    return tuple(factors)


def run_cell(job: CellJob) -> dict[str, object]:
    """Run and unconditionally score one real public-Python coverage cell.

    Args:
        job: Exact design, truth, replicate, and deterministic-seed coordinates.

    Returns:
        Complete aggregate evidence, including all failure and profile buckets.
    """
    started: float = time.perf_counter()
    """Started cell timing outside preparation and simulation setup."""

    if job.replicates < 1 or job.families < 1 or job.truth_index < 0:
        raise ValueError("COVERAGE_CELL_JOB_INVALID")
    if (job.fixture_path is None) != (job.fixture_sha256 is None):
        raise ValueError("COVERAGE_TARGET_FIXTURE_IDENTITY_INCOMPLETE")

    envelope: SimulationEnvelope = (
        standard_envelope(job.families)
        if job.fixture_path is None
        else target_envelope(
            job.fixture_path,
            expected_sha256=str(job.fixture_sha256),
            families=job.families,
        )
    )
    """Built either the standard Rust-port roster or reviewed target envelope."""

    model: asterism.PreparedModel = asterism.prepare(
        envelope.fixed_effects, envelope.relationship
    )
    """Prepared the model through the documented public Python interface only."""

    factors: tuple[npt.NDArray[np.float64], ...] = covariance_factors(
        envelope, job.truth
    )
    """Factored the known generating covariance once per family component."""

    fixed_mean: npt.NDArray[np.float64] = envelope.fixed_effects @ TRUE_BETA
    """Applied the fixed generating effects independently of random outcomes."""

    seed: int = BASE_SEED + job.truth_index * 1_000_003
    """Assigned a disjoint reproducible stream to the explicit truth-grid cell."""

    generator: np.random.Generator = np.random.default_rng(seed)
    """Created the sole outcome stream owned by this scientific cell."""

    covered: int = 0
    """Counted mixture-aware interval membership among every attempted replicate."""

    refused: int = 0
    """Counted public fit refusals as scored coverage misses."""

    refusal_codes: dict[str, int] = {}
    """Counted stable refusal messages without dropping their replicates."""

    nonconverged: int = 0
    """Counted nonconverged free fits as scored coverage misses."""

    profile_failed_replicates: int = 0
    """Counted converged fits carrying any failed constrained profile evaluation."""

    profile_failure_evaluations: int = 0
    """Summed failed constrained evaluations across all attempted profiles."""

    complete_profiles: int = 0
    """Counted converged free fits with no failed profile evaluations."""

    at_zero: int = 0
    """Counted converged estimates on the lower heritability boundary."""

    at_one: int = 0
    """Counted converged estimates on the upper heritability boundary."""

    naive_covered: int = 0
    """Counted endpoint-only containment for transparent boundary comparison."""

    widths: list[float] = []
    """Collected widths only from complete converged profiles."""

    for _ in range(job.replicates):
        response: npt.NDArray[np.float64] = fixed_mean.copy()
        """Started one response at its deterministic fixed-effect mean."""

        offset: int = 0
        """Tracked component slices while adding correlated Gaussian draws."""

        for factor in factors:
            stop: int = offset + factor.shape[0]
            """Located the response slice governed by this covariance factor."""

            response[offset:stop] += factor @ generator.standard_normal(factor.shape[0])
            """Added this family's correlated Gaussian outcome draw."""

            offset = stop
            """Advanced to the next independent family's response slice."""
        """Generated every row from the known block-diagonal covariance."""

        try:
            record: dict[str, object] = model.fit(response, estimator="reml")
            """Fitted and profiled through ``PreparedModel.fit`` only."""
        except ValueError as error:
            refused += 1
            """Counted the refused fit in the unconditional denominator."""

            code: str = str(error)
            """Retained the public refusal code as scientific evidence."""

            refusal_codes[code] = refusal_codes.get(code, 0) + 1
            """Incremented the exact public refusal-code frequency."""
            continue

        if record.get("converged") is not True:
            nonconverged += 1
            """Counted the failed free fit as an unconditional coverage miss."""
            continue

        boundary: object = record.get("boundary")
        """Read the public free-fit boundary classification."""

        at_zero += int(boundary == "lower")
        """Counted lower-bound free estimates among converged fits."""

        at_one += int(boundary == "upper")
        """Counted upper-bound free estimates among converged fits."""

        interval_value: object = record.get("interval")
        """Selected the public profile record for completeness accounting."""

        if not isinstance(interval_value, Mapping):
            profile_failed_replicates += 1
            """Counted a missing profile record as a failed-profile replicate."""

            profile_failure_evaluations += 1
            """Assigned one explicit failure to the malformed public profile."""
            continue

        failures_value: object = interval_value.get("profile_failures")
        """Read the exact number of failed constrained evaluations."""

        if not isinstance(failures_value, int) or failures_value != 0:
            profile_failed_replicates += 1
            """Counted the replicate once when any constrained fit failed."""

            profile_failure_evaluations += (
                failures_value
                if isinstance(failures_value, int) and failures_value > 0
                else 1
            )
            """Accumulated reported failures or one failure for malformed counts."""
            continue

        complete_profiles += 1
        """Counted the converged fit with a fully evaluated profile."""

        covered += int(covers(record, job.truth))
        """Applied mixture-aware containment to the unconditional hit count."""

        lower_value: object = interval_value.get("lower")
        """Read the complete profile's lower endpoint for width and naive scoring."""

        upper_value: object = interval_value.get("upper")
        """Read the complete profile's upper endpoint for width and naive scoring."""

        if isinstance(lower_value, int | float) and isinstance(
            upper_value, int | float
        ):
            lower: float = float(lower_value)
            """Normalised the lower endpoint for deterministic aggregation."""

            upper: float = float(upper_value)
            """Normalised the upper endpoint for deterministic aggregation."""

            widths.append(upper - lower)
            """Retained this complete profile's interval width."""

            naive_covered += int(lower <= job.truth <= upper)
            """Counted endpoint-only containment for transparent comparison."""
        """Retained width and endpoint-only coverage from valid numeric profiles."""
    """Scored every requested replicate into one mutually exclusive fit bucket."""

    decision: dict[str, object] = coverage_decision(covered, job.replicates, job.truth)
    """Applied the fixed asymmetric exact-binomial coverage rule."""

    median_width: float | None = float(np.median(widths)) if widths else None
    """Summarised interval precision only where a complete profile existed."""

    naive_boundary_coverage: float | None = (
        naive_covered / job.replicates if job.truth in (0.0, 1.0) else None
    )
    """Reported the superseded endpoint-only rule only at actual bounds."""

    record: dict[str, object] = {
        "truth": job.truth,
        "n": envelope.relationship.shape[0],
        "families": len(envelope.component_sizes),
        "largest_family": max(envelope.component_sizes),
        "nonzero_relationship_pairs": envelope.nonzero_relationship_pairs,
        "observed_nonzero_relationship_pairs": (
            envelope.observed_nonzero_relationship_pairs
        ),
        "relative_pair_count_discrepancy": (envelope.relative_pair_count_discrepancy),
        "fixture_sha256": envelope.fixture_sha256,
        "seed": seed,
        "estimator": "reml",
        "refused": refused,
        "refusal_codes": refusal_codes,
        "nonconverged": nonconverged,
        "profile_failed_replicates": profile_failed_replicates,
        "profile_failure_evaluations": profile_failure_evaluations,
        "complete_profiles": complete_profiles,
        "fraction_at_zero": at_zero / job.replicates,
        "fraction_at_one": at_one / job.replicates,
        "median_width": median_width,
        "naive_boundary_coverage": naive_boundary_coverage,
        "seconds": time.perf_counter() - started,
    }
    """Recorded design, failure, profile, boundary, and precision evidence."""

    record.update(decision)
    """Attached the independently recomputable coverage decision fields."""
    return record


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Read a fully explicit participant-free coverage command.

    Args:
        arguments: Argument vector excluding the executable name, or ``None``
            to read the process command line.

    Returns:
        Validated replicate, family, worker, truth, and output selections.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built the command contract without scientific defaults."""

    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument("--families", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--truths", required=True, type=float, nargs="+")
    parser.add_argument("--design-fixture", type=Path)
    parser.add_argument("--fixture-sha256")
    parser.add_argument(
        "--no-write",
        required=True,
        action="store_true",
        help="emit evidence to standard output without modifying the checkout",
    )
    parsed: argparse.Namespace = parser.parse_args(arguments)
    """Read every scientific campaign coordinate directly from argv."""

    if parsed.replicates < 1 or parsed.families < 1 or parsed.workers < 1:
        parser.error("--replicates, --families, and --workers must be positive")
    if len(set(parsed.truths)) != len(parsed.truths):
        parser.error("--truths must not contain duplicate cells")
    if any(not 0.0 <= truth <= 1.0 for truth in parsed.truths):
        parser.error("--truths must lie on the closed unit interval")
    if (parsed.design_fixture is None) != (parsed.fixture_sha256 is None):
        parser.error("--design-fixture and --fixture-sha256 must be supplied together")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the exact explicit grid and emit one values-free JSON decision record.

    Args:
        arguments: Argument vector excluding the executable name, or ``None``
            to read the process command line.

    Returns:
        Zero when no cell is anti-conservative, otherwise one.
    """
    parsed: argparse.Namespace = parse_arguments(arguments)
    """Fixed every scientific coordinate before generating any outcomes."""

    jobs: list[CellJob] = [
        CellJob(
            truth=truth,
            truth_index=index,
            replicates=parsed.replicates,
            families=parsed.families,
            fixture_path=parsed.design_fixture,
            fixture_sha256=parsed.fixture_sha256,
        )
        for index, truth in enumerate(parsed.truths)
    ]
    """Enumerated every requested truth cell with a scheduling-independent index."""

    workers_used: int = min(parsed.workers, len(jobs))
    """Avoided starting processes that could own no explicit scientific cell."""

    if workers_used == 1:
        cells: list[dict[str, object]] = [run_cell(job) for job in jobs]
        """Executed the same worker boundary directly for a single-process command."""
    else:
        with ProcessPoolExecutor(max_workers=workers_used) as pool:
            cells = list(pool.map(run_cell, jobs, chunksize=1))
            """Ran independent truth cells in processes without changing their seeds."""
        """Closed every worker before deciding or printing the complete evidence."""

    failures: list[str] = [
        f"h2={cell['truth']}: {cell['verdict']} coverage"
        for cell in cells
        if cell.get("passed") is not True
    ]
    """Collected only anti-conservative cell decisions as validity failures."""

    evidence: dict[str, object] = {
        "check": (
            "coverage_at_the_real_design_point"
            if parsed.design_fixture is not None
            else "coverage"
        ),
        "analysis_id": "one_trait_gaussian_heritability",
        "participant_free": True,
        "public_entry_points": ["asterism.prepare", "asterism.PreparedModel.fit"],
        "design": "reviewed_target_envelope"
        if parsed.design_fixture is not None
        else "repeated_extended_family",
        "fixture_sha256": parsed.fixture_sha256,
        "replicates_per_cell": parsed.replicates,
        "families": parsed.families,
        "workers": parsed.workers,
        "workers_used": workers_used,
        "truths": list(parsed.truths),
        "base_seed": BASE_SEED,
        "cell_seed_increment": 1_000_003,
        "every_replicate_scored": True,
        "boundary_membership": "contains_lower_bound_or_upper_bound",
        "coverage_uncertainty": "two_sided_95_percent_clopper_pearson",
        "acceptable_coverage_band": list(ACCEPTABLE_COVERAGE_BAND),
        "anti_conservative_rule": "clopper_pearson_upper_below_0.94",
        "conservative_rule": "clopper_pearson_lower_above_0.96_report_only",
        "historically_conservative_truths": sorted(HISTORICALLY_CONSERVATIVE_TRUTHS),
        "cells": cells,
        "passed": not failures,
        "failures": failures,
    }
    """Built enough values-free evidence to recompute every release decision."""

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if not failures else 1


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    """Return the exact two-sided 95% interval for a binomial proportion.

    Args:
        successes: Replicates whose profile interval covered the truth.
        trials: All attempted replicates, including failed fits and profiles.

    Returns:
        Exact lower and upper confidence limits.

    Raises:
        ValueError: If the counts do not describe a binomial experiment.
    """
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("COVERAGE_BINOMIAL_COUNTS_INVALID")

    lower: float = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, trials - successes + 1))
    )
    """Computed the exact lower Monte Carlo confidence limit."""

    upper: float = (
        1.0
        if successes == trials
        else float(beta.ppf(0.975, successes + 1, trials - successes))
    )
    """Computed the exact upper Monte Carlo confidence limit."""

    return lower, upper


def coverage_decision(successes: int, trials: int, truth: float) -> dict[str, object]:
    """Apply the fixed validity-first decision to one coverage cell.

    Args:
        successes: Replicates whose intervals covered the generating truth.
        trials: Every attempted replicate in the cell.
        truth: Generating heritability for the cell.

    Returns:
        Exact uncertainty, compatibility classification, and pass decision.

    Notes:
        A cell fails only when its exact upper confidence limit lies below the
        lower edge of ADR 0004's compatibility band. Extra coverage costs power
        rather than validity, so it is reported as conservative and the known
        0.05/0.07 regime receives an explicit historical label.
    """
    interval: tuple[float, float] = clopper_pearson(successes, trials)
    """Quantified Monte Carlo uncertainty without dropping failed replicates."""

    if interval[1] < ACCEPTABLE_COVERAGE_BAND[0]:
        verdict: str = "anti_conservative"
        """Rejected uncertainty lying wholly below the safe coverage band."""
    elif interval[0] > ACCEPTABLE_COVERAGE_BAND[1]:
        verdict = (
            "historically_conservative"
            if truth in HISTORICALLY_CONSERVATIVE_TRUTHS
            else "conservative"
        )
        """Named extra coverage without misclassifying it as invalid inference."""
    else:
        verdict = "compatible"
        """Accepted uncertainty overlapping the pre-written compatibility band."""

    return {
        "attempted": trials,
        "covered": successes,
        "coverage": successes / trials,
        "coverage_clopper_pearson": interval,
        "acceptable_coverage_band": ACCEPTABLE_COVERAGE_BAND,
        "verdict": verdict,
        "passed": verdict != "anti_conservative",
    }


def covers(record: Mapping[str, object], truth: float) -> bool:
    """Return whether one public fit record covers its generating truth.

    Args:
        record: Result returned by ``asterism.prepare(...).fit(...)``.
        truth: Generating heritability on the closed unit interval.

    Returns:
        True only for a converged, complete profile containing the truth under
        ADR 0004's boundary-membership rule.
    """
    if record.get("converged") is not True:
        return False

    interval_value: object = record.get("interval")
    """Selected the public profile record without assuming its shape."""

    if not isinstance(interval_value, Mapping):
        return False
    if interval_value.get("profile_failures") != 0:
        return False

    lower_value: object = interval_value.get("lower")
    """Read the lower endpoint before requiring a finite numeric value."""

    upper_value: object = interval_value.get("upper")
    """Read the upper endpoint before requiring a finite numeric value."""

    if not isinstance(lower_value, int | float) or not isinstance(
        upper_value, int | float
    ):
        return False

    lower: float = float(lower_value)
    """Normalised the numeric lower endpoint for containment comparisons."""

    upper: float = float(upper_value)
    """Normalised the numeric upper endpoint for containment comparisons."""

    if truth == 0.0:
        return interval_value.get("contains_lower_bound") is True
    if truth == 1.0:
        return interval_value.get("contains_upper_bound") is True
    return lower <= truth <= upper


if __name__ == "__main__":
    sys.exit(main())
