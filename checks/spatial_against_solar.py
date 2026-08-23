"""Bound native SOLAR's valid comparison with Asterism's component model.

SOLAR evaluates a loaded covariance matrix within pedigrees and silently drops
cross-pedigree entries. A block-diagonal spatial kernel is therefore a genuine
same-model external comparison. A dense kernel is also supplied to prove and
freeze the native truncation boundary: native SOLAR returns its within-family
answer while Asterism fits the genuinely dense model.

No-argument execution performs the qualified live comparison. Explicit refresh
records raw native outputs; verify recomputes both Asterism fits without SOLAR.
"""

from __future__ import annotations

import argparse
import gzip
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import asterism
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from against_r import extended_family, roster
from solar_reference_adapter import (
    CONTRACT_VERSION,
    SolarReferenceError,
    array_identity,
    bytes_identity,
    check_sex_survived,
    emit_reference_record,
    fixture_mapping,
    json_identity,
    load_fixture,
    parse_reference_arguments,
    record_list,
    require_fixture_inputs,
    require_solar_version,
    required_integer,
    required_number,
    run_solar,
    solar_tool_identity,
    support_source_identity,
)

CHECK_ID: str = "spatial_against_solar"
"""Matched the release-manifest check identifier and fixture filename."""

VARIANCE_TOLERANCE: float = 5e-6
"""Followed native SOLAR's seven-significant-figure parameter output."""

LOGLIK_TOLERANCE: float = 1e-4
"""Bound unexplained likelihood offset after restoring the density constant."""

SQUARE_KM: float = 10.0
"""Set the square containing independently generated spatial positions."""

LAMBDA_PER_KM: float = 0.6
"""Held kernel decay fixed so native SOLAR can represent it as a known matrix."""

FAMILIES: int = 25
"""Retained the historical replicated extended-pedigree count."""

CASES: tuple[tuple[tuple[float, float, float], int], ...] = (
    ((0.5, 0.3, 0.7), 301),
    ((0.2, 0.6, 0.9), 302),
)
"""Fixed component variances and seeds before external outputs were observed."""

SEX: tuple[int, ...] = (2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2)
"""Encoded sex consistently with each generated person's parental role."""

REFERENCE_ACCEPTANCE: dict[str, object] = {
    "maximum_same_model_variance_difference": VARIANCE_TOLERANCE,
    "maximum_unexplained_loglik_offset": LOGLIK_TOLERANCE,
    "maximum_solar_dense_vs_block_variance_difference": VARIANCE_TOLERANCE,
    "minimum_asterism_dense_vs_solar_variance_separation": VARIANCE_TOLERANCE,
    "solar_must_analyse_all_generated_people": True,
}
"""Prewrote same-model agreement and documented truncation-boundary rules."""

RUN: str = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait y
covariate age^1,2#sex
polygsd
matrix load matrix.csv.gz spa
parameter ssd = 0.4 lower 0 upper 100
omega = I*esd*esd + phi2*gsd*gsd + spa*ssd*ssd
outdir out
maximize
exit
"""
"""Defined the native three-component ML model over one loaded known matrix."""


@dataclass(frozen=True)
class SpatialCase:
    """Hold one deterministic participant-free component comparison case."""

    true_variances: tuple[float, float, float]
    """Stored generating additive, spatial and residual variances."""
    seed: int
    """Stored the fixed pseudo-random generator seed."""
    relationship: np.ndarray
    """Held the additive relationship component."""
    dense_kernel: np.ndarray
    """Held the genuinely cross-pedigree spatial component."""
    block_kernel: np.ndarray
    """Held the same kernel restricted within pedigrees."""
    design: np.ndarray
    """Held the six-column fixed-effect design."""
    outcome: np.ndarray
    """Held the generated continuous phenotype."""
    pedigree_csv: str
    """Held the exact generated native pedigree input."""
    phenotype_csv: str
    """Held the exact generated native phenotype input."""
    dense_matrix_csv: bytes
    """Held exact uncompressed native matrix content for the dense kernel."""
    block_matrix_csv: bytes
    """Held exact uncompressed native matrix content for the restricted kernel."""

    @property
    def people(self) -> int:
        """Return the generated sample size."""
        return int(self.outcome.shape[0])


def spatial_kernel(positions: np.ndarray) -> np.ndarray:
    """Return the fixed exponential kernel over generated two-dimensional positions.

    Args:
        positions: Person-by-coordinate array measured in kilometres.

    Returns:
        Dense positive-definite exponential covariance matrix.
    """
    difference: np.ndarray = positions[:, None, :] - positions[None, :, :]
    """Constructed every pairwise coordinate displacement."""

    distance: np.ndarray = np.sqrt((difference**2).sum(axis=2))
    """Reduced displacements to Euclidean distance in kilometres."""

    return np.exp(-LAMBDA_PER_KM * distance)


def within_families(kernel: np.ndarray, block: int) -> np.ndarray:
    """Remove every cross-pedigree covariance from a dense kernel.

    Args:
        kernel: Full person-by-person covariance matrix.
        block: Number of people in each replicated pedigree.

    Returns:
        Positive-definite block-diagonal restriction of the kernel.
    """
    restricted: np.ndarray = np.zeros_like(kernel)
    """Started with all cross-pedigree covariance removed."""

    for start in range(0, kernel.shape[0], block):
        stop: int = start + block
        """Located one complete pedigree block."""

        restricted[start:stop, start:stop] = kernel[start:stop, start:stop]
        """Retained the dense kernel only within this pedigree."""
    """Copied every positive-definite principal family block."""

    return restricted


def matrix_csv(
    identifiers: list[str], family_labels: list[str], matrix: np.ndarray
) -> bytes:
    """Render one known covariance matrix in native SOLAR lower-triangle format.

    Args:
        identifiers: Person identifiers in matrix order.
        family_labels: Pedigree labels in matrix order.
        matrix: Symmetric covariance matrix to load.

    Returns:
        Exact UTF-8 CSV bytes before deterministic gzip compression.
    """
    rows: list[str] = ["id1,id2,famid1,famid2,matrix1"]
    """Started the native matrix table with its required five columns."""

    for first in range(len(identifiers)):
        for second in range(first + 1):
            rows.append(
                f"{identifiers[first]},{identifiers[second]},"
                f"{family_labels[first]},{family_labels[second]},"
                f"{matrix[first, second]:.12f}"
            )
    """Rendered the complete lower triangle including every diagonal."""

    return ("\n".join(rows) + "\n").encode("utf-8")


def generate_case(true_variances: tuple[float, float, float], seed: int) -> SpatialCase:
    """Generate one component-model problem and exact native inputs.

    Args:
        true_variances: Additive, spatial and residual generating variances.
        seed: Fixed pseudo-random generator seed.

    Returns:
        Generated arrays and exact native-SOLAR text and matrix content.
    """
    family: list[tuple[int | None, int | None]] = extended_family()
    """Loaded the established extended-family pedigree structure."""

    block: int = len(family)
    """Counted people per replicated pedigree."""

    relationship: np.ndarray = roster(FAMILIES)
    """Built the block-diagonal additive relationship component."""

    people: int = relationship.shape[0]
    """Counted all generated individuals."""

    identifiers: list[str] = []
    """Started person identifiers in shared matrix order."""

    family_labels: list[str] = []
    """Started pedigree labels in shared matrix order."""

    pedigree_rows: list[str] = ["FAMID,ID,FA,MO,SEX"]
    """Started the generated native pedigree table."""

    for family_index in range(FAMILIES):
        for person_index, (mother, father) in enumerate(family):
            father_id: str = (
                "0" if father is None else f"F{family_index:03d}_{father:02d}"
            )
            """Encoded a founder sentinel or generated father identifier."""

            mother_id: str = (
                "0" if mother is None else f"F{family_index:03d}_{mother:02d}"
            )
            """Encoded a founder sentinel or generated mother identifier."""

            identifier: str = f"F{family_index:03d}_{person_index:02d}"
            """Built the stable generated person identifier."""

            family_label: str = f"F{family_index:03d}"
            """Built the stable replicated-pedigree label."""

            pedigree_rows.append(
                f"{family_label},{identifier},{father_id},{mother_id},{SEX[person_index]}"
            )
            identifiers.append(identifier)
            family_labels.append(family_label)
    """Rendered all pedigrees and recorded their shared person order."""

    positions: np.ndarray = np.random.default_rng(seed + 1).uniform(
        0.0, SQUARE_KM, size=(people, 2)
    )
    """Placed people independently of pedigree so space is not a family indicator."""

    dense_kernel: np.ndarray = spatial_kernel(positions)
    """Built the genuinely cross-pedigree spatial covariance."""

    block_kernel: np.ndarray = within_families(dense_kernel, block)
    """Restricted the same kernel to covariance native SOLAR can represent."""

    age: np.ndarray = np.zeros(people)
    """Allocated the deterministic standardised age covariate."""

    male: np.ndarray = np.zeros(people)
    """Allocated the deterministic binary sex covariate."""

    for family_index in range(FAMILIES):
        for person_index in range(block):
            row: int = family_index * block + person_index
            """Located this person in the shared array order."""

            decade: float = (
                7.0 if person_index < 2 else (4.5 if person_index < 8 else 2.0)
            )
            """Assigned an age level by pedigree generation with fixed variation."""

            age[row] = (decade - 4.5) / 2.0 + ((row % 7) - 3.0) / 10.0
            """Stored this person's standardised deterministic age."""

            male[row] = 1.0 if SEX[person_index] == 1 else 0.0
            """Stored the role-consistent numerical sex covariate."""
    """Filled all covariates in native person order."""

    design: np.ndarray = np.column_stack(
        [np.ones(people), age, age**2, male, age * male, age**2 * male]
    )
    """Built the same six-column fixed-effect span used by native SOLAR."""

    coefficients: np.ndarray = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])
    """Fixed non-zero covariate effects so design disagreement cannot hide."""

    genetic_variance: float = true_variances[0]
    """Read the generating additive component variance."""

    spatial_variance: float = true_variances[1]
    """Read the generating within-family spatial component variance."""

    residual_variance: float = true_variances[2]
    """Read the generating independent residual variance."""

    covariance: np.ndarray = (
        genetic_variance * relationship
        + spatial_variance * block_kernel
        + residual_variance * np.eye(people)
    )
    """Constructed the participant-free generating covariance."""

    generator: np.random.Generator = np.random.default_rng(seed)
    """Created the fixed participant-free random generator."""

    outcome: np.ndarray = design @ coefficients + np.linalg.cholesky(
        covariance
    ) @ generator.standard_normal(people)
    """Generated the continuous phenotype under the restricted spatial model."""

    phenotype_rows: list[str] = ["ID,FAMID,age,sex,y"]
    """Started the generated native phenotype table."""

    for family_index in range(FAMILIES):
        for person_index in range(block):
            row = family_index * block + person_index
            """Located this person in the generated covariate arrays."""

            phenotype_rows.append(
                f"F{family_index:03d}_{person_index:02d},F{family_index:03d},"
                f"{age[row]:.12f},{SEX[person_index]},{outcome[row]:.12f}"
            )
    """Rendered every generated phenotype in native person order."""

    pedigree_csv: str = "\n".join(pedigree_rows) + "\n"
    """Finalised the exact generated native pedigree text."""

    phenotype_csv: str = "\n".join(phenotype_rows) + "\n"
    """Finalised the exact generated native phenotype text."""

    dense_matrix_csv: bytes = matrix_csv(identifiers, family_labels, dense_kernel)
    """Rendered exact native matrix content for the dense spatial kernel."""

    block_matrix_csv: bytes = matrix_csv(identifiers, family_labels, block_kernel)
    """Rendered exact native matrix content for the within-pedigree kernel."""

    return SpatialCase(
        true_variances,
        seed,
        relationship,
        dense_kernel,
        block_kernel,
        design,
        outcome,
        pedigree_csv,
        phenotype_csv,
        dense_matrix_csv,
        block_matrix_csv,
    )


def generated_cases() -> list[SpatialCase]:
    """Return all fixed participant-free component cases.

    Returns:
        Cases in stable fixture order.
    """
    return [generate_case(variances, seed) for variances, seed in CASES]


def reference_input_identities(cases: list[SpatialCase]) -> list[dict[str, str]]:
    """Identify all generated component inputs and adapter mechanics.

    Args:
        cases: Fixed participant-free component cases.

    Returns:
        Stable named SHA-256 records.
    """
    identities: list[dict[str, str]] = [
        support_source_identity(),
        json_identity(
            "design:spatial_against_solar",
            {
                "families": FAMILIES,
                "cases": CASES,
                "sex": SEX,
                "square_km": SQUARE_KM,
                "lambda_per_km": LAMBDA_PER_KM,
            },
        ),
    ]
    """Bound the shared mechanics and all scalar design choices first."""

    for case in cases:
        truth: str = "/".join(f"{value:g}" for value in case.true_variances)
        """Built a stable descriptive truth label for this case."""

        label: str = f"variances={truth},seed={case.seed}"
        """Combined truth and seed into a unique generated-input prefix."""

        identities.extend(
            [
                array_identity(f"relationship:{label}", case.relationship),
                array_identity(f"dense_kernel:{label}", case.dense_kernel),
                array_identity(f"block_kernel:{label}", case.block_kernel),
                array_identity(f"design:{label}", case.design),
                array_identity(f"outcome:{label}", case.outcome),
                json_identity(f"pedigree_csv:{label}", case.pedigree_csv),
                json_identity(f"phenotype_csv:{label}", case.phenotype_csv),
                bytes_identity(f"dense_matrix_csv:{label}", case.dense_matrix_csv),
                bytes_identity(f"block_matrix_csv:{label}", case.block_matrix_csv),
            ]
        )
    """Hashed all arrays and exact native text and matrix inputs for both cases."""

    return identities


def fit_asterism_components(case: SpatialCase, kernel: np.ndarray) -> dict[str, object]:
    """Fit one known-kernel component model through Asterism's public interface.

    Args:
        case: Fixed participant-free component case.
        kernel: Dense or within-pedigree known covariance component.

    Returns:
        Direct variance estimates, likelihood and optimiser diagnostics.
    """
    model: asterism.ComponentModel = asterism.ComponentModel(
        [case.relationship, kernel], case.design
    )
    """Built the documented two-known-component model."""

    fit: dict[str, object] = model.fit(case.outcome, reml=False)
    """Fitted maximum likelihood through the public component interface."""

    variances: object = fit["variances"]
    """Read the three reported component variances."""

    if not isinstance(variances, list | tuple | np.ndarray) or len(variances) != 3:
        raise SolarReferenceError("Asterism returned no three-component variances")
    return {
        "genetic": float(variances[0]),
        "spatial": float(variances[1]),
        "residual": float(variances[2]),
        "loglik": float(fit["loglik"]),
        "scaled_gradient": float(fit["scaled_gradient"]),
        "converged": bool(fit["converged"]),
    }


def fit_asterism_case(case: SpatialCase) -> dict[str, object]:
    """Fit both the same-model and genuinely dense Asterism component cases.

    Args:
        case: Fixed participant-free component case.

    Returns:
        Case identity and both public Asterism fits.
    """
    return {
        "true_variances": list(case.true_variances),
        "seed": case.seed,
        "people": case.people,
        "within": fit_asterism_components(case, case.block_kernel),
        "dense": fit_asterism_components(case, case.dense_kernel),
    }


def write_solar_inputs(
    directory: Path, case: SpatialCase, matrix_content: bytes
) -> None:
    """Write one generated component case into a temporary SOLAR directory.

    Args:
        directory: Fresh native-SOLAR working directory.
        case: Generated case whose exact inputs are bound by fixture hashes.
        matrix_content: Exact uncompressed lower-triangle covariance CSV.
    """
    (directory / "ped.csv").write_text(case.pedigree_csv, encoding="utf-8")
    (directory / "phen.csv").write_text(case.phenotype_csv, encoding="utf-8")
    (directory / "matrix.csv.gz").write_bytes(gzip.compress(matrix_content, mtime=0))
    (directory / "run.tcl").write_text(RUN, encoding="utf-8")
    """Wrote deterministic participant-free inputs and a reproducible gzip file."""


def fit_solar_components(case: SpatialCase, matrix_content: bytes) -> dict[str, object]:
    """Fit one generated known-matrix component model with native SOLAR.

    Args:
        case: Fixed participant-free component case.
        matrix_content: Exact dense or block-diagonal covariance CSV.

    Returns:
        Raw reported native variances, likelihood and analysis counts.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory)
        """Isolated all generated native inputs and outputs outside the workspace."""

        write_solar_inputs(directory, case, matrix_content)
        completed: subprocess.CompletedProcess[str] = run_solar(directory)
        """Ran the qualified executable on the fixed component command."""

        report_path: Path = directory / "out" / "solar.out"
        """Selected native SOLAR's completed optimisation report."""

        if not report_path.is_file():
            raise SolarReferenceError(
                "SOLAR produced no component result: "
                + (completed.stdout + completed.stderr)[-2000:]
            )
        report: str = report_path.read_text(encoding="utf-8")
        """Read the raw native optimisation report."""

        iterations: int = required_integer(
            report, r"Iterations\s*=\s*([0-9]+)", "iteration count"
        )
        """Read the native optimiser iteration count."""

        if iterations < 1:
            raise SolarReferenceError("SOLAR did not complete a maximisation")
        check_sex_survived(directory)
        genetic_deviation: float = required_number(
            report,
            r"^\s+gsd\s+(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)\s",
            "gsd",
        )
        """Read native SOLAR's additive standard-deviation parameter."""

        spatial_deviation: float = required_number(
            report,
            r"^\s+ssd\s+(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)\s",
            "ssd",
        )
        """Read native SOLAR's known-matrix standard-deviation parameter."""

        residual_deviation: float = required_number(
            report,
            r"^\s+esd\s+(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)\s",
            "esd",
        )
        """Read native SOLAR's residual standard-deviation parameter."""

        result: dict[str, object] = {
            "genetic": genetic_deviation**2,
            "spatial": spatial_deviation**2,
            "residual": residual_deviation**2,
            "loglik": required_number(
                report,
                r"Loglikelihood\s*=\s*(-?[0-9.eE+-]+)",
                "log-likelihood",
            ),
            "iterations": iterations,
            "people": required_integer(
                report,
                r"sample size including probands used is\s+([0-9]+)",
                "sample size",
            ),
        }
        """Converted native deviation parameters into directly comparable variances."""

    return result


def fit_solar_case(case: SpatialCase) -> dict[str, object]:
    """Fit both known-matrix inputs with qualified native SOLAR.

    Args:
        case: Fixed participant-free component case.

    Returns:
        Case identity and raw within-pedigree and dense native outputs.
    """
    return {
        "true_variances": list(case.true_variances),
        "seed": case.seed,
        "people": case.people,
        "within": fit_solar_components(case, case.block_matrix_csv),
        "dense": fit_solar_components(case, case.dense_matrix_csv),
    }


def result_mapping(record: dict[str, object], field: str) -> dict[str, object]:
    """Return one required nested result mapping.

    Args:
        record: Case result containing named model fits.
        field: Required nested fit name.

    Returns:
        Validated heterogeneous result mapping.

    Raises:
        SolarReferenceError: If the field is not an object.
    """
    value: object = record.get(field)
    """Read one nested fit without assuming fixture-controlled structure."""

    if not isinstance(value, dict):
        raise SolarReferenceError(f"component result {field} must be an object")
    return value


def spatial_case_identity(
    record: dict[str, object],
) -> tuple[tuple[float, ...], int, int]:
    """Return the deterministic identity carried by one component result.

    Args:
        record: Asterism or native case result.

    Returns:
        Generating variances, seed and sample size.

    Raises:
        SolarReferenceError: If the variance identity is malformed.
    """
    variance_value: object = record.get("true_variances")
    """Read the generating component variances from the result record."""

    if not isinstance(variance_value, list | tuple) or len(variance_value) != 3:
        raise SolarReferenceError("component case needs three true variances")
    variances: tuple[float, ...] = tuple(float(value) for value in variance_value)
    """Normalised the three generating variances into an immutable identity."""

    return variances, int(record["seed"]), int(record["people"])


def compare_outputs(
    asterism_outputs: dict[str, object], external_outputs: dict[str, object]
) -> dict[str, object]:
    """Apply same-model agreement and native truncation-boundary rules.

    Args:
        asterism_outputs: Newly recomputed within and dense Asterism fits.
        external_outputs: Live or frozen raw within and dense native fits.

    Returns:
        Detailed pass/fail record for both scientific claims.
    """
    ours: list[dict[str, object]] = record_list(
        asterism_outputs.get("cases"), "Asterism component"
    )
    """Validated the newly recomputed component case sequence."""

    theirs: list[dict[str, object]] = record_list(
        external_outputs.get("cases"), "SOLAR component"
    )
    """Validated the independent live or frozen component case sequence."""

    failures: list[str] = []
    """Collected identity, same-model and truncation-boundary disagreements."""

    comparisons: list[dict[str, object]] = []
    """Retained transparent derived metrics for both fixed cases."""

    if len(ours) != len(theirs):
        failures.append(
            f"case count differs: Asterism {len(ours)}, SOLAR {len(theirs)}"
        )

    components: tuple[str, str, str] = ("genetic", "spatial", "residual")
    """Named the three directly comparable component variances."""

    for mine, external in zip(ours, theirs, strict=False):
        mine_identity: tuple[tuple[float, ...], int, int] = spatial_case_identity(mine)
        """Read the deterministic Asterism case identity."""

        external_identity: tuple[tuple[float, ...], int, int] = spatial_case_identity(
            external
        )
        """Read the corresponding independent native case identity."""

        if mine_identity != external_identity:
            failures.append(
                f"case identity differs: Asterism {mine_identity}, SOLAR {external_identity}"
            )
            continue
        ours_within: dict[str, object] = result_mapping(mine, "within")
        """Read Asterism's fit to the representable block kernel."""

        ours_dense: dict[str, object] = result_mapping(mine, "dense")
        """Read Asterism's genuinely dense-kernel fit."""

        solar_within: dict[str, object] = result_mapping(external, "within")
        """Read native SOLAR's fit to the representable block kernel."""

        solar_dense: dict[str, object] = result_mapping(external, "dense")
        """Read native SOLAR's nominal dense-kernel fit."""

        people: int = mine_identity[2]
        """Recovered the common generated sample size."""

        if (
            int(solar_within["people"]) != people
            or int(solar_dense["people"]) != people
        ):
            failures.append(
                f"case {mine_identity}: SOLAR did not analyse all {people} people"
            )

        within_differences: dict[str, float] = {}
        """Started direct same-model component variance differences."""

        for component in components:
            difference: float = abs(
                float(ours_within[component]) - float(solar_within[component])
            )
            """Computed one same-model component variance disagreement."""

            within_differences[component] = difference
            """Stored the component-specific same-model difference."""

            if not difference < VARIANCE_TOLERANCE:
                failures.append(
                    f"case {mine_identity}: {component} differs by {difference:.3e}"
                )
        """Applied native printed-precision tolerance to every same-model variance."""

        expected_offset: float = -0.5 * people * math.log(2.0 * math.pi)
        """Computed the ML density constant retained only by Asterism."""

        observed_offset: float = float(ours_within["loglik"]) - float(
            solar_within["loglik"]
        )
        """Computed the observed same-model reported likelihood difference."""

        unexplained: float = abs(observed_offset - expected_offset)
        """Isolated disagreement beyond the known density constant."""

        if not unexplained < LOGLIK_TOLERANCE:
            failures.append(
                f"case {mine_identity}: unexplained log-likelihood offset "
                f"is {unexplained:.3e}"
            )

        solar_truncation_difference: float = max(
            abs(float(solar_within[name]) - float(solar_dense[name]))
            for name in components
        )
        """Measured whether native dense input reproduced its block-kernel answer."""

        if not solar_truncation_difference < VARIANCE_TOLERANCE:
            failures.append(
                f"case {mine_identity}: native dense and block fits differ by "
                f"{solar_truncation_difference:.3e}"
            )

        dense_separation: float = max(
            abs(float(ours_dense[name]) - float(solar_dense[name]))
            for name in components
        )
        """Measured separation between the genuinely dense and truncated fits."""

        if not dense_separation > VARIANCE_TOLERANCE:
            failures.append(
                f"case {mine_identity}: dense fit separation is only {dense_separation:.3e}"
            )

        comparisons.append(
            {
                "identity": [list(mine_identity[0]), mine_identity[1], people],
                "within_variance_differences": within_differences,
                "loglik_offset_unexplained": unexplained,
                "solar_dense_vs_block_variance_difference": (
                    solar_truncation_difference
                ),
                "asterism_dense_vs_solar_variance_separation": dense_separation,
            }
        )
    """Applied all fixed same-model and truncation-boundary checks per case."""

    return {"passed": not failures, "failures": failures, "cases": comparisons}


def reference_record(mode: str, fixture_path: Path | None) -> dict[str, object]:
    """Build one strict component refresh or portable verification record.

    Args:
        mode: Either ``refresh`` or ``verify``.
        fixture_path: Frozen envelope required only for verification.

    Returns:
        Machine record consumed by the strict fixture driver.

    Raises:
        SolarReferenceError: If mode, fixture provenance or tool identity fails.
    """
    cases: list[SpatialCase] = generated_cases()
    """Regenerated all fixed participant-free component inputs."""

    identities: list[dict[str, str]] = reference_input_identities(cases)
    """Hashed every exact generated input and shared adapter mechanics."""

    asterism_outputs: dict[str, object] = {
        "cases": [fit_asterism_case(case) for case in cases]
    }
    """Recomputed both Asterism models before consulting external outputs."""

    if mode == "refresh":
        tools: object = solar_tool_identity()
        """Probed the exact native executable before the live comparison."""

        require_solar_version(tools)
        external_outputs: dict[str, object] = {
            "cases": [fit_solar_case(case) for case in cases]
        }
        """Ran native SOLAR only in explicit live-refresh mode."""

        external_tool_invoked: bool = True
        """Recorded the genuine independent-software execution."""
    elif mode == "verify":
        if fixture_path is None:
            raise SolarReferenceError("verify mode requires a fixture path")
        fixture: dict[str, object] = load_fixture(fixture_path, CHECK_ID)
        """Loaded the frozen envelope without executing native SOLAR."""

        require_fixture_inputs(fixture, identities)
        tools = fixture.get("tools")
        """Read the qualified external-tool identity frozen during refresh."""

        require_solar_version(tools)
        acceptance: dict[str, object] = fixture_mapping(fixture, "acceptance")
        """Read the acceptance rule frozen before the external outputs."""

        if acceptance != REFERENCE_ACCEPTANCE:
            raise SolarReferenceError("fixture acceptance rule differs from adapter")
        external_outputs = fixture_mapping(fixture, "external_outputs")
        """Loaded raw native outputs without invoking the external executable."""

        external_tool_invoked = False
        """Recorded that portable verification used only frozen outputs."""
    else:
        raise SolarReferenceError(f"unsupported reference mode: {mode}")

    comparison: dict[str, object] = compare_outputs(asterism_outputs, external_outputs)
    """Applied the same prewritten rules to live and frozen native results."""

    return {
        "reference_fixture_contract": CONTRACT_VERSION,
        "check_id": CHECK_ID,
        "passed": comparison["passed"],
        "external_tool_invoked": external_tool_invoked,
        "asterism_recomputed": True,
        "input_identities": identities,
        "tools": tools,
        "external_outputs": external_outputs,
        "acceptance": REFERENCE_ACCEPTANCE,
        "asterism_outputs": asterism_outputs,
        "comparison": comparison,
    }


def main() -> int:
    """Run the historical live native-SOLAR component comparison."""
    record: dict[str, object] = reference_record("refresh", None)
    """Performed the same qualified comparison used to create a fixture."""

    return emit_reference_record(record)


def entrypoint() -> int:
    """Dispatch historical live use or the strict two-mode fixture contract."""
    arguments: argparse.Namespace = parse_reference_arguments(
        __doc__ or "native SOLAR component comparison"
    )
    """Parsed the optional explicit fixture operation."""

    if arguments.reference_mode is None:
        return main()
    fixture_path: Path = arguments.reference_fixture
    """Selected the paired fixture path required by strict adapter mode."""

    record: dict[str, object] = reference_record(arguments.reference_mode, fixture_path)
    """Ran exactly one explicit refresh or portable verify operation."""

    return emit_reference_record(record)


if __name__ == "__main__":
    try:
        sys.exit(entrypoint())
    except SolarReferenceError as error:
        raise SystemExit(str(error)) from error
