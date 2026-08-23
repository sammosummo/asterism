"""Compare Asterism's one-trait ML fit with qualified native SOLAR.

The generated extended pedigrees exercise six fixed effects and three genetic
variance settings. Historical no-argument execution still performs the live
comparison. ``--reference-mode refresh`` records qualified external outputs;
``verify`` recomputes Asterism without invoking SOLAR.

SOLAR can silently rewrite sex when it conflicts with parental role. Sex is
therefore generated from role and checked in ``pedindex.out`` before any result
is accepted.
"""

from __future__ import annotations

import argparse
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
    check_sex_survived,
    emit_reference_record,
    fixture_mapping,
    json_identity,
    load_fixture,
    parse_reference_arguments,
    record_list,
    require_fixture_inputs,
    require_solar_version,
    required_number,
    run_solar,
    solar_tool_identity,
    support_source_identity,
)

CHECK_ID: str = "against_solar"
"""Matched the release-manifest check identifier and fixture filename."""

H2_TOLERANCE: float = 5e-7
"""Followed the seven significant figures printed by native SOLAR."""

LOGLIK_TOLERANCE: float = 1e-6
"""Bound unexplained ML log-likelihood offset after restoring the density constant."""

FAMILIES: int = 25
"""Retained the historical generated extended-pedigree design."""

CASES: tuple[tuple[float, int], ...] = ((0.5, 202), (0.2, 203), (0.7, 204))
"""Fixed all simulated truths and seeds before external outputs were observed."""

REFERENCE_ACCEPTANCE: dict[str, object] = {
    "maximum_relative_heritability_difference": H2_TOLERANCE,
    "maximum_unexplained_loglik_offset": LOGLIK_TOLERANCE,
}
"""Prewrote the historical scientific acceptance rule for frozen verification."""

SEX: tuple[int, ...] = (2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2)
"""Encoded sex consistently with each generated person's parental role."""

RUN: str = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait y
covariate age^1,2#sex
outdir out
polygenic
exit
"""
"""Defined the native SOLAR ML analysis over generated temporary inputs."""


@dataclass(frozen=True)
class GaussianCase:
    """Hold one deterministic participant-free one-trait comparison case."""

    true_h2: float
    """Stored the generating heritability used to label the case."""
    seed: int
    """Stored the fixed pseudo-random seed used to reproduce the phenotype."""
    relationship: np.ndarray
    """Held the additive relationship matrix shared by both implementations."""
    design: np.ndarray
    """Held the six-column fixed-effect design shared by both implementations."""
    outcome: np.ndarray
    """Held the generated continuous phenotype shared by both implementations."""
    pedigree_csv: str
    """Held the generated SOLAR pedigree representation."""
    phenotype_csv: str
    """Held the generated SOLAR phenotype representation."""

    @property
    def people(self) -> int:
        """Return the generated sample size."""
        return int(self.outcome.shape[0])


def generate_case(true_h2: float, seed: int) -> GaussianCase:
    """Generate one deterministic extended-pedigree comparison case.

    Args:
        true_h2: Additive variance fraction used to simulate the outcome.
        seed: Fixed NumPy random seed.

    Returns:
        Generated arrays and exact native-SOLAR input text.
    """
    family: list[tuple[int | None, int | None]] = extended_family()
    """Loaded the established fourteen-person extended-family structure."""

    block: int = len(family)
    """Counted people per generated pedigree."""

    relationship: np.ndarray = roster(FAMILIES)
    """Constructed the block-diagonal additive relationship matrix."""

    people: int = relationship.shape[0]
    """Counted all generated individuals across the fixed family count."""

    pedigree_rows: list[str] = ["FAMID,ID,FA,MO,SEX"]
    """Started the native SOLAR pedigree table."""

    for family_index in range(FAMILIES):
        for person_index, (mother, father) in enumerate(family):
            father_id: str = (
                "0" if father is None else f"F{family_index:03d}_{father:02d}"
            )
            """Encoded the generated father identifier or founder sentinel."""

            mother_id: str = (
                "0" if mother is None else f"F{family_index:03d}_{mother:02d}"
            )
            """Encoded the generated mother identifier or founder sentinel."""

            pedigree_rows.append(
                f"F{family_index:03d},F{family_index:03d}_{person_index:02d},"
                f"{father_id},{mother_id},{SEX[person_index]}"
            )
    """Rendered every generated pedigree with role-consistent sex."""

    age: np.ndarray = np.zeros(people)
    """Allocated the standardised age covariate."""

    male: np.ndarray = np.zeros(people)
    """Allocated the binary sex covariate."""

    for family_index in range(FAMILIES):
        for person_index in range(block):
            row: int = family_index * block + person_index
            """Located this person in the shared array ordering."""

            decade: float = (
                7.0 if person_index < 2 else (4.5 if person_index < 8 else 2.0)
            )
            """Assigned age by pedigree generation with a deterministic perturbation."""

            age[row] = (decade - 4.5) / 2.0 + ((row % 7) - 3.0) / 10.0
            """Stored this person's standardised deterministic age."""

            male[row] = 1.0 if SEX[person_index] == 1 else 0.0
            """Stored the role-consistent numeric sex covariate."""
    """Filled covariates identically for Asterism and the SOLAR CSV."""

    design: np.ndarray = np.column_stack(
        [np.ones(people), age, age**2, male, age * male, age**2 * male]
    )
    """Built the same column span as SOLAR's age-polynomial-by-sex covariate."""

    coefficients: np.ndarray = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])
    """Fixed non-zero effects so covariate disagreements cannot hide."""

    generator: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic participant-free random generator."""

    factor: np.ndarray = np.linalg.cholesky(
        true_h2 * relationship + (1.0 - true_h2) * np.eye(people)
    )
    """Factored the generating additive-plus-residual covariance."""

    outcome: np.ndarray = design @ coefficients + factor @ generator.standard_normal(
        people
    )
    """Generated the continuous phenotype from the documented covariance."""

    phenotype_rows: list[str] = ["ID,FAMID,age,sex,y"]
    """Started the native SOLAR phenotype table."""

    for family_index in range(FAMILIES):
        for person_index in range(block):
            row = family_index * block + person_index
            """Located this person in the already generated arrays."""

            phenotype_rows.append(
                f"F{family_index:03d}_{person_index:02d},F{family_index:03d},"
                f"{age[row]:.12f},{SEX[person_index]},{outcome[row]:.12f}"
            )
    """Rendered every generated phenotype in the shared person order."""

    pedigree_csv: str = "\n".join(pedigree_rows) + "\n"
    """Finalised the exact native pedigree input bytes as text."""

    phenotype_csv: str = "\n".join(phenotype_rows) + "\n"
    """Finalised the exact native phenotype input bytes as text."""

    return GaussianCase(
        true_h2,
        seed,
        relationship,
        design,
        outcome,
        pedigree_csv,
        phenotype_csv,
    )


def generated_cases() -> list[GaussianCase]:
    """Return all fixed participant-free one-trait cases.

    Returns:
        Cases in their stable fixture order.
    """
    return [generate_case(true_h2, seed) for true_h2, seed in CASES]


def reference_input_identities(cases: list[GaussianCase]) -> list[dict[str, str]]:
    """Identify every generated input and the SOLAR fixture mechanics.

    Args:
        cases: Deterministic comparison cases.

    Returns:
        Stable named SHA-256 records.
    """
    identities: list[dict[str, str]] = [
        support_source_identity(),
        json_identity(
            "design:against_solar",
            {"families": FAMILIES, "cases": CASES, "sex": SEX},
        ),
    ]
    """Bound the shared mechanics and all scalar design choices first."""

    for case in cases:
        label: str = f"h2={case.true_h2:g},seed={case.seed}"
        """Built a stable human-readable prefix for one generated case."""

        identities.extend(
            [
                array_identity(f"relationship:{label}", case.relationship),
                array_identity(f"design:{label}", case.design),
                array_identity(f"outcome:{label}", case.outcome),
                json_identity(f"pedigree_csv:{label}", case.pedigree_csv),
                json_identity(f"phenotype_csv:{label}", case.phenotype_csv),
            ]
        )
    """Hashed all arrays and exact native input representations for every case."""

    return identities


def write_solar_inputs(directory: Path, case: GaussianCase) -> None:
    """Write one generated case into an isolated native-SOLAR directory.

    Args:
        directory: Fresh temporary analysis directory.
        case: Generated case whose exact CSV content is recorded by hash.
    """
    (directory / "ped.csv").write_text(case.pedigree_csv, encoding="utf-8")
    (directory / "phen.csv").write_text(case.phenotype_csv, encoding="utf-8")
    (directory / "run.tcl").write_text(RUN, encoding="utf-8")
    """Wrote only deterministic participant-free inputs and the fixed command."""


def fit_solar_case(case: GaussianCase) -> dict[str, object]:
    """Fit one generated case with the qualified native SOLAR executable.

    Args:
        case: Generated participant-free one-trait case.

    Returns:
        Raw reported SOLAR quantities used by the frozen comparison.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory)
        """Isolated all generated native inputs and outputs outside the workspace."""

        write_solar_inputs(directory, case)
        completed: subprocess.CompletedProcess[str] = run_solar(directory)
        """Ran the qualified executable on the fixed native command file."""

        output_path: Path = directory / "out" / "polygenic.out"
        """Selected the required primary optimisation report."""

        if not output_path.is_file():
            raise SolarReferenceError(
                "SOLAR produced no result: "
                + (completed.stdout + completed.stderr)[-2000:]
            )
        report: str = output_path.read_text(encoding="utf-8")
        """Read the native optimisation report containing reported estimates."""

        logs_path: Path = directory / "out" / "polygenic.logs.out"
        """Selected the optional native log carrying secondary output fields."""

        if logs_path.is_file():
            report += logs_path.read_text(encoding="utf-8")
            """Included secondary reported fields when the native run wrote them."""

        check_sex_survived(directory)
        result: dict[str, object] = {
            "true_h2": case.true_h2,
            "seed": case.seed,
            "people": case.people,
            "h2": required_number(report, r"H2r is\s+([0-9.eE+-]+)", "H2r"),
            "standard_error": required_number(
                report,
                r"H2r Std\. Error:\s+([0-9.eE+-]+)",
                "H2r standard error",
            ),
            "loglik": required_number(
                report,
                r"Loglikelihood of polygenic model is\s+(-?[0-9.eE+-]+)",
                "polygenic log-likelihood",
            ),
            "sporadic_loglik": required_number(
                report,
                r"Loglikelihood of sporadic model is\s+(-?[0-9.eE+-]+)",
                "sporadic log-likelihood",
            ),
        }
        """Retained raw native fields without derived Asterism comparisons."""

    return result


def fit_asterism_case(case: GaussianCase) -> dict[str, object]:
    """Fit one generated case through Asterism's public one-trait interface.

    Args:
        case: Generated participant-free one-trait case.

    Returns:
        Asterism estimates needed by the prewritten acceptance rule.
    """
    fit: dict[str, object] = asterism.prepare(case.design, case.relationship).fit(
        case.outcome, "ml"
    )
    """Fitted maximum likelihood through the documented prepared-model API."""

    standard_errors: object = fit.get("standard_errors")
    """Read optional standard errors without assuming their runtime shape."""

    standard_error: float | None = None
    """Defaulted the descriptive standard error when the fit omitted it."""

    if isinstance(standard_errors, dict) and standard_errors.get("h2") is not None:
        standard_error = float(standard_errors["h2"])
        """Retained the reported heritability standard error when available."""

    return {
        "true_h2": case.true_h2,
        "seed": case.seed,
        "people": case.people,
        "h2": float(fit["h2"]),
        "standard_error": standard_error,
        "loglik": float(fit["loglik"]),
    }


def compare_outputs(
    asterism_outputs: dict[str, object], external_outputs: dict[str, object]
) -> dict[str, object]:
    """Apply the prewritten one-trait SOLAR acceptance rule.

    Args:
        asterism_outputs: Newly recomputed Asterism results.
        external_outputs: Live or frozen raw native-SOLAR results.

    Returns:
        Detailed pass/fail comparison record.
    """
    ours: list[dict[str, object]] = record_list(
        asterism_outputs.get("cases"), "Asterism"
    )
    """Validated the newly recomputed case sequence."""

    theirs: list[dict[str, object]] = record_list(
        external_outputs.get("cases"), "SOLAR"
    )
    """Validated the independent live or frozen case sequence."""

    failures: list[str] = []
    """Collected every identity or numerical disagreement."""

    comparisons: list[dict[str, object]] = []
    """Retained transparent derived differences without rewriting native outputs."""

    if len(ours) != len(theirs):
        failures.append(
            f"case count differs: Asterism {len(ours)}, SOLAR {len(theirs)}"
        )

    for mine, external in zip(ours, theirs, strict=False):
        mine_identity: tuple[float, int, int] = (
            float(mine["true_h2"]),
            int(mine["seed"]),
            int(mine["people"]),
        )
        """Read the deterministic Asterism case identity."""

        external_identity: tuple[float, int, int] = (
            float(external["true_h2"]),
            int(external["seed"]),
            int(external["people"]),
        )
        """Read the corresponding independent native case identity."""

        if mine_identity != external_identity:
            failures.append(
                f"case identity differs: Asterism {mine_identity}, SOLAR {external_identity}"
            )
            continue
        ours_h2: float = float(mine["h2"])
        """Read Asterism's recomputed heritability."""

        solar_h2: float = float(external["h2"])
        """Read native SOLAR's reported heritability."""

        relative: float = abs(ours_h2 - solar_h2) / (1.0 + abs(solar_h2))
        """Computed the historical scale-stable heritability difference."""

        people: int = mine_identity[2]
        """Recovered the common sample size for the ML density constant."""

        expected_offset: float = -0.5 * people * math.log(2.0 * math.pi)
        """Computed the density constant retained by Asterism and dropped by SOLAR."""

        observed_offset: float = float(mine["loglik"]) - float(external["loglik"])
        """Computed the observed difference between the two reported likelihoods."""

        unexplained: float = abs(observed_offset - expected_offset)
        """Isolated any discrepancy beyond the known density constant."""

        comparisons.append(
            {
                "true_h2": mine_identity[0],
                "seed": mine_identity[1],
                "heritability_relative_difference": relative,
                "loglik_offset_unexplained": unexplained,
            }
        )
        if not relative < H2_TOLERANCE:
            failures.append(
                f"true h2 {mine_identity[0]}: heritability differs by {relative:.3e}"
            )
        if not unexplained < LOGLIK_TOLERANCE:
            failures.append(
                f"true h2 {mine_identity[0]}: unexplained log-likelihood offset "
                f"is {unexplained:.3e}"
            )
    """Compared every aligned case using only the fixed historical thresholds."""

    return {"passed": not failures, "failures": failures, "cases": comparisons}


def reference_record(mode: str, fixture_path: Path | None) -> dict[str, object]:
    """Build one strict live-refresh or portable-verification adapter record.

    Args:
        mode: Either ``refresh`` or ``verify``.
        fixture_path: Frozen envelope required only for verification.

    Returns:
        Machine record consumed by the strict fixture driver.

    Raises:
        SolarReferenceError: If mode, fixture provenance or tool identity fails.
    """
    cases: list[GaussianCase] = generated_cases()
    """Regenerated every participant-free input from fixed seeds."""

    identities: list[dict[str, str]] = reference_input_identities(cases)
    """Hashed exact inputs and shared SOLAR adapter mechanics."""

    asterism_outputs: dict[str, object] = {
        "cases": [fit_asterism_case(case) for case in cases]
    }
    """Recomputed Asterism before consulting live or frozen external outputs."""

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
        """Recorded that portable verification used only the frozen outputs."""
    else:
        raise SolarReferenceError(f"unsupported reference mode: {mode}")

    comparison: dict[str, object] = compare_outputs(asterism_outputs, external_outputs)
    """Applied the same prewritten rule to live and frozen native results."""

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
    """Run the historical live native-SOLAR comparison."""
    record: dict[str, object] = reference_record("refresh", None)
    """Performed the same qualified comparison used to create a fixture."""

    return emit_reference_record(record)


def entrypoint() -> int:
    """Dispatch historical live use or the strict two-mode fixture contract."""
    arguments: argparse.Namespace = parse_reference_arguments(
        __doc__ or "native SOLAR comparison"
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
