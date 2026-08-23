"""Compare Asterism's unbalanced two-trait ML fit with native SOLAR.

Every seventh generated person lacks the second trait, forcing the comparison
through the unbalanced likelihood rather than the simpler Kronecker path.
No-argument execution remains a qualified live comparison. Explicit refresh
and verify modes implement the frozen external-reference contract.
"""

from __future__ import annotations

import argparse
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
    require_fixture_inputs,
    require_solar_version,
    required_number,
    run_solar,
    solar_tool_identity,
    support_source_identity,
)

CHECK_ID: str = "bivariate_against_solar"
"""Matched the release-manifest check identifier and fixture filename."""

TOLERANCE: float = 5e-6
"""Followed the seven significant figures printed by native SOLAR."""

MAXIMUM_SCALED_GRADIENT: float = 1e-7
"""Retained the historical optimiser-quality requirement for Asterism."""

FAMILIES: int = 25
"""Retained the original participant-free extended-pedigree count."""

SEED: int = 411
"""Fixed the pseudo-random design before external outputs were observed."""

TRUTH: dict[str, float] = {"h1": 0.6, "h2": 0.35, "rg": 0.55, "re": 0.25}
"""Fixed the generating bivariate covariance parameters."""

SEX: tuple[int, ...] = (2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2)
"""Encoded sex consistently with each generated person's parental role."""

REFERENCE_ACCEPTANCE: dict[str, object] = {
    "maximum_absolute_reported_difference": TOLERANCE,
    "asterism_must_converge": True,
    "maximum_asterism_scaled_gradient": MAXIMUM_SCALED_GRADIENT,
}
"""Prewrote all historical numerical and optimiser acceptance rules."""

RUN: str = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait y1 y2
covariate age^1,2#sex
outdir out
polygenic
exit
"""
"""Defined the native SOLAR two-trait maximum-likelihood analysis."""


@dataclass(frozen=True)
class BivariateCase:
    """Hold the deterministic participant-free unbalanced bivariate case."""

    relationship: np.ndarray
    """Held the additive relationship matrix shared by both implementations."""
    observed: np.ndarray
    """Held the person-by-trait observation mask."""
    design: np.ndarray
    """Held the row-expanded two-trait fixed-effect design."""
    values: np.ndarray
    """Held observed outcomes in person-major trait order."""
    pedigree_csv: str
    """Held the exact generated native pedigree input."""
    phenotype_csv: str
    """Held the exact generated native phenotype input."""

    @property
    def people(self) -> int:
        """Return the generated number of people."""
        return int(self.observed.shape[0])


def generate_case() -> BivariateCase:
    """Generate the fixed unbalanced two-trait comparison problem.

    Returns:
        Arrays and exact native-SOLAR text generated from the fixed seed.
    """
    family: list[tuple[int | None, int | None]] = extended_family()
    """Loaded the established extended-family pedigree structure."""

    block: int = len(family)
    """Counted people in each replicated pedigree."""

    relationship: np.ndarray = roster(FAMILIES)
    """Built the block-diagonal additive relationship matrix."""

    people: int = relationship.shape[0]
    """Counted the generated individuals."""

    pedigree_rows: list[str] = ["FAMID,ID,FA,MO,SEX"]
    """Started the native pedigree table."""

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

            pedigree_rows.append(
                f"F{family_index:03d},F{family_index:03d}_{person_index:02d},"
                f"{father_id},{mother_id},{SEX[person_index]}"
            )
    """Rendered all replicated pedigrees with role-consistent sex."""

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
            """Assigned an age level by generation with deterministic variation."""

            age[row] = (decade - 4.5) / 2.0 + ((row % 7) - 3.0) / 10.0
            """Stored the generated standardised age."""

            male[row] = 1.0 if SEX[person_index] == 1 else 0.0
            """Stored the role-consistent numerical sex value."""
    """Filled all fixed-effect covariates in native person order."""

    covariates: np.ndarray = np.column_stack(
        [np.ones(people), age, age**2, male, age * male, age**2 * male]
    )
    """Built the same six-column span as native SOLAR."""

    h1: float = TRUTH["h1"]
    """Read the fixed first-trait heritability."""

    h2: float = TRUTH["h2"]
    """Read the fixed second-trait heritability."""

    genetic_correlation: float = TRUTH["rg"]
    """Read the fixed genetic correlation."""

    residual_correlation: float = TRUTH["re"]
    """Read the fixed residual correlation."""

    genetic: np.ndarray = np.array(
        [
            [h1, genetic_correlation * np.sqrt(h1 * h2)],
            [genetic_correlation * np.sqrt(h1 * h2), h2],
        ]
    )
    """Constructed the generating additive trait covariance."""

    residual: np.ndarray = np.array(
        [
            [
                1.0 - h1,
                residual_correlation * np.sqrt((1.0 - h1) * (1.0 - h2)),
            ],
            [
                residual_correlation * np.sqrt((1.0 - h1) * (1.0 - h2)),
                1.0 - h2,
            ],
        ]
    )
    """Constructed the generating residual trait covariance."""

    covariance: np.ndarray = np.kron(genetic, relationship) + np.kron(
        residual, np.eye(people)
    )
    """Combined trait covariance with relationship and individual structure."""

    generator: np.random.Generator = np.random.default_rng(SEED)
    """Created the fixed participant-free random generator."""

    draw: np.ndarray = np.linalg.cholesky(
        covariance + 1e-10 * np.eye(2 * people)
    ) @ generator.standard_normal(2 * people)
    """Generated one stable bivariate residual draw."""

    first_coefficients: np.ndarray = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])
    """Fixed non-zero first-trait covariate effects."""

    second_coefficients: np.ndarray = np.array([-1.0, 0.15, 0.08, -0.30, 0.02, 0.01])
    """Fixed distinct non-zero second-trait covariate effects."""

    first_outcome: np.ndarray = covariates @ first_coefficients + draw[:people]
    """Generated the first continuous trait."""

    second_outcome: np.ndarray = covariates @ second_coefficients + draw[people:]
    """Generated the second continuous trait."""

    observed: np.ndarray = np.array(
        [[True, row % 7 != 0] for row in range(people)], dtype=bool
    )
    """Removed every seventh second-trait value to force unbalanced fitting."""

    phenotype_rows: list[str] = ["ID,FAMID,age,sex,y1,y2"]
    """Started the generated native phenotype table."""

    for family_index in range(FAMILIES):
        for person_index in range(block):
            row = family_index * block + person_index
            """Located this person in the generated trait arrays."""

            second_text: str = f"{second_outcome[row]:.12f}" if observed[row, 1] else ""
            """Rendered the designed second-trait missingness as an empty CSV field."""

            phenotype_rows.append(
                f"F{family_index:03d}_{person_index:02d},F{family_index:03d},"
                f"{age[row]:.12f},{SEX[person_index]},"
                f"{first_outcome[row]:.12f},{second_text}"
            )
    """Rendered both traits in the shared native person order."""

    values: list[float] = []
    """Started outcomes in Asterism's person-major observed-trait order."""

    design_rows: list[np.ndarray] = []
    """Started the matching trait-expanded fixed-effect rows."""

    for row in range(people):
        values.append(float(first_outcome[row]))
        design_rows.append(np.concatenate([covariates[row], np.zeros(6)]))
        if observed[row, 1]:
            values.append(float(second_outcome[row]))
            design_rows.append(np.concatenate([np.zeros(6), covariates[row]]))
    """Expanded observed trait values and designs in documented input order."""

    pedigree_csv: str = "\n".join(pedigree_rows) + "\n"
    """Finalised the exact generated native pedigree text."""

    phenotype_csv: str = "\n".join(phenotype_rows) + "\n"
    """Finalised the exact generated native phenotype text."""

    return BivariateCase(
        relationship,
        observed,
        np.asarray(design_rows),
        np.asarray(values),
        pedigree_csv,
        phenotype_csv,
    )


def reference_input_identities(case: BivariateCase) -> list[dict[str, str]]:
    """Identify all generated bivariate inputs and shared adapter mechanics.

    Args:
        case: Fixed participant-free bivariate case.

    Returns:
        Stable named SHA-256 records.
    """
    return [
        support_source_identity(),
        json_identity(
            "design:bivariate_against_solar",
            {"families": FAMILIES, "seed": SEED, "truth": TRUTH, "sex": SEX},
        ),
        array_identity("relationship", case.relationship),
        array_identity("observed", case.observed),
        array_identity("design", case.design),
        array_identity("values", case.values),
        json_identity("pedigree_csv", case.pedigree_csv),
        json_identity("phenotype_csv", case.phenotype_csv),
    ]


def fit_asterism(case: BivariateCase) -> dict[str, object]:
    """Fit the generated case through Asterism's public bivariate model.

    Args:
        case: Fixed participant-free bivariate case.

    Returns:
        Directly reported bivariate parameters and optimiser diagnostics.
    """
    model: asterism.BivariateModel = asterism.BivariateModel(
        np.ascontiguousarray(case.relationship),
        case.observed.tolist(),
        np.ascontiguousarray(case.design),
    )
    """Built the documented model with the generated unbalanced observation mask."""

    fit: dict[str, object] = model.fit(np.ascontiguousarray(case.values), reml=False)
    """Fitted maximum likelihood through the public Asterism interface."""

    return {
        "h2_first": float(fit["h2_first"]),
        "h2_second": float(fit["h2_second"]),
        "rho_g": float(fit["rho_g"]),
        "rho_e": float(fit["rho_e"]),
        "loglik": float(fit["loglik"]),
        "scaled_gradient": float(fit["scaled_gradient"]),
        "converged": bool(fit["converged"]),
    }


def write_solar_inputs(directory: Path, case: BivariateCase) -> None:
    """Write exact generated bivariate inputs into one temporary directory.

    Args:
        directory: Fresh native-SOLAR working directory.
        case: Generated case whose inputs are bound by fixture hashes.
    """
    (directory / "ped.csv").write_text(case.pedigree_csv, encoding="utf-8")
    (directory / "phen.csv").write_text(case.phenotype_csv, encoding="utf-8")
    (directory / "run.tcl").write_text(RUN, encoding="utf-8")
    """Wrote only deterministic participant-free inputs and the fixed command."""


def fit_solar(case: BivariateCase) -> dict[str, object]:
    """Fit the generated bivariate case with qualified native SOLAR.

    Args:
        case: Fixed participant-free bivariate case.

    Returns:
        Raw directly reported native-SOLAR parameters.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory)
        """Isolated all generated native inputs and outputs outside the workspace."""

        write_solar_inputs(directory, case)
        completed: subprocess.CompletedProcess[str] = run_solar(directory)
        """Ran the qualified executable on the fixed two-trait command."""

        output_path: Path = directory / "out" / "polygenic.out"
        """Selected the required native two-trait report."""

        if not output_path.is_file():
            raise SolarReferenceError(
                "SOLAR produced no bivariate result: "
                + (completed.stdout + completed.stderr)[-2000:]
            )
        report: str = output_path.read_text(encoding="utf-8")
        """Read the raw native parameter report."""

        logs_path: Path = directory / "out" / "polygenic.logs.out"
        """Selected the optional native secondary report."""

        if logs_path.is_file():
            report += logs_path.read_text(encoding="utf-8")
            """Included any parameters emitted only to the secondary report."""

        check_sex_survived(directory)
        result: dict[str, object] = {
            "people": case.people,
            "observations_by_trait": [
                int(case.observed[:, 0].sum()),
                int(case.observed[:, 1].sum()),
            ],
            "h2_first": required_number(
                report, r"H2r\(y1\) is\s+([0-9.eE+-]+)", "H2r(y1)"
            ),
            "h2_second": required_number(
                report, r"H2r\(y2\) is\s+([0-9.eE+-]+)", "H2r(y2)"
            ),
            "rho_g": required_number(report, r"RhoG is\s+(-?[0-9.eE+-]+)", "RhoG"),
            "rho_e": required_number(report, r"RhoE is\s+(-?[0-9.eE+-]+)", "RhoE"),
        }
        """Retained only raw native outputs and their generated-case identity."""

    return result


def compare_outputs(
    asterism_outputs: dict[str, object], external_outputs: dict[str, object]
) -> dict[str, object]:
    """Apply the prewritten unbalanced bivariate acceptance rule.

    Args:
        asterism_outputs: Newly recomputed Asterism result.
        external_outputs: Live or frozen raw native-SOLAR result.

    Returns:
        Detailed pass/fail record for the four directly reported parameters.
    """
    failures: list[str] = []
    """Collected optimiser and numerical disagreements."""

    if asterism_outputs.get("converged") is not True:
        failures.append("Asterism did not declare convergence")
    gradient: float = float(asterism_outputs["scaled_gradient"])
    """Read the public scaled projected-gradient diagnostic."""

    if not gradient < MAXIMUM_SCALED_GRADIENT:
        failures.append(f"Asterism scaled projected gradient was {gradient:.3e}")

    differences: dict[str, float] = {}
    """Retained absolute disagreements for every direct native parameter."""

    for field in ("h2_first", "h2_second", "rho_g", "rho_e"):
        difference: float = abs(
            float(asterism_outputs[field]) - float(external_outputs[field])
        )
        """Computed one direct reported-parameter difference without re-expression."""

        differences[field] = difference
        """Stored the derived difference beside its native parameter name."""

        if not difference < TOLERANCE:
            failures.append(f"{field} differs by {difference:.3e}")
    """Applied the same fixed printed-precision threshold to all four parameters."""

    return {"passed": not failures, "failures": failures, "differences": differences}


def reference_record(mode: str, fixture_path: Path | None) -> dict[str, object]:
    """Build one strict bivariate refresh or portable verification record.

    Args:
        mode: Either ``refresh`` or ``verify``.
        fixture_path: Frozen envelope required only for verification.

    Returns:
        Machine record consumed by the strict fixture driver.

    Raises:
        SolarReferenceError: If mode, fixture provenance or tool identity fails.
    """
    case: BivariateCase = generate_case()
    """Regenerated the fixed participant-free unbalanced case."""

    identities: list[dict[str, str]] = reference_input_identities(case)
    """Hashed all exact generated inputs and shared adapter mechanics."""

    asterism_outputs: dict[str, object] = fit_asterism(case)
    """Recomputed Asterism before consulting live or frozen external outputs."""

    if mode == "refresh":
        tools: object = solar_tool_identity()
        """Probed the exact native executable before the live comparison."""

        require_solar_version(tools)
        external_outputs: dict[str, object] = fit_solar(case)
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
    """Run the historical live native-SOLAR bivariate comparison."""
    record: dict[str, object] = reference_record("refresh", None)
    """Performed the same qualified comparison used to create a fixture."""

    return emit_reference_record(record)


def entrypoint() -> int:
    """Dispatch historical live use or the strict two-mode fixture contract."""
    arguments: argparse.Namespace = parse_reference_arguments(
        __doc__ or "native SOLAR bivariate comparison"
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
