"""Compare Asterism's binary liability model with qualified native SOLAR.

Three deterministic simulated pedigree cases include two informative interior
heritability fits and one zero-heritability control. SOLAR must identify its
discrete liability model itself. Explicit refresh records raw native outputs;
verify recomputes Asterism against them without invoking SOLAR.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

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

CHECK_ID: str = "liability_against_solar"
"""Matched the release-manifest check identifier and fixture filename."""

MAXIMUM_DIFFERENCE_IN_STANDARD_ERRORS: float = 0.25
"""Retained the historical approximation-aware scientific acceptance bar."""

CASES: tuple[tuple[float, float, int, int], ...] = (
    (0.5, 0.25, 30, 8_311),
    (0.4, 0.30, 45, 8_312),
    (0.0, 0.30, 30, 8_313),
)
"""Fixed heritability, prevalence, family count and seed for all three cases."""

SEX: tuple[int, ...] = (2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2)
"""Encoded sex consistently with each generated person's parental role."""

REFERENCE_ACCEPTANCE: dict[str, object] = {
    "maximum_difference_in_solar_standard_errors": (
        MAXIMUM_DIFFERENCE_IN_STANDARD_ERRORS
    ),
    "solar_must_use_liability_model": True,
    "solar_standard_error_required_for_interior_cases": True,
    "zero_control_must_agree_exactly": True,
}
"""Prewrote the scientifically necessary native model and scaled-error rules."""

RUN: str = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait affected
covariate age^1,2#sex
outdir out
polygenic
exit
"""
"""Defined the native SOLAR binary-trait maximum-likelihood analysis."""


@dataclass(frozen=True)
class LiabilityCase:
    """Hold one deterministic participant-free liability comparison case."""

    true_heritability: float
    """Stored the generating additive liability variance fraction."""
    true_prevalence: float
    """Stored the prevalence used to set the generating threshold."""
    families: int
    """Stored the number of replicated extended pedigrees."""
    seed: int
    """Stored the fixed pseudo-random generator seed."""
    relationship: np.ndarray
    """Held the additive relationship matrix shared by both implementations."""
    covariates: np.ndarray
    """Held the fixed-effect design shared by both implementations."""
    affected: np.ndarray
    """Held the generated binary phenotype."""
    pedigree_csv: str
    """Held the exact generated native pedigree input."""
    phenotype_csv: str
    """Held the exact generated native phenotype input."""

    @property
    def people(self) -> int:
        """Return the generated sample size."""
        return int(self.affected.shape[0])


AGE_DECIMALS: int = 6
"""How coarsely the generated age is quantised, and why it has to be.

The native phenotype table renders age as text, and text has no tolerance. A
draw differs between Mac and Linux in its last bit -- about seven parts in a
thousand million million on a number of order fifty -- which is far below
anything scientific and still enough to change the twelfth decimal whenever a
value happens to sit near that rounding boundary. Over six hundred draws one of
them does, so the frozen phenotype table hashed on one platform could never be
regenerated on the other, and this check could only ever verify where it was
made.

Everything else here already survives, because generated arrays are quantised
to twelve decimals before hashing. Only the text escaped that. Six decimals
puts the quantisation nine orders of magnitude above the arithmetic, which
makes a boundary crossing not merely unlikely but not worth thinking about, and
a millionth of a year of age is not a quantity any of this can see.
"""


def generate_case(
    true_heritability: float,
    true_prevalence: float,
    families: int,
    seed: int,
) -> LiabilityCase:
    """Generate one extended-pedigree liability-threshold case.

    Args:
        true_heritability: Additive variance fraction on the liability scale.
        true_prevalence: Population prevalence determining the threshold.
        families: Number of replicated extended pedigrees.
        seed: Fixed pseudo-random generator seed.

    Returns:
        Generated arrays and exact native-SOLAR input text.
    """
    family: list[tuple[int | None, int | None]] = extended_family()
    """Loaded the established extended-family pedigree structure."""

    block: int = len(family)
    """Counted people per replicated pedigree."""

    relationship: np.ndarray = roster(families)
    """Built the block-diagonal additive relationship matrix."""

    people: int = relationship.shape[0]
    """Counted generated individuals in this case."""

    pedigree_rows: list[str] = ["FAMID,ID,FA,MO,SEX"]
    """Started the generated native pedigree table."""

    for family_index in range(families):
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
    """Rendered all generated pedigrees with role-consistent sex."""

    generator: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic participant-free random generator."""

    age: np.ndarray = np.round(generator.uniform(20.0, 70.0, people), AGE_DECIMALS)
    """Generated a non-degenerate continuous age covariate, quantised to survive
    two platforms."""

    male: np.ndarray = np.array(
        [1.0 if SEX[row % block] == 1 else 0.0 for row in range(people)]
    )
    """Generated the role-consistent numerical sex covariate."""

    centred_age: np.ndarray = (age - age.mean()) / age.std()
    """Standardised age for stable liability optimisation."""

    covariates: np.ndarray = np.column_stack(
        [
            np.ones(people),
            centred_age,
            centred_age**2,
            male,
            centred_age * male,
            centred_age**2 * male,
        ]
    )
    """Built the same six-column fixed-effect span used by native SOLAR."""

    covariance: np.ndarray = true_heritability * relationship + (
        1.0 - true_heritability
    ) * np.eye(people)
    """Constructed the additive-plus-residual liability covariance."""

    liability: np.ndarray = np.linalg.cholesky(
        covariance + 1e-10 * np.eye(people)
    ) @ generator.standard_normal(people)
    """Generated a stable correlated latent liability."""

    liability += 0.25 * centred_age + 0.30 * male
    """Added fixed effects so covariate disagreement cannot hide."""

    threshold: float = NormalDist().inv_cdf(1.0 - true_prevalence)
    """Computed the liability cut giving the target population prevalence."""

    affected: np.ndarray = (liability > threshold).astype(float)
    """Thresholded liability into the binary phenotype fitted by both engines."""

    phenotype_rows: list[str] = ["ID,FAMID,age,sex,affected"]
    """Started the generated native phenotype table."""

    for family_index in range(families):
        for person_index in range(block):
            row: int = family_index * block + person_index
            """Located this person in the generated covariate arrays."""

            phenotype_rows.append(
                f"F{family_index:03d}_{person_index:02d},F{family_index:03d},"
                f"{age[row]:.12f},{SEX[person_index]},{int(affected[row])}"
            )
    """Rendered all generated binary phenotypes in native person order."""

    pedigree_csv: str = "\n".join(pedigree_rows) + "\n"
    """Finalised the exact generated native pedigree text."""

    phenotype_csv: str = "\n".join(phenotype_rows) + "\n"
    """Finalised the exact generated native phenotype text."""

    return LiabilityCase(
        true_heritability,
        true_prevalence,
        families,
        seed,
        relationship,
        covariates,
        affected,
        pedigree_csv,
        phenotype_csv,
    )


def generated_cases() -> list[LiabilityCase]:
    """Return all fixed participant-free liability cases.

    Returns:
        Cases in stable fixture order.
    """
    return [generate_case(*case) for case in CASES]


def reference_input_identities(cases: list[LiabilityCase]) -> list[dict[str, str]]:
    """Identify all generated liability inputs and shared adapter mechanics.

    Args:
        cases: Fixed participant-free liability cases.

    Returns:
        Stable named SHA-256 records.
    """
    identities: list[dict[str, str]] = [
        support_source_identity(),
        json_identity(
            "design:liability_against_solar",
            {"cases": CASES, "sex": SEX},
        ),
    ]
    """Bound shared mechanics and scalar design choices first."""

    for case in cases:
        label: str = (
            f"h2={case.true_heritability:g},prevalence={case.true_prevalence:g},"
            f"families={case.families},seed={case.seed}"
        )
        """Built a stable descriptive prefix for one generated case."""

        identities.extend(
            [
                array_identity(f"relationship:{label}", case.relationship),
                array_identity(f"covariates:{label}", case.covariates),
                array_identity(f"affected:{label}", case.affected),
                json_identity(f"pedigree_csv:{label}", case.pedigree_csv),
                json_identity(f"phenotype_csv:{label}", case.phenotype_csv),
            ]
        )
    """Hashed all arrays and exact native text for every generated case."""

    return identities


def fit_asterism_case(case: LiabilityCase) -> dict[str, object]:
    """Fit one generated case through Asterism's public liability model.

    Args:
        case: Deterministic participant-free liability case.

    Returns:
        Asterism estimate, likelihood and optimiser diagnostics.
    """
    model: asterism.LiabilityModel = asterism.LiabilityModel(
        np.ascontiguousarray(case.relationship),
        case.affected,
        case.covariates,
    )
    """Built the documented public liability model on generated inputs."""

    fit: dict[str, object] = model.fit()
    """Fitted maximum likelihood through the public Asterism interface."""

    return {
        "true_heritability": case.true_heritability,
        "true_prevalence": case.true_prevalence,
        "families": case.families,
        "seed": case.seed,
        "people": case.people,
        "observed_prevalence": float(fit["prevalence"]),
        "largest_family": int(fit["largest_family"]),
        "heritability": float(fit["heritability"]),
        "loglik": float(fit["loglik"]),
        "converged": bool(fit["converged"]),
        "scaled_gradient": float(fit["scaled_gradient"]),
    }


def write_solar_inputs(directory: Path, case: LiabilityCase) -> None:
    """Write one generated liability case into a temporary SOLAR directory.

    Args:
        directory: Fresh native-SOLAR working directory.
        case: Generated case whose exact inputs are bound by fixture hashes.
    """
    (directory / "ped.csv").write_text(case.pedigree_csv, encoding="utf-8")
    (directory / "phen.csv").write_text(case.phenotype_csv, encoding="utf-8")
    (directory / "run.tcl").write_text(RUN, encoding="utf-8")
    """Wrote only deterministic participant-free inputs and the fixed command."""


def optional_number(text: str, pattern: str) -> float | None:
    """Return one optional finite numeric native output.

    Args:
        text: Combined native report text.
        pattern: Regular expression containing one numeric capture.

    Returns:
        Parsed value when present, otherwise ``None``.
    """
    found: re.Match[str] | None = re.search(pattern, text)
    """Looked for one secondary output not required to exist by native SOLAR."""

    return None if found is None else float(found.group(1))


def fit_solar_case(case: LiabilityCase) -> dict[str, object]:
    """Fit one generated binary case with qualified native SOLAR.

    Args:
        case: Deterministic participant-free liability case.

    Returns:
        Raw native liability-model outputs used by frozen verification.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory)
        """Isolated all generated native inputs and outputs outside the workspace."""

        write_solar_inputs(directory, case)
        completed: subprocess.CompletedProcess[str] = run_solar(directory)
        """Ran the qualified executable on the fixed binary-trait command."""

        output_path: Path = directory / "out" / "polygenic.out"
        """Selected the required primary native optimisation report."""

        if not output_path.is_file():
            raise SolarReferenceError(
                "SOLAR produced no liability result: "
                + (completed.stdout + completed.stderr)[-2000:]
            )
        report: str = output_path.read_text(encoding="utf-8")
        """Read the primary native liability output."""

        for extra_name in ("polygenic.logs.out", "null0.out"):
            extra_path: Path = directory / "out" / extra_name
            """Selected one optional native secondary output."""

            if extra_path.is_file():
                report += extra_path.read_text(encoding="utf-8")
                """Included secondary evidence of the native discrete model."""
        """Combined all native reports without retaining bulky raw text in fixtures."""

        check_sex_survived(directory)
        used_liability_model: bool = bool(re.search(r"[Dd]iscrete", report))
        """Recorded whether native SOLAR announced its discrete liability machinery."""

        result: dict[str, object] = {
            "true_heritability": case.true_heritability,
            "true_prevalence": case.true_prevalence,
            "families": case.families,
            "seed": case.seed,
            "people": case.people,
            "heritability": required_number(report, r"H2r is\s+([0-9.eE+-]+)", "H2r"),
            "loglik": optional_number(
                report,
                r"Loglikelihood of polygenic model is\s+(-?[0-9.eE+-]+)",
            ),
            "standard_error": optional_number(
                report, r"H2r Std\. Error:\s+([0-9.eE+-]+)"
            ),
            "p_value": optional_number(report, r"H2r is[^)]*?p = ([0-9.eE+-]+)"),
            "used_the_liability_model": used_liability_model,
        }
        """Retained compact raw native fields and generated-case identity."""

    return result


def case_identity(case: dict[str, object]) -> tuple[float, float, int, int, int]:
    """Return the deterministic identity carried by one liability result.

    Args:
        case: Asterism or native result record.

    Returns:
        Truth, prevalence, family count, seed and sample size.
    """
    return (
        float(case["true_heritability"]),
        float(case["true_prevalence"]),
        int(case["families"]),
        int(case["seed"]),
        int(case["people"]),
    )


def compare_outputs(
    asterism_outputs: dict[str, object], external_outputs: dict[str, object]
) -> dict[str, object]:
    """Apply the prewritten liability-scale native-SOLAR acceptance rule.

    Args:
        asterism_outputs: Newly recomputed Asterism cases.
        external_outputs: Live or frozen raw native-SOLAR cases.

    Returns:
        Detailed pass/fail record scaled by native standard errors.
    """
    ours: list[dict[str, object]] = record_list(
        asterism_outputs.get("cases"), "Asterism liability"
    )
    """Validated the newly recomputed liability case sequence."""

    theirs: list[dict[str, object]] = record_list(
        external_outputs.get("cases"), "SOLAR liability"
    )
    """Validated the independent live or frozen liability case sequence."""

    failures: list[str] = []
    """Collected model-identity and scaled numerical disagreements."""

    comparisons: list[dict[str, object]] = []
    """Retained transparent per-case scaled differences."""

    if len(ours) != len(theirs):
        failures.append(
            f"case count differs: Asterism {len(ours)}, SOLAR {len(theirs)}"
        )

    worst_in_errors: float = 0.0
    """Tracked the largest heritability difference in native standard errors."""

    for mine, external in zip(ours, theirs, strict=False):
        mine_identity: tuple[float, float, int, int, int] = case_identity(mine)
        """Read the deterministic Asterism case identity."""

        external_identity: tuple[float, float, int, int, int] = case_identity(external)
        """Read the corresponding independent native case identity."""

        if mine_identity != external_identity:
            failures.append(
                f"case identity differs: Asterism {mine_identity}, SOLAR {external_identity}"
            )
            continue
        if external.get("used_the_liability_model") is not True:
            failures.append(
                f"case {mine_identity}: SOLAR did not use a liability model"
            )
        standard_error_value: object = external.get("standard_error")
        """Read the native uncertainty scale required by the acceptance rule."""

        if standard_error_value is None or float(standard_error_value) <= 0.0:
            zero_control_agreed: bool = (
                mine_identity[0] == 0.0
                and float(mine["heritability"]) == 0.0
                and float(external["heritability"]) == 0.0
            )
            """Recognised the fixed boundary control where no standard error exists."""

            if not zero_control_agreed:
                failures.append(
                    f"case {mine_identity}: SOLAR reported no positive standard error"
                )
            comparisons.append(
                {
                    "identity": list(mine_identity),
                    "absolute_heritability_difference": abs(
                        float(mine["heritability"]) - float(external["heritability"])
                    ),
                    "difference_in_solar_standard_errors": None,
                    "exact_zero_control_agreement": zero_control_agreed,
                }
            )
            continue
        standard_error: float = float(standard_error_value)
        """Validated the positive native heritability uncertainty scale."""

        absolute_difference: float = abs(
            float(mine["heritability"]) - float(external["heritability"])
        )
        """Computed the absolute liability-scale heritability disagreement."""

        difference_in_errors: float = absolute_difference / standard_error
        """Scaled the difference by uncertainty in the external estimate."""

        worst_in_errors = max(worst_in_errors, difference_in_errors)
        """Updated the maximum scaled disagreement across fixed cases."""

        comparisons.append(
            {
                "identity": list(mine_identity),
                "absolute_heritability_difference": absolute_difference,
                "difference_in_solar_standard_errors": difference_in_errors,
            }
        )
    """Compared every aligned case and required native liability-model evidence."""

    if worst_in_errors > MAXIMUM_DIFFERENCE_IN_STANDARD_ERRORS:
        failures.append(
            f"worst difference is {worst_in_errors:.3f} SOLAR standard errors"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "worst_difference_in_solar_standard_errors": worst_in_errors,
        "cases": comparisons,
    }


def reference_record(mode: str, fixture_path: Path | None) -> dict[str, object]:
    """Build one strict liability refresh or portable verification record.

    Args:
        mode: Either ``refresh`` or ``verify``.
        fixture_path: Frozen envelope required only for verification.

    Returns:
        Machine record consumed by the strict fixture driver.

    Raises:
        SolarReferenceError: If mode, fixture provenance or tool identity fails.
    """
    cases: list[LiabilityCase] = generated_cases()
    """Regenerated all fixed participant-free liability inputs."""

    identities: list[dict[str, str]] = reference_input_identities(cases)
    """Hashed exact inputs and shared native adapter mechanics."""

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
        """Read the acceptance rule frozen before external outputs."""

        if acceptance != REFERENCE_ACCEPTANCE:
            raise SolarReferenceError("fixture acceptance rule differs from adapter")
        external_outputs = fixture_mapping(fixture, "external_outputs")
        """Loaded raw native outputs without invoking the external executable."""

        external_tool_invoked = False
        """Recorded that portable verification used only frozen outputs."""
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
    """Run the historical live native-SOLAR liability comparison."""
    record: dict[str, object] = reference_record("refresh", None)
    """Performed the same qualified comparison used to create a fixture."""

    return emit_reference_record(record)


def entrypoint() -> int:
    """Dispatch historical live use or the strict two-mode fixture contract."""
    arguments: argparse.Namespace = parse_reference_arguments(
        __doc__ or "native SOLAR liability comparison"
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
