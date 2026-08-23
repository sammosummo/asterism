"""Compare Asterism's binary-continuous fit with qualified native SOLAR.

The deterministic nuclear-family simulation gives both implementations a
binary liability trait beside a continuous quantity. Both fix liability
variance at one, making the reported heritability and correlations directly
comparable. Explicit refresh invokes native SOLAR; verify uses only its frozen
outputs while recomputing Asterism.
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

CHECK_ID: str = "mixed_bivariate_against_solar"
"""Matched the release-manifest check identifier and fixture filename."""

FAMILIES: int = 300
"""Retained the historical number of generated nuclear families."""

SIBLINGS: int = 4
"""Retained four full siblings beside two founders per family."""

SEED: int = 20_260_818
"""Fixed the participant-free pseudo-random design before external observation."""

TRUE_HERITABILITY: tuple[float, float] = (0.5, 0.5)
"""Fixed liability-scale and continuous-trait heritabilities."""

TRUE_GENETIC_CORRELATION: float = 0.5
"""Fixed the generating cross-trait additive correlation."""

TRUE_RESIDUAL_CORRELATION: float = 0.2
"""Fixed the generating cross-trait residual correlation."""

CONTINUOUS_VARIANCE: float = 3.0
"""Set the scale of the generated continuous quantity."""

PREVALENCE: float = 0.3
"""Set the binary liability threshold through population prevalence."""

TOLERANCE: float = 0.05
"""Retained the historical direct-parameter agreement threshold."""

REFERENCE_ACCEPTANCE: dict[str, object] = {
    "maximum_absolute_reported_difference": TOLERANCE,
    "solar_must_use_discrete_model": True,
}
"""Prewrote the historical model-identity and direct-difference rules."""

RUN: str = """load pedigree ped.csv
load phenotypes phen.csv
model new
trait affected quantity
outdir out
polygenic
exit
"""
"""Defined the native SOLAR mixed binary-continuous analysis."""


@dataclass(frozen=True)
class MixedCase:
    """Hold the deterministic participant-free mixed bivariate case."""

    relationship: np.ndarray
    """Held the additive relationship matrix shared by both implementations."""
    affected: np.ndarray
    """Held the generated binary liability-threshold phenotype."""
    quantity: np.ndarray
    """Held the generated continuous phenotype."""
    pedigree_csv: str
    """Held the exact generated native pedigree input."""
    phenotype_csv: str
    """Held the exact generated native phenotype input."""

    @property
    def people(self) -> int:
        """Return the generated sample size."""
        return int(self.affected.shape[0])


def family_relationship() -> np.ndarray:
    """Return the additive relationship matrix for one generated nuclear family.

    Returns:
        Founder-and-four-full-sibling additive relationship matrix.
    """
    block: int = 2 + SIBLINGS
    """Counted two founders and the fixed number of full siblings."""

    relationship: np.ndarray = np.eye(block)
    """Started with each person's unit additive variance."""

    for child in range(2, block):
        relationship[0, child] = 0.5
        """Set the father's relationship with this child."""

        relationship[child, 0] = 0.5
        """Mirrored the father-child relationship."""

        relationship[1, child] = 0.5
        """Set the mother's relationship with this child."""

        relationship[child, 1] = 0.5
        """Mirrored the mother-child relationship."""

        for sibling in range(2, block):
            if sibling != child:
                relationship[child, sibling] = 0.5
                """Set the full-sibling additive relationship."""
    """Filled all founder-offspring and full-sibling relationships."""

    return relationship


def generate_case() -> MixedCase:
    """Generate the fixed binary-continuous nuclear-family problem.

    Returns:
        Generated arrays and exact native-SOLAR input text.
    """
    block_relationship: np.ndarray = family_relationship()
    """Built the additive covariance for one six-person family."""

    block: int = block_relationship.shape[0]
    """Counted people in each generated family."""

    people: int = FAMILIES * block
    """Counted all generated individuals."""

    relationship: np.ndarray = np.kron(np.eye(FAMILIES), block_relationship)
    """Replicated the additive relationship matrix across independent families."""

    additive_cross: float = TRUE_GENETIC_CORRELATION * np.sqrt(
        TRUE_HERITABILITY[0] * TRUE_HERITABILITY[1] * CONTINUOUS_VARIANCE
    )
    """Computed the generating cross-trait additive covariance."""

    additive: np.ndarray = np.array(
        [
            [TRUE_HERITABILITY[0], additive_cross],
            [additive_cross, TRUE_HERITABILITY[1] * CONTINUOUS_VARIANCE],
        ]
    )
    """Constructed the generating two-trait additive covariance."""

    residual_cross: float = TRUE_RESIDUAL_CORRELATION * np.sqrt(
        (1.0 - TRUE_HERITABILITY[0])
        * (1.0 - TRUE_HERITABILITY[1])
        * CONTINUOUS_VARIANCE
    )
    """Computed the generating cross-trait residual covariance."""

    residual: np.ndarray = np.array(
        [
            [1.0 - TRUE_HERITABILITY[0], residual_cross],
            [
                residual_cross,
                (1.0 - TRUE_HERITABILITY[1]) * CONTINUOUS_VARIANCE,
            ],
        ]
    )
    """Constructed the generating two-trait residual covariance."""

    family_covariance: np.ndarray = np.kron(additive, block_relationship) + np.kron(
        residual, np.eye(block)
    )
    """Combined additive and residual covariance within one family."""

    family_factor: np.ndarray = np.linalg.cholesky(
        family_covariance + 1e-9 * np.eye(2 * block)
    )
    """Factored the small repeated family covariance exactly once."""

    generator: np.random.Generator = np.random.default_rng(SEED)
    """Created the deterministic participant-free random generator."""

    family_draws: np.ndarray = generator.standard_normal((FAMILIES, 2 * block))
    """Drew independent standard-normal vectors for every nuclear family."""

    transformed: np.ndarray = family_draws @ family_factor.T
    """Applied the shared family covariance to every generated draw."""

    latent: np.ndarray = np.empty((2, people))
    """Allocated trait-major outcomes in Asterism's person order."""

    for family_index in range(FAMILIES):
        start: int = family_index * block
        """Located the first array row for this family."""

        stop: int = start + block
        """Located the exclusive last array row for this family."""

        latent[0, start:stop] = transformed[family_index, :block]
        """Stored this family's generated binary-trait liabilities."""

        latent[1, start:stop] = transformed[family_index, block:]
        """Stored this family's generated continuous quantities."""
    """Assembled all independent family draws into the shared person order."""

    threshold: float = NormalDist().inv_cdf(1.0 - PREVALENCE)
    """Computed the liability threshold from the fixed population prevalence."""

    affected: np.ndarray = (latent[0] > threshold).astype(np.int64)
    """Thresholded the generated liabilities into case indicators."""

    quantity: np.ndarray = latent[1].copy()
    """Retained the generated continuous trait in shared person order."""

    pedigree_rows: list[str] = ["FAMID,ID,FA,MO,SEX"]
    """Started the native nuclear-family pedigree table."""

    phenotype_rows: list[str] = ["ID,FAMID,affected,quantity"]
    """Started the matching native mixed-trait phenotype table."""

    for family_index in range(FAMILIES):
        for person_index in range(block):
            identifier: str = f"F{family_index:04d}_{person_index:02d}"
            """Built the stable generated person identifier."""

            if person_index < 2:
                father_id: str = "0"
                """Marked this founder as having no recorded father."""

                mother_id: str = "0"
                """Marked this founder as having no recorded mother."""

                sex: int = 1 if person_index == 0 else 2
                """Assigned founder sex consistently with parental role."""
            else:
                father_id = f"F{family_index:04d}_00"
                """Linked this full sibling to the generated father."""

                mother_id = f"F{family_index:04d}_01"
                """Linked this full sibling to the generated mother."""

                sex = 1 + person_index % 2
                """Alternated sibling sex without contradicting parental roles."""

            pedigree_rows.append(
                f"F{family_index:04d},{identifier},{father_id},{mother_id},{sex}"
            )
            row: int = family_index * block + person_index
            """Located this person in both generated trait arrays."""

            phenotype_rows.append(
                f"{identifier},F{family_index:04d},{affected[row]},{quantity[row]:.12f}"
            )
    """Rendered all pedigrees and traits in the shared native person order."""

    pedigree_csv: str = "\n".join(pedigree_rows) + "\n"
    """Finalised the exact generated native pedigree text."""

    phenotype_csv: str = "\n".join(phenotype_rows) + "\n"
    """Finalised the exact generated native phenotype text."""

    return MixedCase(
        relationship,
        affected,
        quantity,
        pedigree_csv,
        phenotype_csv,
    )


def reference_input_identities(case: MixedCase) -> list[dict[str, str]]:
    """Identify all generated mixed-trait inputs and adapter mechanics.

    Args:
        case: Fixed participant-free mixed bivariate case.

    Returns:
        Stable named SHA-256 records.
    """
    return [
        support_source_identity(),
        json_identity(
            "design:mixed_bivariate_against_solar",
            {
                "families": FAMILIES,
                "siblings": SIBLINGS,
                "seed": SEED,
                "heritability": TRUE_HERITABILITY,
                "genetic_correlation": TRUE_GENETIC_CORRELATION,
                "residual_correlation": TRUE_RESIDUAL_CORRELATION,
                "continuous_variance": CONTINUOUS_VARIANCE,
                "prevalence": PREVALENCE,
            },
        ),
        array_identity("relationship", case.relationship),
        array_identity("affected", case.affected),
        array_identity("quantity", case.quantity),
        json_identity("pedigree_csv", case.pedigree_csv),
        json_identity("phenotype_csv", case.phenotype_csv),
    ]


def fit_asterism(case: MixedCase) -> dict[str, object]:
    """Fit the generated case through Asterism's public mixed interface.

    Args:
        case: Fixed participant-free mixed bivariate case.

    Returns:
        Directly comparable model parameters and optimiser diagnostics.
    """
    people: int = case.people
    """Read the common generated sample size."""

    fit: dict[str, object] = asterism.mixed_bivariate_fit(
        case.relationship,
        {
            "kind": "binary",
            "value": np.full(people, np.nan),
            "censoring": np.where(case.affected == 1, 1, 2).astype(np.int64),
            "limit": np.zeros(people),
        },
        {
            "kind": "continuous",
            "value": case.quantity,
            "censoring": np.zeros(people, dtype=np.int64),
            "limit": np.zeros(people),
        },
        np.ones((people, 1)),
    )
    """Fitted the documented binary-continuous model on generated inputs."""

    heritability: object = fit["heritability"]
    """Read the two reported trait heritabilities."""

    if (
        not isinstance(heritability, list | tuple | np.ndarray)
        or len(heritability) != 2
    ):
        raise SolarReferenceError("Asterism returned no two-trait heritability")
    return {
        "h2_affected": float(heritability[0]),
        "h2_quantity": float(heritability[1]),
        "rho_g": float(fit["genetic_correlation"]),
        "rho_e": float(fit["residual_correlation"]),
        "loglik": float(fit["loglik"]),
        "scaled_gradient": float(fit["scaled_gradient"]),
        "converged": bool(fit["converged"]),
    }


def write_solar_inputs(directory: Path, case: MixedCase) -> None:
    """Write exact generated mixed-trait inputs to a temporary directory.

    Args:
        directory: Fresh native-SOLAR working directory.
        case: Generated case whose exact inputs are bound by fixture hashes.
    """
    (directory / "ped.csv").write_text(case.pedigree_csv, encoding="utf-8")
    (directory / "phen.csv").write_text(case.phenotype_csv, encoding="utf-8")
    (directory / "run.tcl").write_text(RUN, encoding="utf-8")
    """Wrote only deterministic participant-free inputs and the fixed command."""


def fit_solar(case: MixedCase) -> dict[str, object]:
    """Fit the generated mixed-trait case with qualified native SOLAR.

    Args:
        case: Fixed participant-free mixed bivariate case.

    Returns:
        Raw directly reported native parameters and model identity.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory)
        """Isolated all generated native inputs and outputs outside the workspace."""

        write_solar_inputs(directory, case)
        completed: subprocess.CompletedProcess[str] = run_solar(directory)
        """Ran the qualified executable on the fixed mixed-trait command."""

        report: str = ""
        """Started the compact collection of native reports."""

        for report_name in ("polygenic.out", "polygenic.logs.out", "null0.out"):
            report_path: Path = directory / "out" / report_name
            """Selected one possible native mixed-trait report."""

            if report_path.is_file():
                report += report_path.read_text(encoding="utf-8")
                """Included the existing native report in result parsing."""
        """Combined primary and secondary native output without retaining raw text."""

        if not report:
            raise SolarReferenceError(
                "SOLAR produced no mixed result: "
                + (completed.stdout + completed.stderr)[-2000:]
            )
        check_sex_survived(directory)
        used_discrete_model: bool = bool(re.search(r"[Dd]iscrete", report))
        """Recorded whether native SOLAR announced its discrete trait machinery."""

        result: dict[str, object] = {
            "people": case.people,
            "affected": int(case.affected.sum()),
            "h2_affected": required_number(
                report,
                r"H2r\(affected\) is\s+([0-9.eE+-]+)",
                "H2r(affected)",
            ),
            "h2_quantity": required_number(
                report,
                r"H2r\(quantity\) is\s+([0-9.eE+-]+)",
                "H2r(quantity)",
            ),
            "rho_g": required_number(report, r"RhoG is\s+(-?[0-9.eE+-]+)", "RhoG"),
            "rho_e": required_number(report, r"RhoE is\s+(-?[0-9.eE+-]+)", "RhoE"),
            "used_the_discrete_model": used_discrete_model,
        }
        """Retained compact raw native fields and generated-case summaries."""

    return result


def compare_outputs(
    asterism_outputs: dict[str, object], external_outputs: dict[str, object]
) -> dict[str, object]:
    """Apply the prewritten direct mixed-trait acceptance rule.

    Args:
        asterism_outputs: Newly recomputed Asterism result.
        external_outputs: Live or frozen raw native-SOLAR result.

    Returns:
        Detailed pass/fail record for four directly reported parameters.
    """
    failures: list[str] = []
    """Collected model-identity and numerical disagreements."""

    if external_outputs.get("used_the_discrete_model") is not True:
        failures.append("SOLAR did not use its discrete trait model")

    differences: dict[str, float] = {}
    """Retained every direct absolute parameter difference."""

    for field in ("h2_affected", "h2_quantity", "rho_g", "rho_e"):
        difference: float = abs(
            float(asterism_outputs[field]) - float(external_outputs[field])
        )
        """Computed one direct liability-scale or correlation disagreement."""

        differences[field] = difference
        """Stored the derived difference beside its native parameter name."""

        if difference > TOLERANCE:
            failures.append(f"{field} differs by {difference:.4f}, over {TOLERANCE}")
    """Applied the unchanged historical threshold to all direct parameters."""

    return {"passed": not failures, "failures": failures, "differences": differences}


def reference_record(mode: str, fixture_path: Path | None) -> dict[str, object]:
    """Build one strict mixed-trait refresh or portable verification record.

    Args:
        mode: Either ``refresh`` or ``verify``.
        fixture_path: Frozen envelope required only for verification.

    Returns:
        Machine record consumed by the strict fixture driver.

    Raises:
        SolarReferenceError: If mode, fixture provenance or tool identity fails.
    """
    case: MixedCase = generate_case()
    """Regenerated the fixed participant-free mixed-trait inputs."""

    identities: list[dict[str, str]] = reference_input_identities(case)
    """Hashed every exact generated input and shared adapter mechanics."""

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
    """Run the historical live native-SOLAR mixed-trait comparison."""
    record: dict[str, object] = reference_record("refresh", None)
    """Performed the same qualified comparison used to create a fixture."""

    return emit_reference_record(record)


def entrypoint() -> int:
    """Dispatch historical live use or the strict two-mode fixture contract."""
    arguments: argparse.Namespace = parse_reference_arguments(
        __doc__ or "native SOLAR mixed-trait comparison"
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
