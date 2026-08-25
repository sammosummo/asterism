"""The two-trait REML fit against R's `regress`.

This compares Asterism's two-trait REML fit with an equivalent construction in
R `regress`. SOLAR covers the separate ML comparison because its `polygenic`
command does not fit REML.

**`regress` has no bivariate response, and does not need one.** The two-trait
model is linear in six variance components — three for the genetic covariance
and three for the residual — so handing `regress` six structure matrices over
the stacked response fits exactly the same model:

    V = a₁₁·A⊗e₁₁ + a₂₂·A⊗e₂₂ + a₁₂·A⊗e₁₂ + r₁₁·I⊗e₁₁ + r₂₂·I⊗e₂₂ + r₁₂·I⊗e₁₂

That makes it a genuinely independent route rather than a second spelling of the
same one: it carries covariances where Asterism carries correlations, its
constraint set is different, and it uses its own optimiser. Where the two agree,
the agreement means something.

Run with:

    uv run --no-project python checks/bivariate_against_r.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import asterism
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_reference_adapter import (
    CONTRACT_VERSION,
    ReferenceAdapterError,
    array_identity,
    emit_reference_record,
    json_identity,
    load_fixture,
    parse_reference_arguments,
    r_tool_identity,
    require_r_versions,
    support_source_identity,
)

# `regress` reports variance components; the comparison is on the quantities
# anybody reports, which are the ratios.
# The two routes agree to about 2e-8 on this problem, so a tolerance of 1e-4
# would pass through a regression four orders of magnitude larger than
# anything ever seen here. This is set from the measured agreement with room
# for optimiser drift, not from what looks safe.
TOLERANCE: float = 1e-6
"""Fixed the maximum absolute disagreement in reported quantities."""

CHECK_ID: str = "bivariate_against_r"
"""Matched the bivariate manifest check identifier exactly."""

REFERENCE_ACCEPTANCE: dict[str, float | bool] = {
    "maximum_absolute_reported_quantity_difference": TOLERANCE,
    "asterism_must_converge": True,
    "maximum_asterism_scaled_gradient": 1e-7,
}
"""Prewrote the four-quantity agreement and candidate convergence rules."""

QUALIFIED_R_VERSION: str = "4.5.2"
"""Named the R release on which the independent output is qualified."""

QUALIFIED_PACKAGES: dict[str, str] = {"regress": "1.3.22"}
"""Named the exact independent bivariate REML engine used for refresh."""

R_SCRIPT: str = """
suppressMessages(library(regress))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(file.path(args[1], "data.csv"))
n <- nrow(d)
load(file.path(args[1], "structures.RData"))
# Start from something feasible. `regress` otherwise begins with equal weight
# on every component, and the two cross-trait matrices have zero diagonals and
# are indefinite alone, so that starting point is singular.
start <- c(0.5, 0.5, 0.1, 0.5, 0.5, 0.1)
fit <- regress(y ~ 0 + t1 + t2, ~ Va11 + Va22 + Va12 + Vr11 + Vr22 + Vr12,
               identity = FALSE, data = d, start = start,
               tol = 1e-10, maxcyc = 500)
s <- fit$sigma
cat(sprintf('{"a11": %.17g, "a22": %.17g, "a12": %.17g, "r11": %.17g, "r22": %.17g, "r12": %.17g, "llik": %.17g}',
            s[["Va11"]], s[["Va22"]], s[["Va12"]], s[["Vr11"]], s[["Vr22"]], s[["Vr12"]], fit$llik))
"""
"""Defined the fixed independent regress calculation run during live refresh."""


def structures(
    relationship: np.ndarray, observed: np.ndarray
) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    """Construct six component matrices over observed person-trait rows.

    Args:
        relationship: Additive relationship matrix in participant order.
        observed: Boolean person-by-trait observation mask.

    Returns:
        Observed row identities and six independent component matrices.
    """
    rows: list[tuple[int, int]] = [
        (p, t) for p in range(len(observed)) for t in (0, 1) if observed[p, t]
    ]
    """Enumerated observed person-trait rows in interleaved order."""

    size: int = len(rows)
    """Counted observations included in the stacked independent model."""

    people: np.ndarray = np.array([p for p, _ in rows])
    """Extracted participant indices for every observed row."""

    traits: np.ndarray = np.array([t for _, t in rows])
    """Extracted trait indices for every observed row."""

    a: np.ndarray = relationship[np.ix_(people, people)]
    """Expanded the participant relationship matrix over observed trait rows."""

    same_person: np.ndarray = people[:, None] == people[None, :]
    """Marked row pairs belonging to the same participant."""

    first: np.ndarray = (traits[:, None] == 0) & (traits[None, :] == 0)
    """Marked covariance entries within the first trait."""

    second: np.ndarray = (traits[:, None] == 1) & (traits[None, :] == 1)
    """Marked covariance entries within the second trait."""

    cross: np.ndarray = traits[:, None] != traits[None, :]
    """Marked cross-trait covariance entries."""

    return rows, {
        "Va11": a * first,
        "Va22": a * second,
        "Va12": a * cross,
        "Vr11": np.eye(size) * first,
        "Vr22": np.eye(size) * second,
        "Vr12": same_person * cross * 1.0,
    }


def fit_in_r(
    directory: Path,
    matrices: dict[str, np.ndarray],
    y: np.ndarray,
    traits: np.ndarray,
) -> dict[str, Any]:
    """Fit the stacked response through independent R regress.

    Args:
        directory: Temporary directory receiving R inputs and scripts.
        matrices: Six observed-row component matrices.
        y: Stacked observed response vector.
        traits: Trait index for each stacked response row.

    Returns:
        Parsed independent component, likelihood and fixed-effect quantities.

    Raises:
        SystemExit: If the independent R fit exits unsuccessfully.
    """
    lines: list[str] = ["y,t1,t2"]
    """Initialised the stacked response CSV with trait-indicator columns."""

    for value, t in zip(y, traits, strict=True):
        lines.append(f"{value:.17g},{1 if t == 0 else 0},{1 if t == 1 else 0}")
    (directory / "data.csv").write_text("\n".join(lines) + "\n")

    # Write the matrices where R can load them without a text round trip.
    save: list[str] = ["setwd(commandArgs(trailingOnly = TRUE)[1])"]
    """Initialised the R script that serialises component matrices exactly once."""

    for name, matrix in matrices.items():
        np.savetxt(directory / f"{name}.txt", matrix)
        save.append(f'{name} <- as.matrix(read.table("{name}.txt"))')
        save.append(f"dimnames({name}) <- NULL")
    save.append("save(" + ", ".join(matrices) + ', file = "structures.RData")')
    (directory / "save.R").write_text("\n".join(save) + "\n")
    subprocess.run(
        ["Rscript", "--vanilla", str(directory / "save.R"), str(directory)],
        capture_output=True,
        text=True,
        check=True,
    )

    (directory / "fit.R").write_text(R_SCRIPT)
    finished: subprocess.CompletedProcess[str] = subprocess.run(
        ["Rscript", "--vanilla", str(directory / "fit.R"), str(directory)],
        capture_output=True,
        text=True,
        check=False,
    )
    """Ran the fixed independent regress calculation in its temporary directory."""

    if finished.returncode != 0:
        raise SystemExit(f"R failed:\n{finished.stdout}\n{finished.stderr}")
    return json.loads(finished.stdout.strip().splitlines()[-1])


def reference_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    dict[str, np.ndarray],
    dict[str, float],
]:
    """Regenerate the exact participant-free two-trait comparison.

    Returns:
        Relationship, observed mask, observed design, response, row map,
        independent structure matrices and simulation truth.
    """
    families: int = 60
    """Retained the historical sixty independent sibling families."""

    per_family: int = 6
    """Retained the six-person largest-family size named in the manifest."""

    people: int = families * per_family
    """Calculated the fixed participant-free sample size of 360 people."""

    relationship: np.ndarray = np.zeros((people, people))
    """Started the block-diagonal full-sibling relationship matrix."""

    for family in range(families):
        block: slice = slice(family * per_family, (family + 1) * per_family)
        """Selected one independent six-sibling family block."""

        relationship[block, block] = 0.5
        """Assigned full-sibling covariance throughout one family block."""
    np.fill_diagonal(relationship, 1.0)
    truth: dict[str, float] = {"h1": 0.6, "h2": 0.35, "rg": 0.55, "re": 0.25}
    """Fixed both heritabilities and both cross-trait correlations."""

    additive: np.ndarray = np.array(
        [
            [truth["h1"], truth["rg"] * np.sqrt(truth["h1"] * truth["h2"])],
            [truth["rg"] * np.sqrt(truth["h1"] * truth["h2"]), truth["h2"]],
        ]
    )
    """Constructed the two-by-two additive trait covariance."""

    residual: np.ndarray = np.array(
        [
            [
                1 - truth["h1"],
                truth["re"] * np.sqrt((1 - truth["h1"]) * (1 - truth["h2"])),
            ],
            [
                truth["re"] * np.sqrt((1 - truth["h1"]) * (1 - truth["h2"])),
                1 - truth["h2"],
            ],
        ]
    )
    """Constructed the two-by-two residual trait covariance."""

    generator: np.random.Generator = np.random.default_rng(31)
    """Selected the prewritten deterministic phenotype stream."""

    covariance: np.ndarray = np.kron(additive, relationship) + np.kron(
        residual, np.eye(people)
    )
    """Assembled the trait-major two-trait covariance used only for simulation."""

    draw: np.ndarray = np.linalg.cholesky(
        covariance + 1e-10 * np.eye(2 * people)
    ) @ generator.standard_normal(2 * people)
    """Generated the complete participant-free two-trait outcome."""

    observed: np.ndarray = np.array([[True, index % 9 != 0] for index in range(people)])
    """Applied the prewritten unbalanced second-trait missingness pattern."""

    response_full: np.ndarray = np.zeros(2 * people)
    """Allocated the person-major public response layout."""

    response_full[0::2] = draw[:people]
    """Placed the complete first-trait draw into person-major rows."""

    response_full[1::2] = draw[people:]
    """Placed the complete second-trait draw into person-major rows."""

    design_full: np.ndarray = np.zeros((2 * people, 2))
    """Allocated trait-specific intercept columns in person-major order."""

    design_full[0::2, 0] = 1.0
    """Activated the first-trait intercept on first-trait rows."""

    design_full[1::2, 1] = 1.0
    """Activated the second-trait intercept on second-trait rows."""

    # asterism-style: allow missing-following-doc -- the row map and matrices form one observed-data transform
    rows, matrices = structures(relationship, observed)
    index: list[int] = [person * 2 + trait for person, trait in rows]
    """Selected only observed person-trait rows for both implementations."""

    return (
        relationship,
        observed,
        np.ascontiguousarray(design_full[index]),
        np.ascontiguousarray(response_full[index]),
        rows,
        matrices,
        truth,
    )


def reference_input_identities(
    relationship: np.ndarray,
    observed: np.ndarray,
    observed_design: np.ndarray,
    response: np.ndarray,
    rows: list[tuple[int, int]],
    truth: dict[str, float],
) -> list[dict[str, str]]:
    """Return canonical hashes for every generated two-trait input.

    Args:
        relationship: Additive relationship matrix.
        observed: Person-by-trait observation mask.
        observed_design: Fixed-effect rows supplied to the fit.
        response: Observed person-trait outcome vector.
        rows: Observed row map shared with regress structures.
        truth: Simulation covariance parameters.

    Returns:
        Ordered strict input identities.
    """
    return [
        support_source_identity(),
        array_identity("bivariate relationship matrix", relationship),
        array_identity("bivariate observed mask", observed),
        array_identity("bivariate observed fixed-effect design", observed_design),
        array_identity("bivariate observed response", response),
        json_identity("bivariate observed row map", rows),
        json_identity("bivariate simulation truth", truth),
    ]


def fit_asterism_reference(
    relationship: np.ndarray,
    observed: np.ndarray,
    observed_design: np.ndarray,
    response: np.ndarray,
) -> dict[str, Any]:
    """Fit the frozen case through Asterism's public bivariate API.

    Args:
        relationship: Additive relationship matrix.
        observed: Person-by-trait observation mask.
        observed_design: Fixed-effect rows supplied to the fit.
        response: Observed person-trait outcome vector.

    Returns:
        Public fit quantities required by the independent comparison.
    """
    model: asterism.BivariateModel = asterism.BivariateModel(
        np.ascontiguousarray(relationship),
        observed.tolist(),
        observed_design,
    )
    """Constructed the documented public model over observed person-trait rows."""

    fit: dict[str, Any] = model.fit(response, reml=True)
    """Recomputed the candidate REML estimate from canonical participant-free input."""

    return {
        "h2_trait_a": float(fit["h2_first"]),
        "h2_trait_b": float(fit["h2_second"]),
        "rho_g": float(fit["rho_g"]),
        "rho_e": float(fit["rho_e"]),
        "scaled_gradient": float(fit["scaled_gradient"]),
        "converged": bool(fit["converged"]),
    }


def fit_regress_reference(
    matrices: dict[str, np.ndarray],
    response: np.ndarray,
    rows: list[tuple[int, int]],
) -> dict[str, Any]:
    """Run live R regress and retain only raw independent components.

    Args:
        matrices: Six observed-row variance-component structures.
        response: Observed person-trait outcome vector.
        rows: Observed row map carrying the trait indicator.

    Returns:
        Raw regress variance components and likelihood.
    """
    traits: np.ndarray = np.array([trait for _, trait in rows])
    """Presented the same observed trait row ordering to the independent engine."""

    with tempfile.TemporaryDirectory() as temporary:
        result: dict[str, Any] = fit_in_r(Path(temporary), matrices, response, traits)
        """Ran independently authored REML code only during live refresh."""

    return result


def compare_reference(
    ours: dict[str, Any], external_outputs: object
) -> tuple[bool, dict[str, object]]:
    """Apply the immutable bivariate R agreement rule.

    Args:
        ours: Fresh public Asterism result.
        external_outputs: Frozen raw regress variance components.

    Returns:
        Pass decision and all supported-quantity differences.

    Raises:
        ReferenceAdapterError: If the frozen independent output is incomplete.
    """
    required: tuple[str, ...] = ("a11", "a22", "a12", "r11", "r22", "r12")
    """Named the six raw components required to derive four supported ratios."""

    if not isinstance(external_outputs, dict) or any(
        field not in external_outputs for field in required
    ):
        raise ReferenceAdapterError("bivariate regress external_outputs are incomplete")
    a11: float = float(external_outputs["a11"])
    """Read the frozen first-trait additive variance."""

    a22: float = float(external_outputs["a22"])
    """Read the frozen second-trait additive variance."""

    r11: float = float(external_outputs["r11"])
    """Read the frozen first-trait residual variance."""

    r22: float = float(external_outputs["r22"])
    """Read the frozen second-trait residual variance."""

    external_reported: dict[str, float] = {
        "h2_trait_a": a11 / (a11 + r11),
        "h2_trait_b": a22 / (a22 + r22),
        "rho_g": float(external_outputs["a12"]) / np.sqrt(a11 * a22),
        "rho_e": float(external_outputs["r12"]) / np.sqrt(r11 * r22),
    }
    """Converted independent covariance components to Asterism's public quantities."""

    differences: dict[str, float] = {
        quantity: abs(float(ours[quantity]) - value)
        for quantity, value in external_reported.items()
    }
    """Compared every quantity named by the bivariate release contract."""

    failures: list[str] = []
    """Collected convergence, gradient and independent-agreement failures."""

    if REFERENCE_ACCEPTANCE["asterism_must_converge"] and not ours["converged"]:
        failures.append("Asterism did not declare convergence")
    if float(ours["scaled_gradient"]) >= float(
        REFERENCE_ACCEPTANCE["maximum_asterism_scaled_gradient"]
    ):
        failures.append("Asterism scaled gradient exceeds acceptance")
    if max(differences.values()) >= float(
        REFERENCE_ACCEPTANCE["maximum_absolute_reported_quantity_difference"]
    ):
        failures.append("a reported bivariate quantity exceeds agreement tolerance")
    diagnostics: dict[str, object] = {
        "asterism": ours,
        "regress_reported": external_reported,
        "absolute_differences": differences,
        "failures": failures,
    }
    """Returned the exact quantities behind the binary gate decision."""

    return not failures, diagnostics


def reference_record(mode: str, fixture_path: Path) -> dict[str, Any]:
    """Build one live-refresh or portable-verification adapter response.

    Args:
        mode: Strict ``refresh`` or ``verify`` operation.
        fixture_path: Frozen envelope supplied by the external fixture driver.

    Returns:
        Version-one adapter response consumed by the strict driver.
    """
    # asterism-style: allow missing-following-doc -- tuple unpacking names one regenerated problem
    (
        relationship,
        observed,
        observed_design,
        response,
        rows,
        matrices,
        truth,
    ) = reference_problem()
    identities: list[dict[str, str]] = reference_input_identities(
        relationship, observed, observed_design, response, rows, truth
    )
    """Recomputed canonical participant-free inputs in both adapter modes."""

    ours: dict[str, Any] = fit_asterism_reference(
        relationship, observed, observed_design, response
    )
    """Recomputed Asterism through the public BivariateModel API in both modes."""

    tools: list[dict[str, object]] | None = None
    """Held live independent software identity only during explicit refresh."""

    if mode == "refresh":
        tools = r_tool_identity(tuple(QUALIFIED_PACKAGES))
        """Loaded and identified the qualified live regress implementation."""

        require_r_versions(
            tools,
            r_version=QUALIFIED_R_VERSION,
            packages=QUALIFIED_PACKAGES,
        )
        external_outputs: object = fit_regress_reference(matrices, response, rows)
        """Ran R only during explicit live refresh and retained raw components."""
    else:
        fixture: dict[str, Any] = load_fixture(fixture_path, CHECK_ID)
        """Loaded the driver-validated frozen independent envelope without R."""

        if fixture.get("acceptance") != REFERENCE_ACCEPTANCE:
            raise ReferenceAdapterError(
                "bivariate_against_r fixture acceptance was altered"
            )
        external_outputs = fixture.get("external_outputs")
        """Read only the frozen independent values needed by portable comparison."""

    # asterism-style: allow missing-following-doc -- paired decision and diagnostics are one result
    passed, diagnostics = compare_reference(ours, external_outputs)
    result: dict[str, Any] = {
        "reference_fixture_contract": CONTRACT_VERSION,
        "check_id": CHECK_ID,
        "passed": passed,
        "external_tool_invoked": mode == "refresh",
        "asterism_recomputed": True,
        "input_identities": identities,
        "comparison": diagnostics,
    }
    """Declared external execution and public recomputation explicitly."""

    if mode == "refresh":
        result.update(
            {
                "tools": tools,
                "external_outputs": external_outputs,
                "acceptance": REFERENCE_ACCEPTANCE,
            }
        )
    return result


def entrypoint() -> int:
    """Preserve the live no-argument check and expose strict fixture modes."""
    arguments: Any = parse_reference_arguments(__doc__)
    """Selected historical live output or the machine-only adapter contract."""

    if arguments.reference_mode is None:
        return main()
    try:
        record: dict[str, Any] = reference_record(
            arguments.reference_mode, arguments.reference_fixture
        )
        """Ran exactly one strict mode with paired arguments guaranteed by parsing."""
    except (ReferenceAdapterError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return emit_reference_record(record)


def main() -> int:
    """Run the historical live bivariate R comparison and print its receipt."""
    if shutil.which("Rscript") is None:
        raise SystemExit(
            "Rscript is not on the path. This check fails rather than skips."
        )

    families: int = 60
    """Fixed the number of independent sibling families."""

    per_family: int = 6
    """Fixed the number of siblings in each family."""

    n: int = families * per_family
    """Computed the fixed simulated roster size."""

    relationship: np.ndarray = np.zeros((n, n))
    """Initialised the block-diagonal sibling relationship matrix."""

    for f in range(families):
        block: slice = slice(f * per_family, (f + 1) * per_family)
        """Located one independent sibling-family block."""

        relationship[block, block] = 0.5
        """Assigned full-sibling covariance throughout one family block."""
    np.fill_diagonal(relationship, 1.0)

    truth: dict[str, float] = {"h1": 0.6, "h2": 0.35, "rg": 0.55, "re": 0.25}
    """Fixed both heritabilities and both cross-trait correlations."""

    sa: np.ndarray = np.array(
        [
            [truth["h1"], truth["rg"] * np.sqrt(truth["h1"] * truth["h2"])],
            [truth["rg"] * np.sqrt(truth["h1"] * truth["h2"]), truth["h2"]],
        ]
    )
    """Constructed the simulated additive covariance matrix."""

    se: np.ndarray = np.array(
        [
            [
                1 - truth["h1"],
                truth["re"] * np.sqrt((1 - truth["h1"]) * (1 - truth["h2"])),
            ],
            [
                truth["re"] * np.sqrt((1 - truth["h1"]) * (1 - truth["h2"])),
                1 - truth["h2"],
            ],
        ]
    )
    """Constructed the simulated residual covariance matrix."""

    rng: np.random.Generator = np.random.default_rng(31)
    """Created the fixed random-number generator for the comparison response."""

    full: np.ndarray = np.kron(sa, relationship) + np.kron(se, np.eye(n))
    """Constructed the full stacked two-trait observation covariance."""

    draw: np.ndarray = np.linalg.cholesky(
        full + 1e-10 * np.eye(2 * n)
    ) @ rng.standard_normal(2 * n)
    """Drew one deterministic stacked response from the generating covariance."""

    # Unbalanced: every ninth person lacks the second trait.
    observed: np.ndarray = np.array([[True, i % 9 != 0] for i in range(n)])
    """Applied the fixed unbalanced second-trait missingness pattern."""

    y_full: np.ndarray = np.zeros(2 * n)
    """Initialised the interleaved two-trait response vector."""

    y_full[0::2] = draw[:n]
    """Placed first-trait values in the even response positions."""

    y_full[1::2] = draw[n:]
    """Placed second-trait values in the odd response positions."""

    design_full: np.ndarray = np.zeros((2 * n, 2))
    """Initialised separate intercept columns for the two traits."""

    design_full[0::2, 0] = 1.0
    """Activated the first-trait intercept on even rows."""

    design_full[1::2, 1] = 1.0
    """Activated the second-trait intercept on odd rows."""

    rows, matrices = structures(relationship, observed)
    """Built independent component matrices over observed person-trait rows."""

    index: list[int] = [p * 2 + t for p, t in rows]
    """Mapped observed person-trait pairs to interleaved response rows."""

    model: asterism.BivariateModel = asterism.BivariateModel(
        np.ascontiguousarray(relationship),
        observed.tolist(),
        np.ascontiguousarray(design_full[index]),
    )
    """Built the documented bivariate model over the observed person-trait rows."""

    fit: dict[str, Any] = model.fit(np.ascontiguousarray(y_full[index]), reml=True)
    """Fitted the REML model through the public named-record interface."""

    ours: dict[str, Any] = {
        "total_variance_a": fit["total_variance"][0],
        "total_variance_b": fit["total_variance"][1],
        "h2_trait_a": fit["h2_first"],
        "h2_trait_b": fit["h2_second"],
        "rho_g": fit["rho_g"],
        "rho_e": fit["rho_e"],
        "loglik": fit["loglik"],
        "scaled_gradient": fit["scaled_gradient"],
        "converged": fit["converged"],
    }
    """Mapped the public fit fields onto the comparison's historical output names."""
    with tempfile.TemporaryDirectory() as temporary:
        theirs: dict[str, Any] = fit_in_r(
            Path(temporary), matrices, y_full[index], np.array([t for _, t in rows])
        )
        """Fitted the same stacked response through independent R regress."""

    # `regress` gives covariances; the ratios are what anybody reports.
    their_h1: float = theirs["a11"] / (theirs["a11"] + theirs["r11"])
    """Converted independent first-trait components to heritability."""

    their_h2: float = theirs["a22"] / (theirs["a22"] + theirs["r22"])
    """Converted independent second-trait components to heritability."""

    their_rg: float = theirs["a12"] / np.sqrt(theirs["a11"] * theirs["a22"])
    """Converted independent additive covariance to genetic correlation."""

    their_re: float = theirs["r12"] / np.sqrt(theirs["r11"] * theirs["r22"])
    """Converted independent residual covariance to residual correlation."""

    print(
        f"Two traits, REML, n = {n}, "
        f"{observed[:, 0].sum()} with the first and {observed[:, 1].sum()} with the second.\n"
    )
    print(
        f"{'':<10} {'asterism':>12} {'R regress':>12} {'difference':>12} {'truth':>8}"
    )
    failures: list[str] = []
    """Initialised the scientific comparison failure messages."""

    if not ours["converged"]:
        failures.append("Asterism did not declare convergence")
    if not ours["scaled_gradient"] < 1e-7:
        failures.append(
            f"Asterism scaled projected gradient was {ours['scaled_gradient']:.3e}"
        )
    comparisons: tuple[tuple[str, float, float, float], ...] = (
        ("h2 first", ours["h2_trait_a"], their_h1, truth["h1"]),
        ("h2 second", ours["h2_trait_b"], their_h2, truth["h2"]),
        ("rho_g", ours["rho_g"], their_rg, truth["rg"]),
        ("rho_e", ours["rho_e"], their_re, truth["re"]),
    )
    """Paired each Asterism quantity with R and generating truth."""

    differences: dict[str, float] = {}
    """Initialised absolute disagreements for the evidence receipt."""

    for label, ours_value, theirs_value, true_value in comparisons:
        difference: float = abs(ours_value - theirs_value)
        """Computed the absolute disagreement for this reported quantity."""

        differences[label] = difference
        """Recorded the quantity disagreement under its table label."""

        print(
            f"{label:<10} {ours_value:>12.7f} {theirs_value:>12.7f} "
            f"{difference:>12.2e} {true_value:>8.2f}"
        )
        if not difference < TOLERANCE:
            failures.append(f"{label} differs by {difference:.3e}")

    if failures:
        print("\nDISAGREEMENT:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nAgreement within tolerance on every reported quantity.")
    print(f"Asterism scaled projected gradient: {ours['scaled_gradient']:.3e}.")
    print("Different parameterisation, different optimiser, same answer.")
    print("The independently implemented REML calculations agree.")
    print(
        json.dumps(
            {
                "seed": 31,
                "families": families,
                "people_per_family": per_family,
                "people": n,
                "observations_by_trait": [
                    int(observed[:, 0].sum()),
                    int(observed[:, 1].sum()),
                ],
                "estimator": "reml",
                "truth": truth,
                "asterism": ours,
                "regress": {
                    **theirs,
                    "h2_trait_a": their_h1,
                    "h2_trait_b": their_h2,
                    "rho_g": their_rg,
                    "rho_e": their_re,
                },
                "absolute_differences": differences,
                "comparison_tolerance": TOLERANCE,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(entrypoint())
