"""Compare Asterism's REML fit against R's `regress`.

R `regress` independently implements the same published REML algebra with a
different optimiser. This check compares estimates and the identifiable
log-likelihood offset between the two implementations.

Run with:

    uv run --no-project python checks/against_r.py

It needs R with the `regress` package installed, and it fails rather than skips
when R is missing: a check that skips is a check that passes.
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

# The estimates are optimiser-bounded, so the tolerance is 1e-6 relative. The
# log-likelihood is compared as a difference rather than a level for the reason
# given below.
ESTIMATE_TOLERANCE: float = 1e-6
"""Fixed the maximum relative estimate disagreement."""

LOGLIK_DIFFERENCE_TOLERANCE: float = 1e-8
"""Fixed the maximum drift in the identifiable likelihood offset."""

CHECK_ID: str = "against_r"
"""Matched the one-trait manifest check identifier exactly."""

REFERENCE_ACCEPTANCE: dict[str, float] = {
    "maximum_relative_estimate_difference": ESTIMATE_TOLERANCE,
    "maximum_loglik_offset_drift": LOGLIK_DIFFERENCE_TOLERANCE,
    "maximum_unexplained_loglik_offset": 1e-6,
}
"""Prewrote every threshold before the live independent outputs were refreshed."""

QUALIFIED_R_VERSION: str = "4.5.2"
"""Named the R release on which the frozen independent output is qualified."""

QUALIFIED_PACKAGES: dict[str, str] = {"regress": "1.3.22"}
"""Named the exact independent REML implementation used for refresh."""

R_SCRIPT: str = """
suppressMessages(library(regress))
args <- commandArgs(trailingOnly = TRUE)
k <- as.matrix(read.csv(file.path(args[1], "k.csv"), header = FALSE))
d <- read.csv(file.path(args[1], "data.csv"))
covariates <- setdiff(names(d), "y")
formula <- if (length(covariates) == 0) y ~ 1 else
    as.formula(paste("y ~", paste(covariates, collapse = " + ")))
fit <- regress(formula, ~k, identity = TRUE, data = d, tol = 1e-10)
cat(toJSON <- sprintf(
    '{"sigma_k": %.17g, "sigma_e": %.17g, "llik": %.17g, "beta": [%s]}',
    fit$sigma[["k"]], fit$sigma[["In"]], fit$llik,
    paste(sprintf("%.17g", fit$beta), collapse = ",")
))
"""
"""Defined the fixed independent regress calculation run during live refresh."""


def extended_family() -> list[tuple[int | None, int | None]]:
    """Return the three-generation family used by the coverage check.

    Returns:
        Parent-index pairs in deterministic family roster order.
    """
    family: list[tuple[int | None, int | None]] = [(None, None), (None, None)]
    """Started the pedigree with two unrelated founders."""

    family += [(0, 1)] * 3
    """Added three siblings descended from the founder pair."""

    family += [(None, None)] * 3
    """Added unrelated partners for the second generation."""

    for child in range(3):
        family += [(2 + child, 5 + child)] * 2
        """Added two third-generation children for this parental pair."""
    return family


def kinship(people: list[tuple[int | None, int | None]]) -> np.ndarray:
    """Construct the kinship matrix recursively from parent indices.

    Args:
        people: Mother and father indices in topological pedigree order.

    Returns:
        Symmetric kinship coefficients in the supplied roster order.
    """
    n: int = len(people)
    """Counted people in the topologically ordered pedigree."""

    phi: np.ndarray = np.zeros((n, n))
    """Initialised the symmetric kinship matrix."""

    for i in range(n):
        for j in range(i + 1):
            mother, father = people[i]
            """Read this person's parent indices from the ordered pedigree."""

            if i == j:
                value: float = (
                    0.5 if mother is None else 0.5 * (1.0 + phi[mother, father])
                )
                """Computed the diagonal kinship coefficient for this person."""
            else:
                value = (
                    0.0 if mother is None else 0.5 * (phi[mother, j] + phi[father, j])
                )
                """Computed the off-diagonal kinship coefficient for this pair."""
            phi[i, j] = phi[j, i] = value
            """Stored the computed coefficient symmetrically."""
    return phi


def roster(families: int) -> np.ndarray:
    """Repeat the extended family into a block-diagonal relationship roster.

    Args:
        families: Number of independent family blocks to construct.

    Returns:
        Twice-kinship additive relationship matrix for all families.
    """
    family: list[tuple[int | None, int | None]] = extended_family()
    """Loaded the fixed three-generation family structure."""

    size: int = len(family)
    """Recorded the number of people in each independent family."""

    phi: np.ndarray = kinship(family)
    """Constructed the single-family kinship coefficients."""

    k: np.ndarray = np.zeros((families * size, families * size))
    """Initialised the block-diagonal additive relationship matrix."""

    for f in range(families):
        o: int = f * size
        """Located this family's starting row and column."""

        k[o : o + size, o : o + size] = 2.0 * phi
        """Inserted the twice-kinship family block on the matrix diagonal."""
    return k


def design(n: int, block: int) -> np.ndarray:
    """Intercept, age, age squared, sex, and the two age-by-sex products —
    `age_years^1,2#sex`, which is what 1,194 of the lab's 1,246 recorded SOLAR
    runs use. An intercept-only comparison would leave the restricted
    likelihood's determinant term unexercised, which is the term most likely to
    differ between two implementations.

    Args:
        n: Total number of simulated people.
        block: Number of people in each repeated family structure.

    Returns:
        Six-column fixed-effect design in relationship-matrix order.
    """
    x: np.ndarray = np.zeros((n, 6))
    """Initialised the six-column fixed-effect design."""

    for row in range(n):
        within: int = row % block
        """Located the person within the repeated family structure."""

        decade: float = 7.0 if within < 2 else (4.5 if within < 8 else 2.0)
        """Assigned an age band from generation position."""

        age: float = (decade - 4.5) / 2.0 + ((row % 7) - 3.0) / 10.0
        """Constructed centred age with deterministic within-band variation."""

        sex: float = float(row % 2 == 0)
        """Assigned the alternating binary sex covariate."""

        x[row] = [1.0, age, age * age, sex, age * sex, age * age * sex]
        """Stored intercept, polynomial age, sex and interaction columns."""
    return x


def simulate(
    k: np.ndarray, x: np.ndarray, beta: np.ndarray, h2: float, seed: int
) -> np.ndarray:
    """Simulate one continuous trait under the one-component model.

    Args:
        k: Additive relationship matrix.
        x: Fixed-effect design matrix.
        beta: Generating fixed-effect coefficients.
        h2: Generating additive variance share.
        seed: Deterministic random seed for this case.

    Returns:
        Continuous response in relationship-matrix order.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic random-number generator for this case."""

    n: int = k.shape[0]
    """Read the simulated roster size from the relationship matrix."""

    covariance: np.ndarray = h2 * k + (1.0 - h2) * np.eye(n)
    """Constructed the additive-plus-residual response covariance."""

    factor: np.ndarray = np.linalg.cholesky(covariance)
    """Factorised the response covariance for deterministic simulation."""
    return x @ beta + factor @ rng.standard_normal(n)


def fit_in_r(
    directory: Path, k: np.ndarray, x: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    """Fit one generated response through the independent R implementation.

    Args:
        directory: Temporary directory receiving R inputs and script.
        k: Additive relationship matrix.
        x: Fixed-effect design matrix.
        y: Continuous generated response.

    Returns:
        Parsed independent variance, likelihood and fixed-effect quantities.

    Raises:
        SystemExit: If the independent R process exits unsuccessfully.
    """
    np.savetxt(directory / "k.csv", k, delimiter=",")
    header: str = ",".join(["y"] + [f"c{i}" for i in range(1, x.shape[1])])
    """Constructed the independent data-table header without duplicate intercept."""

    columns: np.ndarray = np.column_stack([y] + [x[:, i] for i in range(1, x.shape[1])])
    """Combined the response and non-intercept covariates for R."""

    np.savetxt(
        directory / "data.csv", columns, delimiter=",", header=header, comments=""
    )
    script: Path = directory / "fit.R"
    """Selected the temporary independent-script path."""

    script.write_text(R_SCRIPT)
    finished: subprocess.CompletedProcess[str] = subprocess.run(
        ["Rscript", "--vanilla", str(script), str(directory)],
        capture_output=True,
        text=True,
        check=False,
    )
    """Ran the fixed independent regress calculation in its temporary directory."""

    if finished.returncode != 0:
        raise SystemExit(f"R failed:\n{finished.stdout}\n{finished.stderr}")
    return json.loads(finished.stdout.strip().splitlines()[-1])


def relative(a: float, b: float) -> float:
    """Return the historical scale-stabilised relative difference.

    Args:
        a: Candidate Asterism value.
        b: Independent comparison value.

    Returns:
        Absolute difference divided by one plus the comparison magnitude.
    """
    return abs(a - b) / (1.0 + abs(b))


def reference_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, int]],
    list[np.ndarray],
]:
    """Regenerate the exact participant-free comparison inputs.

    Returns:
        Relationship, design, fixed-effect truth, cases and phenotypes.
    """
    families: int = 25
    """Retained the historical twenty-five independent extended families."""

    block: int = len(extended_family())
    """Recorded the fourteen-person largest-family size used by the manifest."""

    relationship: np.ndarray = roster(families)
    """Built the additive relationship matrix supplied to both implementations."""

    design_matrix: np.ndarray = design(relationship.shape[0], block)
    """Built the six-column fixed-effect design supplied to both implementations."""

    beta: np.ndarray = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])
    """Fixed the participant-free regression coefficients used by every case."""

    cases: list[tuple[float, int]] = [(0.5, 101), (0.25, 102), (0.75, 103)]
    """Retained the three prewritten heritability and random-seed cells."""

    phenotypes: list[np.ndarray] = [
        simulate(relationship, design_matrix, beta, h2, seed) for h2, seed in cases
    ]
    """Generated the exact continuous outcomes independently of any participant data."""

    return relationship, design_matrix, beta, cases, phenotypes


def reference_input_identities(
    relationship: np.ndarray,
    design_matrix: np.ndarray,
    beta: np.ndarray,
    cases: list[tuple[float, int]],
    phenotypes: list[np.ndarray],
) -> list[dict[str, str]]:
    """Return canonical hashes for every generated comparison input.

    Args:
        relationship: Additive relationship matrix.
        design_matrix: Fixed-effect design.
        beta: Simulation coefficient vector.
        cases: Heritability and seed cells.
        phenotypes: Generated response for each cell.

    Returns:
        Ordered strict input identities.
    """
    identities: list[dict[str, str]] = [
        support_source_identity(),
        array_identity("one-trait relationship matrix", relationship),
        array_identity("one-trait fixed-effect design", design_matrix),
        array_identity("one-trait simulation coefficients", beta),
        json_identity("one-trait simulation cells", cases),
    ]
    """Bound shared mechanics and non-response inputs before the case outcomes."""

    identities.extend(
        array_identity(f"one-trait phenotype h2={h2} seed={seed}", phenotype)
        for (h2, seed), phenotype in zip(cases, phenotypes, strict=True)
    )
    """Bound every generated phenotype separately for actionable stale evidence."""

    return identities


def fit_asterism_reference(
    relationship: np.ndarray,
    design_matrix: np.ndarray,
    cases: list[tuple[float, int]],
    phenotypes: list[np.ndarray],
) -> list[dict[str, Any]]:
    """Fit every frozen case through Asterism's public one-trait API.

    Args:
        relationship: Additive relationship matrix.
        design_matrix: Fixed-effect design.
        cases: Heritability and seed cells.
        phenotypes: Generated response for each cell.

    Returns:
        Public fit quantities required by the independent comparison.
    """
    model: asterism.PreparedModel = asterism.prepare(design_matrix, relationship)
    """Prepared the public model once because every cell has the same design."""

    results: list[dict[str, Any]] = []
    """Collected only quantities named by the prewritten comparison."""

    for (h2, seed), phenotype in zip(cases, phenotypes, strict=True):
        fit: dict[str, Any] = model.fit(phenotype, "reml")
        """Recomputed the candidate implementation from the generated response."""

        results.append(
            {
                "true_h2": h2,
                "seed": seed,
                "h2": float(fit["h2"]),
                "total_variance": float(fit["total_variance"]),
                "loglik": float(fit["loglik"]),
                "beta": [float(value) for value in fit["beta"]],
            }
        )
    return results


def fit_regress_reference(
    relationship: np.ndarray,
    design_matrix: np.ndarray,
    cases: list[tuple[float, int]],
    phenotypes: list[np.ndarray],
) -> dict[str, object]:
    """Run live R regress and retain only independent numerical outputs.

    Args:
        relationship: Additive relationship matrix.
        design_matrix: Fixed-effect design.
        cases: Heritability and seed cells.
        phenotypes: Generated response for each cell.

    Returns:
        Frozen independent results keyed by their deterministic cell identity.
    """
    results: list[dict[str, Any]] = []
    """Collected raw regress variances, likelihood and fixed-effect estimates."""

    for (h2, seed), phenotype in zip(cases, phenotypes, strict=True):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fit: dict[str, Any] = fit_in_r(
                Path(temporary_directory), relationship, design_matrix, phenotype
            )
            """Ran the independently authored REML implementation on this cell."""

        results.append({"true_h2": h2, "seed": seed, **fit})
    return {"cases": results}


def compare_reference(
    ours: list[dict[str, Any]],
    external_outputs: object,
    *,
    sample_size: int,
    fixed_effects: int,
) -> tuple[bool, dict[str, object]]:
    """Apply the immutable one-trait R agreement rule.

    Args:
        ours: Fresh public Asterism results.
        external_outputs: Frozen raw regress outputs.
        sample_size: Number of observations in every case.
        fixed_effects: REML design rank used for the named likelihood constant.

    Returns:
        Pass decision and auditable numerical diagnostics.

    Raises:
        ReferenceAdapterError: If frozen independent outputs are malformed.
    """
    if not isinstance(external_outputs, dict) or not isinstance(
        external_outputs.get("cases"), list
    ):
        raise ReferenceAdapterError("against_r external_outputs need a cases list")
    theirs: list[object] = external_outputs["cases"]
    """Read the frozen case list only after validating its container."""

    if len(theirs) != len(ours) or not all(isinstance(case, dict) for case in theirs):
        raise ReferenceAdapterError("against_r external case count or shape is invalid")
    failures: list[str] = []
    """Collected scientific disagreements without silently dropping a quantity."""

    case_metrics: list[dict[str, float | int]] = []
    """Retained each relative difference and likelihood gap for the receipt."""

    gaps: list[float] = []
    """Collected likelihood offsets whose absolute level differs by convention."""

    for our_case, external_case_object in zip(ours, theirs, strict=True):
        external_case: dict[str, Any] = external_case_object  # type: ignore[assignment]
        """Narrowed the shape already established for every frozen case."""

        if (
            external_case.get("true_h2") != our_case["true_h2"]
            or external_case.get("seed") != our_case["seed"]
        ):
            raise ReferenceAdapterError("against_r frozen case identity is stale")
        required: tuple[str, ...] = ("sigma_k", "sigma_e", "llik", "beta")
        """Named every independent quantity used by the prewritten rule."""

        if any(field not in external_case for field in required) or not isinstance(
            external_case["beta"], list
        ):
            raise ReferenceAdapterError("against_r frozen case is incomplete")
        their_total: float = float(external_case["sigma_k"]) + float(
            external_case["sigma_e"]
        )
        """Converted regress variance components to Asterism's reported scale."""

        their_h2: float = float(external_case["sigma_k"]) / their_total
        """Converted the independent components to heritability."""

        beta_difference: float = max(
            relative(float(a), float(b))
            for a, b in zip(our_case["beta"], external_case["beta"], strict=True)
        )
        """Required every fixed effect rather than comparing only an intercept."""

        h2_difference: float = relative(float(our_case["h2"]), their_h2)
        """Measured heritability agreement on the historical relative scale."""

        variance_difference: float = relative(
            float(our_case["total_variance"]), their_total
        )
        """Measured total-variance agreement on the historical relative scale."""

        gap: float = float(our_case["loglik"]) - float(external_case["llik"])
        """Retained the identifiable likelihood offset instead of comparing levels."""

        gaps.append(gap)
        case_metrics.append(
            {
                "seed": int(our_case["seed"]),
                "true_h2": float(our_case["true_h2"]),
                "h2_relative_difference": h2_difference,
                "total_variance_relative_difference": variance_difference,
                "maximum_beta_relative_difference": beta_difference,
                "loglik_gap": gap,
            }
        )
        if max(h2_difference, variance_difference, beta_difference) >= float(
            REFERENCE_ACCEPTANCE["maximum_relative_estimate_difference"]
        ):
            failures.append(f"seed {our_case['seed']} exceeds estimate tolerance")
    drift: float = max(gaps) - min(gaps)
    """Required the convention offset to remain constant across distinct datasets."""

    expected_gap: float = -0.5 * (sample_size - fixed_effects) * np.log(2.0 * np.pi)
    """Named regress's omitted REML normalising constant analytically."""

    unexplained: float = abs(gaps[0] - expected_gap)
    """Rejected a stable but scientifically unexplained likelihood offset."""

    if drift >= REFERENCE_ACCEPTANCE["maximum_loglik_offset_drift"]:
        failures.append("log-likelihood offset drift exceeds tolerance")
    if unexplained >= REFERENCE_ACCEPTANCE["maximum_unexplained_loglik_offset"]:
        failures.append("log-likelihood offset is not the named 2*pi constant")
    diagnostics: dict[str, object] = {
        "cases": case_metrics,
        "loglik_offset_drift": drift,
        "expected_loglik_offset": expected_gap,
        "unexplained_loglik_offset": unexplained,
        "failures": failures,
    }
    """Returned the exact evidence behind the binary gate decision."""

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
    relationship, design_matrix, beta, cases, phenotypes = reference_problem()
    identities: list[dict[str, str]] = reference_input_identities(
        relationship, design_matrix, beta, cases, phenotypes
    )
    """Recomputed canonical participant-free inputs in both adapter modes."""

    ours: list[dict[str, Any]] = fit_asterism_reference(
        relationship, design_matrix, cases, phenotypes
    )
    """Recomputed Asterism through the public prepare/fit API in both modes."""

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
        external_outputs: object = fit_regress_reference(
            relationship, design_matrix, cases, phenotypes
        )
        """Ran R only in explicit live refresh mode and retained raw outputs."""
    else:
        fixture: dict[str, Any] = load_fixture(fixture_path, CHECK_ID)
        """Loaded the driver-validated frozen independent envelope without R."""

        if fixture.get("acceptance") != REFERENCE_ACCEPTANCE:
            raise ReferenceAdapterError("against_r fixture acceptance was altered")
        external_outputs = fixture.get("external_outputs")
        """Read only the frozen independent values needed by portable comparison."""

    # asterism-style: allow missing-following-doc -- paired decision and diagnostics are one result
    passed, diagnostics = compare_reference(
        ours,
        external_outputs,
        sample_size=relationship.shape[0],
        fixed_effects=design_matrix.shape[1],
    )
    response: dict[str, Any] = {
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
        response.update(
            {
                "tools": tools,
                "external_outputs": external_outputs,
                "acceptance": REFERENCE_ACCEPTANCE,
            }
        )
    return response


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
    """Run the historical live R comparison and print its evidence receipt."""
    if shutil.which("Rscript") is None:
        raise SystemExit(
            "Rscript is not on the path. This check fails rather than skips."
        )

    families: int = 25
    """Fixed the number of independent extended families."""

    block: int = len(extended_family())
    """Recorded the number of people in each extended family."""

    k: np.ndarray = roster(families)
    """Built the block-diagonal additive relationship matrix."""

    n: int = k.shape[0]
    """Read the simulated roster size from the relationship matrix."""

    x: np.ndarray = design(n, block)
    """Built the six-column fixed-effect design."""

    beta: np.ndarray = np.array([2.0, 0.30, -0.10, 0.50, 0.05, -0.02])
    """Fixed the generating regression coefficients shared by every case."""

    model: asterism.PreparedModel = asterism.prepare(x, k)
    """Prepared the public one-trait model once for the shared design."""

    cases: list[tuple[float, int]] = [(0.5, 101), (0.25, 102), (0.75, 103)]
    """Fixed the three heritability and random-seed comparison cells."""

    results: list[dict[str, Any]] = []
    """Initialised the complete per-case comparison records."""

    print(f"Asterism against R `regress`, REML, n = {n}, {x.shape[1]} fixed effects.\n")
    print(
        f"{'truth':>6} {'asterism h2':>12} {'regress h2':>12} {'rel':>10} "
        f"{'var rel':>10} {'worst beta rel':>15} {'loglik gap':>13}"
    )

    for h2, seed in cases:
        y: np.ndarray = simulate(k, x, beta, h2, seed)
        """Generated this deterministic response under its known truth."""

        ours: dict[str, Any] = model.fit(y, "reml")
        """Fitted the response through Asterism's public prepared model."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            theirs: dict[str, Any] = fit_in_r(Path(temporary_directory), k, x, y)
            """Fitted the same response through independent R regress."""

        their_total: float = theirs["sigma_k"] + theirs["sigma_e"]
        """Combined independent additive and residual variance estimates."""

        their_h2: float = theirs["sigma_k"] / their_total
        """Converted the independent variance estimates to heritability."""

        beta_rel: float = max(
            relative(a, b) for a, b in zip(ours["beta"], theirs["beta"], strict=True)
        )
        """Measured the worst fixed-effect relative disagreement."""

        gap: float = ours["loglik"] - theirs["llik"]
        """Measured the convention-dependent likelihood offset for this case."""

        results.append(
            {
                "true_h2": h2,
                "seed": seed,
                "asterism": {
                    "h2": ours["h2"],
                    "total_variance": ours["total_variance"],
                    "loglik": ours["loglik"],
                    "beta": ours["beta"],
                },
                "regress": {
                    "h2": their_h2,
                    "total_variance": their_total,
                    "llik": theirs["llik"],
                    "beta": theirs["beta"],
                },
                "h2_relative": relative(ours["h2"], their_h2),
                "total_variance_relative": relative(
                    ours["total_variance"], their_total
                ),
                "worst_beta_relative": beta_rel,
                "loglik_gap": gap,
            }
        )
        print(
            f"{h2:>6.2f} {ours['h2']:>12.9f} {their_h2:>12.9f} "
            f"{relative(ours['h2'], their_h2):>10.2e} "
            f"{relative(ours['total_variance'], their_total):>10.2e} "
            f"{beta_rel:>15.2e} {gap:>13.9f}"
        )

    # The two log-likelihoods use different constants: `regress` drops terms
    # Asterism keeps. A level comparison would therefore mean nothing, but the
    # *offset* must be identical across datasets — if it drifts, the two are not
    # computing the same function of the data.
    gaps: list[float] = [r["loglik_gap"] for r in results]
    """Collected likelihood offsets across distinct generated responses."""

    drift: float = max(gaps) - min(gaps)
    """Measured offset drift that cannot be explained by a fixed convention."""

    # Better than constant: the offset is the constant we can name. `regress`
    # omits the 2*pi term from the restricted likelihood and Asterism keeps it,
    # which is exactly (n - p)/2 * log(2*pi). If the measured offset matches
    # that, the two are computing the same function and the difference is
    # bookkeeping. If it were merely stable but unrecognisable, something else
    # would be going on and it would need finding.
    expected: float = 0.5 * (n - x.shape[1]) * np.log(2.0 * np.pi)
    """Computed regress's omitted restricted-likelihood normalising constant."""

    unexplained: float = abs(gaps[0] + expected)
    """Measured likelihood offset not explained by the named constant."""

    print(f"\nlog-likelihood offset {gaps[0]:.9f}, drift across cases {drift:.3e}")
    print(
        f"expected offset -(n - p)/2 * log(2*pi) = {-expected:.9f}, "
        f"unexplained {unexplained:.3e}"
    )

    failures: list[str] = []
    """Initialised the scientific comparison failure messages."""

    for r in results:
        for name, value, tolerance in (
            ("h2", r["h2_relative"], ESTIMATE_TOLERANCE),
            ("total variance", r["total_variance_relative"], ESTIMATE_TOLERANCE),
            ("beta", r["worst_beta_relative"], ESTIMATE_TOLERANCE),
        ):
            if not value < tolerance:
                failures.append(
                    f"true h2 {r['true_h2']}: {name} differs by {value:.3e}"
                )
    if not drift < LOGLIK_DIFFERENCE_TOLERANCE:
        failures.append(f"log-likelihood offset drifts by {drift:.3e}")
    if not unexplained < 1e-6:
        failures.append(
            f"the offset is stable but not the expected 2*pi term; "
            f"{unexplained:.3e} is unaccounted for"
        )

    print(
        json.dumps(
            {
                "results": results,
                "loglik_offset": gaps[0],
                "loglik_offset_drift": drift,
                "loglik_offset_expected": -expected,
                "loglik_offset_unexplained": unexplained,
            },
            indent=2,
        )
    )

    if failures:
        print("\nDISAGREEMENT:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nAgreement within tolerance on every quantity.")
    print("The independent REML implementations agree on this comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(entrypoint())
