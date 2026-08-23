"""Does the censored likelihood agree with R's `censReg`?

**The comparison is arranged so that only the new part is under test.** With no
relatedness the relationship matrix is the identity, the additive and residual
variances add to one total, and the model is an ordinary Tobit regression --
exactly what `censReg` fits. So a disagreement here is a disagreement about
censoring, not about variance components, which Asterism already checks
elsewhere against `regress` and SOLAR.

That the heritability is unidentified in this design is deliberate rather than
a flaw in it. The likelihood is flat along that coordinate, so the optimiser
must still return the right total variance and the right coefficients while
having nothing to say about the split. A model that cannot do that would be
unusable on any real trait with little family structure.

`censReg` takes the dependent variable with censored values set to the limit
and infers the censoring from equality with it. Asterism takes the status
separately and never infers it, which is the safer contract but means the two
inputs are constructed differently here.

Run with:

    uv run --no-project python checks/tobit_against_censreg.py

It fails rather than skips when R or `censReg` is missing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

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

PEOPLE: int = 4000
"""Unrelated simulated people in the fixed censReg comparison."""

TRUE_INTERCEPT: float = 5.0
"""Intercept used to generate the latent Gaussian response."""

TRUE_SLOPE: float = 1.5
"""Covariate slope used to generate the latent Gaussian response."""

TRUE_SD: float = 2.0
"""Residual standard deviation used for the latent response."""

# A limit a little above the mean, censoring roughly a third.
LIMIT: float = 6.5
"""Common right-censoring limit supplied to both implementations."""

TOLERANCE: float = 1.0e-3
"""Maximum accepted protected relative difference from censReg."""

CHECK_ID: str = "tobit_against_censreg"
"""Matched the unrelated Tobit manifest check identifier exactly."""

REFERENCE_ACCEPTANCE: dict[str, float] = {"maximum_relative_difference": TOLERANCE}
"""Prewrote the four-quantity likelihood agreement threshold."""

QUALIFIED_R_VERSION: str = "4.5.2"
"""Named the R release on which the censored-regression output is qualified."""

QUALIFIED_PACKAGES: dict[str, str] = {"censReg": "0.5.38"}
"""Named the exact independent Tobit implementation used for refresh."""

R_SCRIPT: str = """
suppressMessages(library(censReg))
d <- read.csv("data.csv")
m <- censReg(y ~ x, left = -Inf, right = {limit}, data = d)
co <- coef(m)
cat(sprintf("intercept %.12f\\n", co[["(Intercept)"]]))
cat(sprintf("slope %.12f\\n", co[["x"]]))
cat(sprintf("sigma %.12f\\n", exp(co[["logSigma"]])))
cat(sprintf("loglik %.12f\\n", as.numeric(logLik(m))))
"""
"""R program fitting the independent ordinary Tobit regression."""


def reference_problem() -> dict[str, np.ndarray | float]:
    """Regenerate the exact participant-free unrelated Tobit input.

    Returns:
        Named arrays supplied to Asterism and censReg plus the censoring limit.
    """
    generator: np.random.Generator = np.random.default_rng(20_260_818)
    """Selected the prewritten deterministic simulation stream."""

    covariate: np.ndarray = generator.normal(size=PEOPLE)
    """Generated the continuous covariate shared by both implementations."""

    complete: np.ndarray = (
        TRUE_INTERCEPT
        + TRUE_SLOPE * covariate
        + TRUE_SD * generator.normal(size=PEOPLE)
    )
    """Generated the latent uncensored Gaussian response."""

    censored: np.ndarray = complete >= LIMIT
    """Applied the fixed right-censoring limit as an explicit status mask."""

    relationship: np.ndarray = np.eye(PEOPLE)
    """Removed relatedness so only the censored likelihood differs between engines."""

    status: np.ndarray = np.where(censored, 1, 0).astype(np.int64)
    """Encoded censoring through Asterism's explicit public status contract."""

    limits: np.ndarray = np.full(PEOPLE, LIMIT)
    """Supplied the same fixed upper detection limit for every observation."""

    design_matrix: np.ndarray = np.column_stack([np.ones(PEOPLE), covariate])
    """Built the intercept-and-slope public fixed-effect design."""

    return {
        "covariate": covariate,
        "complete": complete,
        "censored": censored,
        "relationship": relationship,
        "status": status,
        "limits": limits,
        "design": design_matrix,
        "limit": LIMIT,
    }


def reference_input_identities(
    problem: dict[str, np.ndarray | float],
) -> list[dict[str, str]]:
    """Return canonical hashes for every generated Tobit input.

    Args:
        problem: Deterministic problem returned by :func:`reference_problem`.

    Returns:
        Ordered strict input identities.
    """
    return [
        support_source_identity(),
        json_identity(
            "censReg simulation constants",
            {
                "people": PEOPLE,
                "seed": 20_260_818,
                "intercept": TRUE_INTERCEPT,
                "slope": TRUE_SLOPE,
                "sd": TRUE_SD,
                "limit": LIMIT,
            },
        ),
        array_identity("censReg covariate", problem["covariate"]),  # type: ignore[arg-type]
        array_identity("censReg complete response", problem["complete"]),  # type: ignore[arg-type]
        array_identity("censReg censoring mask", problem["censored"]),  # type: ignore[arg-type]
        array_identity("censReg relationship matrix", problem["relationship"]),  # type: ignore[arg-type]
        array_identity("censReg status", problem["status"]),  # type: ignore[arg-type]
        array_identity("censReg limits", problem["limits"]),  # type: ignore[arg-type]
        array_identity("censReg fixed-effect design", problem["design"]),  # type: ignore[arg-type]
    ]


def fit_asterism_reference(
    problem: dict[str, np.ndarray | float],
) -> dict[str, float | bool]:
    """Fit the frozen case through Asterism's public Tobit API.

    Args:
        problem: Deterministic problem returned by :func:`reference_problem`.

    Returns:
        Public likelihood, scale and coefficient quantities under comparison.
    """
    complete: np.ndarray = problem["complete"]  # type: ignore[assignment]
    """Read the finite latent response used to construct explicit missing values."""

    censored: np.ndarray = problem["censored"]  # type: ignore[assignment]
    """Read the generated right-censoring mask."""

    fit: dict[str, object] = asterism.tobit_fit(
        problem["relationship"],
        np.where(censored, np.nan, complete),
        problem["status"],
        problem["limits"],
        problem["design"],
    )  # type: ignore[arg-type]
    """Recomputed the candidate through the public explicit-status Tobit function."""

    effects: list[float] = [float(value) for value in fit["fixed_effects"]]  # type: ignore[union-attr]
    """Normalized public fixed effects for deterministic JSON diagnostics."""

    total_variance: float = float(fit["total_variance"])
    """Read the identifiable variance sum in the unrelated design."""

    return {
        "intercept": effects[0],
        "slope": effects[1],
        "sigma": total_variance**0.5,
        "loglik": float(fit["loglik"]),
        "converged": bool(fit["converged"]),
    }


def fit_censreg_reference(
    problem: dict[str, np.ndarray | float],
) -> dict[str, float]:
    """Run live R censReg and retain only independent numerical outputs.

    Args:
        problem: Deterministic problem returned by :func:`reference_problem`.

    Returns:
        Independent intercept, slope, standard deviation and log likelihood.

    Raises:
        ReferenceAdapterError: If live R fails or omits a required quantity.
    """
    complete: np.ndarray = problem["complete"]  # type: ignore[assignment]
    """Read the complete response before applying censReg's value-based contract."""

    censored: np.ndarray = problem["censored"]  # type: ignore[assignment]
    """Read the explicit status used to place limits into censReg's response."""

    covariate: np.ndarray = problem["covariate"]  # type: ignore[assignment]
    """Read the generated continuous predictor."""

    observed: np.ndarray = np.where(censored, LIMIT, complete)
    """Encoded censored rows at the limit as required by the independent package."""

    with tempfile.TemporaryDirectory() as temporary:
        directory: Path = Path(temporary)
        """Isolated all transient R inputs from versioned repository evidence."""

        rows: list[str] = ["y,x"] + [
            f"{observed[index]:.12f},{covariate[index]:.12f}" for index in range(PEOPLE)
        ]
        """Rendered the historical twelve-decimal censReg CSV deterministically."""

        (directory / "data.csv").write_text("\n".join(rows) + "\n")
        (directory / "run.R").write_text(R_SCRIPT.format(limit=LIMIT))
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            ["Rscript", "--vanilla", "run.R"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        """Ran the independently authored censored-regression implementation."""

    if completed.returncode != 0:
        raise ReferenceAdapterError(
            f"censReg did not run: {completed.stderr or completed.stdout}"
        )
    results: dict[str, float] = {}
    """Collected exactly the four frozen independent quantities required later."""

    for name in ("intercept", "slope", "sigma", "loglik"):
        found: re.Match[str] | None = re.search(
            rf"{name} (-?[0-9.eE+-]+)", completed.stdout
        )
        """Located one named value in the package's captured machine output."""

        if found is None:
            raise ReferenceAdapterError(f"censReg printed no {name}")
        results[name] = float(found.group(1))
        """Stored one exact independent likelihood quantity by stable name."""
    return results


def compare_reference(
    ours: dict[str, float | bool], external_outputs: object
) -> tuple[bool, dict[str, object]]:
    """Apply the immutable censReg likelihood agreement rule.

    Args:
        ours: Fresh public Asterism result.
        external_outputs: Frozen independent censReg quantities.

    Returns:
        Pass decision and per-quantity relative differences.

    Raises:
        ReferenceAdapterError: If the frozen independent output is incomplete.
    """
    names: tuple[str, ...] = ("intercept", "slope", "sigma", "loglik")
    """Named every likelihood quantity retained from the live independent engine."""

    if not isinstance(external_outputs, dict) or any(
        name not in external_outputs for name in names
    ):
        raise ReferenceAdapterError("censReg external_outputs are incomplete")
    differences: dict[str, float] = {
        name: abs(float(ours[name]) - float(external_outputs[name]))
        / max(1.0, abs(float(external_outputs[name])))
        for name in names
    }
    """Applied the historical relative scale to coefficients, sigma and likelihood."""

    worst: float = max(differences.values())
    """Required every independent quantity to pass rather than reporting an average."""

    failures: list[str] = []
    """Collected convergence and independent-likelihood disagreements."""

    if ours["converged"] is not True:
        failures.append("Asterism did not declare convergence")
    if worst > REFERENCE_ACCEPTANCE["maximum_relative_difference"]:
        failures.append("a censReg quantity exceeds the relative tolerance")
    diagnostics: dict[str, object] = {
        "asterism": ours,
        "censreg": external_outputs,
        "relative_differences": differences,
        "worst_relative_difference": worst,
        "failures": failures,
    }
    """Returned the exact quantities behind the binary gate decision."""

    return not failures, diagnostics


def reference_record(mode: str, fixture_path: Path) -> dict[str, object]:
    """Build one live-refresh or portable-verification adapter response.

    Args:
        mode: Strict ``refresh`` or ``verify`` operation.
        fixture_path: Frozen envelope supplied by the external fixture driver.

    Returns:
        Version-one adapter response consumed by the strict driver.
    """
    problem: dict[str, np.ndarray | float] = reference_problem()
    """Regenerated the participant-free comparison in both adapter modes."""

    identities: list[dict[str, str]] = reference_input_identities(problem)
    """Recomputed canonical participant-free inputs before reading frozen values."""

    ours: dict[str, float | bool] = fit_asterism_reference(problem)
    """Recomputed Asterism through the public tobit_fit function in both modes."""

    tools: list[dict[str, object]] | None = None
    """Held live independent software identity only during explicit refresh."""

    if mode == "refresh":
        tools = r_tool_identity(tuple(QUALIFIED_PACKAGES))
        """Loaded and identified the qualified live censReg implementation."""

        require_r_versions(
            tools,
            r_version=QUALIFIED_R_VERSION,
            packages=QUALIFIED_PACKAGES,
        )
        external_outputs: object = fit_censreg_reference(problem)
        """Ran R only during explicit live refresh and retained four raw outputs."""
    else:
        fixture: dict[str, object] = load_fixture(fixture_path, CHECK_ID)
        """Loaded the driver-validated frozen independent envelope without R."""

        if fixture.get("acceptance") != REFERENCE_ACCEPTANCE:
            raise ReferenceAdapterError("tobit censReg fixture acceptance was altered")
        external_outputs = fixture.get("external_outputs")
        """Read only the frozen independent values needed by portable comparison."""

    # asterism-style: allow missing-following-doc -- paired decision and diagnostics are one result
    passed, diagnostics = compare_reference(ours, external_outputs)
    response: dict[str, object] = {
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
    arguments: object = parse_reference_arguments(__doc__)
    """Selected historical live output or the machine-only adapter contract."""

    if not hasattr(arguments, "reference_mode"):
        raise ReferenceAdapterError("argument parser returned no reference_mode")
    if arguments.reference_mode is None:  # type: ignore[attr-defined]
        return main()
    try:
        record: dict[str, object] = reference_record(
            arguments.reference_mode,
            arguments.reference_fixture,  # type: ignore[attr-defined]
        )
        """Ran exactly one strict mode with paired arguments guaranteed by parsing."""
    except (ReferenceAdapterError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return emit_reference_record(record)


def main() -> int:
    """Run the historical live censReg comparison and write its receipt.

    Returns:
        Zero when all four quantities agree within tolerance, otherwise one.

    Raises:
        SystemExit: If R or the independent censReg fit is unavailable.
    """
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on the path")

    rng: np.random.Generator = np.random.default_rng(20_260_818)
    """Selected the historical deterministic simulation stream."""

    x: np.ndarray = rng.normal(size=PEOPLE)
    """Generated the continuous covariate shared by both implementations."""

    complete: np.ndarray = (
        TRUE_INTERCEPT + TRUE_SLOPE * x + TRUE_SD * rng.normal(size=PEOPLE)
    )
    """Generated the complete latent Gaussian response."""

    censored: np.ndarray = complete >= LIMIT
    """Applied the fixed right-censoring limit as an explicit mask."""

    print(
        f"{PEOPLE} unrelated people, {censored.sum()} censored "
        f"({censored.mean():.1%}) at a limit of {LIMIT}"
    )

    # Asterism: the status is given, never inferred from the value.
    relationship: np.ndarray = np.eye(PEOPLE)
    """Removed relatedness so the comparison isolates the censored likelihood."""

    value: np.ndarray = np.where(censored, np.nan, complete)
    """Represented censored latent values as missing for Asterism."""

    status: np.ndarray = np.where(censored, 1, 0).astype(np.int64)
    """Encoded the censoring state explicitly for the public API."""

    limit: np.ndarray = np.full(PEOPLE, LIMIT)
    """Supplied the same upper detection limit for every observation."""

    design: np.ndarray = np.column_stack([np.ones(PEOPLE), x])
    """Built the intercept-and-slope fixed-effect design."""

    ours: dict[str, object] = asterism.tobit_fit(
        relationship,
        value,
        status,
        limit,
        design,
    )
    """Fitted the unrelated censored outcome through Asterism's public API."""

    effects: list[float] = ours["fixed_effects"]
    """Read the fitted intercept and slope in design-column order."""

    total_variance: float = ours["total_variance"]
    """Read the fitted latent total variance."""

    loglik: float = ours["loglik"]
    """Read Asterism's maximised censored-data log likelihood."""

    print(
        f"\nAsterism : intercept {effects[0]:.9f} slope {effects[1]:.9f} "
        f"sigma {total_variance**0.5:.9f}"
    )
    print(
        f"           loglik {loglik:.9f}, converged {ours['converged']}, "
        f"scaled gradient {ours['scaled_gradient']:.2e}"
    )
    print(
        f"           heritability {ours['heritability']:.4f} "
        f"(unidentified here, and expected to be)"
    )

    # censReg: the dependent variable carries the limit where censored.
    with tempfile.TemporaryDirectory() as tmp:
        directory: Path = Path(tmp)
        """Converted the isolated live-comparison workspace to a path object."""

        observed: np.ndarray = np.where(censored, LIMIT, complete)
        """Placed the detection limit in censReg's censored dependent values."""

        rows: list[str] = ["y,x"] + [
            f"{observed[i]:.12f},{x[i]:.12f}" for i in range(PEOPLE)
        ]
        """Rendered both inputs at the historical twelve-decimal precision."""

        (directory / "data.csv").write_text("\n".join(rows) + "\n")
        (directory / "run.R").write_text(R_SCRIPT.format(limit=LIMIT))
        finished: subprocess.CompletedProcess[str] = subprocess.run(
            ["Rscript", "run.R"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        """Ran the independent censReg Tobit likelihood on the shared data."""

        if finished.returncode != 0:
            raise SystemExit(
                f"censReg did not run:\n{finished.stdout}\n{finished.stderr}"
            )
        theirs: dict[str, float] = {}
        """Collected the independent coefficients, scale and log likelihood."""

        for name in ("intercept", "slope", "sigma", "loglik"):
            found: re.Match[str] | None = re.search(
                rf"{name} (-?[0-9.eE+-]+)",
                finished.stdout,
            )
            """Located this required censReg quantity in captured R output."""

            if found is None:
                raise SystemExit(f"censReg printed no {name}:\n{finished.stdout}")
            theirs[name] = float(found.group(1))
            """Stored the independently calculated scalar under its stable name."""

    print(
        f"\ncensReg  : intercept {theirs['intercept']:.9f} "
        f"slope {theirs['slope']:.9f} sigma {theirs['sigma']:.9f}"
    )
    print(f"           loglik {theirs['loglik']:.9f}")

    ours_named: dict[str, float] = {
        "intercept": effects[0],
        "slope": effects[1],
        "sigma": total_variance**0.5,
        "loglik": loglik,
    }
    """Mapped Asterism's outputs to the four directly comparable quantities."""

    print(f"\n{'quantity':<11} | {'Asterism':>16} | {'censReg':>16} | difference")
    print("-" * 68)
    worst: float = 0.0
    """Initialised the maximum protected relative difference."""

    failures: list[str] = []
    """Collected quantities exceeding the prewritten comparison tolerance."""

    for name in ("intercept", "slope", "sigma", "loglik"):
        difference: float = abs(ours_named[name] - theirs[name])
        """Calculated the absolute difference for human-readable output."""

        relative: float = difference / max(1.0, abs(theirs[name]))
        """Scaled the difference by the prewritten protected denominator."""

        worst = max(worst, relative)
        """Updated the maximum comparison difference without changing its rule."""

        print(
            f"{name:<11} | {ours_named[name]:16.9f} | {theirs[name]:16.9f} | "
            f"{difference:.3e}"
        )
        if relative > TOLERANCE:
            failures.append(f"{name} differs by {relative:.3e}, over {TOLERANCE:.0e}")

    receipt: dict[str, object] = {
        "what": "the censored likelihood against R's censReg, with no relatedness",
        "date": date.today().isoformat(),
        "people": PEOPLE,
        "censored": int(censored.sum()),
        "limit": LIMIT,
        "truth": {"intercept": TRUE_INTERCEPT, "slope": TRUE_SLOPE, "sigma": TRUE_SD},
        "asterism": ours_named,
        "censreg": theirs,
        "worst_relative_difference": worst,
        "tolerance": TOLERANCE,
        "passed": not failures,
        "failures": failures,
    }
    """Assembled the dated human-facing comparison receipt without raw inputs."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / (f"tobit-against-censreg-{receipt['date']}.json")
    )
    """Selected the dated receipt path for the ordinary live comparison."""

    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nPASSED: worst relative difference {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(entrypoint())
