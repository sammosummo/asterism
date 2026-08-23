"""Does the censored model agree with MCMCglmm once relatedness is in play?

`checks/tobit_against_censreg.py` compares the censored likelihood on unrelated
people, where the model is an ordinary Tobit regression. That leaves the case
the package actually exists for untested against anything external: censoring
**and** a relationship matrix at the same time. No engine in R does that except
MCMCglmm, whose `cengaussian` family takes a censoring interval per observation
and whose `ginverse` argument takes a relationship matrix.

**This comparison is of a different kind from the others and the criterion has
to match.** MCMCglmm is Bayesian and sampled; Asterism is maximum likelihood and
optimised. They will not agree to eight decimals and it would be suspicious if
they did. What can be asked is whether the maximum-likelihood estimate falls
inside the posterior MCMCglmm draws, which is the strongest statement two
estimators of different kinds can make about each other.

That MCMCglmm is Bayesian is the point rather than a nuisance. Under ADR 0006
agreement proves fidelity only against something arrived at separately, and an
engine using a different inferential framework, a different algorithm and a
different author is about as separate as a comparator gets.

Run with:

    uv run --no-project python checks/tobit_against_mcmcglmm.py

It fails rather than skips when R or MCMCglmm is missing.
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

PAIRS: int = 800
"""Full-sibling pairs in the fixed related Tobit comparison."""

TRUE_HERITABILITY: float = 0.5
"""Latent-scale heritability used to generate the comparison outcome."""

TRUE_VARIANCE: float = 4.0
"""Total latent variance used for the simulated outcome."""

TRUE_MEAN: float = 10.0
"""Latent intercept used for every simulated sibling."""

CENSORED_SHARE: float = 0.25
"""Target upper-tail share hidden behind the empirical censoring limit."""

# Long enough that the posterior is not the limiting uncertainty. Reported with
# the result, because a chain too short is how two engines come to "disagree".
# A posterior wider than this cannot discriminate between two engines, so a
# "pass" against it would be worth nothing.
WIDEST_USEFUL_INTERVAL: float = 0.40
"""Largest posterior heritability width accepted as discriminating evidence."""

ITERATIONS: int = 120_000
"""Total MCMCglmm iterations in the prewritten sampling contract."""

BURNIN: int = 20_000
"""Initial MCMCglmm iterations discarded from posterior summaries."""

THIN: int = 50
"""Sampling interval retained from the MCMCglmm chain."""

CHECK_ID: str = "tobit_against_mcmcglmm"
"""Matched the related censored manifest check identifier exactly."""

REFERENCE_ACCEPTANCE: dict[str, object] = {
    "maximum_h2_posterior_interval_width": WIDEST_USEFUL_INTERVAL,
    "minimum_effective_h2": 200.0,
    "asterism_points_inside_95_percent_hpd": [
        "heritability",
        "total_variance",
        "intercept",
    ],
}
"""Prewrote informativeness and point-inside-posterior requirements."""

QUALIFIED_R_VERSION: str = "4.5.2"
"""Named the R release on which the sampled independent output is qualified."""

QUALIFIED_PACKAGES: dict[str, str] = {
    "MCMCglmm": "2.36",
    "Matrix": "1.7.4",
    "coda": "0.19.4.1",
}
"""Named the exact sampler, sparse-matrix layer and posterior-summary package."""

R_SCRIPT: str = """
suppressMessages(library(MCMCglmm))
suppressMessages(library(Matrix))

d <- read.csv("data.csv")
n <- nrow(d)
# **The levels must be in the matrix's own order, and R will not do that
# by itself.** A factor built from "1".."1600" sorts its levels
# lexicographically -- "1", "10", "100", "1000" -- so naming the matrix
# rows after them labels row 2 as person 10. Every sibling pair is then
# mis-paired, the additive variance collapses, and the heritability comes
# back near nought while everything else still looks right. Stating the
# levels explicitly is what stops it.
d$animal <- factor(as.character(seq_len(n)), levels = as.character(seq_len(n)))
d$upper[!is.finite(d$upper)] <- Inf

A <- as.matrix(read.csv("relationship.csv", header = FALSE))
dimnames(A) <- list(levels(d$animal), levels(d$animal))
Ainv <- as(solve(A), "dgCMatrix")

# **Parameter expanded, not the plain inverse Wishart.** With `nu = 0.002`
# the prior on the additive variance drags it towards nought, and at this
# sample size it dominates a heritability the data barely constrain -- the
# posterior then spans almost the whole range and the comparison proves
# nothing. This is the prior the animal-model literature recommends for
# exactly that reason.
prior <- list(G = list(G1 = list(V = 1, nu = 1, alpha.mu = 0, alpha.V = 1000)),
              R = list(V = 1, nu = 0.002))

set.seed(20260818)
m <- MCMCglmm(cbind(lower, upper) ~ 1,
              random = ~ animal,
              ginverse = list(animal = Ainv),
              family = "cengaussian",
              data = d, prior = prior,
              nitt = {iterations}, burnin = {burnin}, thin = {thin},
              verbose = FALSE)

va <- m$VCV[, "animal"]
ve <- m$VCV[, "units"]
h2 <- va / (va + ve)
total <- va + ve
hpd_h2 <- HPDinterval(mcmc(h2))
hpd_total <- HPDinterval(mcmc(total))
hpd_mean <- HPDinterval(m$Sol[, "(Intercept)"])

cat(sprintf("h2_mean %.9f\\n", mean(h2)))
cat(sprintf("h2_low %.9f\\n", hpd_h2[1]))
cat(sprintf("h2_high %.9f\\n", hpd_h2[2]))
cat(sprintf("total_mean %.9f\\n", mean(total)))
cat(sprintf("total_low %.9f\\n", hpd_total[1]))
cat(sprintf("total_high %.9f\\n", hpd_total[2]))
cat(sprintf("intercept_mean %.9f\\n", mean(m$Sol[, "(Intercept)"])))
cat(sprintf("intercept_low %.9f\\n", hpd_mean[1]))
cat(sprintf("intercept_high %.9f\\n", hpd_mean[2]))
cat(sprintf("effective_h2 %.1f\\n", effectiveSize(h2)))
"""
"""R program fitting and summarising the independent censored animal model."""


def reference_problem() -> dict[str, np.ndarray | float]:
    """Regenerate the exact participant-free related Tobit input.

    Returns:
        Named arrays supplied to Asterism and MCMCglmm plus the censoring limit.
    """
    generator: np.random.Generator = np.random.default_rng(20_260_818)
    """Selected the prewritten deterministic phenotype stream."""

    people: int = 2 * PAIRS
    """Calculated the fixed 1,600-person sibling-pair design."""

    shared: np.ndarray = generator.normal(size=PAIRS) * np.sqrt(TRUE_HERITABILITY / 2.0)
    """Generated one additive half-sibling covariance contribution per pair."""

    own: np.ndarray = generator.normal(size=people) * np.sqrt(
        1.0 - TRUE_HERITABILITY / 2.0
    )
    """Generated independent individual-specific Gaussian contributions."""

    complete: np.ndarray = TRUE_MEAN + np.sqrt(TRUE_VARIANCE) * (
        np.repeat(shared, 2) + own
    )
    """Combined shared and individual effects into the complete latent response."""

    relationship: np.ndarray = np.eye(people)
    """Started the additive relationship matrix with unrelated diagonals."""

    for pair in range(PAIRS):
        first: int = 2 * pair
        """Located the first member of one deterministic sibling pair."""

        second: int = first + 1
        """Located the paired sibling adjacent in the canonical ordering."""

        relationship[first, second] = 0.5
        """Assigned additive half-covariance from the first to the second sibling."""

        relationship[second, first] = 0.5
        """Mirrored the additive half-covariance to preserve symmetry."""

    censoring_limit: float = float(np.quantile(complete, 1.0 - CENSORED_SHARE))
    """Selected the empirical limit yielding the prewritten censoring share."""

    censored: np.ndarray = complete >= censoring_limit
    """Applied the fixed right-censoring rule to the latent response."""

    status: np.ndarray = np.where(censored, 1, 0).astype(np.int64)
    """Encoded censoring explicitly for Asterism's public API."""

    limits: np.ndarray = np.full(people, censoring_limit)
    """Supplied the empirical limit for every complete or censored observation."""

    design_matrix: np.ndarray = np.ones((people, 1))
    """Used the intercept-only mean model shared by both inferential frameworks."""

    return {
        "complete": complete,
        "relationship": relationship,
        "censored": censored,
        "status": status,
        "limits": limits,
        "design": design_matrix,
        "limit": censoring_limit,
    }


def reference_input_identities(
    problem: dict[str, np.ndarray | float],
) -> list[dict[str, str]]:
    """Return canonical hashes for every generated related Tobit input.

    Args:
        problem: Deterministic problem returned by :func:`reference_problem`.

    Returns:
        Ordered strict input identities including the fixed sampling contract.
    """
    return [
        support_source_identity(),
        json_identity(
            "MCMCglmm simulation constants",
            {
                "pairs": PAIRS,
                "seed": 20_260_818,
                "true_heritability": TRUE_HERITABILITY,
                "true_variance": TRUE_VARIANCE,
                "true_mean": TRUE_MEAN,
                "censored_share": CENSORED_SHARE,
            },
        ),
        json_identity(
            "MCMCglmm sampling configuration",
            {
                "seed": 20_260_818,
                "iterations": ITERATIONS,
                "burnin": BURNIN,
                "thin": THIN,
            },
        ),
        array_identity("MCMCglmm complete response", problem["complete"]),  # type: ignore[arg-type]
        array_identity("MCMCglmm relationship matrix", problem["relationship"]),  # type: ignore[arg-type]
        array_identity("MCMCglmm censoring mask", problem["censored"]),  # type: ignore[arg-type]
        array_identity("MCMCglmm status", problem["status"]),  # type: ignore[arg-type]
        array_identity("MCMCglmm limits", problem["limits"]),  # type: ignore[arg-type]
        array_identity("MCMCglmm fixed-effect design", problem["design"]),  # type: ignore[arg-type]
    ]


def fit_asterism_reference(
    problem: dict[str, np.ndarray | float],
) -> dict[str, float | bool]:
    """Fit the frozen case through Asterism's public Tobit API.

    Args:
        problem: Deterministic problem returned by :func:`reference_problem`.

    Returns:
        Public maximum-likelihood points compared with the posterior intervals.
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
    """Recomputed the candidate through the public related Tobit function."""

    effects: list[float] = [float(value) for value in fit["fixed_effects"]]  # type: ignore[union-attr]
    """Normalized the public intercept for deterministic comparison output."""

    return {
        "heritability": float(fit["heritability"]),
        "total_variance": float(fit["total_variance"]),
        "intercept": effects[0],
        "converged": bool(fit["converged"]),
    }


def fit_mcmcglmm_reference(
    problem: dict[str, np.ndarray | float],
) -> dict[str, object]:
    """Run live MCMCglmm and retain posterior summaries needed by verify.

    Args:
        problem: Deterministic problem returned by :func:`reference_problem`.

    Returns:
        Independent posterior endpoints, means, effective size and sampler contract.

    Raises:
        ReferenceAdapterError: If live R fails or omits a required quantity.
    """
    complete: np.ndarray = problem["complete"]  # type: ignore[assignment]
    """Read the complete response before constructing censoring intervals."""

    censored: np.ndarray = problem["censored"]  # type: ignore[assignment]
    """Read the explicit censoring mask shared by both engines."""

    relationship: np.ndarray = problem["relationship"]  # type: ignore[assignment]
    """Read the sibling-pair relationship matrix supplied through ginverse."""

    censoring_limit: float = float(problem["limit"])
    """Read the empirical limit applied to every censored observation."""

    lower: np.ndarray = np.where(censored, censoring_limit, complete)
    """Encoded the lower endpoint of MCMCglmm's cengaussian intervals."""

    upper: np.ndarray = np.where(censored, np.inf, complete)
    """Encoded right-censored rows with infinite upper endpoints."""

    with tempfile.TemporaryDirectory() as temporary:
        directory: Path = Path(temporary)
        """Isolated large transient sampler inputs from versioned evidence."""

        rows: list[str] = ["lower,upper"] + [
            f"{lower[index]:.12f},"
            f"{'Inf' if not np.isfinite(upper[index]) else f'{upper[index]:.12f}'}"
            for index in range(complete.size)
        ]
        """Rendered the historical twelve-decimal censoring intervals."""

        (directory / "data.csv").write_text("\n".join(rows) + "\n")
        (directory / "relationship.csv").write_text(
            "\n".join(
                ",".join(f"{value:.12f}" for value in row) for row in relationship
            )
            + "\n"
        )
        (directory / "run.R").write_text(
            R_SCRIPT.format(iterations=ITERATIONS, burnin=BURNIN, thin=THIN)
        )
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            ["Rscript", "--vanilla", "run.R"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        """Ran the independently authored Bayesian censored animal model."""

    if completed.returncode != 0:
        raise ReferenceAdapterError(
            "MCMCglmm did not run: "
            f"{completed.stderr[-3000:] or completed.stdout[-3000:]}"
        )
    names: tuple[str, ...] = (
        "h2_mean",
        "h2_low",
        "h2_high",
        "total_mean",
        "total_low",
        "total_high",
        "intercept_mean",
        "intercept_low",
        "intercept_high",
        "effective_h2",
    )
    """Named every posterior quantity required for informativeness and agreement."""

    posterior: dict[str, float] = {}
    """Collected exact sampled summaries printed by live MCMCglmm."""

    for name in names:
        found: re.Match[str] | None = re.search(
            rf"{name} (-?[0-9.eE+-]+)", completed.stdout
        )
        """Located one named sampler summary in captured machine output."""

        if found is None:
            raise ReferenceAdapterError(f"MCMCglmm printed no {name}")
        posterior[name] = float(found.group(1))
        """Stored one exact independent posterior summary by stable name."""
    return {
        "posterior": posterior,
        "sampling": {
            "seed": 20_260_818,
            "iterations": ITERATIONS,
            "burnin": BURNIN,
            "thin": THIN,
        },
    }


def compare_reference(
    ours: dict[str, float | bool], external_outputs: object
) -> tuple[bool, dict[str, object]]:
    """Apply the immutable MCMCglmm posterior-agreement rule.

    Args:
        ours: Fresh public Asterism maximum-likelihood result.
        external_outputs: Frozen independent posterior summaries.

    Returns:
        Pass decision and interval-membership diagnostics.

    Raises:
        ReferenceAdapterError: If frozen posterior evidence is incomplete.
    """
    if not isinstance(external_outputs, dict) or not isinstance(
        external_outputs.get("posterior"), dict
    ):
        raise ReferenceAdapterError(
            "MCMCglmm external_outputs need posterior summaries"
        )
    expected_sampling: dict[str, int] = {
        "seed": 20_260_818,
        "iterations": ITERATIONS,
        "burnin": BURNIN,
        "thin": THIN,
    }
    """Reconstructed the prewritten sampler contract independently of the fixture."""

    if external_outputs.get("sampling") != expected_sampling:
        raise ReferenceAdapterError("MCMCglmm sampling configuration is stale")
    posterior: dict[str, object] = external_outputs["posterior"]
    """Read posterior values only after validating their enclosing record."""

    required: tuple[str, ...] = (
        "h2_low",
        "h2_high",
        "total_low",
        "total_high",
        "intercept_low",
        "intercept_high",
        "effective_h2",
    )
    """Named every posterior endpoint and informativeness quantity used."""

    if any(name not in posterior for name in required):
        raise ReferenceAdapterError("MCMCglmm posterior summary is incomplete")
    interval_map: dict[str, tuple[str, str]] = {
        "heritability": ("h2_low", "h2_high"),
        "total_variance": ("total_low", "total_high"),
        "intercept": ("intercept_low", "intercept_high"),
    }
    """Mapped public Asterism points to the independent posterior endpoints."""

    inside: dict[str, bool] = {
        quantity: float(posterior[low])
        <= float(ours[quantity])
        <= float(posterior[high])
        for quantity, (low, high) in interval_map.items()
    }
    """Tested every prewritten point-inside-95-percent-HPD requirement."""

    interval_width: float = float(posterior["h2_high"]) - float(posterior["h2_low"])
    """Measured whether the heritability interval is discriminating enough."""

    failures: list[str] = []
    """Collected convergence, informativeness and posterior-agreement failures."""

    if ours["converged"] is not True:
        failures.append("Asterism did not declare convergence")
    if interval_width > float(
        REFERENCE_ACCEPTANCE["maximum_h2_posterior_interval_width"]
    ):
        failures.append("MCMCglmm heritability interval is too wide to discriminate")
    if float(posterior["effective_h2"]) < float(
        REFERENCE_ACCEPTANCE["minimum_effective_h2"]
    ):
        failures.append("MCMCglmm effective heritability sample size is too small")
    failures.extend(
        f"Asterism {quantity} is outside the MCMCglmm 95% HPD"
        for quantity, is_inside in inside.items()
        if not is_inside
    )
    diagnostics: dict[str, object] = {
        "asterism": ours,
        "posterior": posterior,
        "inside_posterior": inside,
        "h2_interval_width": interval_width,
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
        """Loaded and identified the qualified live MCMCglmm implementation."""

        require_r_versions(
            tools,
            r_version=QUALIFIED_R_VERSION,
            packages=QUALIFIED_PACKAGES,
        )
        external_outputs: object = fit_mcmcglmm_reference(problem)
        """Ran R only during explicit live refresh and retained posterior summaries."""
    else:
        fixture: dict[str, object] = load_fixture(fixture_path, CHECK_ID)
        """Loaded the driver-validated frozen independent envelope without R."""

        if fixture.get("acceptance") != REFERENCE_ACCEPTANCE:
            raise ReferenceAdapterError("tobit MCMCglmm fixture acceptance was altered")
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
    """Run the historical live MCMCglmm comparison and write its receipt.

    Returns:
        Zero when all Asterism points lie inside informative posterior intervals,
        otherwise one.

    Raises:
        SystemExit: If R or the independent MCMCglmm fit is unavailable.
    """
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on the path")

    rng: np.random.Generator = np.random.default_rng(20_260_818)
    """Selected the historical deterministic phenotype stream."""

    n: int = 2 * PAIRS
    """Calculated the fixed number of simulated siblings."""

    shared: np.ndarray = rng.normal(size=PAIRS) * np.sqrt(TRUE_HERITABILITY / 2.0)
    """Generated one additive shared contribution per sibling pair."""

    own: np.ndarray = rng.normal(size=n) * np.sqrt(1.0 - TRUE_HERITABILITY / 2.0)
    """Generated independent individual-specific Gaussian contributions."""

    complete: np.ndarray = TRUE_MEAN + np.sqrt(TRUE_VARIANCE) * (
        np.repeat(shared, 2) + own
    )
    """Combined the shared and individual effects into the latent response."""

    relationship: np.ndarray = np.eye(n)
    """Initialised the sibling relationship matrix at unrelated identities."""

    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Set the forward additive relationship within this sibling pair."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Mirrored the additive relationship to preserve matrix symmetry."""

    limit: float = float(np.quantile(complete, 1.0 - CENSORED_SHARE))
    """Selected the empirical limit yielding the fixed censored share."""

    censored: np.ndarray = complete >= limit
    """Applied right-censoring at the shared empirical limit."""

    print(
        f"{n} people in {PAIRS} sibling pairs, {censored.sum()} censored "
        f"({censored.mean():.1%}) at a limit of {limit:.4f}"
    )
    print(
        f"True heritability {TRUE_HERITABILITY}, variance {TRUE_VARIANCE}, "
        f"mean {TRUE_MEAN}\n"
    )

    ours: dict[str, object] = asterism.tobit_fit(
        relationship,
        np.where(censored, np.nan, complete),
        np.where(censored, 1, 0).astype(np.int64),
        np.full(n, limit),
        np.ones((n, 1)),
    )
    """Fitted the related censored outcome through Asterism's public API."""
    print(
        f"Asterism  : h2 {ours['heritability']:.6f}  "
        f"variance {ours['total_variance']:.6f}  "
        f"intercept {ours['fixed_effects'][0]:.6f}"
    )
    print(
        f"            converged {ours['converged']}, "
        f"scaled gradient {ours['scaled_gradient']:.2e}\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        directory: Path = Path(tmp)
        """Converted the isolated live-comparison workspace to a path object."""

        # `cengaussian` takes the interval the value is known to lie in.
        lower: np.ndarray = np.where(censored, limit, complete)
        """Encoded observed values and censoring limits as interval lower bounds."""

        upper: np.ndarray = np.where(censored, np.inf, complete)
        """Encoded right-censored observations with infinite upper bounds."""

        rows: list[str] = ["lower,upper"] + [
            f"{lower[i]:.12f},{'Inf' if not np.isfinite(upper[i]) else f'{upper[i]:.12f}'}"
            for i in range(n)
        ]
        """Rendered MCMCglmm's historical twelve-decimal censoring intervals."""

        (directory / "data.csv").write_text("\n".join(rows) + "\n")
        (directory / "relationship.csv").write_text(
            "\n".join(",".join(f"{v:.12f}" for v in row) for row in relationship) + "\n"
        )
        (directory / "run.R").write_text(
            R_SCRIPT.format(iterations=ITERATIONS, burnin=BURNIN, thin=THIN)
        )
        print(
            f"sampling {ITERATIONS:,} iterations in MCMCglmm, this takes a while...",
            flush=True,
        )
        finished: subprocess.CompletedProcess[str] = subprocess.run(
            ["Rscript", "run.R"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        """Ran the independently authored Bayesian censored animal model."""

        if finished.returncode != 0:
            raise SystemExit(
                f"MCMCglmm did not run:\n{finished.stdout[-3000:]}\n"
                f"{finished.stderr[-3000:]}"
            )
        theirs: dict[str, float] = {}
        """Collected the independent posterior summaries printed by MCMCglmm."""

        for name in (
            "h2_mean",
            "h2_low",
            "h2_high",
            "total_mean",
            "total_low",
            "total_high",
            "intercept_mean",
            "intercept_low",
            "intercept_high",
            "effective_h2",
        ):
            found: re.Match[str] | None = re.search(
                rf"{name} (-?[0-9.eE+-]+)",
                finished.stdout,
            )
            """Located this required posterior summary in captured R output."""

            if found is None:
                raise SystemExit(
                    f"MCMCglmm printed no {name}:\n{finished.stdout[-2000:]}"
                )
            theirs[name] = float(found.group(1))
            """Stored the independently sampled summary under its stable name."""

    print(
        f"\nMCMCglmm  : h2 {theirs['h2_mean']:.6f} "
        f"[{theirs['h2_low']:.6f}, {theirs['h2_high']:.6f}]"
    )
    print(
        f"            variance {theirs['total_mean']:.6f} "
        f"[{theirs['total_low']:.6f}, {theirs['total_high']:.6f}]"
    )
    print(
        f"            intercept {theirs['intercept_mean']:.6f} "
        f"[{theirs['intercept_low']:.6f}, {theirs['intercept_high']:.6f}]"
    )
    print(f"            effective sample size for h2: {theirs['effective_h2']:.0f}")

    failures: list[str] = []
    """Collected sampler-informativeness and posterior-agreement failures."""

    width: float = theirs["h2_high"] - theirs["h2_low"]
    """Measured the posterior heritability interval's discriminating width."""

    if width > WIDEST_USEFUL_INTERVAL:
        failures.append(
            f"the posterior interval for the heritability spans {width:.3f}, "
            f"wider than {WIDEST_USEFUL_INTERVAL}; a maximum-likelihood estimate "
            "falling inside it would say nothing about either engine"
        )
    if theirs["effective_h2"] < 200:
        failures.append(
            f"the chain's effective sample size for the heritability is "
            f"{theirs['effective_h2']:.0f}, too few to compare against"
        )

    print(
        f"\n{'quantity':<11} | {'Asterism':>11} | {'MCMCglmm 95% interval':>26} | inside"
    )
    print("-" * 66)
    comparisons: list[tuple[str, float, str, str]] = [
        ("heritability", ours["heritability"], "h2_low", "h2_high"),
        ("variance", ours["total_variance"], "total_low", "total_high"),
        ("intercept", ours["fixed_effects"][0], "intercept_low", "intercept_high"),
    ]
    """Mapped each Asterism point to its MCMCglmm posterior interval names."""

    inside_all: dict[str, bool] = {}
    """Accumulated posterior-membership verdicts by scientific quantity."""

    for label, value, low, high in comparisons:
        inside: bool = theirs[low] <= value <= theirs[high]
        """Tested this maximum-likelihood point against its posterior interval."""

        inside_all[label] = inside
        """Recorded the interval-membership verdict under its quantity name."""

        print(
            f"{label:<11} | {value:11.6f} | "
            f"[{theirs[low]:11.6f}, {theirs[high]:9.6f}] | {'yes' if inside else 'NO'}"
        )
        if not inside:
            failures.append(
                f"the maximum-likelihood {label} falls outside the posterior "
                f"interval MCMCglmm drew"
            )

    receipt: dict[str, object] = {
        "what": "the censored model with relatedness, against MCMCglmm's "
        "cengaussian family",
        "date": date.today().isoformat(),
        "people": n,
        "pairs": PAIRS,
        "censored": int(censored.sum()),
        "limit": limit,
        "truth": {
            "heritability": TRUE_HERITABILITY,
            "variance": TRUE_VARIANCE,
            "mean": TRUE_MEAN,
        },
        "asterism": {
            "heritability": ours["heritability"],
            "total_variance": ours["total_variance"],
            "intercept": ours["fixed_effects"][0],
            "converged": ours["converged"],
        },
        "mcmcglmm": theirs,
        "sampling": {"iterations": ITERATIONS, "burnin": BURNIN, "thin": THIN},
        "inside_posterior": inside_all,
        "passed": not failures,
        "failures": failures,
    }
    """Assembled the dated human-facing comparison receipt without raw inputs."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / (f"tobit-against-mcmcglmm-{receipt['date']}.json")
    )
    """Selected the dated receipt path for the ordinary live comparison."""

    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: every maximum-likelihood estimate lies inside the posterior.")
    return 0


if __name__ == "__main__":
    sys.exit(entrypoint())
