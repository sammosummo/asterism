"""An independent bivariate fit, in numpy and scipy.

This NumPy and SciPy implementation provides an independent calculation for the
two-trait model. It was written before the Rust implementation so that the two
routes could be compared without transcribing one into the other.

**The model.** Two traits, additive and residual variance, with the correlations
carried as free parameters:

    Σ_A = [[h₁σ₁,          ρ_G√(h₁σ₁h₂σ₂)],
           [ρ_G√(h₁σ₁h₂σ₂), h₂σ₂         ]]
    Σ_E = [[(1−h₁)σ₁,               ρ_E√((1−h₁)σ₁(1−h₂)σ₂)],
           [ρ_E√((1−h₁)σ₁(1−h₂)σ₂), (1−h₂)σ₂              ]]

    V = Σ_A ⊗ A + Σ_E ⊗ I

Six parameters — σ₁, σ₂, h₁, h₂, ρ_G, ρ_E — and for two traits the constraint
set is simply a box: the variances positive, the heritabilities in [0,1], the
correlations in [−1,1]. Nothing more is needed to keep both covariances
positive semi-definite, which is what makes this parameterisation useful for two
traits even though it does not generalise to three.

**Unbalanced by construction.** A person contributes the rows for the traits
they actually have. Everyone with at least one measured trait is in.

**Checked, twice.** Over 40 replicates at n = 420 with a known truth it recovers
h² 0.590 and 0.384 against 0.6 and 0.4, and correlations 0.693 and 0.199 against
0.7 and 0.2 — biases within one and a half Monte Carlo standard errors, and the
small negative bias on the heritabilities is what maximum likelihood does, which
is why REML exists. Against native SOLAR on real GOBS data, inverse-normalised
height and weight over 1,685 people with 1,673 and 1,684 measured and 1,672
having both, it agrees to 1.6e-07 on the first heritability, 1.7e-06 on the
second, 3.8e-07 on the genetic correlation and 8.3e-07 on the environmental one
— every digit SOLAR prints. SOLAR arrived at the same 1,685-person roster
independently.

**No Kronecker shortcut.** The eigen-rotation that makes the one-trait fit fast
survives extra traits only while every person has every trait. Once missingness
differs by trait the structure is gone, so each family block is built and
factorised directly. On the real rosters the largest family is 160 people, so a
likelihood evaluation is about 0.05 Gflop and this is affordable.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize

LOG_TWO_PI: float = float(np.log(2.0 * np.pi))
"""Cached the Gaussian normalising constant used by every family block."""


def family_blocks(relationship: np.ndarray) -> list[np.ndarray]:
    """Return the index vector for each connected family.

    A relationship matrix over unrelated families is block diagonal, and
    nothing couples the blocks.

    Args:
        relationship: Additive relationship matrix in participant order.

    Returns:
        Connected-component index arrays in deterministic roster order.
    """
    n: int = relationship.shape[0]
    """Read the participant count from the relationship matrix."""

    nonzero: np.ndarray = relationship != 0.0
    """Marked matrix entries that connect participants within a family."""

    seen: np.ndarray = np.zeros(n, dtype=bool)
    """Initialised the participant visitation mask."""

    blocks: list[np.ndarray] = []
    """Initialised the connected-family index arrays."""

    for start in range(n):
        if seen[start]:
            continue
        stack, block = [start], []
        """Started a depth-first traversal from one unseen participant."""

        seen[start] = True
        """Marked the traversal root as visited."""

        while stack:
            index: int = stack.pop()
            """Removed the next participant index from the traversal stack."""

            block.append(index)
            neighbours: np.ndarray = np.flatnonzero(nonzero[index] & ~seen)
            """Found unvisited participants connected to the current person."""

            seen[neighbours] = True
            """Marked newly discovered family members as visited."""

            stack.extend(neighbours.tolist())
        blocks.append(np.array(sorted(block)))
    return blocks


def covariances(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct the two component covariances from six parameters.

    Args:
        theta: Total variances, heritabilities and component correlations.

    Returns:
        The additive and residual two-by-two covariance matrices.
    """
    s1, s2, h1, h2, rg, re = theta
    """Unpacked the fitted parameters in the reference model's fixed order."""

    a1, a2 = h1 * s1, h2 * s2
    """Converted heritabilities to additive variances for both traits."""

    e1, e2 = (1.0 - h1) * s1, (1.0 - h2) * s2
    """Converted complementary shares to residual variances for both traits."""

    sigma_a: np.ndarray = np.array(
        [[a1, rg * np.sqrt(a1 * a2)], [rg * np.sqrt(a1 * a2), a2]]
    )
    """Constructed the additive covariance from variances and correlation."""

    sigma_e: np.ndarray = np.array(
        [[e1, re * np.sqrt(e1 * e2)], [re * np.sqrt(e1 * e2), e2]]
    )
    """Constructed the residual covariance from variances and correlation."""
    return sigma_a, sigma_e


def negative_log_likelihood(
    theta: np.ndarray,
    blocks: list[np.ndarray],
    relationship: np.ndarray,
    observed: np.ndarray,
    y: np.ndarray,
    design: np.ndarray,
    reml: bool,
) -> float:
    """Return the negative profiled log-likelihood.

    Fixed effects are profiled out by generalised least squares at every
    evaluation, as REML requires.

    Args:
        theta: Total variances, heritabilities and component correlations.
        blocks: Connected-family participant index arrays.
        relationship: Additive relationship matrix in participant order.
        observed: Boolean person-by-trait observation mask.
        y: Interleaved two-trait response vector.
        design: Fixed-effect design in the same interleaved row order.
        reml: Whether to include the restricted-likelihood adjustment.

    Returns:
        The negative profiled log-likelihood, or infinity for an infeasible fit.
    """
    sigma_a, sigma_e = covariances(theta)
    """Constructed the additive and residual covariance matrices."""

    # A covariance that is exactly singular is legitimate, not infeasible. At
    # |ρ| = 1 the genetic covariance has rank one — complete pleiotropy — and V
    # is still positive definite, because a semi-definite term plus a definite
    # one is definite. Its smallest eigenvalue comes back a hair negative from
    # rounding, and rejecting on `< 0` refuses exactly the point a test of
    # ρ = ±1 has to evaluate. That is what made the boundary test reject sixty
    # per cent of the time: the constrained refit was handed an infinity and
    # returned the penalty value. The floor is the one `prepared.rs` already
    # uses for the same reason.
    if (
        min(np.linalg.eigvalsh(sigma_a).min(), np.linalg.eigvalsh(sigma_e).min())
        < -1e-9
    ):
        return np.inf

    logdet: float = 0.0
    """Initialised the accumulated observation-covariance log determinant."""

    xvx: np.ndarray = np.zeros((design.shape[1], design.shape[1]))
    """Initialised the generalised least-squares design cross-product."""

    xvy: np.ndarray = np.zeros(design.shape[1])
    """Initialised the generalised least-squares design-response cross-product."""

    yvy: float = 0.0
    """Initialised the generalised least-squares response quadratic form."""

    total_rows: int = 0
    """Initialised the number of observed person-trait rows."""

    for block in blocks:
        # Rows of this family that were actually measured, in (person, trait)
        # order. A person with one trait contributes one row.
        rows: list[tuple[int, int]] = [
            (p, t) for p in block for t in (0, 1) if observed[p, t]
        ]
        """Selected observed person-trait rows within this family."""

        if not rows:
            continue
        size: int = len(rows)
        """Counted the observed rows contributed by this family."""

        total_rows += size
        """Added this family's observed rows to the likelihood total."""

        v: np.ndarray = np.empty((size, size))
        """Initialised this family's observation covariance matrix."""

        for i, (pi, ti) in enumerate(rows):
            for j, (pj, tj) in enumerate(rows):
                v[i, j] = sigma_a[ti, tj] * relationship[pi, pj]
                """Added the relationship-scaled additive covariance entry."""

                if pi == pj:
                    v[i, j] += sigma_e[ti, tj]
                    """Added residual covariance for measurements on one person."""
        try:
            factor: tuple[np.ndarray, bool] = cho_factor(
                v, lower=True, check_finite=False
            )
            """Factorised the positive-definite family covariance."""
        except np.linalg.LinAlgError:
            return np.inf
        logdet += 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
        """Accumulated this family's covariance log determinant."""

        index: list[int] = [p * 2 + t for p, t in rows]
        """Mapped observed person-trait pairs to interleaved response rows."""

        yb: np.ndarray = y[index]
        """Selected this family's observed response values."""

        xb: np.ndarray = design[index]
        """Selected this family's observed fixed-effect rows."""

        vy: np.ndarray = cho_solve(factor, yb, check_finite=False)
        """Applied the inverse family covariance to the response."""

        vx: np.ndarray = cho_solve(factor, xb, check_finite=False)
        """Applied the inverse family covariance to the design."""

        xvx += xb.T @ vx
        """Accumulated the weighted design cross-product."""

        xvy += xb.T @ vy
        """Accumulated the weighted design-response cross-product."""

        yvy += float(yb @ vy)
        """Accumulated the weighted response quadratic form."""

    try:
        beta: np.ndarray = np.linalg.solve(xvx, xvy)
        """Solved the profiled generalised least-squares fixed effects."""
    except np.linalg.LinAlgError:
        return np.inf
    quadratic: float = yvy - float(xvy @ beta)
    """Computed the residual quadratic after profiling fixed effects."""

    if not np.isfinite(quadratic) or quadratic <= 0:
        return np.inf

    p: int = design.shape[1]
    """Read the fixed-effect rank from the validated design columns."""

    value: float = 0.5 * (total_rows * LOG_TWO_PI + logdet + quadratic)
    """Computed the profiled Gaussian negative log-likelihood."""

    if reml:
        sign, logdet_xvx = np.linalg.slogdet(xvx)
        """Computed the sign and log determinant of the weighted design matrix."""

        if sign <= 0:
            return np.inf
        observed_design: np.ndarray = design[observed.reshape(-1)]
        """Selected fixed-effect rows for measurements present in the likelihood."""

        design_sign, logdet_xtx = np.linalg.slogdet(observed_design.T @ observed_design)
        """Computed the ordinary observed-design determinant for basis invariance."""

        if design_sign <= 0:
            return np.inf
        # The restricted likelihood is a density for n - p error contrasts, so
        # it drops p of the 2*pi terms and gains the weighted design determinant.
        # Subtracting the ordinary design determinant makes the reported value
        # invariant to a nonsingular change of fixed-effect basis, matching the
        # one-trait Asterism convention.
        value += 0.5 * (logdet_xvx - logdet_xtx) - 0.5 * p * LOG_TWO_PI
        """Applied the restricted-likelihood and basis-invariance adjustment."""
    return value


def trait_scales(y: np.ndarray, observed: np.ndarray) -> list[float]:
    """Return a variance start for each possibly unbalanced trait.

    Args:
        y: Interleaved two-trait response vector.
        observed: Boolean person-by-trait observation mask.

    Returns:
        One finite variance start for each trait in model order.
    """
    matrix: np.ndarray = y.reshape(observed.shape)
    """Reshaped the interleaved response to person-by-trait form."""
    return [
        float(np.var(matrix[observed[:, trait], trait]))
        if observed[:, trait].any()
        else 1.0
        for trait in (0, 1)
    ]


def fit(
    relationship: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    design: np.ndarray,
    reml: bool = True,
) -> dict[str, float | bool | str]:
    """Fit two traits.

    ``y`` and ``design`` are in person-then-trait order and length ``2n``;
    ``observed`` is ``n`` by two and says which of those rows are real.

    Args:
        relationship: Additive relationship matrix in participant order.
        y: Interleaved two-trait response vector.
        observed: Boolean person-by-trait observation mask.
        design: Fixed-effect design in the same interleaved row order.
        reml: Whether to fit by restricted rather than full maximum likelihood.

    Returns:
        Fitted bivariate parameters, likelihood and convergence diagnostics.
    """
    blocks: list[np.ndarray] = family_blocks(relationship)
    """Split the relationship matrix into independently factorised families."""

    args: tuple[
        list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool
    ] = (blocks, relationship, observed, y, design, reml)
    """Assembled the fixed likelihood arguments shared by every optimiser start."""

    # Start from each trait on its own, which is what the joint fit reduces to
    # when the correlations are zero, and from two additional starting values.
    scale: list[float] = trait_scales(y, observed)
    """Estimated trait-specific total-variance starting scales."""

    starts: list[np.ndarray] = [
        np.array([scale[0], scale[1], 0.5, 0.5, 0.0, 0.0]),
        np.array([scale[0], scale[1], 0.3, 0.3, 0.5, 0.5]),
        np.array([scale[0], scale[1], 0.7, 0.7, -0.3, 0.3]),
    ]
    """Constructed central and correlated multistart parameter vectors."""

    # The direct (h², rho) coordinates are not differentiable where a component
    # variance vanishes, so keep h² just inside its bounds. Exact correlation
    # boundaries remain valid: one component may be rank deficient while their
    # sum, the full observation covariance, is still positive definite.
    edge: float = 1e-5
    """Kept direct heritability coordinates just inside non-differentiable bounds."""

    bounds: list[tuple[float | None, float | None]] = [
        (1e-8, None),
        (1e-8, None),
        (edge, 1.0 - edge),
        (edge, 1.0 - edge),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    """Bounded variances, heritabilities and correlations to their model domains."""

    best: OptimizeResult | None = None
    """Initialised selection of the best finite multistart optimisation result."""

    for start in starts:
        result: OptimizeResult = minimize(
            negative_log_likelihood,
            start,
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
        )
        """Optimised all six parameters from one deterministic starting vector."""

        if best is None or (np.isfinite(result.fun) and result.fun < best.fun):
            best = result
            """Retained the lowest finite objective reached across optimiser starts."""

    s1, s2, h1, h2, rg, re = best.x
    """Unpacked the selected fit in the reference model's parameter order."""
    return {
        "h2_trait_a": h1,
        "h2_trait_b": h2,
        "rho_g": rg,
        "rho_e": re,
        "total_variance_a": s1,
        "total_variance_b": s2,
        "loglik": -best.fun,
        "converged": bool(best.success),
        "estimator": "reml" if reml else "ml",
        "message": best.message,
    }


def fit_fixing(
    relationship: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    design: np.ndarray,
    which: int,
    value: float,
    reml: bool = True,
) -> dict[str, float | bool]:
    """Refit with one parameter held at `value`.

    This is what makes the correlations testable. Carried as free parameters, a
    correlation supports an ordinary likelihood ratio against a constrained
    refit: the constrained model has one fewer free parameter.

    Args:
        relationship: Additive relationship matrix in participant order.
        y: Interleaved two-trait response vector.
        observed: Boolean person-by-trait observation mask.
        design: Fixed-effect design in the same interleaved row order.
        which: Zero-based parameter index to hold fixed.
        value: Exact value imposed on the selected parameter.
        reml: Whether to fit by restricted rather than full maximum likelihood.

    Returns:
        The constrained log-likelihood and convergence verdict.

    Raises:
        ValueError: If the requested fixed value lies outside the model domain.
    """
    blocks: list[np.ndarray] = family_blocks(relationship)
    """Split the relationship matrix into independently factorised families."""

    args: tuple[
        list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool
    ] = (blocks, relationship, observed, y, design, reml)
    """Assembled the fixed likelihood arguments shared by every optimiser start."""

    scale: list[float] = trait_scales(y, observed)
    """Estimated trait-specific total-variance starting scales."""

    edge: float = 1e-5
    """Kept direct heritability coordinates just inside non-differentiable bounds."""

    bounds: list[tuple[float | None, float | None]] = [
        (1e-8, None),
        (1e-8, None),
        (edge, 1.0 - edge),
        (edge, 1.0 - edge),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    """Bounded variances, heritabilities and correlations to their model domains."""

    held: float = float(np.clip(value, bounds[which][0], bounds[which][1] or np.inf))
    """Mapped the requested fixed value through the selected parameter bounds."""

    if held != value:
        raise ValueError(
            "the requested constrained value is outside the exact model domain"
        )
    bounds[which] = (held, held)
    """Collapsed the selected optimiser bound to the exact constrained value."""

    best: OptimizeResult | None = None
    """Initialised selection of the best finite constrained optimisation result."""

    for start in (
        [scale[0], scale[1], 0.5, 0.5, 0.0, 0.0],
        [scale[0], scale[1], 0.3, 0.3, 0.4, 0.4],
    ):
        start: list[float] = list(start)
        """Made the candidate start mutable before imposing the constraint."""

        start[which] = held
        """Inserted the exact constrained value into this starting vector."""

        result: OptimizeResult = minimize(
            negative_log_likelihood,
            np.array(start),
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
        )
        """Optimised the remaining parameters from one deterministic start."""

        if best is None or (np.isfinite(result.fun) and result.fun < best.fun):
            best = result
            """Retained the lowest finite constrained objective across starts."""
    return {"loglik": -best.fun, "converged": bool(best.success)}


def profile_interval(
    relationship: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    design: np.ndarray,
    which: int,
    fitted_value: float,
    reml: bool = True,
    tolerance: float = 1e-4,
) -> dict[str, float | bool]:
    """A 95 per cent profile-likelihood interval, by bisection on the refit.

    This exists to be a second opinion. Every other quantity Asterism reports is
    checked against software written by somebody else — SOLAR for ML, R
    `regress` for REML — but neither of those computes a profile interval for
    these models, so the Rust intervals rested on calibration alone. Calibration
    says the intervals cover; it cannot say the Rust arithmetic matches an
    independent route to the same definition. This is that route.

    **The maximum comes from a constrained refit at the fitted value, not from
    the free fit.** Those are the same number in exact arithmetic and not in
    practice, and taking the free fit's is how the Rust version came to return
    intervals of zero width: its two log-likelihoods were computed in different
    units and the constant that left in the deviance never let it fall below the
    threshold. Deriving both ends the same way makes the difference a deviance
    and nothing else, which is the point.

    Args:
        relationship: Additive relationship matrix in participant order.
        y: Interleaved two-trait response vector.
        observed: Boolean person-by-trait observation mask.
        design: Fixed-effect design in the same interleaved row order.
        which: Zero-based parameter index to profile.
        fitted_value: Selected fit's value of the profiled parameter.
        reml: Whether to use restricted rather than full likelihoods.
        tolerance: Maximum bisection width for each interval endpoint.

    Returns:
        Profile endpoints, bound-limited flags and the fixed confidence level.
    """
    from scipy import stats

    threshold: float = stats.chi2.ppf(0.95, 1)
    """Computed the 95 per cent one-degree-of-freedom deviance threshold."""

    bottom, top = (0.0, 1.0) if which in (2, 3) else (-1.0, 1.0)
    """Selected the exact parameter-space bounds for the profiled quantity."""

    maximum: float = fit_fixing(
        relationship, y, observed, design, which, fitted_value, reml
    )["loglik"]
    """Established the profile maximum through a like-for-like constrained refit."""

    def deviance(value: float) -> float:
        """Return twice the profile log-likelihood loss at one fixed value.

        Args:
            value: Candidate value of the profiled parameter.

        Returns:
            The profile deviance, or infinity where the refit is unavailable.
        """
        try:
            refit: dict[str, float | bool] = fit_fixing(
                relationship, y, observed, design, which, value, reml
            )
            """Refitted the model with the profiled quantity held fixed."""
        except ValueError:
            # `fit_fixing` refuses a heritability of exactly nought or one as
            # outside the exact model domain. A value the model cannot take is
            # a value the interval cannot reach, so it counts as infinitely far
            # rather than as an error. The bisection then converges to just
            # inside the bound, which is where Asterism's own epsilon puts it.
            return np.inf
        if not np.isfinite(refit["loglik"]):
            return np.inf
        return 2.0 * (maximum - refit["loglik"])

    def endpoint(bound: float) -> tuple[float, bool]:
        """Find one profile endpoint by bisection towards a parameter bound.

        Args:
            bound: Lower or upper parameter-space limit.

        Returns:
            The endpoint and whether the parameter bound limited it.
        """
        if deviance(bound) <= threshold:
            # The threshold is never reached: the endpoint is the bound, and the
            # interval is limited by the parameter space rather than the data.
            return bound, True
        inside: float = fitted_value
        """Started the accepted side of the bracket at the fitted value."""

        outside: float = bound
        """Started the rejected side of the bracket at the parameter bound."""

        while abs(outside - inside) > tolerance:
            middle: float = 0.5 * (inside + outside)
            """Bisected the current profile-deviance bracket."""

            if deviance(middle) <= threshold:
                inside = middle
                """Moved the accepted side of the profile bracket inward."""
            else:
                outside = middle
                """Moved the rejected side of the profile bracket inward."""
        return 0.5 * (inside + outside), False

    lower, lower_limited = endpoint(bottom)
    """Located the lower profile endpoint and its bound-limited state."""

    upper, upper_limited = endpoint(top)
    """Located the upper profile endpoint and its bound-limited state."""
    return {
        "lower": lower,
        "upper": upper,
        "lower_limited": lower_limited,
        "upper_limited": upper_limited,
        "level": 0.95,
    }


def correlation_tests(
    relationship: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    design: np.ndarray,
    fitted: dict[str, float | bool | str],
    reml: bool = True,
) -> dict[str, dict[str, float | bool | str]]:
    """The three tests on the genetic correlation, and one on the environmental.

    Against zero the value is interior and the statistic is chi-square on one
    degree of freedom. Against plus or minus one it sits on a bound, and with the
    correlation carried as a free parameter and both variances positive that is a
    single parameter on a smooth one-sided boundary — the well-behaved case, so
    the Self–Liang 50:50 mixture applies.

    The following older simulation used the previous optimiser and approximate
    heritability states. It showed that the old boundary refits were unusable.
    Simulation on 11 August 2026, 120 replicates at n = 240:

        nominal   rho_g = 0   rho_g = 1
           0.01       0.017       0.608
           0.05       0.033       0.733
           0.10       0.133       0.783
           0.25       0.250       0.817
           0.50       0.517       0.825

    In that historical run every interior level sat inside its binomial band.
    The boundary test rejected six times in ten at a nominal one in a hundred —
    it would have called a genetic correlation different from one in most
    samples where it was exactly one.

    So the reasoning that produced it was wrong somewhere. The likely place is
    not the mixture itself but the constrained refit: holding the correlation at
    its bound makes the genetic covariance near-singular, the refit converges
    badly, its log-likelihood comes out too low, and the statistic is inflated by
    the optimiser rather than by the data. A parametric bootstrap is the fallback
    when this boundary approximation does not hold.

    **Recalibrated on 12 August 2026, after the fix, and they now hold.** 200
    replicates at n = 240:

        nominal   rho_g = 0   rho_g = 1   was
           0.01       0.015       0.000   0.608
           0.05       0.035       0.030   0.733
           0.10       0.115       0.085   0.783
           0.25       0.245       0.210   0.817
           0.50       0.495       0.710   0.825

    The interior test is right at every level; the boundary test is right, and
    slightly conservative, at every level anybody reports. It still departs at a
    half, because the estimate lands exactly on the bound in 29 per cent of
    replicates where the mixture expects 50 and 1 − 0.29 = 0.71 is what is
    observed. The search sometimes stops at 0.9999 rather than exactly one. That
    error runs in the safe direction at conventional levels.

    These are the *reference's* tests. Correcting the note that stood here: the
    Rust now has correlation tests of its own and profile intervals, and both are
    calibrated -- `checks/bivariate_calibration.py`, 12 August 2026.

    The Rust is calibrated slightly better at the boundary, and for a reason
    worth recording. The departure above comes from the estimate landing exactly
    on the bound in only 29 per cent of replicates where the mixture expects 50,
    because this implementation's search stops at 0.9999 rather than at one. The
    Rust carries the bound as an exact state rather than approaching it, and its
    atom is 55 per cent -- near enough the theory that its boundary test sits
    inside the band at the levels anybody reports.

    Args:
        relationship: Additive relationship matrix in participant order.
        y: Interleaved two-trait response vector.
        observed: Boolean person-by-trait observation mask.
        design: Fixed-effect design in the same interleaved row order.
        fitted: Unconstrained reference fit supplying the maximum likelihood.
        reml: Whether to use restricted rather than full likelihoods.

    Returns:
        Named genetic and residual correlation tests with reference rules.
    """
    from math import erfc, sqrt

    def chi2_one_tail(statistic: float) -> float:
        """Return a one-degree-of-freedom chi-square upper-tail probability.

        Args:
            statistic: Non-negative likelihood-ratio statistic.

        Returns:
            The exact one-degree-of-freedom upper-tail probability.
        """
        return 1.0 if statistic <= 0 else erfc(sqrt(statistic / 2.0))

    tests: dict[str, dict[str, float | bool | str]] = {}
    """Initialised the named constrained correlation-test records."""

    for label, which, value, interior in (
        ("rho_g = 0", 4, 0.0, True),
        ("rho_g = 1", 4, 1.0, False),
        ("rho_g = -1", 4, -1.0, False),
        ("rho_e = 0", 5, 0.0, True),
    ):
        null: dict[str, float | bool] = fit_fixing(
            relationship, y, observed, design, which, value, reml
        )
        """Refitted the model under this exact correlation constraint."""

        statistic: float = max(0.0, 2.0 * (fitted["loglik"] - null["loglik"]))
        """Computed the non-negative likelihood-ratio statistic."""

        if interior:
            p: float = chi2_one_tail(statistic)
            """Computed the ordinary chi-square tail for an interior null."""

            rule: str = "chi2_1"
            """Named the interior-null reference distribution."""
        else:
            # Half the null's mass sits at zero because the parameter is on its
            # bound; at a statistic of exactly zero nothing can be exceeded.
            p = 1.0 if statistic <= 0.0 else 0.5 * chi2_one_tail(statistic)
            """Computed the half-mixture tail for a correlation-boundary null."""

            rule = "mixture_50_50"
            """Named the one-sided boundary reference distribution."""

        tests[label] = {
            "statistic": statistic,
            "p_value": p,
            "rule": rule,
            "null_loglik": null["loglik"],
            "converged": null["converged"],
            # Both are calibrated at the levels anybody reports. The boundary
            # test is conservative there and departs at a half, for the reason
            # in this function's note.
            "calibrated": True,
        }
        """Recorded this constrained test and its calibrated reference rule."""
    return tests
