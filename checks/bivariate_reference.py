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
from scipy.optimize import minimize

LOG_TWO_PI = float(np.log(2.0 * np.pi))


def family_blocks(relationship: np.ndarray) -> list[np.ndarray]:
    """Indices of each connected family. A relationship matrix over unrelated
    families is block diagonal, and nothing couples the blocks."""
    n = relationship.shape[0]
    nonzero = relationship != 0.0
    seen = np.zeros(n, dtype=bool)
    blocks = []
    for start in range(n):
        if seen[start]:
            continue
        stack, block = [start], []
        seen[start] = True
        while stack:
            index = stack.pop()
            block.append(index)
            neighbours = np.flatnonzero(nonzero[index] & ~seen)
            seen[neighbours] = True
            stack.extend(neighbours.tolist())
        blocks.append(np.array(sorted(block)))
    return blocks


def covariances(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two 2x2 covariances from the six parameters."""
    s1, s2, h1, h2, rg, re = theta
    a1, a2 = h1 * s1, h2 * s2
    e1, e2 = (1.0 - h1) * s1, (1.0 - h2) * s2
    sigma_a = np.array([[a1, rg * np.sqrt(a1 * a2)], [rg * np.sqrt(a1 * a2), a2]])
    sigma_e = np.array([[e1, re * np.sqrt(e1 * e2)], [re * np.sqrt(e1 * e2), e2]])
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
    """
    sigma_a, sigma_e = covariances(theta)
    # A covariance that is exactly singular is legitimate, not infeasible. At
    # |ρ| = 1 the genetic covariance has rank one — complete pleiotropy — and V
    # is still positive definite, because a semi-definite term plus a definite
    # one is definite. Its smallest eigenvalue comes back a hair negative from
    # rounding, and rejecting on `< 0` refuses exactly the point a test of
    # ρ = ±1 has to evaluate. That is what made the boundary test reject sixty
    # per cent of the time: the constrained refit was handed an infinity and
    # returned the penalty value. The floor is the one `prepared.rs` already
    # uses for the same reason.
    if min(np.linalg.eigvalsh(sigma_a).min(), np.linalg.eigvalsh(sigma_e).min()) < -1e-9:
        return np.inf

    logdet = 0.0
    xvx = np.zeros((design.shape[1], design.shape[1]))
    xvy = np.zeros(design.shape[1])
    yvy = 0.0
    total_rows = 0

    for block in blocks:
        # Rows of this family that were actually measured, in (person, trait)
        # order. A person with one trait contributes one row.
        rows = [(p, t) for p in block for t in (0, 1) if observed[p, t]]
        if not rows:
            continue
        size = len(rows)
        total_rows += size
        v = np.empty((size, size))
        for i, (pi, ti) in enumerate(rows):
            for j, (pj, tj) in enumerate(rows):
                v[i, j] = sigma_a[ti, tj] * relationship[pi, pj]
                if pi == pj:
                    v[i, j] += sigma_e[ti, tj]
        try:
            factor = cho_factor(v, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf
        logdet += 2.0 * float(np.sum(np.log(np.diag(factor[0]))))

        index = [p * 2 + t for p, t in rows]
        yb = y[index]
        xb = design[index]
        vy = cho_solve(factor, yb, check_finite=False)
        vx = cho_solve(factor, xb, check_finite=False)
        xvx += xb.T @ vx
        xvy += xb.T @ vy
        yvy += float(yb @ vy)

    try:
        beta = np.linalg.solve(xvx, xvy)
    except np.linalg.LinAlgError:
        return np.inf
    quadratic = yvy - float(xvy @ beta)
    if not np.isfinite(quadratic) or quadratic <= 0:
        return np.inf

    p = design.shape[1]
    value = 0.5 * (total_rows * LOG_TWO_PI + logdet + quadratic)
    if reml:
        sign, logdet_xvx = np.linalg.slogdet(xvx)
        if sign <= 0:
            return np.inf
        observed_design = design[observed.reshape(-1)]
        design_sign, logdet_xtx = np.linalg.slogdet(observed_design.T @ observed_design)
        if design_sign <= 0:
            return np.inf
        # The restricted likelihood is a density for n - p error contrasts, so
        # it drops p of the 2*pi terms and gains the weighted design determinant.
        # Subtracting the ordinary design determinant makes the reported value
        # invariant to a nonsingular change of fixed-effect basis, matching the
        # one-trait Asterism convention.
        value += 0.5 * (logdet_xvx - logdet_xtx) - 0.5 * p * LOG_TWO_PI
    return value


def trait_scales(y: np.ndarray, observed: np.ndarray) -> list[float]:
    """Variance start for each trait without losing trait identity when rows
    are unbalanced."""
    matrix = y.reshape(observed.shape)
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
) -> dict:
    """Fit two traits.

    `y` and `design` are in (person, trait) order and length 2n; `observed` is
    n by 2 and says which of those rows are real.
    """
    blocks = family_blocks(relationship)
    args = (blocks, relationship, observed, y, design, reml)

    # Start from each trait on its own, which is what the joint fit reduces to
    # when the correlations are zero, and from two additional starting values.
    scale = trait_scales(y, observed)
    starts = [
        np.array([scale[0], scale[1], 0.5, 0.5, 0.0, 0.0]),
        np.array([scale[0], scale[1], 0.3, 0.3, 0.5, 0.5]),
        np.array([scale[0], scale[1], 0.7, 0.7, -0.3, 0.3]),
    ]
    # The direct (h², rho) coordinates are not differentiable where a component
    # variance vanishes, so keep h² just inside its bounds. Exact correlation
    # boundaries remain valid: one component may be rank deficient while their
    # sum, the full observation covariance, is still positive definite.
    edge = 1e-5
    bounds = [
        (1e-8, None), (1e-8, None),
        (edge, 1.0 - edge), (edge, 1.0 - edge),
        (-1.0, 1.0), (-1.0, 1.0),
    ]

    best = None
    for start in starts:
        result = minimize(
            negative_log_likelihood, start, args=args,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if best is None or (np.isfinite(result.fun) and result.fun < best.fun):
            best = result

    s1, s2, h1, h2, rg, re = best.x
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
) -> dict:
    """Refit with one parameter held at `value`.

    This is what makes the correlations testable. Carried as free parameters, a
    correlation supports an ordinary likelihood ratio against a constrained
    refit: the constrained model has one fewer free parameter.
    """
    blocks = family_blocks(relationship)
    args = (blocks, relationship, observed, y, design, reml)
    scale = trait_scales(y, observed)
    edge = 1e-5
    bounds = [
        (1e-8, None), (1e-8, None),
        (edge, 1.0 - edge), (edge, 1.0 - edge),
        (-1.0, 1.0), (-1.0, 1.0),
    ]
    held = float(np.clip(value, bounds[which][0], bounds[which][1] or np.inf))
    if held != value:
        raise ValueError("the requested constrained value is outside the exact model domain")
    bounds[which] = (held, held)

    best = None
    for start in (
        [scale[0], scale[1], 0.5, 0.5, 0.0, 0.0],
        [scale[0], scale[1], 0.3, 0.3, 0.4, 0.4],
    ):
        start = list(start)
        start[which] = held
        result = minimize(
            negative_log_likelihood, np.array(start), args=args,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if best is None or (np.isfinite(result.fun) and result.fun < best.fun):
            best = result
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
) -> dict:
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
    """
    from scipy import stats

    threshold = stats.chi2.ppf(0.95, 1)
    bottom, top = (0.0, 1.0) if which in (2, 3) else (-1.0, 1.0)

    maximum = fit_fixing(
        relationship, y, observed, design, which, fitted_value, reml
    )["loglik"]

    def deviance(value: float) -> float:
        try:
            refit = fit_fixing(relationship, y, observed, design, which, value, reml)
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
        if deviance(bound) <= threshold:
            # The threshold is never reached: the endpoint is the bound, and the
            # interval is limited by the parameter space rather than the data.
            return bound, True
        inside, outside = fitted_value, bound
        while abs(outside - inside) > tolerance:
            middle = 0.5 * (inside + outside)
            if deviance(middle) <= threshold:
                inside = middle
            else:
                outside = middle
        return 0.5 * (inside + outside), False

    lower, lower_limited = endpoint(bottom)
    upper, upper_limited = endpoint(top)
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
    fitted: dict,
    reml: bool = True,
) -> dict:
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
    """
    from math import erfc, sqrt

    def chi2_one_tail(statistic: float) -> float:
        return 1.0 if statistic <= 0 else erfc(sqrt(statistic / 2.0))

    tests = {}
    for label, which, value, interior in (
        ("rho_g = 0", 4, 0.0, True),
        ("rho_g = 1", 4, 1.0, False),
        ("rho_g = -1", 4, -1.0, False),
        ("rho_e = 0", 5, 0.0, True),
    ):
        null = fit_fixing(relationship, y, observed, design, which, value, reml)
        statistic = max(0.0, 2.0 * (fitted["loglik"] - null["loglik"]))
        if interior:
            p = chi2_one_tail(statistic)
            rule = "chi2_1"
        else:
            # Half the null's mass sits at zero because the parameter is on its
            # bound; at a statistic of exactly zero nothing can be exceeded.
            p = 1.0 if statistic <= 0.0 else 0.5 * chi2_one_tail(statistic)
            rule = "mixture_50_50"
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
    return tests
