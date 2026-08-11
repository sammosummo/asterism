"""An independent bivariate fit, in numpy and scipy.

`docs/adr/0006` makes an independent implementation a condition of entry for any
capability, and `0001` decision 21 repeats it: a capability arrives with its own
independent check or it does not arrive. This is that check for two traits, and
it exists before the Rust does deliberately — it is what the Rust will be
measured against, and writing it second would make it a transcription rather
than an independent route.

**The model.** Two traits, additive and residual variance, with the correlations
carried as free parameters (`0001` decision 29):

    Σ_A = [[h₁σ₁,          ρ_G√(h₁σ₁h₂σ₂)],
           [ρ_G√(h₁σ₁h₂σ₂), h₂σ₂         ]]
    Σ_E = [[(1−h₁)σ₁,               ρ_E√((1−h₁)σ₁(1−h₂)σ₂)],
           [ρ_E√((1−h₁)σ₁(1−h₂)σ₂), (1−h₂)σ₂              ]]

    V = Σ_A ⊗ A + Σ_E ⊗ I

Six parameters — σ₁, σ₂, h₁, h₂, ρ_G, ρ_E — and for two traits the constraint
set is simply a box: the variances positive, the heritabilities in [0,1], the
correlations in [−1,1]. Nothing more is needed to keep both covariances
positive semi-definite, which is what makes this parameterisation worth having
at two traits even though it fails at three (`0001` decision 5).

**Unbalanced by construction** (`0001` decision 9). A person contributes the
rows for the traits they actually have. Everyone with at least one measured
trait is in.

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
    """Minus the profiled log-likelihood. Fixed effects are profiled out by
    generalised least squares at every evaluation, as REML does anyway
    (`0001` decision 10)."""
    sigma_a, sigma_e = covariances(theta)
    if min(np.linalg.eigvalsh(sigma_a).min(), np.linalg.eigvalsh(sigma_e).min()) < 0:
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
        # The restricted likelihood is a density for n - p error contrasts, so
        # it drops p of the 2*pi terms and gains the design's determinant.
        value += 0.5 * logdet_xvx - 0.5 * p * LOG_TWO_PI
    return value


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
    # when the correlations are zero, and from a couple of insurance starts
    # (`0001` decision 14: one good warm start, not many arbitrary ones).
    scale = [float(np.var(y[observed.reshape(-1)][t::2])) if observed[:, t].any() else 1.0
             for t in (0, 1)]
    starts = [
        np.array([scale[0], scale[1], 0.5, 0.5, 0.0, 0.0]),
        np.array([scale[0], scale[1], 0.3, 0.3, 0.5, 0.5]),
        np.array([scale[0], scale[1], 0.7, 0.7, -0.3, 0.3]),
    ]
    bounds = [
        (1e-8, None), (1e-8, None),
        (0.0, 1.0), (0.0, 1.0),
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
