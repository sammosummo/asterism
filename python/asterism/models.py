"""The models beyond one trait with one component.

Each is a small object holding the matrices, with `fit`, `interval` and a test.
Building one validates; fitting returns a dictionary with named fields rather
than a tuple whose meaning has to be remembered.

**This layer exists because the compiled bindings are positional.**
`_core.component_fit` returns eight values in a fixed order and
`_core.spatial_fit` eight more, and reading the fourth of eight correctly every
time is not a reasonable thing to ask of an analysis script. Nothing here computes anything: every number
comes from the same compiled code, and this only names it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core

__all__ = [
    "ComponentModel",
    "BivariateModel",
    "SpatialModel",
    "kinship_classes",
]


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name.upper()}_NOT_TWO_DIMENSIONAL")
    return array


class ComponentModel:
    """One trait with any number of variance components.

    The residual is added for you and is always last, so a model built with a
    relationship matrix and a household matrix reports three shares: additive,
    household, residual.

    Parameters
    ----------
    matrices
        The structured components, in the order you want them reported.
    x
        The fixed-effect design, one row per person, including its own
        intercept column if one is wanted.
    """

    def __init__(self, matrices: list[Any], x: Any) -> None:
        if not matrices:
            raise ValueError("COMPONENTS_NONE_GIVEN")
        self._matrices = [_matrix(m, f"matrix_{i}") for i, m in enumerate(matrices)]
        self._x = _matrix(x, "design")
        self.components = len(self._matrices) + 1

    def fit(self, y: Any, reml: bool = True) -> dict[str, Any]:
        """Fit, and return the variances and their shares.

        The share is what gets reported: the first share of a
        relationship-plus-residual model is the heritability.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        (
            variances,
            shares,
            total,
            loglik,
            gradient,
            converged,
            effects,
            errors,
        ) = _core.component_fit(self._matrices, self._x, y, reml)
        return {
            "variances": list(variances),
            "shares": list(shares),
            "total_variance": total,
            # The generalised least squares estimates: the best linear unbiased
            # estimator of the fixed effects at the fitted variances, with the
            # standard errors from the diagonal of (X' V^-1 X)^-1.
            "fixed_effects": [
                {"estimate": e, "standard_error": s}
                for e, s in zip(effects, errors)
            ],
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            "estimator": "reml" if reml else "ml",
        }

    def predict(self, y: Any, component: int, reml: bool = True) -> dict[str, Any]:
        """Predict the random effects of one component.

        The best linear unbiased prediction, one value per person, with the
        standard error of prediction beside it.

        **The error is how far the prediction may be from the effect**, not the
        spread of the predictions. A prediction is shrunk toward nought, so its
        own spread is smaller than the effect's; the question worth answering is
        how wrong it might be.

        The variance components are treated as known, though they were estimated
        from the same data, so the errors are a little optimistic. That is the
        usual approximation and the same one the fixed effects make.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        values, errors = _core.component_blup(
            self._matrices, self._x, y, component, reml
        )
        return {
            "component": component,
            "values": list(values),
            "errors": list(errors),
        }

    def interval(self, y: Any, component: int, reml: bool = True) -> dict[str, Any]:
        """A 95 per cent profile interval for one component's share."""
        y = np.ascontiguousarray(y, dtype=np.float64)
        lower, upper, at_lower, at_upper, level = _core.component_interval(
            self._matrices, self._x, y, component, reml
        )
        return {
            "lower": lower,
            "upper": upper,
            "lower_at_bound": at_lower,
            "upper_at_bound": at_upper,
            "level": level,
        }

    def test(self, y: Any, component: int, reml: bool = True) -> dict[str, Any]:
        """Test one component against having no variance at all.

        The null sits on the edge of the parameter space, so the reference is
        the Self–Liang half-and-half mixture and not a plain chi-squared.

        It refuses where another component has itself gone to nought, because
        the mixture assumes only one is on the boundary; a number there would be
        a p-value for a question nobody asked.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        statistic, p_value, rule, null_loglik = _core.component_test(
            self._matrices, self._x, y, component, reml
        )
        return {
            "statistic": statistic,
            "p_value": p_value,
            "rule": rule,
            "null_loglik": null_loglik,
        }


class BivariateModel:
    """Two traits fitted jointly, on possibly unbalanced data.

    Parameters
    ----------
    k
        The relationship matrix, one row per person.
    observed
        One pair of booleans per person, saying which of the two traits they
        have. People missing either trait are the ordinary case.
    design
        The fixed-effect design over the observed person–trait rows, in person
        order with trait within person.
    """

    QUANTITIES = ("h2_first", "h2_second", "rho_g", "rho_e", "rho_p")

    def __init__(self, k: Any, observed: Any, design: Any) -> None:
        self._k = _matrix(k, "relationship")
        self._observed = [[bool(a), bool(b)] for a, b in observed]
        self._design = _matrix(design, "design")

    def fit(self, y: Any, reml: bool = True) -> dict[str, Any]:
        """Fit, and return both heritabilities and all three correlations.

        The phenotypic correlation is derived from the others rather than
        estimated, which is why it has no variance of its own.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        theta, loglik, gradient, converged = _core.bivariate_fit(
            self._k, self._observed, self._design, y, reml
        )
        return {
            "total_variance": [theta[0], theta[1]],
            "h2_first": theta[2],
            "h2_second": theta[3],
            "rho_g": theta[4],
            "rho_e": theta[5],
            "rho_p": theta[6],
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            "estimator": "reml" if reml else "ml",
        }

    def interval(self, y: Any, quantity: str, reml: bool = True) -> dict[str, Any]:
        """A 95 per cent profile interval for one reported quantity."""
        if quantity not in self.QUANTITIES:
            raise ValueError("BIVARIATE_QUANTITY_UNKNOWN")
        y = np.ascontiguousarray(y, dtype=np.float64)
        lower, upper, at_lower, at_upper, level = _core.bivariate_interval(
            self._k, self._observed, self._design, y, quantity, reml
        )
        return {
            "lower": lower,
            "upper": upper,
            "lower_at_bound": at_lower,
            "upper_at_bound": at_upper,
            "level": level,
        }

    def test(
        self, y: Any, quantity: str, null: float = 0.0, reml: bool = True
    ) -> dict[str, Any]:
        """Test one correlation against a fixed value.

        Against nought the value is interior and a plain chi-squared applies;
        against plus or minus one it sits on a bound and the Self–Liang mixture
        does. Heritabilities are not testable this way.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        statistic, p_value, rule, null_loglik = _core.bivariate_correlation_test(
            self._k, self._observed, self._design, y, quantity, null, reml
        )
        return {
            "statistic": statistic,
            "p_value": p_value,
            "rule": rule,
            "null_loglik": null_loglik,
        }


class SpatialModel:
    """One trait with a spatial component whose range is estimated.

    The kernel is ``exp(-λd)`` with distances in kilometres. Any fixed
    components — a relationship matrix, a household matrix — are passed
    alongside and are reported before the spatial share, which is followed by
    the residual.

    **Two cautions that the numbers do not carry themselves.** The range is
    barely estimated: its interval reaches a bound in 98 per cent of calibration
    replicates, so report it as a point estimate or use ``integrated=True`` and
    be rid of it. And an interval for the spatial share reaching nought is not a
    test of whether there is a spatial effect — under that null the range is
    unidentified, which is why the test is a bootstrap.
    """

    def __init__(self, fixed: list[Any], distance: Any, design: Any) -> None:
        self._fixed = [_matrix(m, f"matrix_{i}") for i, m in enumerate(fixed)]
        self._distance = _matrix(distance, "distance")
        self._design = _matrix(design, "design")

    def fit(self, y: Any, reml: bool = True, integrated: bool = False) -> dict[str, Any]:
        """Fit, taking the range as a free parameter or integrating it out.

        With ``integrated=True`` the range is averaged over rather than
        maximised over, and comes back as ``None``: there is nothing estimated
        to report, which is the point.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        (
            variances,
            shares,
            total,
            lam,
            half,
            loglik,
            gradient,
            converged,
            effects,
            errors,
        ) = _core.spatial_fit(self._fixed, self._distance, self._design, y, reml, integrated)
        return {
            "variances": list(variances),
            "shares": list(shares),
            "total_variance": total,
            # With the range integrated out these carry the extra uncertainty of
            # not knowing it: the standard errors are the average of the
            # within-range ones plus the spread of the estimates across ranges.
            "fixed_effects": [
                {"estimate": e, "standard_error": s}
                for e, s in zip(effects, errors)
            ],
            "decay_per_km": None if integrated else lam,
            "half_distance_km": None if integrated else half,
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            "estimator": "reml" if reml else "ml",
            "range_treatment": "integrated" if integrated else "profile",
        }

    def predict(
        self, y: Any, component: int, reml: bool = True, integrated: bool = False
    ) -> dict[str, Any]:
        """Predict the random effects of one component.

        `component` indexes the fixed components first and then the spatial one.
        With the range integrated out this is refused: there is no single kernel
        to predict from, and averaging predictions across the grid is a
        different quantity that has not been calibrated.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        values, errors = _core.spatial_blup(
            self._fixed, self._distance, self._design, y, component, reml, integrated
        )
        return {"component": component, "values": list(values), "errors": list(errors)}

    def interval(
        self, y: Any, quantity: str, reml: bool = True, integrated: bool = False
    ) -> dict[str, Any]:
        """An interval for a component's share, or for the range.

        ``quantity`` is a component index as a string, or ``"lambda"``. Asking
        for the range when it has been integrated out is refused rather than
        answered.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        lower, upper, at_lower, at_upper, level = _core.spatial_interval(
            self._fixed, self._distance, self._design, y, quantity, reml, integrated
        )
        return {
            "lower": lower,
            "upper": upper,
            "lower_at_bound": at_lower,
            "upper_at_bound": at_upper,
            "level": level,
        }

    def statistic(self, y: Any, reml: bool = True, integrated: bool = False) -> float:
        """The likelihood ratio against no spatial variance.

        A statistic and not a p-value: with the range unidentified under that
        null there is no closed-form reference, so a p-value takes `bootstrap`.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        return _core.spatial_statistic(
            self._fixed, self._distance, self._design, y, reml, integrated
        )

    def bootstrap(
        self,
        y: Any,
        replicates: int = 199,
        seed: int = 1,
        reml: bool = True,
        integrated: bool = False,
    ) -> dict[str, Any]:
        """The parametric bootstrap p-value for no spatial variance.

        The p-value adds one to both counts, so it can never be nought however
        extreme the statistic; the smallest it can report is
        ``1 / (replicates + 1)``.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        observed, exceedances, used, requested, p_value, rule = _core.spatial_bootstrap(
            self._fixed,
            self._distance,
            self._design,
            y,
            replicates,
            seed,
            reml,
            integrated,
        )
        return {
            "statistic": observed,
            "exceedances": exceedances,
            "replicates": used,
            "requested": requested,
            "p_value": p_value,
            "rule": rule,
            "seed": seed,
            "smallest_reportable": 1.0 / (used + 1),
        }


class GxeModel:
    """One trait whose genetic effects may act differently across an environment.

    The genetic covariance between two people becomes their relationship times a
    surface in their two environments, and the residual variance a surface in
    one. Two surfaces are available and they are given the same data and asked
    the same questions:

    - ``"exponential"``: the genetic and residual variances are log-linear in
      the environment, and genetic effects a distance apart in the environment
      correlate as ``exp(-λ|Δ|)``. Five parameters.
    - ``"random_regression"``: a smooth quadratic surface on each covariance,
      held by its Cholesky factor so it stays a covariance. Six parameters.

    Nothing is reported in either surface's own coordinates. What comes back is
    the heritability at each environment you ask about and the genetic
    correlation between each pair — quantities that mean the same thing whichever
    surface produced them.

    **Only the smooth surface can represent a crossover** — a genotype that
    helps in one environment and harms in another, so the genetic correlation
    falls below nought rather than merely below one. The exponential surface
    correlates two environments as ``exp(-λ|Δ|)``, which is positive at every
    rate, so it will report a correlation near nought where the truth is near
    minus one. If a crossover is on the table, the choice is already made.

    **Choose the surface before looking at the answer, and know what it costs to
    choose wrongly.** Neither family contains the other, and a rank-one genetic
    surface from one is misspecified for the other. The misfit goes into the one
    parameter that is free under the alternative and pinned under the null, so a
    surface that cannot bend its variance function the way the data does will
    bend its correlation instead. In calibration over 2000 samples with a linear rank-one genetic
    surface and no reordering at all, the exponential surface rejected
    ``test(y, "correlation")`` on 10.7 per cent of them against a nominal 5. The
    smooth surface was conservative rather than anti-conservative in the mirror
    case, which is why it is the default.
    Fitting both and reporting whichever rejects is not a defensible procedure.

    **A correlation below one is not by itself evidence of an interaction.** The
    estimate cannot exceed one, so under the null every departure runs downward:
    on the exponential surface a tenth of null samples came back below 0.25. Use
    ``test``.
    """

    def __init__(self, relationship: Any, environment: Any, design: Any,
                 surface: str = "random_regression") -> None:
        if surface not in ("exponential", "random_regression"):
            raise ValueError(
                f"surface must be exponential or random_regression, not {surface!r}"
            )
        self._relationship = _matrix(relationship, "relationship")
        self._environment = [float(v) for v in np.asarray(environment).ravel()]
        self._design = _matrix(design, "design")
        self._surface = surface

    def fit(self, y: Any, grid: Any = (-1.0, 0.0, 1.0), reml: bool = True) -> dict[str, Any]:
        """Fit, and report the surface at the environments in ``grid``.

        ``grid`` is in the environment's own units, so it should be chosen from
        the data — quantiles of the observed environment usually. The genetic
        correlations come back as a square list of lists in the grid's order.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        grid = [float(v) for v in np.asarray(grid).ravel()]
        (
            parameters,
            loglik,
            converged,
            gradient,
            effects,
            errors,
            genetic,
            residual,
            heritability,
            correlations,
        ) = _core.gxe_fit(
            self._relationship, self._environment, self._design, y,
            self._surface, grid, reml,
        )
        width = len(grid)
        return {
            "surface": self._surface,
            # Kept so a fit can be reproduced and inspected. They are not the
            # answer and do not mean the same thing across surfaces.
            "parameters": list(parameters),
            "environment": grid,
            "genetic_variance": list(genetic),
            "residual_variance": list(residual),
            "heritability": list(heritability),
            "genetic_correlation": [
                list(correlations[row * width:(row + 1) * width]) for row in range(width)
            ],
            "fixed_effects": [
                {"estimate": e, "standard_error": s} for e, s in zip(effects, errors)
            ],
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            "estimator": "reml" if reml else "ml",
        }

    def test(self, y: Any, null: str = "correlation", reml: bool = True) -> dict[str, Any]:
        """Test one of the two genotype-by-environment nulls.

        - ``"correlation"``: the genetic effects at any two environments are the
          same effects. The genetic variance may still change; what is ruled out
          is a change in *which* genes matter. This is the narrower claim and
          usually the interesting one — a heritability that rises with an
          environment can follow from a change of scale in the measurement, and a
          correlation below one cannot.
        - ``"interaction"``: the genetic covariance does not involve the
          environment at all. Rejecting says something about genes and
          environment together, but not what.
        - ``"variance"``: the genetic variance does not change with the
          environment. This is the recovered SOLAR model's own ``gamma_G = 0``
          null, and on the exponential surface it is one interior coordinate
          referred to chi-square on one degree of freedom. On the smooth
          surface it is not a separate test — a quadratic genetic variance is
          constant only when both its shape coordinates are nought, which is
          the interaction null — so it returns that instead of a differently
          named copy.

        The residual surface is free under both nulls, so a residual variance
        that changes with the environment is not mistaken for a genetic one.
        """
        if null not in ("interaction", "correlation", "variance"):
            raise ValueError(
                f"null must be interaction, correlation or variance, not {null!r}"
            )
        y = np.ascontiguousarray(y, dtype=np.float64)
        statistic, p_value, rule, null_loglik, alternative_loglik = _core.gxe_test(
            self._relationship, self._environment, self._design, y,
            self._surface, null, reml,
        )
        return {
            "surface": self._surface,
            "null": null,
            "statistic": statistic,
            "p_value": p_value,
            # `mixture_50_50` for the correlation null, which holds one bounded
            # coordinate; `half_chi2_1_half_chi2_2` for the interaction null,
            # which holds one bounded and one free; `chi2_1` for the variance
            # null on the exponential surface, which holds one interior one.
            "rule": rule,
            "null_loglik": null_loglik,
            "alternative_loglik": alternative_loglik,
            "estimator": "reml" if reml else "ml",
        }

    def interval(
        self,
        y: Any,
        quantity: str = "heritability",
        first: float = 0.0,
        second: float = 0.0,
        reml: bool = True,
    ) -> dict[str, Any]:
        """A 95 per cent profile-likelihood interval for a reported quantity.

        ``"heritability"`` uses ``first`` as the environment; ``"correlation"``
        uses both, and is the genetic correlation between them.

        The quantity is held by solving one coordinate of the surface for it and
        re-maximising over the rest, so the interval is a likelihood one and not
        a Wald one — it is not symmetric about the estimate and does not have to
        be. ``lower_at_bound`` or ``upper_at_bound`` says the endpoint ran to
        the edge of what the quantity can be rather than to a likelihood
        crossing, which is a limit of the model rather than a measurement.
        """
        if quantity not in ("heritability", "correlation"):
            raise ValueError(
                f"quantity must be heritability or correlation, not {quantity!r}"
            )
        y = np.ascontiguousarray(y, dtype=np.float64)
        estimate, lower, upper, at_lower, at_upper = _core.gxe_interval(
            self._relationship, self._environment, self._design, y,
            self._surface, quantity, float(first), float(second), reml,
        )
        return {
            "surface": self._surface,
            "quantity": quantity,
            "environment": [first] if quantity == "heritability" else [first, second],
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "lower_at_bound": at_lower,
            "upper_at_bound": at_upper,
            "level": 0.95,
            "estimator": "reml" if reml else "ml",
        }


def kinship_classes(
    ids: list[str],
    father: list[str | None],
    mother: list[str | None],
    sex: list[str | None],
    keep: list[str] | None = None,
) -> dict[str, Any]:
    """Split the relationship matrix by the kind of parent–offspring tie.

    Returns the five matrices the class-weighted kinship model wants —
    everything else, then mother–son, mother–daughter, father–son,
    father–daughter — with the order their rows are in, the class names, and how
    many pairs fell in each class.

    Hand ``matrices`` to `ComponentModel` and report each class as a **share**.
    A weight, being a ratio of two estimated variances, comes back near three
    when the truth is one on a design of this size; the shares are unbiased.

    The pair counts are worth reading before the answer is. They are rarely
    balanced, and a class with few pairs is a class whose share is least
    determined.
    """
    matrices, order, names, pairs = _core.kinship_classes(
        ids, father, mother, sex, list(keep or [])
    )
    return {
        "matrices": [np.ascontiguousarray(m) for m in matrices],
        "order": order,
        "class_names": names,
        "pairs": dict(zip(names, pairs)),
    }
