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

    def equality_test(
        self, y: Any, components: list[int] | None = None, reml: bool = True
    ) -> dict[str, Any]:
        """Test whether several components share one variance.

        **This is the question a split matrix asks, and `test` is not it.**
        Splitting a relationship matrix by class of parent–offspring tie gives
        four classes, and every one carries variance if the trait is heritable
        at all — so testing each against nought returns a p-value near nought
        for anything heritable and answers nothing. Whether a mother resembles
        her son by as much as a father resembles his daughter is the classes
        being equal to *one another*.

        The null pools the named components by adding their matrices, which is
        exact: the pieces came from splitting a matrix, so their sum is that
        matrix. Pool all of them and the null is the ordinary additive model.
        Defaults to every structured component.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        if components is None:
            components = list(range(len(self._matrices)))
        statistic, p_value, rule, null_loglik = _core.component_equality_test(
            self._matrices, self._x, y, [int(c) for c in components], reml
        )
        return {
            "components": list(components),
            "statistic": statistic,
            "p_value": p_value,
            # Chi-square on one fewer degree of freedom than components pooled;
            # nothing sits on a bound under this null.
            "rule": rule,
            "null_loglik": null_loglik,
            "estimator": "reml" if reml else "ml",
        }

    def contrasts(
        self, y: Any, classes: list[int] | None = None, reml: bool = True
    ) -> list[dict[str, Any]]:
        """Every class's deviation from the average class, with an interval.

        **This is what `equality_test` cannot give.** The omnibus says the
        classes are not all alike and stops; a contrast against the others
        *pooled* couples them, because raising one class raises the pool the
        rest are measured against — in calibration a single lifted class made a
        second reject half the time.

        The deviations are constrained to sum to nought, so the baseline is the
        average class variance and each deviation is a departure from it. The
        obvious alternative, a free baseline plus free differences, is rank
        deficient: a constant moved from the baseline into every difference
        changes nothing.

        ``classes`` must name only things that are classes of one split.
        Anything not named keeps its own variance, which is what should happen
        to a remainder component — pooling "everything that is not a
        parent–child tie" into the baseline would compare siblings with parents.

        With only two classes this says less than it appears to: the deviations
        must be mirror images, so "the first is above average" and "the second
        is below" are one statement.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        if classes is None:
            classes = list(range(len(self._matrices)))
        rows = _core.component_contrasts(
            self._matrices, self._x, y, [int(c) for c in classes], reml
        )
        return [
            {
                "class": int(c),
                "deviation": deviation,
                "lower": lower,
                "upper": upper,
                "lower_limited": at_lower,
                "upper_limited": at_upper,
                "p_value": p_value,
                "statistic": statistic,
                "level": 0.95,
            }
            for c, (deviation, lower, upper, at_lower, at_upper, p_value, statistic)
            in zip(classes, rows)
        ]

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
      held by loadings so it stays a covariance. Six parameters.
    - ``"powered_exponential"``: the exponential with the decay taken to a
      frozen power, ``exp(-λ|Δ|^κ)``. The same five free parameters, with
      ``shape`` chosen from 0.5, 1.0, 1.5 or 2.0 — **chosen and not fitted**,
      because a shape and a decay rate trade off against each other and a search
      over both wanders. At ``shape=1.0`` it is the exponential surface exactly.

    **The shape is barely identified and strongly changes the answer.** On one
    simulated set the four shapes spanned 0.31 in log likelihood — a deviance of
    0.62, which is nothing — while the genetic correlation they reported ran
    from 0.92 to 0.45. A shape chosen to suit the answer would be invisible in
    the fit, so choose it for a reason outside the data and report which one was
    used beside the result.

    Provenance differs between them and is worth knowing. The exponential form
    is a *recovered* method — it reproduces the restored SOLAR covariance to
    better than 1e-11, and the powered one reproduces the source at all four
    shapes. Random regression is not recovered: the source-fidelity crate holds
    it in a module whose own header says nothing in it is a recovered or
    qualified method. It is a proposal, and this reproduces that proposal.

    Nothing is reported in either surface's own coordinates. What comes back is
    the heritability at each environment you ask about and the genetic
    correlation between each pair — quantities that mean the same thing whichever
    surface produced them.

    **Only the random-regression surface can represent a crossover** — a genotype that
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
    random-regression surface was conservative rather than anti-conservative in the mirror
    case, which is why it is the default.
    Fitting both and reporting whichever rejects is not a defensible procedure.

    **A correlation below one is not by itself evidence of an interaction.** The
    estimate cannot exceed one, so under the null every departure runs downward:
    on the exponential surface a tenth of null samples came back below 0.25. Use
    ``test``.
    """

    def __init__(self, relationship: Any, environment: Any, design: Any,
                 surface: str = "random_regression", shape: float = 1.0) -> None:
        if surface not in ("exponential", "random_regression", "powered_exponential"):
            raise ValueError(
                "surface must be exponential, random_regression or "
                f"powered_exponential, not {surface!r}"
            )
        if surface == "powered_exponential" and float(shape) not in (0.5, 1.0, 1.5, 2.0):
            raise ValueError(
                f"shape must be one of 0.5, 1.0, 1.5, 2.0, not {shape!r}. It is "
                "chosen rather than fitted."
            )
        self._relationship = _matrix(relationship, "relationship")
        self._environment = [float(v) for v in np.asarray(environment).ravel()]
        self._design = _matrix(design, "design")
        self._surface = surface
        self._shape = float(shape)

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
            self._surface, grid, self._shape, reml,
        )
        width = len(grid)
        return {
            "surface": self._surface,
            "shape": self._shape if self._surface == "powered_exponential" else None,
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
          referred to chi-square on one degree of freedom. On the random-regression
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
            self._surface, null, self._shape, reml,
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
            self._surface, quantity, float(first), float(second), self._shape, reml,
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


class GxsModel:
    """One trait whose genes may act differently in the two sexes.

    This is the discrete case of genotype-by-environment, with sex as the
    environment. Because the environment takes two values rather than a range,
    nothing is smoothed and no surface has to be chosen: the model carries one
    genetic standard deviation per sex, one residual standard deviation per sex,
    and one genetic correlation between them.

    ``group`` is one value per person, 1 or 2 and nothing else. **Use the
    pedigree sex.** A missing or unknown sex is refused rather than swept into a
    group, because a model that quietly puts the unknowns together is estimating
    a correlation with a third group in it.

    **Two findings live here and they are not the same.**

    *The heritability differs between the sexes.* The genetic variance is larger
    in one than the other. That is a difference of scale, and a difference of
    scale can come from the measurement rather than the genetics — men are
    larger, so a volume in millimetres varies more in men whether or not the
    genes differ. ``test(y, "genetic")``.

    *The genes differ between the sexes.* The genetic correlation across the
    sexes is below one, so the genes that matter in men are not exactly those
    that matter in women. No change of units can produce this, and it is usually
    the interesting claim. ``test(y, "correlation")``.

    Unlike the kernel surfaces in :class:`GxeModel`, the correlation here is a
    parameter rather than a function of distance, so it is free to be negative:
    a genotype raising a trait in one sex and lowering it in the other is
    reachable.

    **The two residual standard deviations are free, and they should be.** A
    trait simply noisier in one sex would otherwise push its extra variance into
    the genetic term, and a gene-by-sex test would then reject because of
    measurement rather than because of genes.

    **Read the headline test first.** ``test(y, "gene_by_sex")`` puts the
    genetic constraints back at once — same variance, same genes — while leaving
    the two residual variances free. It is what stops several tests on one trait
    being read as several findings.

    **Do not use** ``test(y, "any_difference")`` **as the headline.** It is the
    null the recovered code tested and it ties the residual variances too, so a
    trait merely measured more noisily in one sex rejects it hard with nothing
    genetic happening. In simulation on the GOBS pedigree a sex difference in
    measurement error alone rejected it at p = 1e-34 while every genetic test
    correctly reported nothing.
    """

    def __init__(self, relationship: Any, group: Any, design: Any) -> None:
        self._relationship = _matrix(relationship, "relationship")
        self._group = np.ascontiguousarray(
            np.asarray(group).ravel(), dtype=np.float64
        )
        self._design = _matrix(design, "design")

    def fit(self, y: Any, reml: bool = True) -> dict[str, Any]:
        """Fit, with everything free.

        ``counts`` comes back with the answer because a correlation estimated
        across a group of thirty is not the same claim as one across a thousand,
        and the fit itself cannot tell you which you have.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        (
            genetic,
            residual,
            heritability,
            correlation,
            effects,
            errors,
            loglik,
            converged,
            gradient,
            counts,
        ) = _core.gxs_fit(self._relationship, self._group, self._design, y, reml)
        return {
            "genetic_variance": list(genetic),
            "residual_variance": list(residual),
            "heritability": list(heritability),
            "genetic_correlation": correlation,
            "fixed_effects": [
                {"estimate": e, "standard_error": s} for e, s in zip(effects, errors)
            ],
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            "counts": list(counts),
            "estimator": "reml" if reml else "ml",
        }

    def test(self, y: Any, null: str = "gene_by_sex", reml: bool = True) -> dict[str, Any]:
        """Test one of the five nulls.

        - ``"gene_by_sex"``: no gene-by-sex effect of any kind — the same
          variance and the same genes in both sexes — with the two residual
          variances left free. Two constraints, one of which sits on a bound, so
          the reference is an even mixture of chi-square on one and on two
          degrees of freedom. **Read this one first.**
        - ``"any_difference"``: nothing differs between the sexes at all,
          residual included. Three constraints on an even mixture of chi-square
          on two and on three. This is the null the recovered code tested, and
          it is **not** a gene-by-sex test: a noisier sex rejects it.
        - ``"correlation"``: the same genes act in both sexes. This is the
          gene-by-sex question proper. The null puts the correlation at the edge
          of what it may be, so the reference is the even mixture of a point
          mass at nought with chi-square on one degree of freedom. A plain
          chi-square would roughly double the p-value.
        - ``"genetic"``: the same genetic variance in both sexes. Interior, so
          chi-square on one degree of freedom.
        - ``"residual"``: the same residual variance in both sexes. Report it
          beside the others as a measurement fact, not as a genetic finding.

        ``rule`` names the reference distribution the p-value is a tail of, so a
        reader need not take it on trust.
        """
        allowed = (
            "gene_by_sex", "any_difference", "correlation", "genetic", "residual"
        )
        if null not in allowed:
            raise ValueError(
                f"null must be one of {', '.join(allowed)}, not {null!r}"
            )
        y = np.ascontiguousarray(y, dtype=np.float64)
        statistic, p_value, rule, null_loglik, alternative_loglik = _core.gxs_test(
            self._relationship, self._group, self._design, y, null, reml
        )
        return {
            "null": null,
            "statistic": statistic,
            "p_value": p_value,
            "rule": rule,
            "null_loglik": null_loglik,
            "alternative_loglik": alternative_loglik,
            "estimator": "reml" if reml else "ml",
        }


class LiabilityModel:
    """One binary trait on a pedigree, through a liability threshold.

    Every person carries an unobserved liability and is a case when it crosses
    a threshold. Only the sign is ever seen, so the liability's variance is
    fixed at one and the threshold at nought, with the intercept carrying it.

    **The heritability is of the liability, not of the observed status**, which
    is what anybody means by the heritability of a disease. It is not comparable
    with the REML heritabilities the rest of this package reports, and the fit
    record says ``estimator: ml`` so the difference is visible rather than
    remembered — there is no REML here, because there is no response to project
    onto the null space of the design.

    A family's likelihood is the probability of an orthant: exact at one and two
    people, and the Mendell–Elston sequential truncation above that, taking the
    rarer class first. Agreement with native SOLAR is exact where no
    approximation is used and within a tenth of a standard error where it is.

    ``build`` refuses a relationship above 0.9 off the diagonal. Twins and
    duplicated people push the liability correlation to ``h2`` rather than
    ``h2 / 2``, which is where the two-person quadrature starts losing digits.
    """

    def __init__(self, relationship: Any, status: Any, design: Any) -> None:
        self._relationship = _matrix(relationship, "relationship")
        self._status = np.ascontiguousarray(
            np.asarray(status, dtype=np.float64).ravel()
        )
        self._design = _matrix(design, "design")

    def fit(self) -> dict[str, Any]:
        """Fit, by maximum likelihood because nothing else is available."""
        (
            heritability,
            effects,
            loglik,
            converged,
            gradient,
            prevalence,
            largest_family,
        ) = _core.liability_fit(self._relationship, self._status, self._design)
        return {
            "heritability": heritability,
            "scale": "liability, not observed status",
            # The first is an intercept only in the sense that it carries the
            # threshold: Phi(intercept) is the prevalence a covariate-free model
            # implies.
            "fixed_effects": list(effects),
            "loglik": loglik,
            "converged": converged,
            "scaled_gradient": gradient,
            "prevalence": prevalence,
            "largest_family": largest_family,
            "estimator": "ml",
        }

    def interval(self) -> dict[str, Any]:
        """A 95 per cent profile interval for the liability heritability."""
        estimate, lower, upper, at_lower, at_upper = _core.liability_interval(
            self._relationship, self._status, self._design
        )
        return {
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "lower_at_bound": at_lower,
            "upper_at_bound": at_upper,
            "level": 0.95,
        }

    def test(self) -> dict[str, Any]:
        """Test the liability heritability against nought.

        A heritability of nought sits on a bound, so the reference is the even
        mixture of a point mass and chi-square on one degree of freedom. That
        was assumed from the Gaussian case rather than derived for a liability,
        and then measured: on the real pedigree it rejects 0.055 of the time at
        a nominal 0.05.
        """
        statistic, p_value, rule, null_loglik = _core.liability_test(
            self._relationship, self._status, self._design
        )
        return {
            "statistic": statistic,
            "p_value": p_value,
            "rule": rule,
            "null_loglik": null_loglik,
            "estimator": "ml",
        }


class AssociationModel:
    """Many markers, one at a time, each a fixed effect in a polygenic model.

    The model is ``y = X0 b + m_j c + g + e``, where ``X0`` carries the
    intercept, the ancestry components and any other covariate. The polygenic
    term is what makes it worth doing: relatedness and population structure
    inflate an association test, and the relationship matrix absorbs both.

    **The variance components are fitted once under the null and then held**,
    which is what makes a scan take minutes rather than hours — each marker
    becomes a weighted least squares on rotated data. That is an assumption, not
    a trick: it is good when no single marker explains much of the variance,
    which is the situation a scan is in and exactly not the situation for a
    marker of large effect. Pass ``variance="refitted"`` to refit under every
    marker, which is around a hundred times slower.

    **Wald and the likelihood ratio are the same number when held.** The profile
    log likelihood in the fixed effects is exactly quadratic when the covariance
    is known, so the likelihood ratio is the Wald statistic squared. Both come
    back because both are asked for; they differ only under ``refitted``.

    **The marker under test is inside the relationship matrix.** Leaving its
    chromosome out is the usual answer and is not done, so every test is biased
    towards the null.
    """

    def __init__(self, relationship: Any, design: Any, y: Any) -> None:
        self._relationship = _matrix(relationship, "relationship")
        self._design = _matrix(design, "design")
        self._y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())

    def sweep(
        self,
        markers: Any,
        variance: str = "held",
        refit_below: float | None = None,
    ) -> dict[str, Any]:
        """Test every column of ``markers``.

        With ``refit_below`` set, the sweep runs held and then refits only the
        markers whose held p-value falls under it — which is how a scan should
        be run. Refitting everything costs about twenty times as much and the
        two modes agree to two decimal places on the markers nobody cares about.

        **Set ``refit_below`` some way above the threshold you will report
        against.** Held is conservative — measured across four hundred real
        markers it was never smaller than refitted, with the gap growing from a
        ratio of 1.00 above p = 0.01 to 1.10 below 1e-06 — so a marker can have
        a refitted p under your threshold while its held p sits above it.
        Screening at exactly the threshold would miss it; ten times the
        threshold is ample and costs almost nothing.

        Each marker says whether it was ``refitted``, because a two-stage result
        that does not is one nobody can check.

        A marker that cannot be tested — one with no variation, or one leaving
        the design rank deficient — comes back with a stable code in place of
        its numbers rather than stopping the sweep.
        """
        if variance not in ("held", "refitted"):
            raise ValueError(f"variance must be held or refitted, not {variance!r}")
        markers = np.ascontiguousarray(np.asarray(markers, dtype=np.float64))
        if markers.ndim == 1:
            markers = markers.reshape(-1, 1)
        heritability, null_loglik, rows, covariates = _core.association_sweep(
            self._relationship, self._design, self._y, markers, variance, refit_below
        )
        return {
            "null_heritability": heritability,
            "null_loglik": null_loglik,
            "variance_components": variance,
            "leave_one_chromosome_out": False,
            # The covariates' own effects, from the null model rather than from
            # any one marker. They were previously computed and discarded, so a
            # caller who wanted to know what age or sex was doing had to fit a
            # second model to find out.
            "covariates": [
                {"estimate": estimate, "standard_error": error, "p_value": p_value}
                for estimate, error, p_value in covariates
            ],
            "markers": [
                {
                    "effect": effect,
                    "standard_error": error,
                    "wald": wald,
                    "likelihood_ratio": ratio,
                    "p_value": p_value,
                    "refitted": refitted,
                    "refused": code or None,
                }
                for effect, error, wald, ratio, p_value, refitted, code in rows
            ],
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
