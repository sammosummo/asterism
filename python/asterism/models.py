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
        ) = _core.spatial_fit(self._fixed, self._distance, self._design, y, reml, integrated)
        return {
            "variances": list(variances),
            "shares": list(shares),
            "total_variance": total,
            "decay_per_km": None if integrated else lam,
            "half_distance_km": None if integrated else half,
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            "estimator": "reml" if reml else "ml",
            "range_treatment": "integrated" if integrated else "profile",
        }

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
