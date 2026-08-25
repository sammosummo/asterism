"""The models beyond one trait with one component.

Each is a small object holding the matrices, with `fit`, `interval` and a test.
The compiled calculation validates the numerical model before fitting and
returns a dictionary with named fields rather than a tuple whose meaning has to
be remembered.

**This layer exists because the compiled bindings are positional.**
`_core.component_fit` returns eleven values in a fixed order and
`_core.spatial_fit` eight more, and reading the fourth of eleven correctly every
time is not a reasonable thing to ask of an analysis script. Nothing here computes anything: every number
comes from the same compiled code, and this only names it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _core
from .analysis import build_identity

__all__: list[str] = [
    "AssociationModel",
    "AutoregressiveModel",
    "BivariateModel",
    "ComponentModel",
    "DiscreteGxeModel",
    "GxeModel",
    "LiabilityModel",
    "SpatialModel",
    "VariantSetModel",
    "kinship_classes",
    "mixed_bivariate_fit",
    "mixed_bivariate_interval",
    "mixed_bivariate_test",
    "region_log_probability",
    "tobit_fit",
    "tobit_interval",
    "tobit_test",
]
"""Exported the documented model interfaces from this module."""


# asterism-style: allow private-helper -- reused exact matrix-conversion boundary
def _matrix(value: Any, name: str) -> npt.NDArray[np.float64]:
    """Convert one array-like value to a contiguous binary64 matrix."""
    array: npt.NDArray[np.float64] = np.ascontiguousarray(value, dtype=np.float64)
    """Converted the caller's value without taking an ownership copy."""

    if array.ndim != 2:
        raise ValueError(f"{name.upper()}_NOT_TWO_DIMENSIONAL")
    return array


# asterism-style: allow private-helper -- reused immutable model-input boundary
def _owned_matrix(value: Any, name: str) -> npt.NDArray[np.float64]:
    """Take a private, read-only copy of a caller's matrix.

    `np.ascontiguousarray` returns the caller's own object when it is already
    C-contiguous binary64, so storing that would leave the model holding a live
    view of an array the caller can still change. A model that answered
    differently after a later, unrelated write would break the record's claim to
    be reproducible from what it reports, and would do it silently.
    """
    array: npt.NDArray[np.float64] = _matrix(value, name).copy(order="C")
    """Copied the matrix so later caller mutation cannot change the model."""

    array.setflags(write=False)
    """Made the model-owned matrix read-only."""
    return array


SUBJECT_ORDER_SHA256_PATTERN: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")
"""Recognised the exact lowercase hexadecimal commitment carried by fit records."""


# asterism-style: allow private-helper -- reused fit-identity validation boundary
def _validated_subject_order_sha256(value: str | None) -> str | None:
    """Validate one optional subject-order commitment for a fit route."""
    if value is not None and (
        not isinstance(value, str)
        or SUBJECT_ORDER_SHA256_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("FIT_SUBJECT_ORDER_SHA256_INVALID")
    """Refused malformed or non-lowercase commitments before numerical fitting."""
    return value


class ComponentModel:
    """One trait with any number of variance components.

    The residual is added for you and is always last, so a model built with a
    relationship matrix and a household matrix reports three raw coefficient
    proportions: additive, household, residual. When every structured matrix
    has a positive finite mean diagonal, the fit also reports each coefficient
    times its matrix's mean diagonal. These mean-diagonal contributions and
    their proportions do not change if a matrix is multiplied by a positive
    constant and its fitted coefficient changes reciprocally.

    Parameters
    ----------
    matrices
        The structured components, in the order you want them reported. Each
        is copied at construction, so later caller mutation cannot change the
        fitted model. Row alignment is positional and is the caller's
        responsibility.
    x
        The fixed-effect design, one row per person, including its own
        intercept column if one is wanted. Its values are copied at
        construction.
    subject_order_sha256
        Optional lowercase SHA-256 from ``subject_order_commitment`` for the
        exact fitted row order. It is echoed on every fit record.
    """

    def __init__(
        self,
        matrices: list[Any],
        x: Any,
        *,
        subject_order_sha256: str | None = None,
    ) -> None:
        self._subject_order_sha256 = _validated_subject_order_sha256(
            subject_order_sha256
        )
        """Stored the validated row-order commitment without participant identifiers."""

        if not matrices:
            raise ValueError("COMPONENTS_NONE_GIVEN")
        converted: list[npt.NDArray[np.float64]] = [
            _owned_matrix(matrix, f"matrix_{i}") for i, matrix in enumerate(matrices)
        ]
        """Copied every structured relationship matrix into model-owned storage."""

        self._matrices = converted
        """Stored the structured matrices in their caller-defined order."""

        mean_diagonals: list[float] = [
            float(np.mean(np.diag(matrix))) for matrix in converted
        ]
        """Measured each structured matrix's marginal diagonal scale."""

        self._structured_mean_diagonals = (
            mean_diagonals
            if all(np.isfinite(value) and value > 0.0 for value in mean_diagonals)
            else None
        )
        """Retained usable scales only when every structured diagonal mean was valid."""

        self._x = _owned_matrix(x, "design")
        """Stored an immutable copy of the fixed-effect design."""

        self.components = len(self._matrices) + 1
        """Counted the structured components together with the residual."""

    def fit(self, y: Any, reml: bool = True) -> dict[str, Any]:
        """Fit, and return variance coefficients and their proportions.

        ``raw_coefficient_proportions`` depend on matrix scale. They are useful
        for inspecting the optimiser parameterisation but are not generic
        variance shares. When all structured matrices have positive finite mean
        diagonals, ``mean_diagonal_proportions`` reports the corresponding
        scale-invariant marginal contributions.

        Matrix, design, and response alignment is positional and is the
        caller's responsibility.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response to the contiguous binary64 input expected by Rust."""

        (
            variances,
            proportions,
            raw_coefficient_total,
            loglik,
            gradient,
            converged,
            effects,
            errors,
            stop_code,
            stop_message,
            polished,
        ) = _core.component_fit(self._matrices, self._x, y, reml)
        """Fitted the component model and unpacked its named scientific fields."""

        record: dict[str, object] = {
            "variances": list(variances),
            "raw_coefficient_proportions": list(proportions),
            "raw_coefficient_total": raw_coefficient_total,
            # The generalised least squares estimates: the best linear unbiased
            # estimator of the fixed effects at the fitted variances, with the
            # standard errors from the diagonal of (X' V^-1 X)^-1.
            "fixed_effects": [
                {"estimate": e, "standard_error": s}
                for e, s in zip(effects, errors, strict=True)
            ],
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            # Why the search stopped, in its own words. Nought means one of its
            # tolerances fired, one means it ran out of iterations. This is a
            # different question from `converged`, which is decided afterwards
            # on the projected gradient: the search stops once the objective
            # has settled, and down a long flat valley that happens well before
            # the gradient vanishes. A fit can therefore stop cleanly and still
            # report `converged: False`. Read the two together.
            "stop_code": stop_code,
            "stop_message": stop_message,
            # True where the gradient test failed on the first search and a
            # second was run from that point with the objective tolerance off.
            # It fires nowhere else, so `False` here means this is the fit the
            # package gave before the polish existed, to the last bit.
            "polished": polished,
            "estimator": "reml" if reml else "ml",
            "build": build_identity(),
            "subject_order_sha256": self._subject_order_sha256,
        }
        """Named the positional compiled result for stable Python consumption."""

        if self._structured_mean_diagonals is not None:
            mean_diagonal_contributions: list[float] = [
                variance * mean_diagonal
                for variance, mean_diagonal in zip(
                    variances[:-1], self._structured_mean_diagonals, strict=True
                )
            ]
            """Scaled each structured coefficient by its matrix's mean diagonal."""

            mean_diagonal_contributions.append(variances[-1])
            """Added the unit-diagonal residual contribution."""

            mean_diagonal_total: float = sum(mean_diagonal_contributions)
            """Summed the scale-invariant marginal covariance contributions."""

            record["mean_diagonal_component_contributions"] = (
                mean_diagonal_contributions
            )
            """Exposed the component contributions in fitted component order."""

            record["mean_diagonal_total"] = mean_diagonal_total
            """Exposed their total marginal variance."""

            record["mean_diagonal_proportions"] = [
                contribution / mean_diagonal_total
                for contribution in mean_diagonal_contributions
            ]
            """Normalised the contributions into scale-invariant proportions."""
        return record

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
        """Converted the response to the compiled predictor's binary64 layout."""

        values, errors = _core.component_blup(
            self._matrices, self._x, y, component, reml
        )
        """Predicted the selected component and its per-person prediction errors."""
        return {
            "component": component,
            "values": list(values),
            "errors": list(errors),
        }

    def interval(
        self,
        y: Any,
        component: int,
        reml: bool = True,
        quantity: str = "mean_diagonal_proportion",
    ) -> dict[str, Any]:
        """A 95 per cent profile interval for one component proportion.

        The default profiles the scale-invariant ``mean_diagonal_proportion``
        reported by :meth:`fit`. The diagnostic
        ``raw_coefficient_proportion`` remains available explicitly; it changes
        when a relationship matrix is rescaled and must not be reported as a
        generic variance share.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response for repeated constrained component fits."""

        if quantity == "mean_diagonal_proportion":
            (
                estimate,
                lower,
                upper,
                lower_limited,
                upper_limited,
                level,
                failures,
            ) = _core.component_mean_diagonal_interval(
                self._matrices, self._x, y, component, reml
            )
            """Profiled the scale-invariant marginal component proportion."""
        elif quantity == "raw_coefficient_proportion":
            (
                lower,
                upper,
                lower_limited,
                upper_limited,
                level,
                failures,
            ) = _core.component_interval(self._matrices, self._x, y, component, reml)
            """Profiled the diagnostic proportion of raw fitted coefficients."""

            estimate: float = float(
                self.fit(y, reml)["raw_coefficient_proportions"][component]
            )
            """Recovered the raw-proportion point estimate matching the profile."""
        else:
            raise ValueError("COMPONENTS_INTERVAL_QUANTITY_UNKNOWN")
        return {
            "quantity": quantity,
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "lower_limited": lower_limited,
            "upper_limited": upper_limited,
            "level": level,
            "profile_failures": failures,
            # No component-proportion coverage simulation has yet scored a
            # boundary. Absent means unmeasured, not inapplicable.
            "contains_lower_bound": None,
            "contains_upper_bound": None,
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
        """Converted the response for the pooled-component comparison."""

        if components is None:
            components = list(range(len(self._matrices)))
            """Selected every structured component when no subset was supplied."""

        statistic, p_value, rule, null_loglik = _core.component_equality_test(
            self._matrices, self._x, y, [int(c) for c in components], reml
        )
        """Compared the free variances with their exact pooled-component null."""
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
        """Converted the response for the class-contrast profiles."""

        if classes is None:
            classes = list(range(len(self._matrices)))
            """Selected every structured class when no subset was supplied."""

        rows: list[tuple[float, float, float, bool, bool, float, float]] = (
            _core.component_contrasts(
                self._matrices, self._x, y, [int(c) for c in classes], reml
            )
        )
        """Profiled each class's deviation from the constrained average class."""
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
            for c, (
                deviation,
                lower,
                upper,
                at_lower,
                at_upper,
                p_value,
                statistic,
            ) in zip(classes, rows, strict=True)
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
        """Converted the response for the boundary component test."""

        statistic, p_value, rule, null_loglik = _core.component_test(
            self._matrices, self._x, y, component, reml
        )
        """Compared the fitted component with the nested model that omits it."""
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
    subject_order_sha256
        Optional lowercase SHA-256 from ``subject_order_commitment`` for the
        exact person order. It is echoed on every fit record.
    """

    QUANTITIES: tuple[str, ...] = (
        "h2_first",
        "h2_second",
        "rho_g",
        "rho_e",
        "rho_p",
    )
    """Named every bivariate quantity with a supported profile interval."""

    def __init__(
        self,
        k: Any,
        observed: Any,
        design: Any,
        *,
        subject_order_sha256: str | None = None,
    ) -> None:
        self._subject_order_sha256 = _validated_subject_order_sha256(
            subject_order_sha256
        )
        """Stored the validated row-order commitment without participant identifiers."""

        self._k = _owned_matrix(k, "relationship")
        """Stored an immutable copy of the relationship matrix."""

        self._observed = [[bool(a), bool(b)] for a, b in observed]
        """Normalised the per-person trait availability indicators."""

        self._design = _owned_matrix(design, "design")
        """Stored an immutable copy of the observed-row fixed-effect design."""

    def fit(self, y: Any, reml: bool = True) -> dict[str, Any]:
        """Fit, and return both heritabilities and all three correlations.

        The phenotypic correlation is derived from the others rather than
        estimated, which is why it has no variance of its own.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the stacked observed responses to contiguous binary64."""

        theta, loglik, gradient, converged = _core.bivariate_fit(
            self._k, self._observed, self._design, y, reml
        )
        """Fitted both traits jointly and unpacked their covariance summaries."""
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
            "build": build_identity(),
            "subject_order_sha256": self._subject_order_sha256,
        }

    def interval(self, y: Any, quantity: str, reml: bool = True) -> dict[str, Any]:
        """A 95 per cent profile interval for one reported quantity."""
        if quantity not in self.QUANTITIES:
            raise ValueError("BIVARIATE_QUANTITY_UNKNOWN")
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response for constrained bivariate profile fits."""

        (
            estimate,
            lower,
            upper,
            lower_limited,
            upper_limited,
            level,
            failures,
            contains_lower_bound,
            contains_upper_bound,
        ) = _core.bivariate_interval(
            self._k, self._observed, self._design, y, quantity, reml
        )
        """Profiled the requested bivariate reportable quantity."""
        return {
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "lower_limited": lower_limited,
            "upper_limited": upper_limited,
            "level": level,
            "profile_failures": failures,
            "contains_lower_bound": contains_lower_bound,
            "contains_upper_bound": contains_upper_bound,
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
        """Converted the response for the constrained correlation comparison."""

        (
            statistic,
            p_value,
            rule,
            null_loglik,
            alternative_loglik,
        ) = _core.bivariate_correlation_test(
            self._k, self._observed, self._design, y, quantity, null, reml
        )
        """Compared the free correlation with its fixed-null bivariate fit."""
        return {
            "statistic": statistic,
            "p_value": p_value,
            "rule": rule,
            "null_loglik": null_loglik,
            "alternative_loglik": alternative_loglik,
        }


class SpatialModel:
    """One trait with a spatial component whose range is estimated.

    The kernel is ``exp(-λd)`` with distances in kilometres. Any fixed
    components — a relationship matrix, a household matrix — are passed
    alongside and are reported before the spatial component, which is followed
    by the residual. ``raw_coefficient_proportions`` divide their fitted
    covariance coefficients by their sum, so they depend on fixed-component
    matrix scaling. When every fixed component has a positive finite mean
    diagonal, ``mean_diagonal_proportions`` reports scale-invariant marginal
    covariance contributions instead.

    **Two cautions that the numbers do not carry themselves.** The range is
    barely estimated: its interval reaches a bound in 98 per cent of calibration
    replicates, so report it as a point estimate or use ``integrated=True`` and
    be rid of it. And an interval for the spatial raw coefficient proportion
    reaching nought is not a test of whether there is a spatial effect — under
    that null the range is unidentified, which is why the test is a bootstrap.

    ``subject_order_sha256`` optionally carries the lowercase SHA-256 from
    ``subject_order_commitment`` for the exact fitted row order. Every fit
    record echoes it without retaining identifiers.
    """

    def __init__(
        self,
        fixed: list[Any],
        distance: Any,
        design: Any,
        *,
        subject_order_sha256: str | None = None,
    ) -> None:
        self._subject_order_sha256 = _validated_subject_order_sha256(
            subject_order_sha256
        )
        """Stored the validated row-order commitment without participant identifiers."""

        self._fixed = [
            _owned_matrix(matrix, f"matrix_{i}") for i, matrix in enumerate(fixed)
        ]
        """Copied the fixed covariance-component matrices into model-owned storage."""

        mean_diagonals: list[float] = [
            float(np.mean(np.diag(matrix))) for matrix in self._fixed
        ]
        """Measured the marginal diagonal scale of every fixed component."""

        self._fixed_mean_diagonals = (
            mean_diagonals
            if all(np.isfinite(value) and value > 0.0 for value in mean_diagonals)
            else None
        )
        """Retained the fixed-component scales only when all were usable."""

        self._distance = _owned_matrix(distance, "distance")
        """Stored an immutable copy of the pairwise distance matrix."""

        self._design = _owned_matrix(design, "design")
        """Stored an immutable copy of the fixed-effect design."""

    def fit(
        self, y: Any, reml: bool = True, integrated: bool = False
    ) -> dict[str, Any]:
        """Fit, taking the range as a free parameter or integrating it out.

        ``raw_coefficient_proportions`` and ``raw_coefficient_total`` depend
        on fixed-component matrix scaling. When the fixed component diagonals
        are positive and finite, ``mean_diagonal_component_contributions`` and
        ``mean_diagonal_proportions`` instead report the scale-invariant
        marginal covariance decomposition. The spatial kernel and residual
        identity both have unit diagonals.

        With ``integrated=True`` the range is averaged over rather than
        maximised over, and comes back as ``None``: there is nothing estimated
        to report, which is the point.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response to contiguous binary64 for the spatial fit."""

        (
            variances,
            raw_coefficient_proportions,
            raw_coefficient_total,
            lam,
            half,
            loglik,
            gradient,
            converged,
            polished,
            effects,
            errors,
        ) = _core.spatial_fit(
            self._fixed, self._distance, self._design, y, reml, integrated
        )
        """Fitted the spatial model and unpacked its covariance and range summaries."""

        record: dict[str, object] = {
            "variances": list(variances),
            "raw_coefficient_proportions": list(raw_coefficient_proportions),
            "raw_coefficient_total": raw_coefficient_total,
            # With the range integrated out these carry the extra uncertainty of
            # not knowing it: the standard errors are the average of the
            # within-range ones plus the spread of the estimates across ranges.
            "fixed_effects": [
                {"estimate": e, "standard_error": s}
                for e, s in zip(effects, errors, strict=True)
            ],
            "decay_per_km": None if integrated else lam,
            "half_distance_km": None if integrated else half,
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            # True where the gradient test failed on the first search and a
            # second was run from that point with the objective tolerance
            # switched off. It fires nowhere else, so `False` means this is the
            # fit the package gave before the polish existed, to the last bit.
            "polished": polished,
            "estimator": "reml" if reml else "ml",
            "range_treatment": "integrated" if integrated else "profile",
            "build": build_identity(),
            "subject_order_sha256": self._subject_order_sha256,
        }
        """Named the positional compiled result for stable Python consumption."""

        if self._fixed_mean_diagonals is not None:
            mean_diagonal_contributions: list[float] = [
                variance * mean_diagonal
                for variance, mean_diagonal in zip(
                    variances[: len(self._fixed)],
                    self._fixed_mean_diagonals,
                    strict=True,
                )
            ]
            """Scaled fixed coefficients by their matrices' mean diagonals."""

            mean_diagonal_contributions.extend(variances[len(self._fixed) :])
            """Added the unit-diagonal spatial and residual contributions."""

            mean_diagonal_total: float = sum(mean_diagonal_contributions)
            """Summed the scale-invariant marginal covariance contributions."""

            record["mean_diagonal_component_contributions"] = (
                mean_diagonal_contributions
            )
            """Exposed all marginal contributions in fitted component order."""

            record["mean_diagonal_total"] = mean_diagonal_total
            """Exposed their total marginal variance."""

            record["mean_diagonal_proportions"] = [
                contribution / mean_diagonal_total
                for contribution in mean_diagonal_contributions
            ]
            """Normalised the contributions into scale-invariant proportions."""
        return record

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
        """Converted the response for the compiled spatial predictor."""

        values, errors = _core.spatial_blup(
            self._fixed, self._distance, self._design, y, component, reml, integrated
        )
        """Predicted the selected covariance component and its prediction errors."""
        return {"component": component, "values": list(values), "errors": list(errors)}

    def interval(
        self, y: Any, quantity: str, reml: bool = True, integrated: bool = False
    ) -> dict[str, Any]:
        """An interval for a raw coefficient proportion, or for the range.

        ``quantity`` is a component index as a string, or ``"lambda"``. Asking
        for the range when it has been integrated out is refused rather than
        answered.

        **A component index profiles the raw proportion, which is not the
        headline the fit reports.** ``fit`` leads with
        ``mean_diagonal_proportions`` where those are defined, because they do
        not move when a matrix is rescaled; the raw proportion does, and the two
        can differ by orders of magnitude for the same component of the same
        fit. The estimate these endpoints are actually around is returned beside
        them as ``estimate``, so the pair reads together and cannot be
        mismatched by taking the headline from one dictionary and the endpoints
        from the other.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response for constrained spatial profile fits."""

        (
            estimate,
            lower,
            upper,
            lower_limited,
            upper_limited,
            level,
            profile_failures,
            contains_lower_bound,
            contains_upper_bound,
        ) = _core.spatial_interval(
            self._fixed, self._distance, self._design, y, quantity, reml, integrated
        )
        """Profiled the requested spatial coefficient or decay parameter."""
        return {
            "quantity": (
                "decay_per_km" if quantity == "lambda" else "raw_coefficient_proportion"
            ),
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "lower_limited": lower_limited,
            "upper_limited": upper_limited,
            "level": level,
            "profile_failures": profile_failures,
            "contains_lower_bound": contains_lower_bound,
            "contains_upper_bound": contains_upper_bound,
            "estimator": "reml" if reml else "ml",
        }

    def statistic(self, y: Any, reml: bool = True, integrated: bool = False) -> float:
        """The likelihood ratio against no spatial variance.

        A statistic and not a p-value: with the range unidentified under that
        null there is no closed-form reference, so a p-value takes `bootstrap`.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response for the spatial null comparison."""
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
        """Converted the observed response used by every bootstrap comparison."""

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
        """Ran the requested null replicates and counted statistics at least as large."""
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


class AutoregressiveModel:
    """One trait with a separable first-order autoregressive effect on a grid.

    Two people in cells ``(r_i, c_i)`` and ``(r_j, c_j)`` share
    ``sigma_s^2 * rho_row^|r_i-r_j| * rho_col^|c_i-c_j|``. This is the model
    Stopher and colleagues fitted to the red deer of Rum, and the standard one
    for field trials laid out in rows and columns.

    **It is not** :class:`SpatialModel`. That one estimates the range of an
    isotropic kernel in true distance; this one takes the cell size as given
    and estimates only how fast correlation falls off per cell. The scale is
    therefore a choice the caller makes, and the two rates mean nothing without
    the cell size that produced them — so sweep the cell size rather than
    reporting one grid's answer.

    **At both rates nought the kernel is the same-cell indicator**, because
    ``0**0`` is one and ``0**k`` is nought. So this model contains a plain
    shared-cell random effect as a special case, and asking whether the rates
    are nought asks whether smooth decay across neighbouring cells buys
    anything over shared membership.

    Parameters
    ----------
    fixed
        The matrices whose variances are estimated but whose shape is given —
        a relationship matrix, a household matrix.
    row, column
        Cell coordinates per person, as whole numbers of cells.
    design
        The fixed-effect design.
    """

    def __init__(self, fixed: list[Any], row: Any, column: Any, design: Any) -> None:
        self._fixed = [
            _owned_matrix(matrix, f"matrix_{i}") for i, matrix in enumerate(fixed)
        ]
        """Copied the fixed covariance-component matrices into model-owned storage."""

        self._row = [int(v) for v in np.asarray(row).ravel()]
        """Normalised row coordinates to whole grid cells."""

        self._column = [int(v) for v in np.asarray(column).ravel()]
        """Normalised column coordinates to whole grid cells."""

        if len(self._row) != len(self._column):
            raise ValueError("AUTOREGRESSIVE_CELLS_WRONG_LENGTH")
        self._design = _owned_matrix(design, "design")
        """Stored an immutable copy of the fixed-effect design."""

    @property
    def cells(self) -> int:
        """How many distinct cells the coordinates hold.

        Worth reading before paying for a fit. A grid so fine that almost
        everybody is alone in a cell has nothing to say about neighbours, and
        one so coarse that everybody shares a cell has nothing to say at all.
        """
        return _core.autoregressive_cells(self._row, self._column)

    def fit(self, y: Any, reml: bool = True) -> dict[str, Any]:
        """Fit, and report the variances and the two rates."""
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response to contiguous binary64 for the grid fit."""

        (
            variances,
            proportions,
            total,
            rates,
            loglik,
            converged,
            polished,
            gradient,
            effects,
            errors,
        ) = _core.autoregressive_fit(
            self._fixed, self._row, self._column, self._design, y, reml
        )
        """Fitted the separable grid covariance and unpacked its diagnostics."""

        rho_row, rho_column, half_row, half_column = rates
        """Named the two cell correlations and their half-correlation distances."""
        return {
            "variances": list(variances),
            "raw_coefficient_proportions": list(proportions),
            "raw_coefficient_total": total,
            # Correlation between cells one step apart, down and across. These
            # are per cell: they mean nothing without the cell size.
            "rho_row": rho_row,
            "rho_column": rho_column,
            # The same thing in a unit people think in — how many cells apart
            # the correlation halves. Nought where a rate is nought, infinite
            # where it is one.
            "half_cells_row": half_row,
            "half_cells_column": half_column,
            "cells": self.cells,
            "fixed_effects": [
                {"estimate": e, "standard_error": s}
                for e, s in zip(effects, errors, strict=True)
            ],
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            "polished": polished,
            "estimator": "reml" if reml else "ml",
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
      fixed power, ``exp(-λ|Δ|^κ)``. The same five free parameters, with
      ``shape`` chosen from 0.5, 1.0, 1.5 or 2.0 — **chosen and not fitted**,
      because a shape and a decay rate trade off against each other and a search
      over both wanders. At ``shape=1.0`` it is the exponential surface exactly.

    **The shape is barely identified and strongly changes the answer.** On one
    simulated set the four shapes spanned 0.31 in log likelihood — a deviance of
    0.62, which is nothing — while the genetic correlation they reported ran
    from 0.92 to 0.45. A shape chosen to suit the answer would be invisible in
    the fit, so choose it for a reason outside the data and report which one was
    used beside the result.

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

    ``subject_order_sha256`` optionally carries the lowercase SHA-256 from
    ``subject_order_commitment`` for the exact fitted row order. Every fit
    record echoes it without retaining identifiers.
    """

    def __init__(
        self,
        relationship: Any,
        environment: Any,
        design: Any,
        surface: str = "random_regression",
        shape: float = 1.0,
        *,
        subject_order_sha256: str | None = None,
    ) -> None:
        self._subject_order_sha256 = _validated_subject_order_sha256(
            subject_order_sha256
        )
        """Stored the validated row-order commitment without participant identifiers."""

        if surface not in ("exponential", "random_regression", "powered_exponential"):
            raise ValueError(
                "surface must be exponential, random_regression or "
                f"powered_exponential, not {surface!r}"
            )
        if surface == "powered_exponential" and float(shape) not in (
            0.5,
            1.0,
            1.5,
            2.0,
        ):
            raise ValueError(
                f"shape must be one of 0.5, 1.0, 1.5, 2.0, not {shape!r}. It is "
                "chosen rather than fitted."
            )
        self._relationship = _owned_matrix(relationship, "relationship")
        """Stored an immutable copy of the relationship matrix."""

        self._environment = [float(v) for v in np.asarray(environment).ravel()]
        """Normalised the per-person environment values to binary64 scalars."""

        self._design = _owned_matrix(design, "design")
        """Stored an immutable copy of the fixed-effect design."""

        self._surface = surface
        """Recorded the pre-selected covariance-surface family."""

        self._shape = float(shape)
        """Recorded the fixed powered-exponential shape when applicable."""

    def fit(
        self, y: Any, grid: Any = (-1.0, 0.0, 1.0), reml: bool = True
    ) -> dict[str, Any]:
        """Fit, and report the surface at the environments in ``grid``.

        ``grid`` is in the environment's own units, so it should be chosen from
        the data — quantiles of the observed environment usually. The genetic
        correlations come back as a square list of lists in the grid's order.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response to contiguous binary64 for the surface fit."""

        grid = [float(v) for v in np.asarray(grid).ravel()]
        """Normalised the requested reporting environments to binary64 scalars."""

        (
            parameters,
            loglik,
            converged,
            polished,
            gradient,
            effects,
            errors,
            genetic,
            residual,
            heritability,
            correlations,
        ) = _core.gxe_fit(
            self._relationship,
            self._environment,
            self._design,
            y,
            self._surface,
            grid,
            self._shape,
            reml,
        )
        """Fitted the selected surface and evaluated it across the reporting grid."""

        width: int = len(grid)
        """Recorded the side length of the flattened correlation matrix."""
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
                list(correlations[row * width : (row + 1) * width])
                for row in range(width)
            ],
            "fixed_effects": [
                {"estimate": e, "standard_error": s}
                for e, s in zip(effects, errors, strict=True)
            ],
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            # True where the gradient test failed on the first search and a
            # second was run from that point with the objective tolerance
            # switched off. It fires nowhere else, so `False` means this is
            # the fit the package gave before the polish existed.
            "polished": polished,
            "estimator": "reml" if reml else "ml",
            "build": build_identity(),
            "subject_order_sha256": self._subject_order_sha256,
        }

    def test(
        self, y: Any, null: str = "correlation", reml: bool = True
    ) -> dict[str, Any]:
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
        """Converted the response for the constrained surface comparison."""

        statistic, p_value, rule, null_loglik, alternative_loglik = _core.gxe_test(
            self._relationship,
            self._environment,
            self._design,
            y,
            self._surface,
            null,
            self._shape,
            reml,
        )
        """Compared the fitted surface with the requested biological null."""
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
        be. ``lower_limited`` or ``upper_limited`` says the endpoint ran to
        the edge of what the quantity can be rather than to a likelihood
        crossing, which is a limit of the model rather than a measurement.
        """
        if quantity not in ("heritability", "correlation"):
            raise ValueError(
                f"quantity must be heritability or correlation, not {quantity!r}"
            )
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response for constrained surface profile fits."""

        (
            estimate,
            lower,
            upper,
            lower_limited,
            upper_limited,
            level,
            failures,
            contains_lower_bound,
            contains_upper_bound,
        ) = _core.gxe_interval(
            self._relationship,
            self._environment,
            self._design,
            y,
            self._surface,
            quantity,
            float(first),
            float(second),
            self._shape,
            reml,
        )
        """Profiled the requested surface-derived heritability or correlation."""
        return {
            "surface": self._surface,
            "quantity": quantity,
            "environment": [first] if quantity == "heritability" else [first, second],
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "lower_limited": lower_limited,
            "upper_limited": upper_limited,
            "level": level,
            "estimator": "reml" if reml else "ml",
            "profile_failures": failures,
            "contains_lower_bound": contains_lower_bound,
            "contains_upper_bound": contains_upper_bound,
        }


class DiscreteGxeModel:
    """One trait whose genes may act differently in two environments.

    This is the discrete case of genotype-by-environment: the environment is a
    binary label rather than a measured range, so nothing is smoothed and no
    surface has to be chosen. The model carries one genetic standard deviation
    per environment, one residual standard deviation per environment, and one
    genetic correlation between them. Sex is the canonical environment; any
    other binary label — an exposure, a cohort, a diagnosis — works the same
    way.

    ``environment`` is one label per person and must take exactly two distinct
    finite values, compared exactly. **Name them through ``levels`` if you can**
    — the column is then checked against what you expected, so a third value or
    a missing code where a group should be is refused rather than fitted. Left
    unnamed the two are inferred, and a negative one is refused, because ``-9``
    is the missing code in every pedigree format and far likelier a sentinel
    than a group. Nought is left alone, since a 0/1 exposure is ordinary; a
    caller who genuinely means -1 and +1 says so through ``levels``. The people carrying the smaller label form
    the first group everywhere in the results. **A missing or unknown label
    must be resolved or removed before building**, because a model that quietly
    puts the unknowns together is estimating a correlation with a third group
    in it.

    **Two findings live here and they are not the same.**

    *The heritability differs between the environments.* The genetic variance
    is larger in one than the other. That is a difference of scale, and a
    difference of scale can come from the measurement rather than the genetics
    — men are larger, so a volume in millimetres varies more in men whether or
    not the genes differ. ``test(y, "genetic")``.

    *The genes differ between the environments.* The genetic correlation across
    the environments is below one, so the genes that matter in one are not
    exactly those that matter in the other. No change of units can produce
    this, and it is usually the interesting claim. ``test(y, "correlation")``.

    Unlike the kernel surfaces in :class:`GxeModel`, the correlation here is a
    parameter rather than a function of distance, so it is free to be negative:
    a genotype raising a trait in one environment and lowering it in the other
    is reachable.

    **The two residual standard deviations are free, and they should be.** A
    trait simply noisier in one environment would otherwise push its extra
    variance into the genetic term, and the genetic tests would then reject
    because of measurement rather than because of genes.

    **Read the headline test first.** ``test(y, "gene_by_environment")`` puts
    the genetic constraints back at once — same variance, same genes — while
    leaving the two residual variances free. It is what stops several tests on
    one trait being read as several findings.

    **Do not use** ``test(y, "any_difference")`` **as the headline.** It ties
    the residual variances too, so a trait merely measured more noisily in one
    environment rejects it hard with nothing genetic happening. In simulation
    on the GOBS pedigree a sex difference in measurement error alone rejected
    it at p = 1e-34 while every genetic test correctly reported nothing.

    ``subject_order_sha256`` optionally carries the lowercase SHA-256 from
    ``subject_order_commitment`` for the exact fitted row order. Every fit
    record echoes it without retaining identifiers.
    """

    def __init__(
        self,
        relationship: Any,
        environment: Any,
        design: Any,
        levels: tuple[float, float] | None = None,
        *,
        subject_order_sha256: str | None = None,
    ) -> None:
        self._subject_order_sha256 = _validated_subject_order_sha256(
            subject_order_sha256
        )
        """Stored the validated row-order commitment without participant identifiers."""

        self._relationship = _owned_matrix(relationship, "relationship")
        """Stored an immutable copy of the relationship matrix."""

        self._environment = np.ascontiguousarray(
            np.asarray(environment).ravel(), dtype=np.float64
        )
        """Normalised the two-valued environment labels to contiguous binary64."""

        self._design = _owned_matrix(design, "design")
        """Stored an immutable copy of the fixed-effect design."""

        self._levels = None if levels is None else (float(levels[0]), float(levels[1]))
        """Recorded explicit environment levels when the caller supplied them."""

    def fit(self, y: Any, reml: bool = True) -> dict[str, Any]:
        """Fit, with everything free.

        ``counts`` comes back with the answer because a correlation estimated
        across a group of thirty is not the same claim as one across a thousand,
        and the fit itself cannot tell you which you have.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response to contiguous binary64 for the discrete fit."""

        (
            genetic,
            residual,
            heritability,
            correlation,
            effects,
            errors,
            loglik,
            converged,
            polished,
            gradient,
            counts,
            levels,
        ) = _core.discrete_gxe_fit(
            self._relationship, self._environment, self._design, y, reml, self._levels
        )
        """Fitted separate group variances and their cross-group genetic correlation."""
        return {
            "levels": list(levels),
            "genetic_variance": list(genetic),
            "residual_variance": list(residual),
            "heritability": list(heritability),
            "genetic_correlation": correlation,
            "fixed_effects": [
                {"estimate": e, "standard_error": s}
                for e, s in zip(effects, errors, strict=True)
            ],
            "loglik": loglik,
            "scaled_gradient": gradient,
            "converged": converged,
            # True where the gradient test failed on the first search and a
            # second was run from that point with the objective tolerance
            # switched off. It fires nowhere else, so `False` means this is
            # the fit the package gave before the polish existed.
            "polished": polished,
            "counts": list(counts),
            "estimator": "reml" if reml else "ml",
            "build": build_identity(),
            "subject_order_sha256": self._subject_order_sha256,
        }

    def test(
        self, y: Any, null: str = "gene_by_environment", reml: bool = True
    ) -> dict[str, Any]:
        """Test one of the five nulls.

        - ``"gene_by_environment"``: no genetic difference of any kind — the
          same variance and the same genes in both environments — with the two
          residual variances left free. Two constraints, one of which sits on a
          bound, so the reference is an even mixture of chi-square on one and
          on two degrees of freedom. **Read this one first.**
        - ``"any_difference"``: nothing differs between the environments at
          all, residual included. Three constraints on an even mixture of
          chi-square on two and on three. It is **not** a genetic test: a
          noisier environment rejects it.
        - ``"correlation"``: the same genes act in both environments. This is
          the gene-by-environment question proper. The null puts the
          correlation at the edge of what it may be, so the reference is the
          even mixture of a point mass at nought with chi-square on one degree
          of freedom. A plain chi-square would roughly double the p-value.
        - ``"genetic"``: the same genetic variance in both environments.
          Interior, so chi-square on one degree of freedom.
        - ``"residual"``: the same residual variance in both environments.
          Report it beside the others as a measurement fact, not as a genetic
          finding.

        ``rule`` names the reference distribution the p-value is a tail of, so a
        reader need not take it on trust.
        """
        allowed: tuple[str, ...] = (
            "gene_by_environment",
            "any_difference",
            "correlation",
            "genetic",
            "residual",
        )
        """Listed the five scientifically distinct constrained comparisons."""

        if null not in allowed:
            raise ValueError(f"null must be one of {', '.join(allowed)}, not {null!r}")
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response for the selected discrete-model comparison."""

        statistic, p_value, rule, null_loglik, alternative_loglik = (
            _core.discrete_gxe_test(
                self._relationship,
                self._environment,
                self._design,
                y,
                null,
                reml,
                self._levels,
            )
        )
        """Compared the free group covariance model with the requested null."""
        return {
            "null": null,
            "statistic": statistic,
            "p_value": p_value,
            "rule": rule,
            "null_loglik": null_loglik,
            "alternative_loglik": alternative_loglik,
            "estimator": "reml" if reml else "ml",
        }

    def correlation_interval(self, y: Any, reml: bool = True) -> dict[str, Any]:
        """A 95 per cent profile interval for the genetic correlation.

        The correlation is a parameter here rather than a function of one, so
        the interval comes from pinning it and refitting everything else, with
        endpoints where twice the drop in log likelihood reaches 3.8415.

        **The reference is the ordinary chi-square on one degree of freedom,
        not the mixture** ``test(y, "correlation")`` **uses.** That test asks
        about a correlation of exactly one, which is the edge of the parameter
        space; an interval is a statement about interior values and takes the
        interior reference. Borrowing the test's mixture would give a narrower
        interval than the coverage it claims.

        ``lower_limited`` and ``upper_limited`` say whether an endpoint sat at
        the edge of what a correlation may be rather than where the likelihood
        fell away. An interval reaching a bound is coverage without precision,
        and that is worth knowing before it is quoted.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        """Converted the response for genetic-correlation profile fits."""

        (
            estimate,
            lower,
            upper,
            lower_limited,
            upper_limited,
            level,
            profile_failures,
            contains_lower_bound,
            contains_upper_bound,
        ) = _core.discrete_gxe_correlation_interval(
            self._relationship,
            self._environment,
            self._design,
            y,
            reml,
            self._levels,
        )
        """Profiled the discrete model's directly parameterised genetic correlation."""
        return {
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "lower_limited": lower_limited,
            "upper_limited": upper_limited,
            "level": level,
            "profile_failures": profile_failures,
            "contains_lower_bound": contains_lower_bound,
            "contains_upper_bound": contains_upper_bound,
            "rule": "chi2_1",
            "estimator": "reml" if reml else "ml",
        }


class VariantSetModel:
    """Score a whole set of variants at once, in the famSKAT form.

    Testing rare variants one at a time finds nothing, because each has a
    handful of carriers. This asks instead whether the variants in a set — a
    gene, a pathway — carry more trait variance together than chance allows,
    without committing to which of them matters or which way each pushes.

    ``backgrounds`` are the covariance bases carrying everything that is not
    the set under test, and ``design`` must include its own intercept.

    **Pass ``Z = G * w``, not the kernel.** One row per person, one column per
    variant, with the column weights already applied. Nothing is lost — the
    kernel is ``Z Z'`` — and nothing ``n by n`` is ever formed, so the memory
    is one column per variant rather than one per person squared and the
    eigenvalues come from a matrix the size of the set rather than the roster.

    **The weights are your choice and they are not innocent.** Squaring is
    implicit: a column multiplier ``w`` is a variance weight of ``w**2``. The
    usual rare-focused choice is ``Beta(1, 25)`` evaluated at each minor allele
    frequency, but it encodes a belief about which variants matter, and a
    different belief gives a different answer. Running a small pre-specified
    set of weightings and combining them is more honest than picking one.

    **Why a score test and not a likelihood ratio.** The null sits on a
    boundary, and Asterism's usual 50:50 reference is right only when the
    tested matrix spreads across many eigenvalues. A variant-set kernel does
    not — a burden kernel has rank one. Measured under the null on a
    rare-variant kernel, the likelihood ratio rejected 0.020 against a nominal
    0.05, where this reached 0.0467. The null is also fitted once for a whole
    scan rather than refitted per set.

    Nothing here corrects for testing many sets.
    """

    def __init__(
        self,
        backgrounds: Sequence[Any],
        design: Any,
        y: Any,
        reml: bool = True,
    ) -> None:
        self._backgrounds = [_owned_matrix(b, "background") for b in backgrounds]
        """Copied every background covariance matrix into model-owned storage."""

        self._design = _owned_matrix(design, "design")
        """Stored an immutable copy of the null fixed-effect design."""

        self._y = np.ascontiguousarray(y, dtype=np.float64)
        """Stored the fixed response shared by every set in the scan."""

        self._reml = bool(reml)
        """Recorded the estimator used for the shared null fit."""

    def scan(self, roots: Sequence[Any]) -> list[dict[str, Any]]:
        """Score every set, fitting the null once.

        Each record carries the statistic, the p-value, the chi-square mixture
        weights it was read against, and ``trustworthy``, which is false where
        the tail is small enough that cancellation has eaten the digits. A set
        nobody carries returns ``code`` of ``VARIANT_SET_NO_CARRIERS`` rather
        than a statistic of nought dressed up as a result.
        """
        prepared: list[npt.NDArray[np.float64]] = [
            np.ascontiguousarray(np.asarray(r, dtype=np.float64), dtype=np.float64)
            for r in roots
        ]
        """Converted every weighted genotype root to contiguous binary64."""

        for root in prepared:
            if root.ndim != 2:
                raise ValueError("VARIANT_SET_ROOT_WRONG_SHAPE")
        """Refused roots that did not provide one person-by-variant matrix."""

        records: list[dict[str, object]] = _core.variant_set_scan(
            self._backgrounds, self._design, self._y, prepared, self._reml
        )
        """Fitted the null once and scored every supplied variant set."""
        return [dict(r) for r in records]

    def test(self, root: Any) -> dict[str, Any]:
        """Score one set."""
        return self.scan([root])[0]

    def scan_family(
        self,
        roots: Sequence[Any],
        correlations: Sequence[float] = (0.0, 0.01, 0.04, 0.09, 0.25, 0.5, 0.9),
    ) -> list[dict[str, Any]]:
        """Score every set across a family of assumptions, and combine them.

        Two tests bet on different truths about a set. A **burden** test
        assumes every variant pushes the trait the same way and adds them into
        one score: powerful when true, blind when half raise the trait and half
        lower it, because they cancel. A **variance-component** test assumes
        nothing about direction and asks only whether the effects are more
        scattered than chance allows: robust to a mixture, weaker when they
        genuinely agree.

        They are two ends of one dial, and ``correlations`` is that dial — the
        assumed correlation between variant effects, nought giving the
        variance-component test and approaching one giving burden.

        **Taking the best of several tests inflates a p-value unless the
        looking is paid for.** These tests are strongly dependent, being one
        score read under different assumptions, so the combination is the
        Cauchy method, whose tail is right whatever the dependence.
        ``strongest_correlation`` comes back because it says something about
        the set, but reporting its p-value alone would be exactly the inflation
        this exists to avoid: report ``p_value``.

        The same mechanism combines across weightings, which is the honest
        answer to a weight being an arbitrary choice: run several and combine,
        rather than fitting one, which the null does not identify.
        """
        prepared: list[npt.NDArray[np.float64]] = [
            np.ascontiguousarray(np.asarray(r, dtype=np.float64), dtype=np.float64)
            for r in roots
        ]
        """Converted every weighted genotype root to contiguous binary64."""

        for root in prepared:
            if root.ndim != 2:
                raise ValueError("VARIANT_SET_ROOT_WRONG_SHAPE")
        """Refused roots that did not provide one person-by-variant matrix."""

        records: list[dict[str, object]] = _core.variant_set_family_scan(
            self._backgrounds,
            self._design,
            self._y,
            prepared,
            [float(c) for c in correlations],
            self._reml,
        )
        """Scored each set across the correlation family and combined its tests."""
        return [dict(r) for r in records]

    def test_family(
        self,
        root: Any,
        correlations: Sequence[float] = (0.0, 0.01, 0.04, 0.09, 0.25, 0.5, 0.9),
    ) -> dict[str, Any]:
        """Score one set across the family."""
        return self.scan_family([root], correlations)[0]


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

    ``subject_order_sha256`` optionally carries the lowercase SHA-256 from
    ``subject_order_commitment`` for the exact fitted row order. Every fit
    record echoes it without retaining identifiers.
    """

    def __init__(
        self,
        relationship: Any,
        status: Any,
        design: Any,
        *,
        subject_order_sha256: str | None = None,
    ) -> None:
        self._subject_order_sha256 = _validated_subject_order_sha256(
            subject_order_sha256
        )
        """Stored the validated row-order commitment without participant identifiers."""

        self._relationship = _owned_matrix(relationship, "relationship")
        """Stored an immutable copy of the relationship matrix."""

        self._status = np.ascontiguousarray(
            np.asarray(status, dtype=np.float64).ravel()
        )
        """Normalised the binary case statuses to contiguous binary64."""

        self._design = _owned_matrix(design, "design")
        """Stored an immutable copy of the liability-scale fixed-effect design."""

    def fit(self) -> dict[str, Any]:
        """Fit, by maximum likelihood because nothing else is available."""
        (
            heritability,
            effects,
            loglik,
            converged,
            polished,
            gradient,
            prevalence,
            largest_family,
        ) = _core.liability_fit(self._relationship, self._status, self._design)
        """Fitted the binary trait on its fixed unit-variance liability scale."""
        return {
            "heritability": heritability,
            "scale": "liability, not observed status",
            # The first is an intercept only in the sense that it carries the
            # threshold: Phi(intercept) is the prevalence a covariate-free model
            # implies.
            "fixed_effects": list(effects),
            "loglik": loglik,
            "converged": converged,
            # True where the gradient test failed on the first search and a
            # second was run from that point with the objective tolerance
            # switched off. It fires nowhere else.
            "polished": polished,
            "scaled_gradient": gradient,
            "prevalence": prevalence,
            "largest_family": largest_family,
            "estimator": "ml",
            "build": build_identity(),
            "subject_order_sha256": self._subject_order_sha256,
        }

    def interval(self) -> dict[str, Any]:
        """A 95 per cent profile interval for the liability heritability."""
        (
            estimate,
            lower,
            upper,
            lower_limited,
            upper_limited,
            level,
            failures,
            contains_lower_bound,
            contains_upper_bound,
        ) = _core.liability_interval(self._relationship, self._status, self._design)
        """Profiled the liability heritability with explicit boundary verdicts."""
        return {
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "lower_limited": lower_limited,
            "upper_limited": upper_limited,
            "level": level,
            "profile_failures": failures,
            "contains_lower_bound": contains_lower_bound,
            "contains_upper_bound": contains_upper_bound,
            "estimator": "ml",
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
        """Compared the fitted liability model with zero additive variance."""
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
        self._relationship = _owned_matrix(relationship, "relationship")
        """Stored an immutable copy of the polygenic relationship matrix."""

        self._design = _owned_matrix(design, "design")
        """Stored an immutable copy of the marker-free fixed-effect design."""

        self._y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        """Stored the fixed response shared by every marker test."""

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
        """Converted the marker matrix to the compiled scan's binary64 layout."""

        if markers.ndim == 1:
            markers = markers.reshape(-1, 1)
            """Promoted a single marker vector to a one-column marker matrix."""

        heritability, null_loglik, rows, covariates = _core.association_sweep(
            self._relationship, self._design, self._y, markers, variance, refit_below
        )
        """Fitted the null and tested every marker under the selected variance rule."""
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

    Hand ``matrices`` to `ComponentModel` and report the fitted covariance
    coefficients, the omnibus equality test, and class contrasts. The class
    matrices have zero diagonals, so neither raw coefficient proportions nor
    mean-diagonal proportions are interpretable as shares of phenotypic
    variance.

    The pair counts are worth reading before the answer is. They are rarely
    balanced, and a class with few pairs has the least precise coefficient and
    contrasts involving it.
    """
    matrices, order, names, pairs = _core.kinship_classes(
        ids, father, mother, sex, list(keep or [])
    )
    """Built the residual relationship matrix and four parent-offspring classes."""
    return {
        "matrices": [np.ascontiguousarray(m) for m in matrices],
        "order": order,
        "class_names": names,
        "pairs": dict(zip(names, pairs, strict=True)),
    }


def tobit_fit(
    relationship: Any,
    value: Any,
    censoring: Any,
    limit: Any,
    design: Any,
    *,
    subject_order_sha256: str | None = None,
) -> dict[str, Any]:
    """Fit one trait whose measurement stops at a limit.

    ``censoring`` is 0 where the value was measured, 1 where it lies at or
    above its limit, and 2 where it lies at or below it. **The status is given
    rather than inferred**, because a censored value can carry the same number
    as a measured one — extended high-frequency audiometry records several
    limits within one frequency, and measured values coincide with them.

    ``value`` is read only where the status says measured, and ``limit`` only
    where it does not.

    The heritability that comes back is the heritability of the *complete*
    variable — the number you would have had if the instrument reached far
    enough. It is comparable with an ordinary heritability of an uncensored
    trait, and not with one fitted to values where the censored ones were
    replaced by the limit. It is maximum likelihood, never REML, so it must not
    be placed beside a REML heritability as though the two were the same.

    ``subject_order_sha256`` optionally carries the lowercase SHA-256 from
    ``subject_order_commitment`` for the exact fitted row order. The fit record
    echoes it without retaining identifiers.
    """
    subject_order_sha256 = _validated_subject_order_sha256(subject_order_sha256)
    """Validated the row-order commitment before starting numerical fitting."""

    (
        heritability,
        total_variance,
        fixed_effects,
        loglik,
        converged,
        scaled_gradient,
        censored_share,
        largest_family,
    ) = _core.tobit_fit(
        np.ascontiguousarray(relationship, dtype=float),
        np.ascontiguousarray(value, dtype=float),
        np.ascontiguousarray(censoring, dtype=np.int64),
        np.ascontiguousarray(limit, dtype=float),
        np.ascontiguousarray(design, dtype=float),
    )
    """Fitted the latent complete trait while retaining every censoring limit."""
    return {
        "heritability": heritability,
        "total_variance": total_variance,
        "fixed_effects": fixed_effects,
        "loglik": loglik,
        "converged": converged,
        "scaled_gradient": scaled_gradient,
        "censored_share": censored_share,
        "largest_family": largest_family,
        "estimator": "ml",
        "build": build_identity(),
        "subject_order_sha256": subject_order_sha256,
    }


def mixed_bivariate_fit(
    relationship: Any,
    first: dict[str, Any],
    second: dict[str, Any],
    design: Any,
    *,
    subject_order_sha256: str | None = None,
) -> dict[str, Any]:
    """Fit two traits whose measurements need not be of the same kind.

    Each trait is a dictionary with ``kind`` (``"continuous"``, ``"binary"`` or
    ``"censored"``), ``value``, ``censoring`` and ``limit``. For a binary trait
    a censoring code of 1 is a case, and the limit is nought because the
    threshold is carried by the intercept.

    **A binary trait's variance is fixed at one** and comes back as one, because
    only the sign of a liability is ever seen. Its heritability is therefore a
    liability heritability, while a continuous or censored trait's is not; the
    two must not be read as the same quantity. The genetic correlation is
    unaffected, which is what makes a mixed pair worth fitting.

    ``subject_order_sha256`` optionally carries the lowercase SHA-256 from
    ``subject_order_commitment`` for the exact fitted row order. The fit record
    echoes it without retaining identifiers.
    """
    subject_order_sha256 = _validated_subject_order_sha256(subject_order_sha256)
    """Validated the row-order commitment before starting numerical fitting."""

    kinds: dict[str, int] = {"continuous": 0, "binary": 1, "censored": 2}
    """Mapped public trait-kind names to their compiled representation."""

    def unpack(
        each: dict[str, Any],
    ) -> tuple[
        int,
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ]:
        """Convert one public trait specification to compiled array inputs."""
        if each["kind"] not in kinds:
            raise ValueError("MIXED_BIVARIATE_TRAIT_KIND_UNKNOWN")
        return (
            kinds[each["kind"]],
            np.ascontiguousarray(each["value"], dtype=float),
            np.ascontiguousarray(each["censoring"], dtype=np.int64),
            np.ascontiguousarray(each["limit"], dtype=float),
        )

    first_kind, first_value, first_censoring, first_limit = unpack(first)
    """Converted the first trait specification to compiled inputs."""

    second_kind, second_value, second_censoring, second_limit = unpack(second)
    """Converted the second trait specification to compiled inputs."""

    (
        heritability,
        total_variance,
        genetic_correlation,
        residual_correlation,
        first_effects,
        second_effects,
        loglik,
        converged,
        scaled_gradient,
        largest_family,
    ) = _core.mixed_bivariate_fit(
        np.ascontiguousarray(relationship, dtype=float),
        first_kind,
        first_value,
        first_censoring,
        first_limit,
        second_kind,
        second_value,
        second_censoring,
        second_limit,
        np.ascontiguousarray(design, dtype=float),
    )
    """Fitted both possibly unlike traits through one joint family likelihood."""
    return {
        "heritability": heritability,
        "total_variance": total_variance,
        "genetic_correlation": genetic_correlation,
        "residual_correlation": residual_correlation,
        "fixed_effects": [first_effects, second_effects],
        "loglik": loglik,
        "converged": converged,
        "scaled_gradient": scaled_gradient,
        "largest_family": largest_family,
        "kinds": [first["kind"], second["kind"]],
        "estimator": "ml",
        "build": build_identity(),
        "subject_order_sha256": subject_order_sha256,
    }


def tobit_interval(
    relationship: Any,
    value: Any,
    censoring: Any,
    limit: Any,
    design: Any,
) -> dict[str, Any]:
    """A 95 per cent profile-likelihood interval for the censored heritability.

    ``lower_limited`` and ``upper_limited`` say whether an end sits on the
    parameter's own bound rather than where the profile fell away. An end on a
    bound means **the data did not rule that end out**, which is a different
    statement from the interval stopping there.

    ``profile_failures`` counts fits along the profile that failed or did not
    converge. Each one widened the interval rather than narrowing it, which is
    the safe direction, but a large count means the interval rests on fewer
    points than its width suggests.

    Read ``censored_share`` beside the answer. On simulated data the model
    recovers the truth to three quarters censored; on real extended
    high-frequency thresholds it degrades past about half, where too little of
    the upper tail is left to estimate a variance from.
    """
    (
        estimate,
        lower,
        upper,
        lower_limited,
        upper_limited,
        level,
        contains_lower_bound,
        contains_upper_bound,
        profile_failures,
        censored_share,
    ) = _core.tobit_interval(
        np.ascontiguousarray(relationship, dtype=float),
        np.ascontiguousarray(value, dtype=float),
        np.ascontiguousarray(censoring, dtype=np.int64),
        np.ascontiguousarray(limit, dtype=float),
        np.ascontiguousarray(design, dtype=float),
    )
    """Profiled the complete-trait heritability under the observed censoring."""
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "lower_limited": lower_limited,
        "upper_limited": upper_limited,
        "level": level,
        # Whether the bound itself belongs to the interval, decided by the
        # Self-Liang mixture rather than by the end having landed on it. Absent
        # where the end is not on its bound, and absent where the fit there
        # could not be made -- which means nobody measured it, not that the
        # question does not apply.
        "contains_lower_bound": contains_lower_bound,
        "contains_upper_bound": contains_upper_bound,
        "profile_failures": profile_failures,
        "censored_share": censored_share,
        "estimator": "ml",
    }


def tobit_test(
    relationship: Any,
    value: Any,
    censoring: Any,
    limit: Any,
    design: Any,
) -> dict[str, Any]:
    """Test the censored heritability against nought.

    The null holds the heritability at nought, which is its own bound, so the
    reference is the Self-Liang 50:50 mixture of chi-square on nought and one
    degrees of freedom rather than a plain chi-square. ``rule`` says which was
    used, as data rather than as a promise.
    """
    statistic, p_value, rule, null_loglik, alternative_loglik = _core.tobit_test(
        np.ascontiguousarray(relationship, dtype=float),
        np.ascontiguousarray(value, dtype=float),
        np.ascontiguousarray(censoring, dtype=np.int64),
        np.ascontiguousarray(limit, dtype=float),
        np.ascontiguousarray(design, dtype=float),
    )
    """Compared the censored trait model with zero additive variance."""
    return {
        "statistic": statistic,
        "p_value": p_value,
        "rule": rule,
        "null_loglik": null_loglik,
        "alternative_loglik": alternative_loglik,
        "estimator": "ml",
    }


def mixed_bivariate_test(
    relationship: Any,
    first: dict[str, Any],
    second: dict[str, Any],
    design: Any,
    coordinate: str = "genetic_correlation",
    null: float = 0.0,
) -> dict[str, Any]:
    """Test one correlation of the mixed bivariate model against a fixed value.

    ``coordinate`` is ``genetic_correlation`` or ``residual_correlation``, the
    same names the interval takes.

    ``null`` is the value tested against, and the two worth asking are the ends
    of the question. Against nought: do these traits share any genes at all? An
    estimate with an interval does not answer that. Against one: are they the
    same genes?

    The reference distribution follows from which. Nought is interior to a
    correlation's range, so a plain chi-square on one degree of freedom applies
    and no boundary mixture is needed. Plus or minus one is the edge of that
    range, so the null rests on a bound and takes the Self-Liang even mixture,
    which is reported as ``mixture_50_50`` rather than ``chi2_1``.
    """
    coordinates: dict[str, int] = {
        "genetic_correlation": 4,
        "residual_correlation": 5,
    }
    """Mapped the two testable public correlations to compiled coordinates."""

    if coordinate not in coordinates:
        raise ValueError("MIXED_BIVARIATE_COORDINATE_HAS_NO_TEST")
    if not np.isfinite(null) or abs(float(null)) > 1.0:
        raise ValueError("MIXED_BIVARIATE_NULL_OUTSIDE_CORRELATION_RANGE")
    kinds: dict[str, int] = {"continuous": 0, "binary": 1, "censored": 2}
    """Mapped public trait-kind names to their compiled representation."""

    def unpack(
        each: dict[str, Any],
    ) -> tuple[
        int,
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ]:
        """Convert one public trait specification to compiled array inputs."""
        if each["kind"] not in kinds:
            raise ValueError("MIXED_BIVARIATE_TRAIT_KIND_UNKNOWN")
        return (
            kinds[each["kind"]],
            np.ascontiguousarray(each["value"], dtype=float),
            np.ascontiguousarray(each["censoring"], dtype=np.int64),
            np.ascontiguousarray(each["limit"], dtype=float),
        )

    first_kind, first_value, first_censoring, first_limit = unpack(first)
    """Converted the first trait specification to compiled inputs."""

    second_kind, second_value, second_censoring, second_limit = unpack(second)
    """Converted the second trait specification to compiled inputs."""

    what, statistic, p_value, rule, null_loglik, alternative_loglik = (
        _core.mixed_bivariate_test(
            np.ascontiguousarray(relationship, dtype=float),
            first_kind,
            first_value,
            first_censoring,
            first_limit,
            second_kind,
            second_value,
            second_censoring,
            second_limit,
            np.ascontiguousarray(design, dtype=float),
            coordinates[coordinate],
            float(null),
        )
    )
    """Compared the free correlation with its zero-correlation joint model."""
    return {
        "what": what,
        "statistic": statistic,
        "p_value": p_value,
        "rule": rule,
        "null_loglik": null_loglik,
        "alternative_loglik": alternative_loglik,
        "estimator": "ml",
    }


def mixed_bivariate_interval(
    relationship: Any,
    first: dict[str, Any],
    second: dict[str, Any],
    design: Any,
    coordinate: str = "genetic_correlation",
) -> dict[str, Any]:
    """A 95 per cent profile-likelihood interval for one bivariate coordinate.

    ``coordinate`` is ``"heritability_one"``, ``"heritability_two"``,
    ``"genetic_correlation"`` or ``"residual_correlation"``.

    **The variances have no interval on purpose.** A binary trait's is fixed at
    one because a liability has no scale of its own, so an interval on it would
    describe that assumption rather than the data.

    ``lower_limited`` and ``upper_limited`` say whether an end sits on the
    coordinate's own bound — nought or one for a heritability, minus one or one
    for a correlation — rather than where the profile fell away. An end on a
    bound means the data did not rule that end out, which is a different
    statement from the interval stopping there.
    """
    coordinates: dict[str, int] = {
        "heritability_one": 0,
        "heritability_two": 1,
        "genetic_correlation": 4,
        "residual_correlation": 5,
    }
    """Mapped every interval-bearing public quantity to its compiled coordinate."""

    if coordinate not in coordinates:
        raise ValueError("MIXED_BIVARIATE_COORDINATE_HAS_NO_INTERVAL")
    kinds: dict[str, int] = {"continuous": 0, "binary": 1, "censored": 2}
    """Mapped public trait-kind names to their compiled representation."""

    def unpack(
        each: dict[str, Any],
    ) -> tuple[
        int,
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ]:
        """Convert one public trait specification to compiled array inputs."""
        return (
            kinds[each["kind"]],
            np.ascontiguousarray(each["value"], dtype=float),
            np.ascontiguousarray(each["censoring"], dtype=np.int64),
            np.ascontiguousarray(each["limit"], dtype=float),
        )

    first_kind, first_value, first_censoring, first_limit = unpack(first)
    """Converted the first trait specification to compiled inputs."""

    second_kind, second_value, second_censoring, second_limit = unpack(second)
    """Converted the second trait specification to compiled inputs."""

    (
        what,
        estimate,
        lower,
        upper,
        lower_limited,
        upper_limited,
        level,
        profile_failures,
        contains_lower_bound,
        contains_upper_bound,
    ) = _core.mixed_bivariate_interval(
        np.ascontiguousarray(relationship, dtype=float),
        first_kind,
        first_value,
        first_censoring,
        first_limit,
        second_kind,
        second_value,
        second_censoring,
        second_limit,
        np.ascontiguousarray(design, dtype=float),
        coordinates[coordinate],
    )
    """Profiled the requested joint-model heritability or correlation."""
    return {
        "what": what,
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "lower_limited": lower_limited,
        "upper_limited": upper_limited,
        "level": level,
        "profile_failures": profile_failures,
        "contains_lower_bound": contains_lower_bound,
        "contains_upper_bound": contains_upper_bound,
        "estimator": "ml",
    }


def region_log_probability(mean: Any, sign: Any, covariance: Any) -> float:
    """The conditional region log-probability the censored models rest on.

    **Exposed so that it can be checked, not so that it can be used.** This uses
    the univariate normal calculation at one coordinate, fixed sixteen-point
    quadrature at two, and Mendell-Elston sequential truncation above two. Every
    censored heritability in the package rests on it. All the evidence for those
    models was generated on pairs, where the sequential branch never runs at
    all, so the only way to learn how it behaves in a large family is to call it
    beside an independent reference.
    ``checks/sequential_against_ghk.py`` is that reference.

    ``mean`` is each coordinate's mean already centred on its own limit, and
    ``sign`` is 1.0 where the value lies above that limit and -1.0 where it
    lies below — the convention the censored model builds. The region is
    therefore about nought, and the probability returned is that of the whole
    orthant, jointly and not coordinate by coordinate.
    """
    return float(
        _core.region_log_probability(
            np.ascontiguousarray(mean, dtype=float),
            np.ascontiguousarray(sign, dtype=float),
            np.ascontiguousarray(covariance, dtype=float),
        )
    )
