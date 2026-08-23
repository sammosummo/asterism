"""Python interface for Asterism's latent mediation model.

This module normalises an ergonomic list of family mappings. Covariance
construction, conditioning, integration, ascertainment, likelihood arithmetic,
and optimisation remain in Rust.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from ._core import LatentMediationCore
from ._core import latent_mediation_simulate as _simulate

__all__: list[str] = ["LatentMediationModel"]
"""Exported the public latent-mediation model."""

_MISSING: object = object()
"""Distinguished an omitted optional field from an explicit null value."""


# asterism-style: allow private-helper -- shared canonical-field validation primitive
def _field(
    family: Mapping[str, Any],
    name: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """Return one canonical field with a stable missing-field code."""
    if name in family:
        return family[name]
    if default is not _MISSING:
        return default
    raise ValueError(f"LATENT_MEDIATION_{name.upper()}_MISSING")


# asterism-style: allow private-helper -- reused sequence and numeric validation boundary
def _sequence(
    value: Any,
    code: str,
    *,
    numeric: bool = False,
    allow_none: bool = False,
) -> list[Any]:
    """Copy a Python or NumPy sequence without performing arithmetic."""
    if np.ma.isMaskedArray(value):
        if np.any(np.ma.getmaskarray(value)):
            raise ValueError("LATENT_MEDIATION_ARRAY_MASKED")
        value = np.ma.getdata(value)
        """Removed an all-false mask while retaining the underlying values."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "c":
            raise ValueError("LATENT_MEDIATION_NUMERIC_NOT_REAL")
        copied: list[Any] = value.tolist()
        """Copied the NumPy sequence into caller-independent Python values."""
        if not isinstance(copied, list):
            raise ValueError(code)
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(code)
    else:
        copied = list(value)
        """Copied a general Python sequence before normalisation."""
    if numeric:
        normalised: list[Any] = []
        """Accumulated real numeric values after rejecting ambiguous booleans."""
        for item in copied:
            if item is None and allow_none:
                normalised.append(None)
                continue
            if isinstance(item, (bool, np.bool_)):
                raise ValueError("LATENT_MEDIATION_NUMERIC_BOOLEAN")
            if not isinstance(item, Real):
                raise ValueError("LATENT_MEDIATION_NUMERIC_NOT_REAL")
            try:
                normalised.append(float(item))
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError("LATENT_MEDIATION_NUMERIC_NOT_REAL") from error
        return normalised
    return copied


# asterism-style: allow private-helper -- reused person-alignment validation primitive
def _person_vector(
    family: Mapping[str, Any],
    name: str,
    size: int,
    *,
    scalar: bool = False,
    allow_none: bool = False,
) -> list[Any]:
    """Expand an allowed scalar or copy a person-aligned sequence."""
    value: Any = _field(family, name)
    """Read the scalar or person-aligned field to expand."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("LATENT_MEDIATION_NUMERIC_BOOLEAN")
    if scalar and isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        return [value] * size
    return _sequence(
        value,
        f"LATENT_MEDIATION_{name.upper()}_NOT_A_SEQUENCE",
        numeric=True,
        allow_none=allow_none,
    )


# asterism-style: allow private-helper -- reused binary-status validation primitive
def _status_vector(family: Mapping[str, Any], name: str, size: int) -> list[Any]:
    """Copy one status vector while refusing Python and NumPy booleans."""
    values: list[Any] = _sequence(
        _field(family, name),
        f"LATENT_MEDIATION_{name.upper()}_NOT_A_SEQUENCE",
    )
    """Copied the person-aligned status sequence before binary validation."""
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("LATENT_MEDIATION_STATUS_BOOLEAN")
        if value is not None and not isinstance(value, (int, np.integer)):
            raise ValueError("LATENT_MEDIATION_STATUS_NOT_BINARY")
    return [None if value is None else int(value) for value in values]


class LatentMediationModel:
    """A collection of independent families for evaluation and fitting."""

    __slots__: tuple[str, ...] = ("_core",)
    """Restricted each wrapper to its sealed Rust model."""

    def __init__(
        self,
        families: Sequence[Mapping[str, Any]],
        *,
        qmc_points: int = 8192,
    ) -> None:
        """Validate and prepare independent-family contributions in Rust.

        Args:
            families: Flattened family specifications and observations. Every
                vector is person-aligned; ``latent_mean`` is process-major. Family
                size is inferred from the relationship matrix.
            qmc_points: Deterministic Genz-Halton points for rectangles above
                two dimensions.
        """
        if isinstance(families, (str, bytes)) or not isinstance(families, Sequence):
            raise ValueError("LATENT_MEDIATION_FAMILIES_NOT_A_SEQUENCE")
        if not families:
            raise ValueError("LATENT_MEDIATION_NO_FAMILIES")

        relationships: list[list[list[Any]]] = []
        """Accumulated one copied relationship matrix per family."""
        latent_means: list[list[Any]] = []
        """Accumulated process-major latent means per family."""
        mediator_measurements: list[list[Any]] = []
        """Accumulated person-aligned mediator measurements."""
        mediator_measurement_error_variances: list[list[Any]] = []
        """Accumulated known mediator measurement-error variances."""
        mediator_proxy_statuses: list[list[Any]] = []
        """Accumulated observed binary mediator-proxy statuses."""
        outcome_statuses: list[list[Any]] = []
        """Accumulated observed binary or staged outcome statuses."""
        mediator_thresholds: list[list[Any]] = []
        """Accumulated mediator-proxy thresholds."""
        outcome_thresholds: list[list[Any]] = []
        """Accumulated outcome thresholds."""
        mediator_proxy_sensitivities: list[list[Any]] = []
        """Accumulated known mediator-proxy sensitivities."""
        mediator_proxy_specificities: list[list[Any]] = []
        """Accumulated known mediator-proxy specificities."""
        ascertainments: list[str] = []
        """Accumulated each family's ascertainment rule."""
        proband_indices: list[Any] = []
        """Accumulated optional within-family proband indices."""
        mediator_designs: list[list[list[Any]]] = []
        """Accumulated mediator fixed-effect designs."""
        outcome_designs: list[list[list[Any]]] = []
        """Accumulated outcome fixed-effect designs."""
        outcome_prevalences: list[list[Any]] = []
        """Accumulated binary outcome prevalence assumptions."""
        # Per family, per person, the share in each ordered outcome category.
        # An empty list for a person means their outcome is binary, which is
        # what every family written before staging existed carries.
        outcome_category_prevalences: list[list[Any]] = []
        """Accumulated ordered-category prevalence assumptions."""
        ascertainment_categories: list[Any] = []
        """Accumulated optional staged-outcome ascertainment categories."""

        for family in families:
            if not isinstance(family, Mapping):
                raise ValueError("LATENT_MEDIATION_FAMILY_NOT_A_MAPPING")
            relationship_rows: list[Any] = _sequence(
                _field(family, "relationship"),
                "LATENT_MEDIATION_RELATIONSHIP_NOT_A_SEQUENCE",
            )
            """Copied the submitted relationship rows before validating each row."""
            relationship: list[list[Any]] = [
                _sequence(
                    row,
                    "LATENT_MEDIATION_RELATIONSHIP_ROW_NOT_A_SEQUENCE",
                    numeric=True,
                )
                for row in relationship_rows
            ]
            """Normalised the family's relationship matrix to real Python values."""
            relationships.append(relationship)
            size: int = len(relationship)
            """Inferred the family size from its square relationship matrix."""
            for name, into in (
                ("mediator_design", mediator_designs),
                ("outcome_design", outcome_designs),
            ):
                rows: Any = _field(family, name, default=[])
                """Read one optional fixed-effect design before row validation."""
                into.append(
                    [
                        _sequence(
                            row,
                            "LATENT_MEDIATION_DESIGN_ROW_NOT_A_SEQUENCE",
                            numeric=True,
                        )
                        for row in _sequence(
                            rows, "LATENT_MEDIATION_DESIGN_NOT_A_SEQUENCE"
                        )
                    ]
                )
            latent_means.append(
                _sequence(
                    _field(family, "latent_mean", default=[0.0] * (2 * size)),
                    "LATENT_MEDIATION_LATENT_MEAN_NOT_A_SEQUENCE",
                    numeric=True,
                )
            )
            mediator_measurements.append(
                _person_vector(family, "mediator_measurement", size, allow_none=True)
            )
            mediator_measurement_error_variances.append(
                _person_vector(
                    family,
                    "mediator_measurement_error_variance",
                    size,
                    allow_none=True,
                )
            )
            mediator_proxy_statuses.append(
                _status_vector(family, "mediator_proxy_status", size)
            )
            outcome_statuses.append(_status_vector(family, "outcome_status", size))
            mediator_thresholds.append(
                _person_vector(family, "mediator_threshold", size, scalar=True)
            )
            outcome_prevalences.append(
                _sequence(
                    _field(family, "outcome_prevalence", default=[]),
                    "LATENT_MEDIATION_OUTCOME_PREVALENCE_NOT_A_SEQUENCE",
                    numeric=True,
                    allow_none=True,
                )
            )
            staged: Any = _field(family, "outcome_category_prevalence", default=[])
            """Read optional ordered-category prevalence rows."""
            outcome_category_prevalences.append(
                [
                    _sequence(
                        person,
                        "LATENT_MEDIATION_OUTCOME_CATEGORY_PREVALENCE_NOT_A_SEQUENCE",
                        numeric=True,
                    )
                    for person in _sequence(
                        staged,
                        "LATENT_MEDIATION_OUTCOME_CATEGORY_PREVALENCE_NOT_A_SEQUENCE",
                    )
                ]
            )
            ascertainment_categories.append(
                _field(family, "ascertainment_category", default=None)
            )
            outcome_thresholds.append(
                _person_vector(
                    family,
                    "outcome_threshold",
                    size,
                    scalar=True,
                )
            )
            mediator_proxy_sensitivities.append(
                _person_vector(family, "mediator_proxy_sensitivity", size, scalar=True)
            )
            mediator_proxy_specificities.append(
                _person_vector(family, "mediator_proxy_specificity", size, scalar=True)
            )
            ascertainment: Any = _field(family, "ascertainment")
            """Read the family's named ascertainment rule."""
            if not isinstance(ascertainment, str):
                raise ValueError("LATENT_MEDIATION_ASCERTAINMENT_NOT_A_STRING")
            ascertainments.append(ascertainment)
            proband_index: Any = family.get("proband_index")
            """Read the optional named-proband position."""
            if isinstance(proband_index, (bool, np.bool_)):
                raise ValueError("LATENT_MEDIATION_PROBAND_BOOLEAN")
            proband_indices.append(
                int(proband_index)
                if isinstance(proband_index, np.integer)
                else proband_index
            )

        self._core = LatentMediationCore._build(
            relationships,
            latent_means,
            mediator_measurements,
            mediator_measurement_error_variances,
            mediator_proxy_statuses,
            outcome_statuses,
            mediator_thresholds,
            outcome_thresholds,
            mediator_proxy_sensitivities,
            mediator_proxy_specificities,
            ascertainments,
            proband_indices,
            qmc_points,
            mediator_designs,
            outcome_designs,
            outcome_prevalences,
            outcome_category_prevalences,
            ascertainment_categories,
        )
        """Built the sealed Rust model from the fully normalised family arrays."""

    def evaluate(
        self,
        *,
        a: float,
        b: float,
        c_prime: float,
        d: float,
        sigma_m2: float,
    ) -> dict[str, Any]:
        """Evaluate the Rust likelihood at one structural-parameter point."""
        return self._core.evaluate(
            a=a,
            b=b,
            c_prime=c_prime,
            d=d,
            sigma_m2=sigma_m2,
        )

    def test_vertical(self, *, bootstrap_replicates: int = 200) -> dict[str, Any]:
        """Test the vertical estimand ``a * b`` against nought.

        **The null is a union, not a point.** ``a * b = 0`` holds whenever the
        mediator carries no inherited signal (``a = 0``) *or* the mediator does
        not reach the outcome (``b = 0``), and those are different models. One
        likelihood ratio has no reference distribution across a union, which is
        why this model reported point estimates and no p-value until now.

        The construction is the intersection-union test: reject the union only
        when both parts are rejected, so the p-value is the larger of the two.
        That is exactly level ``alpha``, at the cost of being conservative --
        most so near ``a = b = 0``, where both parts are true at once.

        Each part carries its own reference. ``a`` is bounded below at nought,
        so its null sits on a boundary and takes the even mixture of a point
        mass with chi-square on one; ``b`` is signed and interior, so it takes
        an ordinary chi-square on one. Both are reported beside the combined
        p-value, because which of them binds says what the data could not
        establish.

        This tests the mediated path, not whether mediation is the right
        account. A trait and a mediator sharing inherited causes will reject
        this null with nothing being mediated, and no likelihood separates
        those two stories.
        """
        return self._core.test_vertical(bootstrap_replicates)

    def test_horizontal(self) -> dict[str, Any]:
        """Test the horizontal estimand ``c_prime`` against nought.

        **The null is a point, not a union**, which is what makes this the
        simpler of the two tests. The direct inherited effect is a single
        signed coordinate -- an effect outside measured hearing may run either
        way -- so it is interior, and the ordinary chi-square on one degree of
        freedom is the whole of the reference. There is no boundary mixture to
        choose and no simulated reference, so this costs two fits rather than
        the few hundred the vertical test can cost.

        **It refuses where the mediator loading is at nought.** There the
        inherited covariance carries the direct path and the outcome loading
        only as ``c'^2 + d^2``, the two rotate freely against each other, and a
        p-value would report which of the pair the optimiser happened to pick.
        Both the free and the held fit are checked, because the loading can be
        positive when free and fall to its bound once the direct path is held.

        A refusal here is the answer, not a gap: a horizontal component that
        fails its identification diagnostics is not separately estimable, which
        is a different statement from its being nought.
        """
        return self._core.test_horizontal()

    def horizontal_set(self, *, searched_to: float = 4.0) -> dict[str, Any]:
        """A 97.5 per cent confidence set for the horizontal estimand.

        Built by inverting the same likelihood ratio :meth:`test_horizontal`
        computes, at chi-square on one at 0.975 because that is what a
        two-sided set at the Bonferroni .025 gives.

        **An end that did not close is ``None``, not a number.** Reporting the
        edge of the search would be a statement about how far the search went
        rather than about the data. ``unbounded`` is true when neither end
        closed, and that is the identification diagnostic in its most useful
        form: a profile that falls away in neither direction is what a direct
        path the data cannot locate looks like from the data's side. It sees
        the near-boundary case that :meth:`test_horizontal` cannot, because
        that refuses only when the loading rests exactly on its bound.

        ``searched_to`` is the distance either side of the estimate that is
        searched. Widening it costs fits and can only close an end that a
        narrower search left open.
        """
        return self._core.horizontal_set(searched_to)

    def vertical_set(
        self,
        *,
        searched_to: float = 1.0,
        bootstrap_replicates: int = 200,
    ) -> dict[str, Any]:
        """A 97.5 per cent confidence set for the vertical estimand.

        **Nought is decided differently from everywhere else, and has to be.**
        Away from nought, holding the estimand is one constraint on a curve --
        the estimand is a product, so a held value fixes the path at the value
        over the loading -- and the likelihood ratio has an ordinary
        chi-square reference. At nought the null is a union, the loading is
        nought or the path is, and no single ratio spans it, so membership
        there comes from the intersection-union test instead.

        **Read ``disjoint`` before treating ``lower`` and ``upper`` as an
        interval.** The profile can admit values either side of nought while
        the union test excludes nought itself, and then the set is genuinely
        two pieces and the values between the ends are not all in it. The
        application requires such a set to be retained rather than reported as
        the interval that covers both.

        This is the expensive one. Holding a product means scanning the loading
        and taking the best of two dozen fits per value examined, where the
        horizontal set needs one.
        """
        return self._core.vertical_set(searched_to, bootstrap_replicates)

    def fit(self) -> dict[str, Any]:
        """Fit the five structural parameters with a fixed numerical recipe.

        Rust owns parameter scaling, bounds, deterministic starts, optimisation,
        convergence checks, integration diagnostics, and record construction.
        """
        return self._core.fit()


def simulate(
    *,
    relationship: Sequence[Sequence[float]],
    a: float,
    b: float,
    c_prime: float,
    d: float,
    sigma_m2: float,
    families: int,
    seed: int,
    outcome_threshold: float | Sequence[float] = 0.0,
    mediator_threshold: float | Sequence[float] = 0.0,
    measurement_error_variance: float | Sequence[float | None] | None = 0.15,
    observe_mediator_proxy: bool | Sequence[bool] = False,
    sensitivity: float | Sequence[float] = 0.8,
    specificity: float | Sequence[float] = 0.85,
    observe_outcome: bool | Sequence[bool] = True,
    ascertainment: str = "population_unconditioned",
    proband_index: int | None = None,
    mediator_design: Sequence[Sequence[float]] | None = None,
    mediator_coefficients: Sequence[float] | None = None,
    outcome_design: Sequence[Sequence[float]] | None = None,
    outcome_coefficients: Sequence[float] | None = None,
    outcome_prevalence: float | Sequence[float | None] | None = None,
    outcome_category_prevalence: Sequence[float]
    | Sequence[Sequence[float]]
    | None = None,
    ascertainment_category: int | None = None,
) -> list[dict[str, Any]]:
    """Draw families from the model, for calibration, coverage and power work.

    One family's shape in, that many draws out, as the same dictionaries
    :class:`LatentMediationModel` takes — so a campaign is ``simulate`` then
    fit, with nothing in between to get wrong.

    Anything person-specific may be given as one value for everybody or as a
    list the length of the family.

    **The draws use the same covariance construction as the likelihood.** A
    simulator that built it its own way would make a calibration measure the
    agreement between two constructions rather than the behaviour of the test;
    the construction itself is checked against an independently written
    evaluation, which is where that assurance belongs.

    ``ascertainment="condition_on_named_proband_case"`` draws and redraws until
    the named person is a case, which is what the model's denominator assumes
    and what a clinic roster actually is. Drawing unconditionally and keeping
    the cases would be a different design.

    Args:
        relationship: One family's relationship matrix, as twice the kinship.
        a: Mediator loading on the inherited factor.
        b: Path from mediator to outcome.
        c_prime: Direct inherited effect on the outcome.
        d: Residual inherited-outcome loading used by the structural model.
        sigma_m2: Residual mediator variance.
        families: Number of independent families to draw.
        seed: Fixed random seed making a campaign exactly reproducible.
        outcome_threshold: Scalar or person-specific binary-outcome threshold.
        mediator_threshold: Scalar or person-specific mediator-proxy threshold.
        measurement_error_variance: Known scalar or person-specific mediator
            measurement-error variance; ``None`` marks no measurement.
        observe_mediator_proxy: Scalar or person-specific proxy-observation flag.
        sensitivity: Known scalar or person-specific proxy sensitivity.
        specificity: Known scalar or person-specific proxy specificity.
        observe_outcome: Scalar or person-specific outcome-observation flag.
        ascertainment: Population or named-proband ascertainment rule.
        proband_index: Within-family proband position for conditioned sampling.
        mediator_design: Optional person-by-covariate mediator design.
        mediator_coefficients: Coefficients matching ``mediator_design``.
        outcome_design: Optional person-by-covariate outcome design.
        outcome_coefficients: Coefficients matching ``outcome_design``.
        outcome_prevalence: Scalar or person-specific binary prevalence.
        outcome_category_prevalence: Optional ordered-category prevalence rows.
        ascertainment_category: Ordered outcome category defining ascertainment.

    Returns:
        Family dictionaries accepted directly by :class:`LatentMediationModel`.

    Raises:
        ValueError: With a stable code when the design does not describe a
            family, a conditioned proband is absent, or conditioning cannot be
            satisfied.
    """
    size: int = len(relationship)
    """Read the number of people whose scalar settings may need expansion."""

    def spread(value: Any) -> list[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return [value] * size
        if len(value) != size:
            raise ValueError("LATENT_MEDIATION_DESIGN_SHAPE_INVALID")
        return list(value)

    return list(
        _simulate(
            [list(map(float, row)) for row in relationship],
            [list(map(float, row)) for row in (mediator_design or [])],
            [float(v) for v in (mediator_coefficients or [])],
            [list(map(float, row)) for row in (outcome_design or [])],
            [float(v) for v in (outcome_coefficients or [])],
            [None if v is None else float(v) for v in spread(outcome_prevalence)],
            [float(v) for v in spread(mediator_threshold)],
            [float(v) for v in spread(outcome_threshold)],
            [
                None if v is None else float(v)
                for v in spread(measurement_error_variance)
            ],
            [bool(v) for v in spread(observe_mediator_proxy)],
            [float(v) for v in spread(sensitivity)],
            [float(v) for v in spread(specificity)],
            [bool(v) for v in spread(observe_outcome)],
            float(a),
            float(b),
            float(c_prime),
            float(d),
            float(sigma_m2),
            int(families),
            int(seed),
            ascertainment,
            proband_index,
            None
            if outcome_category_prevalence is None
            else [
                [float(share) for share in person]
                for person in (
                    [outcome_category_prevalence] * size
                    if outcome_category_prevalence
                    and not isinstance(outcome_category_prevalence[0], Sequence)
                    else outcome_category_prevalence
                )
            ],
            None if ascertainment_category is None else int(ascertainment_category),
        )
    )
