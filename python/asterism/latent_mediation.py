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

__all__ = ["LatentMediationModel"]

_MISSING = object()


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
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "c":
            raise ValueError("LATENT_MEDIATION_NUMERIC_NOT_REAL")
        copied = value.tolist()
        if not isinstance(copied, list):
            raise ValueError(code)
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(code)
    else:
        copied = list(value)
    if numeric:
        normalised: list[Any] = []
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


def _person_vector(
    family: Mapping[str, Any],
    name: str,
    size: int,
    *,
    scalar: bool = False,
    allow_none: bool = False,
) -> list[Any]:
    """Expand an allowed scalar or copy a person-aligned sequence."""
    value = _field(family, name)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("LATENT_MEDIATION_NUMERIC_BOOLEAN")
    if scalar and isinstance(value, Real) and not isinstance(
        value, (bool, np.bool_)
    ):
        return [value] * size
    return _sequence(
        value,
        f"LATENT_MEDIATION_{name.upper()}_NOT_A_SEQUENCE",
        numeric=True,
        allow_none=allow_none,
    )


def _status_vector(
    family: Mapping[str, Any], name: str, size: int
) -> list[Any]:
    """Copy one status vector while refusing Python and NumPy booleans."""
    values = _sequence(
        _field(family, name),
        f"LATENT_MEDIATION_{name.upper()}_NOT_A_SEQUENCE",
    )
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("LATENT_MEDIATION_STATUS_BOOLEAN")
        if value is not None and not isinstance(value, (int, np.integer)):
            raise ValueError("LATENT_MEDIATION_STATUS_NOT_BINARY")
    return [None if value is None else int(value) for value in values]


class LatentMediationModel:
    """A collection of independent families for evaluation and fitting."""

    __slots__ = ("_core",)

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
        if isinstance(families, (str, bytes)) or not isinstance(
            families, Sequence
        ):
            raise ValueError("LATENT_MEDIATION_FAMILIES_NOT_A_SEQUENCE")
        if not families:
            raise ValueError("LATENT_MEDIATION_NO_FAMILIES")

        relationships: list[list[list[Any]]] = []
        latent_means: list[list[Any]] = []
        mediator_measurements: list[list[Any]] = []
        mediator_measurement_error_variances: list[list[Any]] = []
        mediator_proxy_statuses: list[list[Any]] = []
        outcome_statuses: list[list[Any]] = []
        mediator_thresholds: list[list[Any]] = []
        outcome_thresholds: list[list[Any]] = []
        mediator_proxy_sensitivities: list[list[Any]] = []
        mediator_proxy_specificities: list[list[Any]] = []
        ascertainments: list[str] = []
        proband_indices: list[Any] = []
        mediator_designs: list[list[list[Any]]] = []
        outcome_designs: list[list[list[Any]]] = []

        for family in families:
            if not isinstance(family, Mapping):
                raise ValueError("LATENT_MEDIATION_FAMILY_NOT_A_MAPPING")
            relationship_rows = _sequence(
                _field(family, "relationship"),
                "LATENT_MEDIATION_RELATIONSHIP_NOT_A_SEQUENCE",
            )
            relationship = [
                _sequence(
                    row,
                    "LATENT_MEDIATION_RELATIONSHIP_ROW_NOT_A_SEQUENCE",
                    numeric=True,
                )
                for row in relationship_rows
            ]
            relationships.append(relationship)
            size = len(relationship)
            for name, into in (
                ("mediator_design", mediator_designs),
                ("outcome_design", outcome_designs),
            ):
                rows = _field(family, name, default=[])
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
                _person_vector(
                    family, "mediator_measurement", size, allow_none=True
                )
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
                _person_vector(
                    family, "mediator_threshold", size, scalar=True
                )
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
                _person_vector(
                    family, "mediator_proxy_sensitivity", size, scalar=True
                )
            )
            mediator_proxy_specificities.append(
                _person_vector(
                    family, "mediator_proxy_specificity", size, scalar=True
                )
            )
            ascertainment = _field(family, "ascertainment")
            if not isinstance(ascertainment, str):
                raise ValueError("LATENT_MEDIATION_ASCERTAINMENT_NOT_A_STRING")
            ascertainments.append(ascertainment)
            proband_index = family.get("proband_index")
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
        )

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

    def test_vertical(self) -> dict[str, Any]:
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
        return self._core.test_vertical()

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

    Parameters
    ----------
    relationship
        One family's relationship matrix — twice the kinship.
    a, b, c_prime, d, sigma_m2
        The truth to draw from. ``a`` is the mediator's loading on the
        inherited factor, ``b`` the path from mediator to outcome,
        ``c_prime`` the direct inherited effect on the outcome.
    families
        How many to draw.
    seed
        Fixed, so a campaign reruns exactly.
    measurement_error_variance
        The known error variance where the mediator is measured; ``None`` where
        it is not measured at all. At least one family must measure it
        somewhere or the mediator scale is not identified.

    Raises
    ------
    ValueError
        With a stable code where the design does not describe a family, where a
        proband is asked for and not named, or where the conditioning cannot be
        satisfied — a threshold so far out that a case essentially never occurs.
    """
    size = len(relationship)

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
            [float(v) for v in spread(mediator_threshold)],
            [float(v) for v in spread(outcome_threshold)],
            [None if v is None else float(v) for v in spread(measurement_error_variance)],
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
        )
    )
