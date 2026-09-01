"""Immutable build and subject-order identity on public fit records."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import asterism
import asterism.analysis as analysis_implementation
import asterism.models as model_implementation
import numpy as np
import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the exact release contract and dependency locks under test."""


def test_public_build_identity_binds_release_and_dependency_manifests() -> None:
    """Build identity committed to the exact release and dependency inputs."""
    identity: dict[str, Any] = asterism.build_identity()
    """Read commitments calculated from bytes embedded in the extension."""

    for field, filename in (
        ("release_manifest_sha256", "release.toml"),
        ("cargo_lock_sha256", "Cargo.lock"),
        ("uv_lock_sha256", "uv.lock"),
    ):
        expected: str = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        """Hashed the corresponding checkout input independently of the package."""

        assert identity[field] == expected
    """Required every embedded commitment to match the authoritative checkout bytes."""


def test_installed_extension_digest_follows_the_loaded_module_not_a_decoy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hash the binary the import system selected, not a sibling lookalike."""
    loaded: Path = tmp_path / "_core.loaded.so"
    """Represented the exact native module already imported by this process."""

    loaded.write_bytes(b"loaded numerical extension")
    decoy: Path = tmp_path / "_core.decoy.so"
    """Placed another extension-shaped file beside the loaded module."""

    decoy.write_bytes(b"different bytes that must not be hashed")
    monkeypatch.setattr(
        analysis_implementation._core,
        "__spec__",
        SimpleNamespace(origin=str(loaded)),
    )
    """Made the import specification identify the selected test binary exactly."""

    expected: str = hashlib.sha256(loaded.read_bytes()).hexdigest()
    """Calculated the selected binary's digest independently of the helper."""

    assert asterism.installed_extension_sha256() == expected
    assert (
        asterism.installed_extension_sha256()
        != hashlib.sha256(decoy.read_bytes()).hexdigest()
    )


def test_prepared_fit_echoes_build_and_subject_order_identity() -> None:
    """A fit remained attributable when separated from its analysis receipt."""
    order_sha256: str = asterism.subject_order_commitment(
        ["participant-a", "participant-b", "participant-c", "participant-d"]
    )
    """Committed to the row order without giving identifiers to the fit record."""
    relationship: np.ndarray = np.eye(4)
    """Constructed a small positive-definite relationship matrix."""
    design: np.ndarray = np.ones((4, 1))
    """Constructed an intercept-only fixed-effect design in the same order."""
    response: np.ndarray = np.array([-1.0, -0.2, 0.4, 1.3])
    """Constructed a nonconstant quantitative response in the same order."""

    record: dict[str, Any] = dict(
        asterism.prepare(
            design,
            relationship,
            subject_order_sha256=order_sha256,
        ).fit(response)
    )
    """Fitted through the documented prepared-model interface."""

    assert record["build"] == asterism.build_identity()
    assert record["subject_order_sha256"] == order_sha256
    assert "participant-a" not in repr(record)
    assert "participant-b" not in repr(record)


def test_prepare_refuses_an_invalid_subject_order_commitment() -> None:
    """A malformed commitment never entered a prepared model."""
    with pytest.raises(ValueError, match="PREPARE_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.prepare(
            np.ones((4, 1)),
            np.eye(4),
            subject_order_sha256="not-a-sha256",
        )
    """Required validation at the model-construction boundary."""


def test_every_supported_fit_route_carries_build_and_subject_order_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every supported 0.1 numerical fit remained independently attributable."""
    commitment: str = asterism.subject_order_commitment(["synthetic-a", "synthetic-b"])
    """Committed to the exact two-row synthetic order used by every route."""

    relationship: np.ndarray = np.eye(2)
    """Constructed a valid two-person relationship matrix for public constructors."""

    design: np.ndarray = np.ones((2, 1))
    """Constructed the matching intercept-only fixed-effect design."""

    response: np.ndarray = np.array([-0.5, 0.5])
    """Constructed a nonconstant response accepted by every fitted wrapper."""

    component_core: Mock = Mock(
        return_value=(
            [0.4, 0.6],
            [0.4, 0.6],
            1.0,
            -1.0,
            0.0,
            True,
            [0.0],
            [0.1],
            0,
            "converged",
            False,
        )
    )
    """Stubbed only the compiled component calculation beneath the public wrapper."""
    monkeypatch.setattr(model_implementation._core, "component_fit", component_core)

    bivariate_core: Mock = Mock(
        return_value=([1.0, 1.0, 0.4, 0.5, 0.2, 0.1, 0.15], -2.0, 0.0, True)
    )
    """Stubbed only the compiled bivariate calculation beneath the public wrapper."""
    monkeypatch.setattr(model_implementation._core, "bivariate_fit", bivariate_core)

    spatial_core: Mock = Mock(
        return_value=(
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
            1.0,
            0.4,
            1.7,
            -3.0,
            0.0,
            True,
            False,
            [0.0],
            [0.1],
        )
    )
    """Stubbed only the compiled spatial calculation beneath the public wrapper."""
    monkeypatch.setattr(model_implementation._core, "spatial_fit", spatial_core)

    gxe_core: Mock = Mock(
        return_value=(
            [0.0] * 6,
            -4.0,
            True,
            False,
            0.0,
            [0.0],
            [0.1],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.4, 0.4, 0.4],
            [1.0, 0.5, 0.2, 0.5, 1.0, 0.5, 0.2, 0.5, 1.0],
        )
    )
    """Stubbed only the compiled continuous-GxE calculation beneath its wrapper."""
    monkeypatch.setattr(model_implementation._core, "gxe_fit", gxe_core)

    discrete_core: Mock = Mock(
        return_value=(
            [0.4, 0.5],
            [0.6, 0.5],
            [0.4, 0.5],
            0.25,
            [0.0],
            [0.1],
            -5.0,
            True,
            False,
            0.0,
            [1, 1],
            [0.0, 1.0],
        )
    )
    """Stubbed only the compiled discrete-GxE calculation beneath its wrapper."""
    monkeypatch.setattr(model_implementation._core, "discrete_gxe_fit", discrete_core)

    liability_core: Mock = Mock(
        return_value=(0.4, [0.0], -6.0, True, False, 0.0, 0.5, 2)
    )
    """Stubbed only the compiled liability calculation beneath the public wrapper."""
    monkeypatch.setattr(model_implementation._core, "liability_fit", liability_core)

    tobit_core: Mock = Mock(return_value=(0.4, 1.0, [0.0], -7.0, True, 0.0, 0.5, 2))
    """Stubbed only the compiled Tobit calculation beneath the public wrapper."""
    monkeypatch.setattr(model_implementation._core, "tobit_fit", tobit_core)

    mixed_core: Mock = Mock(
        return_value=(
            [0.4, 0.5],
            [1.0, 1.2],
            0.3,
            0.1,
            [0.0],
            [0.0],
            -8.0,
            True,
            0.0,
            2,
        )
    )
    """Stubbed only the compiled mixed-trait calculation beneath the public wrapper."""
    monkeypatch.setattr(model_implementation._core, "mixed_bivariate_fit", mixed_core)

    component: asterism.ComponentModel = asterism.ComponentModel(
        [relationship], design, subject_order_sha256=commitment
    )
    """Built the public several-component route with the shared commitment."""

    bivariate: asterism.BivariateModel = asterism.BivariateModel(
        relationship,
        [[True, True], [True, True]],
        np.tile(np.eye(2), (2, 1)),
        subject_order_sha256=commitment,
    )
    """Built the public two-trait route with the shared commitment."""

    spatial: asterism.SpatialModel = asterism.SpatialModel(
        [relationship],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        design,
        subject_order_sha256=commitment,
    )
    """Built the public spatial route with the shared commitment."""

    continuous_gxe: asterism.GxeModel = asterism.GxeModel(
        relationship,
        [-1.0, 1.0],
        design,
        subject_order_sha256=commitment,
    )
    """Built the public continuous-GxE route with the shared commitment."""

    discrete_gxe: asterism.DiscreteGxeModel = asterism.DiscreteGxeModel(
        relationship,
        [0.0, 1.0],
        design,
        subject_order_sha256=commitment,
    )
    """Built the public discrete-GxE route with the shared commitment."""

    liability: asterism.LiabilityModel = asterism.LiabilityModel(
        relationship,
        [0.0, 1.0],
        design,
        subject_order_sha256=commitment,
    )
    """Built the public binary-liability route with the shared commitment."""

    first_trait: dict[str, Any] = {
        "kind": "binary",
        "value": [np.nan, np.nan],
        "censoring": [2, 1],
        "limit": [0.0, 0.0],
    }
    """Specified the binary member of the supported mixed-trait pairing."""

    second_trait: dict[str, Any] = {
        "kind": "censored",
        "value": [0.0, np.nan],
        "censoring": [0, 1],
        "limit": [1.0, 1.0],
    }
    """Specified the censored member of the supported mixed-trait pairing."""

    records: list[dict[str, Any]] = [
        component.fit(response),
        bivariate.fit(np.array([-0.5, 0.1, 0.5, 0.2])),
        spatial.fit(response),
        continuous_gxe.fit(response),
        discrete_gxe.fit(response),
        liability.fit(),
        asterism.tobit_fit(
            relationship,
            [0.0, np.nan],
            [0, 1],
            [1.0, 1.0],
            design,
            subject_order_sha256=commitment,
        ),
        asterism.mixed_bivariate_fit(
            relationship,
            first_trait,
            second_trait,
            design,
            subject_order_sha256=commitment,
        ),
    ]
    """Exercised every non-prepared supported fit through its public Python route."""

    expected_build: dict[str, Any] = asterism.build_identity()
    """Read the immutable producer identity independently of every fit wrapper."""

    for record in records:
        assert record["build"] == expected_build
        assert record["subject_order_sha256"] == commitment
    """Required uniform identity fields across all supported fit-record shapes."""


def test_subject_order_commitment_is_keyword_only_on_every_supported_route() -> None:
    """The additive identity argument could not reinterpret an existing call."""
    routes: tuple[Callable[..., object], ...] = (
        asterism.prepare,
        asterism.ComponentModel,
        asterism.BivariateModel,
        asterism.SpatialModel,
        asterism.GxeModel,
        asterism.DiscreteGxeModel,
        asterism.LiabilityModel,
        asterism.tobit_fit,
        asterism.mixed_bivariate_fit,
    )
    """Listed exactly the nine supported 0.1 numerical fit entry points."""

    for route in routes:
        parameter: inspect.Parameter = inspect.signature(route).parameters[
            "subject_order_sha256"
        ]
        """Read the additive argument without depending on parameter position."""

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    """Protected every existing positional call from reinterpretation."""


def test_nonprepared_routes_refuse_invalid_commitments_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed commitments stopped before any supported numerical calculation."""
    relationship: np.ndarray = np.eye(2)
    """Constructed a small relationship matrix for every public constructor."""

    design: np.ndarray = np.ones((2, 1))
    """Constructed a matching fixed-effect design."""

    invalid: str = "A" * 64
    """Used the correct length with forbidden uppercase hexadecimal characters."""

    with pytest.raises(ValueError, match="FIT_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.ComponentModel([relationship], design, subject_order_sha256=invalid)
    with pytest.raises(ValueError, match="FIT_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.BivariateModel(
            relationship,
            [[True, True], [True, True]],
            np.tile(np.eye(2), (2, 1)),
            subject_order_sha256=invalid,
        )
    with pytest.raises(ValueError, match="FIT_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.SpatialModel(
            [relationship], relationship, design, subject_order_sha256=invalid
        )
    with pytest.raises(ValueError, match="FIT_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.GxeModel(
            relationship, [-1.0, 1.0], design, subject_order_sha256=invalid
        )
    with pytest.raises(ValueError, match="FIT_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.DiscreteGxeModel(
            relationship, [0.0, 1.0], design, subject_order_sha256=invalid
        )
    with pytest.raises(ValueError, match="FIT_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.LiabilityModel(
            relationship, [0.0, 1.0], design, subject_order_sha256=invalid
        )
    """Required every class-based route to reject at model construction."""

    tobit_core: Mock = Mock(side_effect=AssertionError("Tobit fitting was reached"))
    """Guarded the compiled Tobit calculation against accidental entry."""
    monkeypatch.setattr(model_implementation._core, "tobit_fit", tobit_core)

    with pytest.raises(ValueError, match="FIT_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.tobit_fit(
            relationship,
            [0.0, np.nan],
            [0, 1],
            [1.0, 1.0],
            design,
            subject_order_sha256=invalid,
        )
    assert tobit_core.call_count == 0
    """Proved commitment validation preceded the compiled Tobit fit."""

    mixed_core: Mock = Mock(
        side_effect=AssertionError("mixed-bivariate fitting was reached")
    )
    """Guarded the compiled mixed-trait calculation against accidental entry."""
    monkeypatch.setattr(model_implementation._core, "mixed_bivariate_fit", mixed_core)

    first_trait: dict[str, Any] = {
        "kind": "binary",
        "value": [np.nan, np.nan],
        "censoring": [2, 1],
        "limit": [0.0, 0.0],
    }
    """Specified a valid binary trait so only the commitment could fail."""

    second_trait: dict[str, Any] = {
        "kind": "censored",
        "value": [0.0, np.nan],
        "censoring": [0, 1],
        "limit": [1.0, 1.0],
    }
    """Specified a valid censored trait so only the commitment could fail."""

    with pytest.raises(ValueError, match="FIT_SUBJECT_ORDER_SHA256_INVALID"):
        asterism.mixed_bivariate_fit(
            relationship,
            first_trait,
            second_trait,
            design,
            subject_order_sha256=invalid,
        )
    assert mixed_core.call_count == 0
    """Proved commitment validation preceded the compiled mixed-trait fit."""
