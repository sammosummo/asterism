"""CLI behaviour for the optional R package comparisons."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SKAT_CHECK = PROJECT / "checks" / "against_skat_builders.py"
RRES_CHECK = PROJECT / "checks" / "against_rres.py"

FAKE_RSCRIPT = r'''#!/usr/bin/env python3
import json
import os
import sys

mode = os.environ["ASTERISM_FAKE_R_MODE"]
package = "SKAT" if mode.startswith("skat") else "rres"
if mode.endswith("missing"):
    print(f"ASTERISM_REQUIRED_R_PACKAGE_MISSING:{package}", file=sys.stderr)
    raise SystemExit(86)
if mode.endswith("malformed"):
    print("ASTERISM_JSON:{not-json")
    raise SystemExit(0)

if mode.startswith("skat"):
    result = {
        "r_version": "R version fake-4.5.2",
        "package_version": "2.2.5",
        "linear_q": 12.234623015873012,
        "burden_q": 0.9171626984126969,
        "s2": 1.75,
        "residuals": [
            -1.9166666666666665,
            -0.41666666666666663,
            1.0833333333333335,
            2.0833333333333335,
            -1.4166666666666665,
            0.5833333333333334,
        ],
        "linear_markers": 3,
        "burden_markers": 3,
    }
    if mode == "skat_disagreement":
        result["linear_q"] += 0.25
    if mode == "skat_wrong_version":
        result["package_version"] = "9.9.9"
    if mode == "skat_wrong_markers":
        result["linear_markers"] = 2
    if mode == "skat_nonfinite":
        result["burden_q"] = float("nan")
    if mode == "skat_wrong_residual_shape":
        result["residuals"].pop()
else:
    result = {
        "r_version": "R version fake-4.5.2",
        "package_version": "1.1",
        "hard_matrices": [
            [1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0],
        ],
        "posterior_matrix": [
            1.0, 0.125, 0.0, 0.125, 1.0, 0.75, 0.0, 0.75, 2.0
        ],
        "normalised_draw_weights": [0.25, 0.75],
    }
    if mode == "rres_disagreement":
        result["hard_matrices"][0][1] = 0.25
    if mode == "rres_wrong_version":
        result["package_version"] = "9.9.9"
    if mode == "rres_wrong_dimensions":
        result["hard_matrices"] = [[1.0, 0.5]]
    if mode == "rres_nonfinite":
        result["hard_matrices"][0][0] = float("nan")
    if mode == "rres_posterior_disagreement":
        result["posterior_matrix"][1] = 0.375

print("ASTERISM_JSON:" + json.dumps(result, separators=(",", ":")))
'''


@pytest.fixture
def fake_rscript(tmp_path: Path) -> Path:
    executable = tmp_path / "Rscript"
    executable.write_text(FAKE_RSCRIPT)
    executable.chmod(0o755)
    return executable


def run_check(
    script: Path,
    tmp_path: Path,
    fake_rscript: Path,
    mode: str,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_rscript.parent}{os.pathsep}{environment['PATH']}"
    environment["ASTERISM_FAKE_R_MODE"] = mode
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("script", "mode", "package", "rerun"),
    [
        (SKAT_CHECK, "skat_missing", "SKAT", "checks/against_skat_builders.py"),
        (RRES_CHECK, "rres_missing", "rres", "checks/against_rres.py"),
    ],
)
def test_missing_r_package_fails_clearly(
    script: Path,
    mode: str,
    package: str,
    rerun: str,
    tmp_path: Path,
    fake_rscript: Path,
):
    finished = run_check(script, tmp_path, fake_rscript, mode)

    assert finished.returncode != 0
    assert f"required R package '{package}' is not installed" in finished.stderr
    assert rerun in finished.stderr
    assert not finished.stdout


@pytest.mark.parametrize(
    ("script", "name"),
    [(SKAT_CHECK, "SKAT"), (RRES_CHECK, "rres")],
)
def test_missing_rscript_fails_clearly(script: Path, name: str, tmp_path: Path):
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    environment = os.environ.copy()
    environment["PATH"] = str(empty_path)

    finished = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert finished.returncode != 0
    assert f"{name} comparison cannot run: Rscript is not on PATH" in finished.stderr


def test_skat_check_compares_weighted_linear_and_burden_scores(
    tmp_path: Path,
    fake_rscript: Path,
):
    finished = run_check(SKAT_CHECK, tmp_path, fake_rscript, "skat_agreement")

    assert finished.returncode == 0, finished.stderr
    report = json.loads(finished.stdout)
    assert report["status"] == "agreement"
    assert report["package"] == {"name": "SKAT", "version": "2.2.5"}
    linear = report["comparisons"]["gene_linear"]
    burden = report["comparisons"]["gene_burden"]
    assert linear["skat_r_corr"] == 0.0
    assert burden["skat_r_corr"] == 1.0
    assert linear["matrix_export_compared"] is False
    assert burden["matrix_export_compared"] is False
    assert linear["absolute_difference"] < report["tolerance"]
    assert burden["absolute_difference"] < report["tolerance"]
    assert linear["asterism_score_q"] != burden["asterism_score_q"]
    assert "not full-matrix parity" in report["scope"]


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [
        ("skat_disagreement", "disagreement"),
        ("skat_wrong_markers", "disagreement"),
    ],
)
def test_skat_numerical_or_marker_count_disagreement_is_a_failure(
    mode: str,
    expected_status: str,
    tmp_path: Path,
    fake_rscript: Path,
):
    finished = run_check(SKAT_CHECK, tmp_path, fake_rscript, mode)

    assert finished.returncode == 1
    report = json.loads(finished.stdout)
    assert report["status"] == expected_status
    if mode == "skat_wrong_markers":
        assert report["comparisons"]["gene_linear"]["markers_reported_by_skat"] == 2


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("skat_malformed", "returned malformed JSON"),
        ("skat_nonfinite", "non-finite score statistic"),
        ("skat_wrong_residual_shape", "wrong number of null residuals"),
    ],
)
def test_skat_rejects_malformed_nonfinite_or_wrong_shape_results(
    mode: str,
    message: str,
    tmp_path: Path,
    fake_rscript: Path,
):
    finished = run_check(SKAT_CHECK, tmp_path, fake_rscript, mode)

    assert finished.returncode != 0
    assert message in finished.stderr
    assert not finished.stdout


@pytest.mark.parametrize(
    ("script", "mode", "package", "version"),
    [
        (SKAT_CHECK, "skat_wrong_version", "SKAT", "2.2.5"),
        (RRES_CHECK, "rres_wrong_version", "rres", "1.1"),
    ],
)
def test_unpinned_r_package_version_fails_clearly(
    script: Path,
    mode: str,
    package: str,
    version: str,
    tmp_path: Path,
    fake_rscript: Path,
):
    finished = run_check(script, tmp_path, fake_rscript, mode)

    assert finished.returncode != 0
    assert f"R package '{package}' version 9.9.9" in finished.stderr
    assert f"adapter is pinned to {version}" in finished.stderr


def test_rres_check_compares_every_hard_cell_and_weighted_posterior(
    tmp_path: Path,
    fake_rscript: Path,
):
    finished = run_check(RRES_CHECK, tmp_path, fake_rscript, "rres_agreement")

    assert finished.returncode == 0, finished.stderr
    report = json.loads(finished.stdout)
    assert report["status"] == "agreement"
    assert report["package"] == {"name": "rres", "version": "1.1"}
    hard = report["comparisons"]["hard_local_ibd"]
    posterior = report["comparisons"]["posterior_local_ibd"]
    assert hard["draws_compared"] == 2
    assert hard["cells_compared"] == 18
    assert hard["autozygous_diagonal_values"] == [2.0, 2.0]
    assert hard["maximum_absolute_difference"] < report["tolerance"]
    assert posterior["normalised_draw_weights"] == [0.25, 0.75]
    assert posterior["cells_compared"] == 9
    assert posterior["maximum_absolute_difference"] < report["tolerance"]


def test_rres_hard_or_posterior_disagreement_is_a_failure(
    tmp_path: Path,
    fake_rscript: Path,
):
    hard_finished = run_check(
        RRES_CHECK, tmp_path, fake_rscript, "rres_disagreement"
    )
    posterior_finished = run_check(
        RRES_CHECK, tmp_path, fake_rscript, "rres_posterior_disagreement"
    )

    assert hard_finished.returncode == 1
    hard_report = json.loads(hard_finished.stdout)
    assert hard_report["status"] == "disagreement"
    assert hard_report["comparisons"]["hard_local_ibd"][
        "maximum_absolute_difference"
    ] == 0.25
    assert posterior_finished.returncode == 1
    posterior_report = json.loads(posterior_finished.stdout)
    assert posterior_report["status"] == "disagreement"
    assert posterior_report["comparisons"]["hard_local_ibd"][
        "maximum_absolute_difference"
    ] == 0.0
    assert posterior_report["comparisons"]["posterior_local_ibd"][
        "maximum_absolute_difference"
    ] == 0.25


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("rres_malformed", "returned malformed JSON"),
        ("rres_wrong_dimensions", "wrong matrix dimensions"),
        ("rres_nonfinite", "non-finite relationship value"),
    ],
)
def test_rres_rejects_malformed_nonfinite_or_wrong_shape_results(
    mode: str,
    message: str,
    tmp_path: Path,
    fake_rscript: Path,
):
    finished = run_check(RRES_CHECK, tmp_path, fake_rscript, mode)

    assert finished.returncode != 0
    assert message in finished.stderr
    assert not finished.stdout
