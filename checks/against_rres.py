"""Compare Asterism's hard and posterior local-IBD builders with R ``rres``.

For each synthetic founder-genome-label draw, ``rres::fgl2relatedness`` scores
every ordered row pair.  The check compares every hard-matrix cell, including
one autozygous diagonal, and independently checks the weighted posterior mean.

Run with:

    uv run --locked --no-sync python checks/against_rres.py

The check installs nothing, prints one JSON report, and exits successfully only
when every hard, weighting, aggregation, and posterior comparison agrees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import asterism


TOLERANCE = 1e-12
ACCEPTED_RRES_VERSION = "1.1"
LINEAGE_DRAWS = np.array(
    [
        [[1, 2], [1, 3], [4, 4]],
        [[1, 2], [3, 4], [4, 4]],
    ],
    dtype=np.uint64,
)
DRAW_WEIGHTS = np.array([1.0, 3.0])
AUTOZYGOUS_ROW_INDEX = 2

MISSING_RSCRIPT_MESSAGE = (
    "rres comparison cannot run: Rscript is not on PATH. Install R and the "
    "R package 'rres', then rerun checks/against_rres.py."
)
MISSING_RRES_MESSAGE = (
    "rres comparison cannot run: required R package 'rres' is not installed "
    "for Rscript. Install rres in that R library, then rerun "
    "checks/against_rres.py."
)

R_SCRIPT = r'''
if (!requireNamespace("rres", quietly = TRUE)) {
    cat("ASTERISM_REQUIRED_R_PACKAGE_MISSING:rres\n", file = stderr())
    quit(save = "no", status = 86L)
}

args <- commandArgs(trailingOnly = TRUE)
directory <- args[[1]]
lineages <- read.csv(
    file.path(directory, "lineages.csv"), header = TRUE, check.names = FALSE
)
draw_ids <- sort(unique(lineages$draw))
row_indices <- sort(unique(lineages$row))
hard <- lapply(draw_ids, function(draw_id) {
    one <- lineages[lineages$draw == draw_id, ]
    one <- one[order(one$row), ]
    if (!identical(as.integer(one$row), as.integer(row_indices))) {
        stop("rows are incomplete or reordered within a lineage draw")
    }
    result <- matrix(0.0, nrow = nrow(one), ncol = nrow(one))
    for (i in seq_len(nrow(one))) {
        for (j in seq_len(nrow(one))) {
            result[i, j] <- rres::fgl2relatedness(
                one$lineage_1[i], one$lineage_2[i],
                one$lineage_1[j], one$lineage_2[j]
            )
        }
    }
    result
})
draw_weights <- scan(file.path(directory, "draw_weights.txt"), quiet = TRUE)
if (length(draw_weights) != length(hard)) {
    stop("draw-weight count does not match the lineage-draw count")
}
if (any(!is.finite(draw_weights)) || any(draw_weights < 0) ||
        sum(draw_weights) <= 0) {
    stop("draw weights must be finite, nonnegative and not all zero")
}
normalised_draw_weights <- draw_weights / sum(draw_weights)
posterior <- Reduce(
    "+",
    Map(
        function(value, weight) value * weight,
        hard,
        normalised_draw_weights
    )
)

matrix_json <- function(value) {
    paste0(
        "[",
        paste(sprintf("%.17g", as.vector(t(value))), collapse = ","),
        "]"
    )
}
payload <- sprintf(
    paste0(
        '{"r_version":"%s","package_version":"%s",',
        '"hard_matrices":[%s],"posterior_matrix":%s,',
        '"normalised_draw_weights":[%s]}'
    ),
    R.version.string,
    as.character(utils::packageVersion("rres")),
    paste(vapply(hard, matrix_json, character(1)), collapse = ","),
    matrix_json(posterior),
    paste(sprintf("%.17g", normalised_draw_weights), collapse = ",")
)
cat("ASTERISM_JSON:", payload, "\n", sep = "")
'''


def _external_json(finished: subprocess.CompletedProcess[str]) -> dict:
    """Return the one explicitly marked JSON payload from R."""
    for line in reversed(finished.stdout.splitlines()):
        if line.startswith("ASTERISM_JSON:"):
            try:
                value = json.loads(line.removeprefix("ASTERISM_JSON:"))
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"rres comparison returned malformed JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise SystemExit("rres comparison returned JSON that is not an object.")
            return value
    raise SystemExit(
        "rres comparison returned no marked JSON result. "
        f"R stdout:\n{finished.stdout}\nR stderr:\n{finished.stderr}"
    )


def run_rres() -> dict:
    """Run rres once with all synthetic inputs in one temporary directory."""
    if shutil.which("Rscript") is None:
        raise SystemExit(MISSING_RSCRIPT_MESSAGE)

    with tempfile.TemporaryDirectory(prefix="asterism-rres-") as temporary:
        directory = Path(temporary)
        rows = []
        for draw_index, draw in enumerate(LINEAGE_DRAWS):
            for row_index, pair in enumerate(draw):
                rows.append((draw_index, row_index, int(pair[0]), int(pair[1])))
        np.savetxt(
            directory / "lineages.csv",
            np.asarray(rows, dtype=np.uint64),
            delimiter=",",
            fmt="%d",
            header="draw,row,lineage_1,lineage_2",
            comments="",
        )
        np.savetxt(directory / "draw_weights.txt", DRAW_WEIGHTS)
        script = directory / "compare.R"
        script.write_text(R_SCRIPT)
        finished = subprocess.run(
            ["Rscript", "--vanilla", str(script), str(directory)],
            capture_output=True,
            text=True,
            check=False,
        )

    if "ASTERISM_REQUIRED_R_PACKAGE_MISSING:rres" in finished.stderr:
        raise SystemExit(MISSING_RRES_MESSAGE)
    if finished.returncode != 0:
        raise SystemExit(
            f"rres comparison failed in R (exit {finished.returncode}). "
            f"R stdout:\n{finished.stdout}\nR stderr:\n{finished.stderr}"
        )
    return _external_json(finished)


def make_report(external: dict) -> dict:
    """Compare every hard and posterior-mean cell with the package output."""
    required = {
        "r_version",
        "package_version",
        "hard_matrices",
        "posterior_matrix",
        "normalised_draw_weights",
    }
    missing = sorted(required.difference(external))
    if missing:
        raise SystemExit(f"rres comparison result is missing fields: {missing}")

    package_version = str(external["package_version"])
    if package_version != ACCEPTED_RRES_VERSION:
        raise SystemExit(
            f"rres comparison cannot run with R package 'rres' version "
            f"{package_version}; this adapter is pinned to "
            f"{ACCEPTED_RRES_VERSION}. Install that exact version and rerun "
            "checks/against_rres.py."
        )

    try:
        package_hard = np.asarray(external["hard_matrices"], dtype=np.float64)
        package_posterior = np.asarray(
            external["posterior_matrix"], dtype=np.float64
        )
        package_weights = np.asarray(
            external["normalised_draw_weights"], dtype=np.float64
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SystemExit("rres comparison returned malformed numeric values.") from error

    expected_shape = (
        LINEAGE_DRAWS.shape[0],
        LINEAGE_DRAWS.shape[1],
        LINEAGE_DRAWS.shape[1],
    )
    if package_hard.size == int(np.prod(expected_shape)):
        package_hard = package_hard.reshape(expected_shape)
    if package_hard.shape != expected_shape:
        raise SystemExit(
            "rres comparison returned the wrong matrix dimensions: "
            f"expected {expected_shape}, received {package_hard.shape}."
        )
    if not np.isfinite(package_hard).all():
        raise SystemExit("rres comparison returned a non-finite relationship value.")

    posterior_shape = expected_shape[1:]
    if package_posterior.size == int(np.prod(posterior_shape)):
        package_posterior = package_posterior.reshape(posterior_shape)
    if package_posterior.shape != posterior_shape:
        raise SystemExit(
            "rres comparison returned the wrong posterior-matrix dimensions: "
            f"expected {posterior_shape}, received {package_posterior.shape}."
        )
    if package_weights.shape != (LINEAGE_DRAWS.shape[0],):
        raise SystemExit(
            "rres comparison returned the wrong number of normalised draw "
            f"weights: expected {LINEAGE_DRAWS.shape[0]}, received "
            f"{package_weights.size}."
        )
    if (
        not np.isfinite(package_posterior).all()
        or not np.isfinite(package_weights).all()
        or np.any(package_weights < 0.0)
        or package_weights.sum() <= 0.0
    ):
        raise SystemExit(
            "rres comparison returned a non-finite posterior value or invalid "
            "normalised draw weight."
        )

    asterism_hard = np.stack(
        [np.asarray(asterism.local_ibd_matrix(draw)) for draw in LINEAGE_DRAWS]
    )
    asterism_posterior = np.asarray(
        asterism.posterior_local_ibd_matrix(
            LINEAGE_DRAWS,
            draw_weights=DRAW_WEIGHTS,
        )
    )
    if (
        asterism_hard.shape != expected_shape
        or asterism_posterior.shape != posterior_shape
        or not np.isfinite(asterism_hard).all()
        or not np.isfinite(asterism_posterior).all()
    ):
        raise SystemExit("Asterism returned an invalid local-IBD relationship matrix.")

    hard_differences = np.abs(asterism_hard - package_hard)
    hard_maximum = float(hard_differences.max())
    expected_weights = DRAW_WEIGHTS / DRAW_WEIGHTS.sum()
    weight_maximum = float(np.max(np.abs(package_weights - expected_weights)))
    posterior_from_hard = np.average(
        package_hard,
        axis=0,
        weights=package_weights,
    )
    aggregation_maximum = float(
        np.max(np.abs(package_posterior - posterior_from_hard))
    )
    posterior_differences = np.abs(asterism_posterior - package_posterior)
    posterior_maximum = float(posterior_differences.max())
    autozygous = package_hard[:, AUTOZYGOUS_ROW_INDEX, AUTOZYGOUS_ROW_INDEX]
    agreement = (
        hard_maximum < TOLERANCE
        and posterior_maximum < TOLERANCE
        and weight_maximum < TOLERANCE
        and aggregation_maximum < TOLERANCE
        and np.array_equal(autozygous, np.full(LINEAGE_DRAWS.shape[0], 2.0))
    )

    return {
        "status": "agreement" if agreement else "disagreement",
        "package": {"name": "rres", "version": package_version},
        "r_version": str(external["r_version"]),
        "rows": LINEAGE_DRAWS.shape[1],
        "lineage_draws": LINEAGE_DRAWS.shape[0],
        "tolerance": TOLERANCE,
        "comparisons": {
            "hard_local_ibd": {
                "external_function": "rres::fgl2relatedness",
                "draws_compared": LINEAGE_DRAWS.shape[0],
                "cells_compared": int(package_hard.size),
                "maximum_absolute_difference": hard_maximum,
                "maximum_absolute_difference_by_draw": [
                    float(difference.max()) for difference in hard_differences
                ],
                "autozygous_row_index": AUTOZYGOUS_ROW_INDEX,
                "autozygous_diagonal_values": [
                    float(value) for value in autozygous
                ],
            },
            "posterior_local_ibd": {
                "external_construction": (
                    "weighted mean of the per-draw rres relationship matrices"
                ),
                "submitted_draw_weights": [float(value) for value in DRAW_WEIGHTS],
                "normalised_draw_weights": [
                    float(value) for value in package_weights
                ],
                "expected_normalised_draw_weights": [
                    float(value) for value in expected_weights
                ],
                "draw_weight_maximum_absolute_difference": weight_maximum,
                "external_aggregation_maximum_absolute_difference": (
                    aggregation_maximum
                ),
                "cells_compared": int(package_posterior.size),
                "maximum_absolute_difference": posterior_maximum,
            },
        },
        "scope": (
            "This checks hard-lineage relationship values and their explicitly "
            "weighted posterior mean. It does not validate how founder lineages "
            "were inferred or calibrate a linkage scan."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reject arguments: this check has no output-file or runtime options."""
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    report = make_report(run_rres())
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "agreement" else 1


if __name__ == "__main__":
    sys.exit(main())
