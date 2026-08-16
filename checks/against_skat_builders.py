"""Compare Asterism's two gene builders with public SKAT score statistics.

The same complete synthetic dosage matrix and column multipliers are submitted
to Asterism and SKAT.  SKAT does not expose its subject-by-subject kernel here,
so this check compares the public score ``Q`` with ``r' K r / (2 s2)`` using
SKAT's own null residuals and scale.  Score agreement is not full-matrix parity.

Run with:

    uv run --locked --no-sync python checks/against_skat_builders.py

The check installs nothing, prints one JSON report, and exits successfully only
when both score comparisons and both marker-count checks agree.
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


TOLERANCE = 1e-10
ACCEPTED_SKAT_VERSION = "2.2.5"
GENOTYPES = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
    ]
)
VARIANT_WEIGHTS = np.array([0.5, 1.25, 2.0])
PHENOTYPE = np.array([-1.0, 0.5, 2.0, 3.0, -0.5, 1.5])

MISSING_RSCRIPT_MESSAGE = (
    "SKAT comparison cannot run: Rscript is not on PATH. Install R and the "
    "R package 'SKAT', then rerun checks/against_skat_builders.py."
)
MISSING_SKAT_MESSAGE = (
    "SKAT comparison cannot run: required R package 'SKAT' is not installed "
    "for Rscript. Install SKAT in that R library, then rerun "
    "checks/against_skat_builders.py."
)

R_SCRIPT = r'''
if (!requireNamespace("SKAT", quietly = TRUE)) {
    cat("ASTERISM_REQUIRED_R_PACKAGE_MISSING:SKAT\n", file = stderr())
    quit(save = "no", status = 86L)
}

args <- commandArgs(trailingOnly = TRUE)
directory <- args[[1]]
genotypes <- as.matrix(read.csv(
    file.path(directory, "genotypes.csv"), header = FALSE, check.names = FALSE
))
storage.mode(genotypes) <- "double"
weights <- scan(file.path(directory, "weights.txt"), quiet = TRUE)
phenotype <- scan(file.path(directory, "phenotype.txt"), quiet = TRUE)

null <- SKAT::SKAT_Null_Model(
    phenotype ~ 1, out_type = "C", n.Resampling = 0, Adjustment = FALSE
)
linear <- SKAT::SKAT(
    genotypes, null, kernel = "linear.weighted", method = "liu",
    weights = weights, r.corr = 0, is_check_genotype = FALSE,
    is_dosage = FALSE
)
burden <- SKAT::SKAT(
    genotypes, null, kernel = "linear.weighted", method = "liu",
    weights = weights, r.corr = 1, is_check_genotype = FALSE,
    is_dosage = FALSE
)

payload <- sprintf(
    paste0(
        '{"r_version":"%s","package_version":"%s",',
        '"linear_q":%.17g,"burden_q":%.17g,"s2":%.17g,',
        '"residuals":[%s],"linear_markers":%d,"burden_markers":%d}'
    ),
    R.version.string,
    as.character(utils::packageVersion("SKAT")),
    as.numeric(linear$Q),
    as.numeric(burden$Q),
    as.numeric(null$s2),
    paste(sprintf("%.17g", as.numeric(null$res)), collapse = ","),
    as.integer(linear$param$n.marker.test),
    as.integer(burden$param$n.marker.test)
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
                    f"SKAT comparison returned malformed JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise SystemExit("SKAT comparison returned JSON that is not an object.")
            return value
    raise SystemExit(
        "SKAT comparison returned no marked JSON result. "
        f"R stdout:\n{finished.stdout}\nR stderr:\n{finished.stderr}"
    )


def run_skat() -> dict:
    """Run SKAT once with all synthetic inputs in one temporary directory."""
    if shutil.which("Rscript") is None:
        raise SystemExit(MISSING_RSCRIPT_MESSAGE)

    with tempfile.TemporaryDirectory(prefix="asterism-skat-builders-") as temporary:
        directory = Path(temporary)
        np.savetxt(directory / "genotypes.csv", GENOTYPES, delimiter=",")
        np.savetxt(directory / "weights.txt", VARIANT_WEIGHTS)
        np.savetxt(directory / "phenotype.txt", PHENOTYPE)
        script = directory / "compare.R"
        script.write_text(R_SCRIPT)
        finished = subprocess.run(
            ["Rscript", "--vanilla", str(script), str(directory)],
            capture_output=True,
            text=True,
            check=False,
        )

    if "ASTERISM_REQUIRED_R_PACKAGE_MISSING:SKAT" in finished.stderr:
        raise SystemExit(MISSING_SKAT_MESSAGE)
    if finished.returncode != 0:
        raise SystemExit(
            f"SKAT comparison failed in R (exit {finished.returncode}). "
            f"R stdout:\n{finished.stdout}\nR stderr:\n{finished.stderr}"
        )
    return _external_json(finished)


def _marker_count(value: object, name: str) -> int:
    """Return one exact nonnegative integer marker count."""
    if isinstance(value, bool):
        raise SystemExit(f"SKAT comparison returned an invalid {name} marker count.")
    try:
        numeric = float(value)
        count = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SystemExit(
            f"SKAT comparison returned an invalid {name} marker count."
        ) from error
    if not np.isfinite(numeric) or numeric != count or count < 0:
        raise SystemExit(f"SKAT comparison returned an invalid {name} marker count.")
    return count


def make_report(external: dict) -> dict:
    """Compare SKAT's public Q values with Asterism-built matrices."""
    required = {
        "r_version",
        "package_version",
        "linear_q",
        "burden_q",
        "s2",
        "residuals",
        "linear_markers",
        "burden_markers",
    }
    missing = sorted(required.difference(external))
    if missing:
        raise SystemExit(f"SKAT comparison result is missing fields: {missing}")

    package_version = str(external["package_version"])
    if package_version != ACCEPTED_SKAT_VERSION:
        raise SystemExit(
            f"SKAT comparison cannot run with R package 'SKAT' version "
            f"{package_version}; this adapter is pinned to "
            f"{ACCEPTED_SKAT_VERSION}. Install that exact version and rerun "
            "checks/against_skat_builders.py."
        )

    try:
        residuals = np.asarray(external["residuals"], dtype=np.float64)
        scale = float(external["s2"])
        theirs_linear = float(external["linear_q"])
        theirs_burden = float(external["burden_q"])
    except (TypeError, ValueError, OverflowError) as error:
        raise SystemExit("SKAT comparison returned malformed numeric values.") from error
    if residuals.shape != (GENOTYPES.shape[0],):
        raise SystemExit(
            "SKAT comparison returned the wrong number of null residuals: "
            f"expected {GENOTYPES.shape[0]}, received {residuals.size}."
        )
    if not np.isfinite(residuals).all() or not np.isfinite(scale) or scale <= 0.0:
        raise SystemExit("SKAT comparison returned non-finite residuals or invalid s2.")
    if not np.isfinite([theirs_linear, theirs_burden]).all():
        raise SystemExit("SKAT comparison returned a non-finite score statistic.")

    linear_markers = _marker_count(external["linear_markers"], "linear")
    burden_markers = _marker_count(external["burden_markers"], "burden")
    built_linear = np.asarray(
        asterism.gene_linear_matrix(
            GENOTYPES,
            variant_weights=VARIANT_WEIGHTS,
        ),
        dtype=np.float64,
    )
    built_burden = np.asarray(
        asterism.gene_burden_matrix(
            GENOTYPES,
            variant_weights=VARIANT_WEIGHTS,
        ),
        dtype=np.float64,
    )
    expected_shape = (GENOTYPES.shape[0], GENOTYPES.shape[0])
    if (
        built_linear.shape != expected_shape
        or built_burden.shape != expected_shape
        or not np.isfinite(built_linear).all()
        or not np.isfinite(built_burden).all()
    ):
        raise SystemExit("Asterism returned an invalid gene relationship matrix.")

    denominator = 2.0 * scale
    ours_linear = float(residuals @ built_linear @ residuals / denominator)
    ours_burden = float(residuals @ built_burden @ residuals / denominator)
    linear_difference = abs(ours_linear - theirs_linear)
    burden_difference = abs(ours_burden - theirs_burden)
    agreement = (
        linear_markers == GENOTYPES.shape[1]
        and burden_markers == GENOTYPES.shape[1]
        and linear_difference < TOLERANCE
        and burden_difference < TOLERANCE
    )

    return {
        "status": "agreement" if agreement else "disagreement",
        "package": {"name": "SKAT", "version": package_version},
        "r_version": str(external["r_version"]),
        "subjects": GENOTYPES.shape[0],
        "variants": GENOTYPES.shape[1],
        "tolerance": TOLERANCE,
        "comparisons": {
            "gene_linear": {
                "construction": "K = (G diag(w)) (G diag(w))'",
                "quantity_compared": "Q = r' K r / (2 s2)",
                "skat_kernel": "linear.weighted",
                "skat_r_corr": 0.0,
                "matrix_export_compared": False,
                "asterism_score_q": ours_linear,
                "skat_score_q": theirs_linear,
                "absolute_difference": linear_difference,
                "markers_submitted": GENOTYPES.shape[1],
                "markers_reported_by_skat": linear_markers,
            },
            "gene_burden": {
                "construction": "K = (G w) (G w)'",
                "quantity_compared": "Q = r' K r / (2 s2)",
                "skat_kernel": "linear.weighted",
                "skat_r_corr": 1.0,
                "matrix_export_compared": False,
                "asterism_score_q": ours_burden,
                "skat_score_q": theirs_burden,
                "absolute_difference": burden_difference,
                "markers_submitted": GENOTYPES.shape[1],
                "markers_reported_by_skat": burden_markers,
            },
        },
        "scope": (
            "This is score parity for the stated multipliers, not full-matrix "
            "parity or inferential calibration. An intercept-only null also makes "
            "its residual orthogonal to a constant column, so this comparison "
            "cannot distinguish raw from column-centred G."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reject arguments: this check has no output-file or runtime options."""
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    report = make_report(run_skat())
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "agreement" else 1


if __name__ == "__main__":
    sys.exit(main())
