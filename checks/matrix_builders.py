"""Independently calculate the two variant-set relationship matrices.

This check uses NumPy only for the reference calculations and calls Asterism
only for the values being compared. It compares the results of the equations in
the two implementations; it does not test a real variant mask.

Run with:

    uv run --locked --no-sync python checks/matrix_builders.py
"""

from __future__ import annotations

import json

import numpy as np

import asterism


TOLERANCE = 1e-12
GENOTYPES = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
VARIANT_WEIGHTS = np.array([1.0, 2.0])


def compare(name: str, observed: np.ndarray, expected: np.ndarray) -> float:
    difference = float(np.max(np.abs(observed - expected)))
    if difference > TOLERANCE:
        raise SystemExit(f"{name} differs by {difference:.3g}")
    if np.linalg.eigvalsh(observed).min() < -TOLERANCE:
        raise SystemExit(f"{name} is not positive semidefinite")
    return difference


def main() -> int:
    # An implementation independent of Rust's matrix multiplication route.
    linear_reference = np.zeros((3, 3))
    burden = np.zeros(3)
    for subject in range(3):
        for variant in range(2):
            burden[subject] += GENOTYPES[subject, variant] * VARIANT_WEIGHTS[variant]
        for other in range(3):
            for variant in range(2):
                linear_reference[subject, other] += (
                    GENOTYPES[subject, variant]
                    * GENOTYPES[other, variant]
                    * VARIANT_WEIGHTS[variant] ** 2
                )
    burden_reference = np.outer(burden, burden)

    built = {
        "gene_linear": asterism.gene_linear_matrix(
            GENOTYPES, variant_weights=VARIANT_WEIGHTS
        ),
        "gene_burden": asterism.gene_burden_matrix(
            GENOTYPES, variant_weights=VARIANT_WEIGHTS
        ),
    }
    references = {
        "gene_linear": linear_reference,
        "gene_burden": burden_reference,
    }
    differences = {
        name: compare(name, result, references[name])
        for name, result in built.items()
    }

    report = {
        "what": "variant-set matrix builders against independent NumPy calculations",
        "synthetic_subjects": GENOTYPES.shape[0],
        "synthetic_variants": GENOTYPES.shape[1],
        "tolerance": TOLERANCE,
        "maximum_absolute_differences": differences,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
