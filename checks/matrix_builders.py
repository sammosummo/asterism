"""Independently calculate the four relationship-matrix constructions.

This check uses NumPy only for the reference calculations and calls Asterism
only for the values being compared. It compares the results of the equations in
the two implementations; it does not test a real variant mask or IBD
constructor.

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
LINEAGE_DRAWS = np.array(
    [
        [[1, 2], [1, 3], [4, 4]],
        [[1, 2], [3, 4], [4, 4]],
    ],
    dtype=np.int64,
)
DRAW_WEIGHTS = np.array([1.0, 3.0])


def reference_local_ibd(lineages: np.ndarray) -> np.ndarray:
    """Construct H H' / 2 through an explicit lineage-count matrix."""
    labels = sorted(set(int(value) for value in lineages.ravel()))
    column = {label: index for index, label in enumerate(labels)}
    counts = np.zeros((lineages.shape[0], len(labels)))
    for subject, pair in enumerate(lineages):
        for label in pair:
            counts[subject, column[int(label)]] += 1.0
    return counts @ counts.T / 2.0


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

    local_references = [reference_local_ibd(draw) for draw in LINEAGE_DRAWS]
    posterior_reference = np.average(
        np.stack(local_references), axis=0, weights=DRAW_WEIGHTS
    )

    built = {
        "gene_linear": asterism.gene_linear_matrix(
            GENOTYPES, variant_weights=VARIANT_WEIGHTS
        ),
        "gene_burden": asterism.gene_burden_matrix(
            GENOTYPES, variant_weights=VARIANT_WEIGHTS
        ),
        "local_ibd": asterism.local_ibd_matrix(LINEAGE_DRAWS[0]),
        "local_ibd_posterior_mean": asterism.posterior_local_ibd_matrix(
            LINEAGE_DRAWS, draw_weights=DRAW_WEIGHTS
        ),
    }
    references = {
        "gene_linear": linear_reference,
        "gene_burden": burden_reference,
        "local_ibd": local_references[0],
        "local_ibd_posterior_mean": posterior_reference,
    }
    differences = {
        name: compare(name, result, references[name])
        for name, result in built.items()
    }

    if built["local_ibd"][2, 2] != 2.0:
        raise SystemExit("local autozygosity was not retained")

    report = {
        "what": "relationship-matrix builders against independent NumPy calculations",
        "synthetic_subjects": GENOTYPES.shape[0],
        "synthetic_variants": GENOTYPES.shape[1],
        "synthetic_lineage_draws": LINEAGE_DRAWS.shape[0],
        "tolerance": TOLERANCE,
        "maximum_absolute_differences": differences,
        "local_autozygous_diagonal": float(built["local_ibd"][2, 2]),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
