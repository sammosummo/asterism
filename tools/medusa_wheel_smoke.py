"""Participant-free import-and-fit smoke for the saved portable wheel.

Run this with the interpreter of an environment where the manylinux wheel has
been installed with `--no-deps`, never against a source build. It prints one
JSON record of host facts and one fitted result, and touches no participant
data of any kind: the pedigree and the trait are both generated from a fixed
seed inside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import asterism
import numpy as np

parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Run the participant-free smoke for one installed manylinux wheel."
)
"""Configured the public target-host smoke command interface."""

parser.add_argument("--wheel", type=Path, required=True)
arguments: argparse.Namespace = parser.parse_args()
"""Required the saved artifact whose exact bytes the external result attests."""

if not arguments.wheel.is_file() or arguments.wheel.suffix != ".whl":
    parser.error("--wheel must name the installed saved wheel")
"""Refused a missing or non-wheel artifact before fitting anything."""

rng: np.random.Generator = np.random.default_rng(20260824)
"""Fixed the generator so the smoke is reproducible from its seed alone."""

families: int = 60
"""Chose enough families to fit a variance component without being slow."""

per_family: int = 6
"""Chose two founders and four offspring, which is a real pedigree shape."""

founders: list[tuple[str, str]] = [
    (f"f{family}-father", f"f{family}-mother") for family in range(families)
]
"""Named two unparented founders for every family."""

ids: list[str] = [name for pair in founders for name in pair] + [
    f"f{family}-child{child}"
    for family in range(families)
    for child in range(per_family - 2)
]
"""Listed every founder first, then every offspring, in one stable order."""

father: list[str | None] = [None] * (2 * families) + [
    founders[family][0] for family in range(families) for _ in range(per_family - 2)
]
"""Left founders unparented and gave each child its family's father."""

mother: list[str | None] = [None] * (2 * families) + [
    founders[family][1] for family in range(families) for _ in range(per_family - 2)
]
"""Left founders unparented and gave each child its family's mother."""

kinship, order = asterism.relationship_matrix(ids, father, mother)
"""Constructed the additive relationship matrix and the row order it is in."""

people: int = len(order)
"""Counted the rows the matrix actually carries, rather than assuming."""

trait: np.ndarray = rng.normal(size=people)
"""Drew a null trait, so the fit should rest heritability on its lower bound."""

covariates: np.ndarray = np.ones((people, 1))
"""Supplied an intercept only, which is the smallest honest design."""

model: asterism.ComponentModel = asterism.ComponentModel([kinship], covariates)
"""Built the public component model over the synthetic relationship matrix."""

fit: dict[str, object] = model.fit(trait)
"""Fitted one variance-components model through the public entry point."""

record: dict[str, object] = {
    "machine": platform.machine(),
    "glibc_version": platform.libc_ver()[1],
    "python_version": platform.python_version(),
    "asterism_version": asterism.__version__,
    "build_identity": asterism.build_identity(),
    "wheel_sha256": hashlib.sha256(arguments.wheel.read_bytes()).hexdigest(),
    "people": int(people),
    "largest_family": int(per_family),
    "converged": bool(fit["converged"]),
    "loglik": float(fit["loglik"]),
    "mean_diagonal_proportions": [
        float(value) for value in fit["mean_diagonal_proportions"]
    ],
}
"""Collected exactly the values-free facts the release blocker asks to retain."""

print(json.dumps(record, indent=2, sort_keys=True))
