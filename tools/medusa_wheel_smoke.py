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
import zipfile
from pathlib import Path, PurePosixPath

import asterism
import numpy as np
from asterism import _core

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

extension_path: Path = Path(str(_core.__file__)).resolve()
"""Located the installed native module imported by this exact interpreter."""

if not extension_path.is_file():
    parser.error("the imported installed extension does not exist")
extension_bytes: bytes = extension_path.read_bytes()
"""Read the native bytes that will execute the synthetic fit below."""

try:
    with zipfile.ZipFile(arguments.wheel) as archive:
        native_members: list[zipfile.ZipInfo] = []
        """Collected only native-module members at Asterism's package root."""

        for member in archive.infolist():
            member_path: PurePosixPath = PurePosixPath(member.filename)
            """Interpreted one wheel member using the archive's POSIX paths."""

            if (
                not member.is_dir()
                and member_path.parent == PurePosixPath("asterism")
                and member_path.name.startswith("_core.")
                and member_path.suffix in {".pyd", ".so"}
            ):
                native_members.append(member)
        """Selected compiled `_core` members without accepting arbitrary payloads."""

        if len(native_members) != 1:
            parser.error("--wheel must contain exactly one Asterism native module")
        wheel_extension_bytes: bytes = archive.read(native_members[0])
        """Read the sole archived native member, including its integrity check."""
except (OSError, zipfile.BadZipFile, RuntimeError) as error:
    parser.error(f"--wheel must be a readable wheel archive: {error}")
"""Required an inspectable archive and one unambiguous installed module candidate."""

extension_sha256: str = hashlib.sha256(extension_bytes).hexdigest()
"""Identified the imported native module that will perform the fit."""

wheel_extension_sha256: str = hashlib.sha256(wheel_extension_bytes).hexdigest()
"""Identified the independently read native member in the selected wheel."""

if wheel_extension_sha256 != extension_sha256:
    parser.error("--wheel native module does not match the imported extension")
"""Proved that the selected archive supplied the exact executable bytes in use."""

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
    "extension_sha256": extension_sha256,
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
