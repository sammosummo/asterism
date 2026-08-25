"""A binary trait on a pedigree, through a liability threshold.

Run it with `uv run python examples/liability.py`.

The heritability is of the *liability*, not of the observed status, which is
what anybody means by the heritability of a disease. It is not comparable with
the REML heritabilities the rest of this package reports.
"""

from __future__ import annotations

import asterism
import numpy as np

from synthetic import pedigree, trait

relationship, order = pedigree(families=250, children=2)
"""Built two hundred and fifty families of four people each."""

people: int = len(order)
"""Read the roster size from the returned order."""

liability: np.ndarray = trait(relationship, heritability=0.5, seed=17)
"""Drew the unobserved liability, of which only the sign is ever seen."""

prevalence: float = 0.2
"""Chose a disease one person in five has."""

threshold: float = float(np.quantile(liability, 1.0 - prevalence))
"""Placed the threshold so the prevalence comes out as chosen."""

status: np.ndarray = (liability >= threshold).astype(np.int64)
"""Recorded who is a case. The liability itself is never observed again."""

design: np.ndarray = np.ones((people, 1))
"""Supplied an intercept, which carries the threshold."""

model: asterism.LiabilityModel = asterism.LiabilityModel(relationship, status, design)
"""Built the liability model over the observed statuses."""

fit: dict[str, object] = model.fit()
"""Fitted the liability-scale heritability. It takes no response argument."""

print(f"people:                 {people}")
print(f"cases:                  {int(status.sum())} of {people}")
print(f"h2 liability (true 0.5): {float(fit['heritability']):.3f}")
print(f"converged:              {fit['converged']}")

interval: dict[str, object] = model.interval()
"""Profile interval on the liability scale."""

print(f"95% interval:           [{interval['lower']:.3f}, {interval['upper']:.3f}]")
