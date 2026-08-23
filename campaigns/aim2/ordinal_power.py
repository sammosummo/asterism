"""What would an ordinal outcome be worth in power, on this design?

The liability model already supposes a continuous latent liability; a diagnosis
is that liability cut once. How much cutting it costs is not a matter of
opinion: for the correlation between a normal variable and a categorised version
of another, the information relative to the uncut variable is

    sum over categories of  [phi(t_lower) - phi(t_upper)]^2 / P(category)

with the outer thresholds at infinity. Information behaves like sample size in a
likelihood ratio test, so the ratio converts into the figure a grant quotes.

**The comparison has to hold the disease fixed.** A first version of this
compared the design's binary at 16 per cent against an ordinal whose graded
categories summed to something else, so it was varying the definition of
dementia and the number of cuts at once, and reported the ordinal as *worse*
than the binary when the graded stages thinned -- which was an artefact of the
dementia fraction moving, not a fact about staging. Here CDR 1 or worse is held
at the design's 16 per cent in every row, and what changes is only how finely
the same people are staged.

Asymptotic and for a small correlation, so this sizes the prize rather than
promising a number for this design. Measuring it properly needs the ordinal
outcome implemented, which the integrator can already carry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2, norm

CRITICAL: float = float(chi2.ppf(1 - 0.025, 1))
"""Calculated the chi-squared critical value for the one-sided 0.025 level."""
DEMENTIA: float = 0.16  # CDR 1 or worse, matching the campaign's case rate
"""Fixed the disease fraction to the campaign design's binary case rate."""
MEASURED: float = 0.545  # audiogram arm, mediated effect 0.24
"""Recorded measured binary-outcome power at the target mediated effect."""


def kept(probabilities: Sequence[float]) -> float:
    """Calculate latent-normal information retained after categorisation.

    Args:
        probabilities: Ordered category probabilities summing to a positive value.

    Returns:
        Information relative to observing the continuous latent variable.
    """
    p: np.ndarray = np.asarray(probabilities, dtype=float)
    """Converted the ordered category probabilities to a numeric array."""
    p = p / p.sum()
    """Normalised the category probabilities to sum exactly to one."""
    cuts: np.ndarray = norm.ppf(np.cumsum(p)[:-1])
    """Mapped cumulative category probabilities to latent-normal thresholds."""
    edges: np.ndarray = np.concatenate(([-np.inf], cuts, [np.inf]))
    """Added the two infinite outer thresholds."""
    density: np.ndarray = norm.pdf(edges)
    """Evaluated the standard-normal density at every category boundary."""
    density[0] = density[-1] = 0.0
    """Set the limiting density at both infinite boundaries to nought."""
    return float(sum((density[k] - density[k + 1]) ** 2 / p[k] for k in range(len(p))))


def power_at(noncentrality: float) -> float:
    """Calculate power at the campaign's fixed significance level.

    Args:
        noncentrality: Non-centrality parameter of the chi-squared statistic.

    Returns:
        Upper-tail rejection probability at the fixed critical value.
    """
    return float(ncx2.sf(CRITICAL, 1, noncentrality))


def noncentrality_for(power: float) -> float:
    """Find the non-centrality yielding a requested power.

    Args:
        power: Target rejection probability at the fixed significance level.

    Returns:
        Non-centrality parameter producing the target power.
    """
    return brentq(lambda lam: ncx2.sf(CRITICAL, 1, lam) - power, 1e-9, 400.0)


def staged(questionable: float) -> list[float]:
    """Construct staged CDR probabilities while holding dementia fixed.

    Args:
        questionable: Fraction assigned to the CDR 0.5 category.

    Returns:
        Probabilities for CDR 0, 0.5, 1, 2 and 3.
    """
    severe: np.ndarray = np.array([0.09, 0.05, 0.02])  # 1, 2, 3
    """Set the relative frequencies of the three dementia-severity categories."""
    severe = severe / severe.sum() * DEMENTIA
    """Scaled the severity categories to preserve the fixed dementia fraction."""
    return [1.0 - DEMENTIA - questionable, questionable, *severe]


here: float = noncentrality_for(MEASURED)
"""Converted the measured binary power to its implied non-centrality."""
binary: float = kept([1.0 - DEMENTIA, DEMENTIA])
"""Calculated the information retained by the binary diagnosis."""

print(f"measured power on the design as it stands: {MEASURED:.1%}")
print(
    f"implied non-centrality {here:.2f}, against the "
    f"{noncentrality_for(0.80):.2f} that 80 per cent needs"
)
print(f"dementia held at {DEMENTIA:.0%} throughout\n")

print(f"{'outcome':<38}{'info kept':>11}{'x binary':>10}{'power':>9}")
print("-" * 68)
rows: list[tuple[str, list[float] | None]] = [
    ("binary: dementia or not", [1.0 - DEMENTIA, DEMENTIA]),
    ("+ a questionable band at 9%", staged(0.09)),
    ("+ severity split, 0/0.5/1/2/3", staged(0.18)),
    ("fully continuous liability", None),
]
"""Specified the outcome resolutions compared while holding disease fixed."""
for name, probabilities in rows:
    row_share: float = 1.0 if probabilities is None else kept(probabilities)
    """Calculated information retained by this outcome resolution."""
    row_gain: float = row_share / binary
    """Expressed retained information relative to the binary diagnosis."""
    print(
        f"{name:<38}{row_share:>10.1%}{row_gain:>10.2f}"
        f"{power_at(here * row_gain):>9.1%}"
    )
"""Displayed the information and implied power of each outcome resolution."""

print(
    "\nThe graded band only pays if people are in it. Most of Acoustic is "
    "under 65\nand will stage at nought, so this is the number to argue "
    "about:\n"
)
print(f"{'CDR 0.5 band':<38}{'info kept':>11}{'x binary':>10}{'power':>9}")
print("-" * 68)
for questionable in (0.20, 0.12, 0.06, 0.03, 0.01):
    band_share: float = kept(staged(questionable))
    """Calculated information retained at this questionable-band prevalence."""
    band_gain: float = band_share / binary
    """Expressed the staged outcome's information relative to binary diagnosis."""
    print(
        f"{questionable:>10.0%} of the sample{'':<15}{band_share:>10.1%}"
        f"{band_gain:>10.2f}{power_at(here * band_gain):>9.1%}"
    )
"""Displayed the value of staging across plausible questionable-band sizes."""

print("\nStaging and sample size are not alternatives:\n")
print(f"{'':<22}{'binary':>10}{'staged':>10}")
for factor, label in (
    (1.0, "sample as planned"),
    (1.3, "sample x1.3"),
    (1.7, "sample x1.7"),
):
    staging_gain: float = kept(staged(0.12)) / binary
    """Calculated the fixed staging benefit at a 12 per cent questionable band."""
    print(
        f"{label:<22}{power_at(here * factor):>10.1%}"
        f"{power_at(here * factor * staging_gain):>10.1%}"
    )
"""Displayed how staging and sample expansion combine rather than compete."""
