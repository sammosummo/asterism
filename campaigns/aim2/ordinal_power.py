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

import numpy as np
from scipy.stats import chi2, ncx2, norm
from scipy.optimize import brentq

CRITICAL = chi2.ppf(1 - 0.025, 1)
DEMENTIA = 0.16          # CDR 1 or worse, matching the campaign's case rate
MEASURED = 0.545         # audiogram arm, mediated effect 0.24


def kept(probabilities) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p / p.sum()
    cuts = norm.ppf(np.cumsum(p)[:-1])
    edges = np.concatenate(([-np.inf], cuts, [np.inf]))
    density = norm.pdf(edges)
    density[0] = density[-1] = 0.0
    return float(sum((density[k] - density[k + 1]) ** 2 / p[k]
                     for k in range(len(p))))


def power_at(noncentrality: float) -> float:
    return float(ncx2.sf(CRITICAL, 1, noncentrality))


def noncentrality_for(power: float) -> float:
    return brentq(lambda lam: ncx2.sf(CRITICAL, 1, lam) - power, 1e-9, 400.0)


def staged(questionable: float) -> list[float]:
    """CDR 0 / 0.5 / 1 / 2 / 3, with dementia held at the design's rate."""
    severe = np.array([0.09, 0.05, 0.02])          # 1, 2, 3
    severe = severe / severe.sum() * DEMENTIA
    return [1.0 - DEMENTIA - questionable, questionable, *severe]


here = noncentrality_for(MEASURED)
binary = kept([1.0 - DEMENTIA, DEMENTIA])

print(f"measured power on the design as it stands: {MEASURED:.1%}")
print(f"implied non-centrality {here:.2f}, against the "
      f"{noncentrality_for(0.80):.2f} that 80 per cent needs")
print(f"dementia held at {DEMENTIA:.0%} throughout\n")

print(f"{'outcome':<38}{'info kept':>11}{'x binary':>10}{'power':>9}")
print("-" * 68)
rows = [
    ("binary: dementia or not", [1.0 - DEMENTIA, DEMENTIA]),
    ("+ a questionable band at 9%", staged(0.09)),
    ("+ severity split, 0/0.5/1/2/3", staged(0.18)),
    ("fully continuous liability", None),
]
for name, probabilities in rows:
    share = 1.0 if probabilities is None else kept(probabilities)
    gain = share / binary
    print(f"{name:<38}{share:>10.1%}{gain:>10.2f}{power_at(here * gain):>9.1%}")

print("\nThe graded band only pays if people are in it. Most of Acoustic is "
      "under 65\nand will stage at nought, so this is the number to argue "
      "about:\n")
print(f"{'CDR 0.5 band':<38}{'info kept':>11}{'x binary':>10}{'power':>9}")
print("-" * 68)
for questionable in (0.20, 0.12, 0.06, 0.03, 0.01):
    share = kept(staged(questionable))
    gain = share / binary
    print(f"{questionable:>10.0%} of the sample{'':<15}{share:>10.1%}"
          f"{gain:>10.2f}{power_at(here * gain):>9.1%}")

print("\nStaging and sample size are not alternatives:\n")
print(f"{'':<22}{'binary':>10}{'staged':>10}")
for factor, label in ((1.0, "sample as planned"), (1.3, "sample x1.3"),
                      (1.7, "sample x1.7")):
    gain = kept(staged(0.12)) / binary
    print(f"{label:<22}{power_at(here * factor):>10.1%}"
          f"{power_at(here * factor * gain):>10.1%}")
