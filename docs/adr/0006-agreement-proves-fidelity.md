# 6. Agreement between implementations proves fidelity, never correctness

**Status:** accepted 7 August 2026 in Astrarium, adopted into Asterism on
11 August 2026. Decision 3 of `0001` says correctness is independent rather than
parity with SOLAR; this is the sharper version, and decision 15's three rules for
fast routes are the same principle arrived at separately.

Agreement between the crate and an independent check is stated in advance to be
evidence of exactly one thing: that the crate faithfully implements the
mathematics the check implements. It is never evidence that the mathematics is
right. The predecessor's 2e-11 agreement was correlated error, and six
algebraically identical routes in binary64 on one machine spread by 3.11e-08.

**Correctness belongs to external comparisons, with a fixed division of labour:
SOLAR for ML, R `regress` for REML, and the coverage check's h² = 0 cell for the
boundary null.**

The comparison covers **every field both implementations fill** — which the
record's both-implementations rule (`0003`) makes possible field by field, with
no translation — at tolerances set from measured floors:

- log-likelihood within 1e-8·(1+|ref|), deliberately looser than the measured
  2.8e-14, because a tighter claim institutionalises luck;
- variance components, fixed effects and both interval endpoints within
  1e-6·(1+|ref|), which is optimiser-bounded, and the endpoints are the point of
  the duplication;
- exact equality for converged, boundary, the limited flags, warnings, and which
  standard errors are present.

Inputs come in two tiers: a standing seeded battery that runs in seconds and is
never skipped, and the full battery through both implementations when the
coverage check runs.

## Consequences worth calling out

- **These tolerances are the definition of done** for any implementation work
  that claims to have finished.
- **A breach fails the suite, and the external comparison arbitrates** which side
  is wrong — never the other implementation, which is how correlated error
  survives review.
- **Tolerances move only with a recorded reason**, never to turn a failing suite
  green. No record blocks a verified fix.
- Decision 21 of `0001` makes this a condition of entry: a capability added to
  Asterism arrives with its own independent check or it does not arrive.

## Both comparisons run, 11 August 2026

`checks/against_r.py` and `checks/against_solar.py`, results in `evidence/`.
Both use a six-column design — an intercept, age, age squared, sex and the two
age-by-sex products — because an intercept-only comparison leaves the
determinant term unexercised, and that term is the part most likely to differ
between two implementations.

- **REML against R `regress`.** Heritability agrees to about 5e-9 relative,
  total variance to 4e-9, every fixed effect to 1e-9.
- **ML against native SOLAR.** Heritability agrees to about 1e-8 relative, which
  is every digit SOLAR prints, and the standard errors agree to five figures
  despite being computed differently.

### The log-likelihood constant, and why the checks assert it

Neither comparator's printed log-likelihood equals Asterism's, and the
difference is a constant Asterism keeps and they drop:

| comparison | estimator | offset | equals |
| --- | --- | --- | --- |
| R `regress` | REML | −316.114855 | −(n − p)/2 · log(2π) |
| SOLAR | ML | −321.628487 | −n/2 · log(2π) |

It is the Gaussian normalising constant. A Gaussian log-density carries
−½ log(2π) per observation. ML is a density for the n observations, so it
carries n of them. REML is a density for n − p error contrasts rather than for
the data, so it carries n − p — which is why the same idea gives two different
numbers, and why a check written for one comparator would silently pass or fail
for the wrong reason against the other.

**Why they drop it.** Nothing anyone does with a likelihood notices an additive
constant. It does not move the maximum, it cancels from every likelihood ratio,
it cancels from a profile interval, and its derivative is zero so it is absent
from the score and the information. Dropping it is free for all of that.

**Why Asterism keeps it.** The printed number is then the actual log-likelihood
— a log-density — so it can be compared with anything else that reports one, and
an AIC or BIC computed from it lands on the usual scale. A log-likelihood with
the constant dropped is only meaningful against another number from the same
program at the same n and p.

**The consequence to carry: never compare Asterism's printed log-likelihood
directly with SOLAR's or `regress`'s.** Subtract n/2 · log(2π) to reach SOLAR's
convention, or (n − p)/2 · log(2π) to reach `regress`'s. Differences between two
models fitted in the same program need no correction at all, which is why the
likelihood ratio tests agree without any of this.

Both checks assert the offset is that named constant rather than merely that it
is stable. An offset that held steady but could not be identified would mean the
two were computing different functions that happened to differ by a fixed
amount, and that is worth finding rather than tolerating.

### One trap, found the hard way

The first SOLAR run disagreed — 0.4424 against 0.4380 — and the cause was in the
harness rather than either estimator. **SOLAR silently overrules a sex that
contradicts a parental role.** A founder written as male who appears as
somebody's mother becomes female in `pedindex.out`, with no error and no
warning. The design matrix handed to Asterism still carried the original coding,
so the two fits used different covariates. `checks/against_solar.py` now derives
sex from the parental role and, after running, compares `pedindex.out` against
what was declared, failing if SOLAR changed anything.
