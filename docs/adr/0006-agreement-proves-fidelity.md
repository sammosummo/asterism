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
