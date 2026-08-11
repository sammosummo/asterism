# 4. The interval is a profile likelihood with a mixture-calibrated boundary point

**Status:** accepted 7 August 2026 in Astrarium, adopted into Asterism on
11 August 2026. Decision 5 of `0001` defers to this for the one-trait case; the
general case still needs re-deriving.

The 95 per cent interval for h² is the **profile-likelihood interval** of the
fitted log-likelihood — deviance crossing the χ²₁ threshold of 3.8415, found by
Brent, with endpoints that reach a bound without a crossing carrying an explicit
flag. Whether a boundary *point* — h² of exactly 0 or 1 — belongs to the interval
is decided by the Self–Liang 50:50 mixture critical value of 2.7055, which is the
correct null for a parameter sitting on its bound.

**Decided by measurement rather than argument.** On 2,000 REML replicates per
cell at n = 1909, scored unconditionally, this was the only candidate inside the
binomial band in every cell — including truth exactly at h² = 0, where the
estimator piles onto the boundary half the time, Wald over-covers at 0.988 and
the predecessor's transformed Wald collapses to 0.0005. The mixture rule moves no
endpoint, so it costs no width; it repairs plain profile-χ²₁'s single fault, a
conservative 0.980 at h² = 0.

## Consequences worth calling out

- **The recipe name is data on the record and never a contract term.** Nothing
  may require the string "profile_mixture". The predecessor died when a frozen
  method name blocked the fix to this exact quantity. The coverage check scores
  whatever interval the package actually produces.
- **The coverage evidence is for the REML likelihood**, which is the default. An
  ML fit gets the same construction on its own likelihood, with no separate
  coverage claim.
- The mixture verdict is carried on the record as `contains_lower_bound` and
  `contains_upper_bound` — a boolean when the corresponding endpoint sits on its
  bound and absent otherwise. This was added after an adversarial review found
  the recipe label otherwise unimplementable from the record alone.
- **The evidence is not carried across.** The head-to-head sweep lived in
  Astrarium's `.scratch`, and under decision 26 of `0001` nothing depends on
  Astrarium surviving. The numbers above are the record; the sweep is
  reconstructable if it is ever wanted, and it would have to be rerun against
  Asterism's own code in any case.
