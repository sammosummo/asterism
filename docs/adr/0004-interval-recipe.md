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
  Astrarium surviving. The numbers above are the record; the sweep would have
  had to be rerun against Asterism's own code in any case.

## Re-earned on 11 August 2026, against this code

`src/bin/coverage.rs`, 8,000 replicates in each of twelve cells, on a generated
roster of three-generation families — a founding couple, three children, three
spouses married in, six grandchildren — carrying parent–offspring, sibling,
grandparental, avuncular and cousin relationships. Results in
`evidence/coverage-2026-08-11.json`.

**At n = 1400, all twelve cells pass.** At n = 350, ten of twelve do; the two
that fail are truths of 0.05 and 0.07, at 0.979 and 0.974. Both **over**-cover,
so the intervals there are too wide rather than too narrow.

### Why it over-covers just above zero, and why it is not simply made correct

The interval is every h² whose deviance from the fitted maximum is at most
3.8415. Coverage at a true h₀ is therefore the chance that

    T = 2 [ ℓ(ĥ²) − ℓ(h₀) ]

falls below 3.8415, where ĥ² is the maximiser **constrained to [0, 1]**.

Write ĥ²ᵤ for the maximiser without that constraint, which may be negative.
When ĥ²ᵤ ≥ 0 the two agree and T is the ordinary likelihood ratio statistic,
which is χ²₁ and gives exactly 95 per cent. When ĥ²ᵤ < 0 the constrained fit
stops at zero, so ℓ(ĥ²) is **smaller** than ℓ(ĥ²ᵤ) and therefore T is smaller
than it would otherwise have been. T is never larger, and is strictly smaller
whenever the estimate is pinned. A statistic that is stochastically too small,
compared against a χ²₁ critical value, falls below it too often — so the
interval contains the truth too often.

That is the whole mechanism, and the numbers follow it exactly. At a true 0.05
the estimate pins at zero 28.6 per cent of the time at n = 350 and coverage is
0.979; at n = 1400 it pins 10.1 per cent of the time and coverage is 0.962. The
conservatism tracks the pinning rate, and both shrink as the truth moves away
from the bound in units of its own standard error.

**Making it exactly correct is a calibration problem, not a bug fix.** The right
critical value is not 3.8415 but some c(h₀) with P(T ≤ c) = 0.95 under h₀, and
that distribution is neither χ²₁ — right far from the bound — nor the 50:50
mixture — right exactly at it — but a continuum between the two, indexed by how
many standard errors h₀ sits above zero. There is no closed form, so c(h₀) has
to be found by simulation.

**It cannot be fixed by tightening the interval everywhere.** Coverage is
already exactly right at interior truths, so any uniform narrowing would push
those below 95 per cent — trading a safe error for an unsafe one. The correction
has to vary along the profile, which is decision 12's third tier: calibrate once
per design, as SOLAR does with `lodadj`. The machinery is anticipated; it is
simply not built.

**Until it is, the cost is power rather than validity.** An interval near zero
is wider than it needs to be, so a real but small heritability will be called
not significant more often than it should be. **Report an interval near zero at
SAFS scale knowing it is conservative, and say so in the methods.**

**The mixture rule earns its place, measured rather than argued.** Taking a
lower endpoint of nought to mean the interval contains nought — the obvious
rule — gives 0.977 at a true heritability of zero where the mixture gives 0.953,
and 0.973 against 0.950 at a true heritability of one. The obvious rule is
outside the band at both bounds and the mixture is inside at both. Half the
fits sit on the bound in those cells, so this is not a corner case.

One coincidence, recorded so that nobody later reads it as a number that was
copied rather than computed: the truth-zero cell at n = 350 came out at
0.952875, which is exactly the figure in Astrarium's old results file for a
different roster at n = 1909. Different pedigree, different seed, different
code. At this variance an exact match has about a two per cent chance, and this
is it.
