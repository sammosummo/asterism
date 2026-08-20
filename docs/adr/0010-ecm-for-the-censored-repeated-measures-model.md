# 10. ECM for the censored repeated-measures model

**Extends ADR 0001 under decision 20. Amends nothing. Confirms decision 16's
reasoning for a third time, in the same direction as ADR 0007.**

**Status:** proposed 19 August 2026, before any code for it existed. **Accepted
20 August 2026**, once the ladder this record made a precondition had been
climbed and the complete-data skeleton agreed with `ComponentModel`. Amended the
same day by *What the skeleton changed*, below.

> **This model has no name yet.** It is described here by what it does, because
> naming it is deferred until it works and the name is meant to say what the
> biology is supposed to be. Nothing in this record depends on the name.

## The analysis this names

Decision 20 says a capability enters when a planned analysis needs it and the
amendment naming that analysis exists, before any code for it does. The analysis
is **the heritability of the extended high-frequency audiogram**, and a
follow-up paper on the same data.

`acoustic_audiometry_thresholds` in SAFS.db holds 402 people, in 50 families of
which the largest holds 75, measured at 19 frequencies in each ear. Two
frequencies go: 1500 Hz was tested on one person, and 20 kHz is 88% censored,
which leaves 93 measured ear-records out of 799 and no scale worth estimating.
That leaves 17 frequencies from 125 Hz to 18 kHz, 13,654 observations, 1,556 of
them censored.

The censoring is the point. An audiometer stops at its maximum output, and the
share of thresholds that reach it climbs with frequency — 0.2% at 500 Hz, 1.9%
at 8 kHz, 16% at 12.5 kHz, 30% at 14 kHz, 52% at 16 kHz, 75% at 18 kHz. The
frequencies that are most heritable are the ones most often unmeasurable, so an
analysis that substitutes the limit is biased hardest exactly where the result
is. `TobitModel` already answers that one frequency at a time. What it cannot do
is say how the frequencies relate to each other, which is the audiogram's shape
and therefore the finding.

## What the model is

Traits are the 17 frequencies. The ear is a replicate *within* the person rather
than a second set of 17 traits, on the ground that a genotype does not know left
from right — so the loadings are shared across ears and left-right asymmetry
gets a level of its own instead of being averaged away.

Each variance component carries a free variance at every frequency and a
correlation across frequencies

```text
corr(f, f') = c + (1 - c) exp(-lambda d(f, f'))
```

where `d` is separation on the ERB-number scale and each component has its own
`c` and `lambda`. The floor `c` is not decoration. In these data the correlation
between thresholds falls with separation and then flattens near 0.3 rather than
decaying to nought: 0.96 at one ERB, 0.83 at six, 0.56 at eleven, 0.29 at
thirty-one. A bare exponential cannot represent that and a free factor model
throws away the ordering. The floor is a common factor and the exponential is
the local decay; `lambda` large gives a pure factor model and `c` at nought a
pure distance model, so the fit chooses rather than the analyst. ERB is used
rather than log frequency because it predicts the observed decay slightly better
(−0.76 against −0.71), and the conversion belongs to the caller — the crate
takes positions on a line and knows nothing about hearing.

Three components to begin: additive genetic, person-level environment, and
ear-level. A fourth arrives when a household, childhood-household or distance
matrix exists to supply, which today none does.

Covariates enter the likelihood rather than being removed beforehand, with free
coefficients per frequency, because **a censored observation cannot be
residualised**. There is no value to subtract a fitted mean from, so
residualising first silently forces the analyst back to substituting the limit.

## Why the obvious route does not work

That is 68 fixed effects, 51 variances and 6 kernel parameters: **125
parameters**.

Every non-Gaussian model in Asterism takes its gradient by central difference,
because as `src/tobit.rs` puts it, the region probability has no derivative
worth writing. At 125 parameters that is 250 evaluations per gradient.

One evaluation of the observed-data likelihood on this design is about 9.4
Gflop, of which 84% is the two largest families alone. So one gradient step is
about 2,350 Gflop, one fit at 100 to 300 L-BFGS-B iterations is 20 to 60 hours
at 10 Gflop/s and five to ten days at 2, and the profile intervals the paper
needs multiply that again. These are operation counts rather than timings and
the real figures could differ by a factor of several, but not by the factor that
would make this route reasonable.

## Why the complete-data problem is cheap

If nothing were censored and nothing missing, the covariance is

```text
V = A ⊗ J₂ ⊗ Σ_A  +  I ⊗ J₂ ⊗ Σ_C  +  I ⊗ I₂ ⊗ Σ_D
```

where `J₂` is the two-by-two matrix of ones, because both ears share whatever
belongs to the person. Rotating the two ears into their sum and difference makes
`J₂` into `diag(2, 0)`, which splits the problem in half: the difference channel
carries only the ear-level covariance and is block diagonal by person, and the
sum channel is an ordinary two-component Kronecker form that the eigenvectors of
the relationship matrix diagonalise. What is left is at most 804 blocks of
17 × 17 — about 1.3 Mflop, against 9.4 Gflop, with the one 402 × 402
eigendecomposition done once per fit rather than once per evaluation.

That factor of roughly 1,800 is available to any algorithm that only ever
touches complete data.

## The decision

**Fit by ECM. Keep the observed-data likelihood as the definition of correct.**

The E-step fills in each censored value with its conditional expectation given
the measured values in the same family, which is what the sequential truncation
already computes — `mendell_elston` maintains a running mean and covariance
updated for having conditioned on each coordinate in turn, and those updated
moments are the E-step. The M-step then works on complete data at the cost
above. One ECM iteration costs about one likelihood evaluation instead of 250.

Values that were never tested are imputed in the same step rather than dropped.
There are only 17 of them and the statistics do not care, but unbalanced data is
exactly what breaks the rotation, and `src/bivariate.rs` already records that
lesson: the eigen-rotation "survives extra traits only while everybody has every
trait". Seventeen dropped observations would cost a factor of a thousand.

Starting values come from 17 independent univariate `TobitModel` fits, which are
cheap and are the paper's first table anyway.

**ECM is a route and not a definition.** Under decision 15's rules for fast
routes, a route that disagrees with the general estimator has the bug. So the
fit is not finished when ECM stops: the direct observed-data objective is
evaluated at the ECM fixed point and its central-difference gradient recorded in
the fit record. One number decides whether the route arrived where it claimed.

## What could make this wrong

The E-step moments are approximate, because Mendell-Elston is. An approximate
E-step is not guaranteed to climb the likelihood it claims to maximise, and its
fixed point need not be the maximum. The gradient check above is what detects
that, and it is a check rather than a proof.

Beneath that sits a worse exposure. **Every piece of evidence Asterism has for
the censored machinery was generated on pairs**, where the region probability is
exact and the sequential approximation never runs at all. Here the largest
family needs it at 221 censored dimensions conditional on 2,328 measured values,
and sequential truncation is documented to degrade as coordinates correlate —
these correlate at 0.96 between neighbouring frequencies. The liability model
has been calibrated on the real pedigree at 180 dimensions, so the approximation
is not unqualified in general, but it has never been checked in the conditional
form this model uses.

So the first piece of work is not this model. It is a ladder: conditional
log-probabilities from the sequential update against `rectangle_probability`'s
quasi-Monte Carlo, at 5, 20, 50, 100 and 221 censored dimensions, at the
correlations this model actually produces. The comparison that matters is the
error in the log-probability against the difference in log-likelihood that a
heritability of 0.1 would make, since that is the difference the estimate rests
on.

**If the ladder fails, the model is cut to the frequencies where it passes and
that is said out loud.** The route is not switched to quasi-Monte Carlo, which
costs at least five hundred evaluations per family however few members it has.
A model that runs and is wrong is worse than a smaller one that runs and is
right.

## What is in scope, and what is deliberately not

**In**: positions on an arbitrary ordered continuum, one replicate level within
the person, any number of supplied relationship matrices, right and left
censoring per observation, missing-at-random observations, maximum likelihood.

**Out, each for a reason**:

- **A name.** Deferred to its own session once the thing works, because the name
  is meant to carry the biology and the biology is what the fit will show.
- **The longitudinal capability.** This model could serve longitudinal data —
  positions on a line are positions on a line — but it will not have been
  checked for it. `CONTEXT.md`'s deferred list stays unamended until an analysis
  needs it.
- **Selecting among shared-environment components.** Current household,
  childhood household and shared tract are nested relations, and at 402 people a
  simultaneous fit would produce a division nobody could defend. Each goes in
  alone, against the three-component model, reported as a series with intervals
  rather than reduced to a winner. The likelihood is approximate in two ways at
  once, so a few points of AIC between non-nested component sets is not a number
  to trust.
- **REML.** For the reason it is absent from every non-Gaussian model here: a
  region has no response to project onto the null space of the design. The fit
  record says `ml`.

## Consequences

- The fit record carries the diagnostics that let a reader judge it, not merely
  the estimates: censored share per position, largest family, **the sequential
  approximation dimension actually reached**, ECM iterations, and the verifying
  gradient. The dimension is there because the whole result leans on it and a
  reader should not have to infer it.
- Genetic correlations between positions come back as a function to evaluate
  rather than a 17 × 17 matrix, so that it stays obvious they came from a kernel
  with two parameters and not from 136 free estimates.
- This is the first model in Asterism above two traits. It is also the first to
  be fitted by anything other than L-BFGS-B, which is why this record exists.
- The censored models are reached as functions — `tobit_fit`,
  `mixed_bivariate_fit` — where every other model is a class. This one should be
  a class, like the rest.
- The independent check required of any new capability is four things, since no
  single comparator covers it: `TobitModel` reproducing a one-position slice,
  `mixed_bivariate_fit` reproducing a two-position slice, `MCMCglmm` with a
  `cengaussian` family as the external comparator on an unstructured two- or
  three-position model, and a coverage simulation on the real 50-family roster
  with the real censoring pattern. The external one cannot check the kernel,
  only the likelihood — which is the part that could be wrong quietly.

## What the skeleton changed

*Amendment, 20 August 2026, after building the complete-data half of the model
in `src/repeated.rs`. The decision above stands; three things about it turned
out to be either simpler or harder than written.*

**The ladder passed, and it passed for a reason worth keeping in view.**
Conditioned on the rest of the family, the sequential approximation's error at
221 censored dimensions is 0.062 log units, about 0.09 of what a heritability
step of 0.1 does to the same quantity. But conditioning on a nearly complete
audiogram leaves a mean absolute correlation of 0.009 between the censored
residuals, and sequential truncation is exact when coordinates are independent.
On the same number of coordinates with nothing conditioned away the error is 84
times larger. So the result is a statement about this design and not about the
routine, and the case it does not cover is a person whose audiogram is mostly
unmeasurable — who is exactly the person the model exists to use.
`checks/sequential_against_ghk.py` is the check and it should be run again
whenever the design moves.

**It is EM and not ECM, because the joint M-step exists.** ECM is for when the
complete-data maximisation has to be done in conditional blocks. Here it does
not: the complete-data likelihood separates into one closed form per component,
and the fixed effects and the replicate-level covariance maximise together, the
first without reference to the second — the design is shared across positions
and the coefficients are free at every one, so the covariance cancels out of the
normal equations and what is left is ordinary least squares. Plain EM is the
stronger statement and is what the code does. The title of this record is left
as it was written.

**Plain EM does not reach a boundary, and the boundary is not an exotic case.**
A component whose covariance has gone to nought, or lost a direction, is what a
fit says when the data do not support that component, and this model has three
components and will have a covariance apiece at seventeen positions. EM
approaches such a point geometrically and never arrives: measured on twenty
people with no signal in them, plain EM was still at `1.5e-4` after twenty
thousand iterations for a variance whose maximum is exactly nought, against
L-BFGS-B's `0.0`, and reported itself unconverged — correctly, and uselessly.

Two additions fix it, and both are guarded by the likelihood rather than
trusted.

- **Extrapolation.** Two EM steps from a point lie on a line, and where the
  sequence is geometric the rest of that line can be taken at once. This is the
  `S3` scheme of Varadhan and Roland's SQUAREM. The proposal is projected back
  onto the covariances, followed by an ordinary EM step so that what is returned
  is always an EM iterate, kept only where the observed-data likelihood is at
  least what two plain steps would have given, and backed off towards the plain
  step where it is not. Monotonicity is preserved whatever the extrapolation
  does.
- **Resting on the bound.** A direction whose variance falls below `1e-6` on the
  standardised scale is put exactly on nought. This is the same idea as
  `resting_on_zero` in `src/components.rs`, needed for the same reason and at a
  much looser tolerance, because EM leaves a component further from its bound
  than a bounded search does. It is a proposal and not a decision: the
  likelihood is evaluated at the boundary and the move refused if it falls,
  which is precisely the case of a direction that was small rather than absent.
  A covariance's null space is preserved by the M-step, so nought is an exact
  fixed point and this is never undone.

Together they took that fit from 20,000 iterations and a wrong answer to 3,166
iterations and a variance agreeing with L-BFGS-B to twelve significant figures.
**Neither is an optimisation.** Without them the fit is wrong at a boundary and
says so; the cost argument for this whole route assumed EM iterations are cheap,
and twenty thousand of them are not.

**What is still to come, in order**: censoring, then the kernel, then the four
checks. The skeleton takes complete balanced data and a free covariance per
component, which at seventeen positions is 153 numbers nobody should read. It is
there to show the machinery lands where the answer is already known, and it
does: at one position it reproduces `ComponentModel` in the estimates, the fixed
effects and the log-likelihood, by an arithmetic that shares nothing with it.

**One thing in the decision above is not yet honoured.** It says any number of
supplied relationship matrices is in scope. The rotation cannot do that: two
structured matrices share no eigenbasis, which is the same fact `src/components.rs`
opens with. `RepeatedModel::build` checks the second component is diagonal in
the first's basis and refuses otherwise, rather than approximating. A household
matrix, when one exists, will need a dense per-family route and the cost that
goes with it — which is a decision to take then, with the matrix in hand.
