# 10. ECM for the censored repeated-measures model

**Extends ADR 0001 under decision 20. Amends nothing. Confirms decision 16's
reasoning for a third time, in the same direction as ADR 0007.**

**Status:** proposed 19 August 2026, before any code for it existed. **Accepted
20 August 2026**, once the ladder this record made a precondition had been
climbed and the complete-data skeleton agreed with `ComponentModel`. Amended the
same day by *What the skeleton changed*, *What censoring changed*, *What the
kernel changed* and *What the checks changed*, below.

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

## What censoring changed

*Amendment, 20 August 2026, after building the expectation step over censored
and never-measured values. The decision above stands. Two things in it need
saying more precisely, one of the four checks it asks for cannot be run as
written, and its arithmetic about the route it rejects was pessimistic.*

**The moments were there and were being thrown away.** The record says the
E-step "is what the sequential truncation already computes -- `mendell_elston`
maintains a running mean and covariance updated for having conditioned on each
coordinate in turn". True, and that function returned only the probability and
discarded the rest. `truncated_moments` in `src/liability.rs` is the same
arithmetic keeping it, and a test asserts the two return the same log
probability **to the last bit**, so they cannot drift apart quietly. The one
difference is that it updates every coordinate rather than only those it has not
yet reached, which changes no probability -- the value at step `i` reads only
coordinate `i`'s own mean and variance -- and is what lets a coordinate already
processed pick up what conditioning on a later one says about it.

**The likelihood and the expectation step use different routines, on purpose.**
The likelihood takes its region probability from `region_log_probability`, which
is exact at one and two coordinates where the sequential update is not, and
which is what every other censored model in the crate reports. The expectation
step has only the sequential update, because only it has moments. So the two
halves of an iteration are not approximations of the same order, and where a
family carries one censored value the expectation step is not an approximation
at all -- it is the ordinary truncated normal, and expectation-maximisation is
exact, and the likelihood cannot fall. Above one it can, and the fit record's
`monotone` is there to say whether it did.

**Filling a value in is not the same as knowing it.** Each rotated row carries
the covariance of its own imputation, and every statistic the maximisation forms
is quadratic in the data, so each takes that covariance in place of an outer
product it would otherwise treat as certain. Leave it out and the variances come
back too small. The test that would catch it is the sharpest one here: a value
that was never measured contributes nothing to the likelihood, so a fit that
imputes it must land exactly where `ComponentModel` given only the rest lands.
It does, in the variances, the fixed effects and the log-likelihood.

### The `TobitModel` check cannot be run as this record specifies

The list of four checks asks for "`TobitModel` reproducing a one-position
slice". It cannot, for a structural reason rather than an incidental one.

`TobitModel` refuses a relationship matrix with an off-diagonal above 0.9,
deliberately: its two-person quadrature loses accuracy as the correlation
approaches one. **A replicate design always produces exactly one there**, because
the two ears of a person share the whole of that person's genotype. Written over
rows, `A ⊗ J₂` carries ones off the diagonal wherever two rows are the same
person, so the two models cannot be pointed at the same matrix. Relaxing the
guard is not the answer; it is there for a reason that has nothing to do with
this model.

They can be pointed at the same *analysis*. Leave the second replicate of every
person unmeasured and what remains is one record per person with a genetic
component and a residual, on the plain relationship matrix `TobitModel` accepts.
The repeated model still has to impute the untested ear, condition the censored
values on what was measured and get the region right; it simply has an
independent answer to be checked against while doing it. With one censored
record per family the two agree to 5e-3 in the heritability and 1e-2 in the
log-likelihood. With two per family, where the expectation step is approximate
and the likelihood still is not, they agree to 0.02 in the heritability.

That is weaker than the record asked for and the difference is worth being plain
about: **it does not exercise the replicate structure at all.** What does is a
test that writes the censored likelihood out by hand for one family of four
people with two replicates apiece at two positions, and finds it equal to the
density of what was measured times the probability of the region the one
censored value lies in **given** that. The conditioning is the part that is easy
to get wrong and the part that stops a censored record being counted twice.

### What it costs, measured rather than counted

On a roster with SAFS's family sizes -- 402 people, largest family 75, two
replicates, seventeen positions, 13,668 observations of which 2,073 reached a
limit, and 386 censored values in the largest family:

| | |
|---|---|
| one evaluation of the observed-data likelihood | **130 ms** |
| one EM iteration, which is three of them | **0.4 s** |
| the gradient reading at the fixed point | **2.3 minutes** |

That is about 77 Gflop/s against this record's estimate of 9.4 Gflop per
evaluation, and the two agree well. **The throughput assumption did not.** This
record put the direct route at "20 to 60 hours" per fit by assuming 10 Gflop/s;
at the rate actually reached it is nearer one to three hours. The decision is
unchanged -- an ECM fit is minutes where the direct route is hours, and the
profile intervals the paper needs multiply both -- but the record overstated the
margin and should not be quoted for the larger number.

Part of the difference is a fix rather than a measurement. The expectation step
factorises one dense covariance per family, 2,550 rows for the largest, and it
was using `nalgebra` for it. `src/dense.rs` had already measured `faer` at
**eighteen times** `nalgebra` by 1,800 rows and provided `DenseFactor` to choose
between them by size. Using it was a one-line change to the one factorisation
that decides what a fit costs.

**The gradient reading, not the iterations, is what the kernel has to fix.** With
a free covariance at seventeen positions the fit carries 459 variances and 68
fixed effects, so the reading is 1,055 evaluations -- the same arithmetic this
record used to reject the direct route, with the difference that ECM needs it
once rather than once per step. It is affordable once. It stops being a rounding
error if a fit converges quickly, and it is a second reason for the covariance
kernel beyond the statistical one: 459 variances become 51 and 6.

Nobody should fit the unstructured model at seventeen positions in any case.
Four hundred and two people do not determine 459 variances, the module says so,
and a fit of it was left running for an hour without converging, which is what
being badly posed looks like from the outside.

**There is no repeatable check of any of this yet.** The numbers above came from
a scratch harness that was deleted, because the crate's cost checks live in
`checks/` and are reached through Python, and this model has no Python interface
until the kernel gives it something worth calling.

## What the kernel changed

*Amendment, 20 August 2026, after building the covariance kernel. The decision
above stands and one sentence in it needs qualifying.*

**The kernel is optional and both halves are kept.** `build` leaves every
covariance free; `build_on_a_line` takes the positions' coordinates and shapes
them. That is not indecision. The kernel family sits inside the free one, so the
free fit is the only thing that says whether the shape cost anything, and a test
asserts the restricted fit never beats it.

**ECM is now ECM.** The earlier amendment noted that with free covariances the
joint M-step exists in closed form, so what the skeleton did was plain EM. With
a kernel it does not: each component's maximiser is the nearest member of the
kernel family in the same Wishart likelihood, found by a small bounded search
over nineteen numbers with an analytic gradient, started from where that
component already is. That is a conditional maximisation, and it is what the
`CM` in the title of this record means. It does not have to arrive — the
likelihood only has to rise, and starting from the current parameters guarantees
it cannot fall — which is what makes it affordable at every iteration for every
component.

**The parameter count closes a loop.** With the kernel the fit carries 51
variances, 6 kernel parameters and 68 fixed effects: 125, exactly the count this
record opened with. So the gradient reading at the fixed point is 251
evaluations against the unstructured skeleton's 1,055, and at the 130 ms
measured earlier that is about half a minute. **It is also, to the evaluation,
one gradient step of the route this record rejected.** ECM's whole saving is
that it needs that once rather than once per step, and it is worth seeing the
two quantities be the same number.

### The floor and the rate are not separately estimable

This record says the floor is a common factor and the exponential is the local
decay, that a large `lambda` gives a pure factor model and a `c` of nought a
pure distance model, and that **"the fit chooses rather than the analyst"**. The
fit does choose. What needs saying is how weakly.

Over a finite span the two parameters trade off against each other almost
exactly. A floor of 0.35 with a rate of 0.09, and no floor at all with a rate of
0.043, agree to within a twentieth of a correlation at every separation out to
22 units. On data simulated from the first, at 600 people, the fit converged to
a scaled gradient of 2.8e-08 — a genuine maximum, not a search that stopped
early — and returned the second. The correlation it drew was right at every
separation, within 0.05 of the truth and closer still to what a free covariance
on the same data reported. The parameters were not; the curve was.

Two things follow, and the first was already the decision here.

- **The correlation comes back as a function of separation.** This record
  already required that, on the ground that two parameters are not 136
  estimates. The stronger ground is that the two parameters are not the
  estimable thing at all. `RepeatedFit::correlation` is the interface and the
  floors and rates behind it are reported for completeness rather than for
  reading.
- **A profile interval on a floor alone would be wide, and would not mean what
  it looked like.** The identified quantity is the curve, so an interval belongs
  on `corr(d)` at separations that matter and not on `c`. Nothing has been built
  either way yet; this is a note for whoever builds it.

Where the decay is fast relative to the span the two do separate. In the same
simulation the replicate level, whose rate was 0.8 across separations starting
at 2.1, came back at 0.88 with its floor at 0.066 against a true 0.05. So this
is a statement about slow decay over a short line, which is the genetic
component's case and not the replicate level's.

**None of this makes the kernel the wrong model.** It buys what it was for: a
correlation that respects the ordering of the positions, from a fit whose own
gradient is affordable to read, containing both the factor model and the
distance model as special cases. It simply does not license reading `c` as the
size of a common factor.

## What the checks changed

*Amendment, 20 August 2026, after building the Python interface and running the
checks this record asks for. Three of the four now stand. The fourth cannot be
run, for a reason worth recording, and the external one found a fault and then
measured something underneath it.*

**The interface is a class.** This record noted that the censored models are
reached as functions where every other model is a class, and that this one
should be a class. It is: preparing it does an eigendecomposition and works out
the family blocks, and none of that depends on the response, so a caller fitting
the same roster twice should pay for it once. An observation arrives as a status
of nought, one, two or **three**, the last being a value never measured at all,
which is a fourth code the crate did not have.

### Where the four checks stand

| check | state |
| --- | --- |
| `TobitModel` on a one-position slice | stands, with the structural caveat recorded above |
| `mixed_bivariate_fit` on a two-position slice | stands, uncensored and censored |
| MCMCglmm, external | stands, and found a fault |
| a coverage simulation | **cannot be run** |

`MixedBivariateModel` is the one that covers the covariance *between* positions,
which the `TobitModel` comparison cannot: it has one position and so says
nothing about the thing this model adds. Uncensored the two agree to 5e-3 in
both heritabilities and both correlations; with one position censored, to 0.03
and 0.05.

**The coverage simulation cannot be run because there is no interval, and there
is no interval because what it should be *of* is unsettled.** The previous
amendment found that the kernel's floor and rate are not separately estimable
while the correlation they describe is. An interval on a floor would be wide and
would not mean what it looked like. So this is a decision outstanding, not a
function unwritten, and nothing from this model should be reported with an
interval attached until it is taken. `tests/test_interval_baseline.py` carries
the same reason at the point where it exempts the model.

### The external check found a fault, and then something under it

The search used to stop the first time the observed-data likelihood fell. With
free covariances and complete data it cannot fall, so no internal check ever
noticed. With thirteen censored coordinates in a family it can, and did, at the
seventh pass -- returning estimates that agreed with MCMCglmm anyway, which is
exactly how a fault like that survives an agreement check. The search now
carries on through a fall and reports the best point it saw rather than the last
one it reached.

Carrying on moved the fit from 23 iterations to 75 and left the gradient where
it was. **That is the other thing this record warned about**: an approximate
expectation step's fixed point need not be at the maximum of the likelihood it
reports. Swept over how much of the data is censored, on 480 people in families
of four:

| censored | dimension | scaled gradient | monotone |
| --- | --- | --- | --- |
| none | 0 | 1.2e-07 | yes |
| 2% | 4 | 1.7e-04 | yes |
| 5% | 7 | 6.1e-04 | yes |
| 10% | 8 | 1.2e-03 | no |
| 25% | 13 | 5.7e-03 | no |
| 50% | 16 | 1.4e-02 | no |

Two things follow.

- **The convergence flag means something different for this model than for
  every other one in the package.** Elsewhere it is a statement about whether
  the search arrived. Here it is that only where nothing is censored; with
  censoring it measures the approximation. The check requires the uncensored
  case to converge, because there is nothing approximate in it, and reports the
  rest. `README.md` tells a reader to read `converged` against
  `censored_shares`.
- **The likelihood and the expectation step use different approximations, and
  that is why the gradient does not go to nought.** The likelihood's region
  probability is exact at one and two coordinates; the expectation step's
  moments are sequential throughout. Making them the same would drive the
  gradient to nought and would be the wrong trade: the reported likelihood would
  stop being the best available and would stop agreeing with `TobitModel`. This
  record chose "the observed-data likelihood stays the definition of correct",
  and the gradient reading is the price and the diagnostic at once.

### What the displacement costs, measured

The sweep says the gradient moves. `checks/repeated_bias.py` says what that
does to the estimates, on 200 replicates with the cells **paired** -- a
replicate index gives the same data at every censoring share, only the limits
differ, so the shift between two cells is a paired difference and the sampling
variation that dominates either cell on its own cancels out of it. Every cell is
judged against the cell with nothing censored rather than against the truth,
because maximum likelihood is already biased downward for a variance component
and that is not the question.

| censored | gradient | heritability | genetic corr | replicate corr |
| --- | --- | --- | --- | --- |
| none | 1.5e-07 | -- | -- | -- |
| 10% | 1.3e-03 | +0.0002 ± 0.0004 | −0.0001 ± 0.0006 | −0.0005 ± 0.0006 |
| 25% | 4.2e-03 | −0.0000 ± 0.0008 | +0.0023 ± 0.0014 | −0.0016 ± 0.0013 |
| 50% | 1.1e-02 | −0.0034 ± 0.0017 | +0.0145 ± 0.0033 | −0.0118 ± 0.0023 |

**The gradient grows five orders of magnitude and the largest thing that reaches
an estimate is 0.015 in a correlation.** So the displacement this record
predicted is real, the reading put in to detect it does detect it, and what it
costs is about two orders of magnitude smaller than the reading makes it look.

It is not nothing. At half censored every shift is several standard errors from
nought and they have a direction -- the genetic correlation rising while the
replicate-level one falls, as though the approximation moves covariance from the
lower level to the upper one. At a tenth censored nothing is detectable at all.
No fit was refused in any cell.

**The number to read a position against is how much of that position was
measured**, not the share over the whole design. This model's own data is about
11 per cent censored overall and 52 and 75 per cent at 16 and 18 kHz, so the top
two frequencies sit in the row where the shift is measurable and the rest do
not. A genetic correlation quoted between 16 kHz and anything else carries it;
one between 1 and 4 kHz does not.

**This does not replace the coverage simulation and does not need an interval.**
It says the estimator is very nearly unbiased where it is used. Whether an
interval built on it covers is a separate question that cannot be asked until
there is an interval to ask it of.
