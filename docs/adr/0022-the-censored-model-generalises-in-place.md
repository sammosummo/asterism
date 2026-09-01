# 22. The censored model generalises in place

**Status: accepted.** The censored model takes any number of variance
components through the model that already existed, rather than through a second
model beside it. `TobitModel` holds a `Vec<DMatrix<f64>>` where it held one
matrix, and everything else — the region probability, the block factorisation,
the profile interval, the boundary test — is the code that was already there.

## What was decided against

A separate `CensoredComponentModel` in Rust, leaving the one-component censored
model untouched. It is the obvious move: the released analysis keeps its exact
code path, and nothing that is qualified can be disturbed by work that is not.

## Why in place

**The one-component case is not a case.** With a single matrix the parameter
layout is the same slots in the same order it always was: the shares first, the
log total variance, then the fixed effects. At one component the search is
byte-identical to what it was, which is why the six censored pass rules were
re-run and came back unmoved rather than merely close. A model that reproduces
the old one exactly is the old one.

**A copy would drift, and the expensive parts are shared.** The censored
likelihood's cost is the region probability, reached by sequential truncation
with a bivariate integral below three coordinates and a scaled logarithm in the
deep tail. That routine took several corrections during this work — a floor that
turned six error codes into a finite log-likelihood the optimiser would walk on,
and a deep tail that returned the floor where the truth was −32.89. Each of
those was fixed once. In two models one of them would have been fixed once and
the other would still be wrong, and nothing would have said which.

**The seam the components change is small.** What differs between one component
and several is the covariance being a sum, and the likelihood factorises over
blocks of that sum rather than of any one term. That is one function,
`union_blocks`, and it is right for both.

## Decision 16 stands

[ADR 0001](0001-variance-components-estimator-parameterisation.md) decision 16 says the general
estimator assumes a Gaussian likelihood, and that the threshold, ordered,
ascertained and Student-t objectives stay as separate special cases rather than
being folded into the general design. **Nothing here changes that.** The
censored model is still a special case outside the general estimator. Taking a
list of components instead of one matrix makes it resemble the general
estimator's interface; it does not make it the general estimator, and the two
still share no likelihood.

The reason is the same reason decision 16 gave. Fixed effects cannot be profiled
out by least squares here, because a censored row has no residual to project
onto the null space of the design — so this model is maximum likelihood and
never REML, where the general estimator offers both. The boundary geometry
differs too, and this work measured how: with several components more than one
share can rest on nought at once, which is not the case the one-component
coverage rule scored, so the interval's Self–Liang verdict is deliberately left
absent at several components rather than filled in from a measurement of a
different model.

## What it costs

**The released analysis and the unreleased one now share a code path.** A change
made for several components can break one. That is accepted because the
alternative was two copies of the region routine, and it is paid for by re-running
the six censored pass rules whenever that path changes; they were re-run for this
work and did not move.

**The boundary test is explicitly asymptotic.** Its finite-sample level can vary
with the design, so a target campaign records how the approximation behaved on
that target. It does not create a portable censoring limit or gate the method on
other data. A direct failure does, however, apply to the p-value for the exact
analysis that failed. [ADR 0020](0020-no-design-range-gates-a-result.md) keeps
that named limitation separate from a general design range.

## Amendment: the historical censoring campaigns did not test a fixed instrument

**Added on 31 August 2026.** The design decision above still stands, but its
qualification claims do not. The simulation generators chose each censoring
limit from the realised response in order to force an exact censoring count.
That makes the measurement rule depend on the outcome. It therefore does not
establish estimation, interval coverage, or test level for a design in which
the instrument's limit is fixed before the response is observed. The recorded
campaigns remain historical measurements, but the affected pass rules must be
rerun with limits derived from design facts alone.

The share of likelihood-ratio statistics resting on the bound is useful as a
diagnostic. It is not itself a finite-sample pass rule: the 50:50 mass is an
asymptotic reference, and the realised atom can differ substantially while the
rejection level remains calibrated. The earlier target-design gate on a half
atom, and the level conclusions quoted above, are therefore retired.

For several censored covariance components, the analytic 50:50 test is exposed
with the explicit rule `asymptotic_mixture_50_50`. It refuses when a nuisance
component or the residual rests on its bound, because then the one-boundary
reference is not the stated null problem. At the time of this amendment its
fixed-instrument target campaign remained outstanding; that campaign was to
measure the rejection level directly, with the LRT atom only diagnostic.

The development tree also exposes a constrained-null parametric bootstrap with
an add-one p-value. It is an experimental, design-specific alternative rather
than a 0.2 release claim. Its inner draws are independent and parallelised from
coordinate-derived streams, so an optional large calibration need not inherit
the former serial 999-draw floor.

The one-component numerical fit path used by the released model is unchanged.
That is a statement about the implementation, not a replacement qualification
claim: its affected simulation evidence also needs the fixed-limit rerun.

## Amendment: the fixed-instrument target identified one exact p-value limitation

**Added on 1 September 2026.** The analytic fixed-instrument target completed all
800 attempts. It rejected 0.055 of null datasets at 52% expected censoring and
0.095 at 75%. The higher-censoring cell failed its predeclared rule because the
one-sided exact lower confidence bound was 0.0631, above nominal 0.05. The
descriptive two-sided interval began at 0.0582. The LRT atoms were 0.465 and
0.390 and remain descriptive only. Interval coverage was 0.980 and 0.960, and
the largest censored block was 313.

The optional constrained-null bootstrap then completed all 800 requested outer
coordinates with 999 inner draws requested for each null fit. Eleven attempts,
all in the 75% null cell, returned `TOBIT_BOOTSTRAP_REPLICATE_FAILED`; the other
189 supplied p-values. Its no-write merge therefore failed the predeclared rule
that permits no failed attempt and published no qualifying evidence. This does
not change the decision to expose the bootstrap experimentally.

The analytic test remains available for several components under the explicit
rule `asymptotic_mixture_50_50` when every nuisance variance is interior. The
failed cell was the exact 1,909-person, four-component fixed-instrument target at
75% expected censoring. Its analytic p-value must not be reported without a
design-specific simulated null tied to the model, design, source and seed. This
is not a universal censoring threshold. Fits and intervals at that target, the
test on other designs, and the released one-component route are unaffected; the
bootstrap remains experimental.
