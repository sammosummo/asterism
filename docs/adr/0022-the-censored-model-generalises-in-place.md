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

[ADR 0001](0001-asterism-is-a-fresh-start.md) decision 16 says the general
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

**The boundary test's level has to be measured per design rather than inherited.**
The target-design campaign found it holding at 0.055 with half the rows censored
and rejecting 0.100 of the time with three quarters censored, against a nominal
0.05; estimates and intervals held at both. The one-component model's evidence
does not cover this, because at one component the whole of the information goes
to the component being tested. That is recorded as a measured level beside the
quantity, not as grounds for withholding it: [ADR
0020](0020-no-design-range-gates-a-result.md) settled that no design range gates
a result.
