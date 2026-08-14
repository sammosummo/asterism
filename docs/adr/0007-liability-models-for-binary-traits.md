# 7. Liability models for binary traits

**Amends decision 16 of ADR 0001, and confirms the rest of it.**

## The analysis this names

Decision 20 says a capability enters when a planned analysis needs it and the
amendment naming that analysis exists, before any code for it does. The analysis
is **the heritability of lifetime major depression in GOBS**.

`gobs_mdd_life` in SAFS.db holds 485 cases and 1,425 non-cases across 1,910
people, a prevalence of 0.254, on the reviewed pedigree. It is a real result the
workspace does not have, and no Gaussian model can produce it: a heritability
fitted to an observed 0/1 variable is a different quantity from a heritability
of the liability behind it, and is not what anybody means by the heritability of
depression.

## What decision 16 said, and what survives

Decision 16 held that the general estimator is Gaussian, and that "the
threshold, ordered-threshold, ascertained and Student-t objectives stay as
separate one- and two-trait special cases and are not folded into the general
design". Its reasons were that REML has no clean meaning for a liability model,
that fixed effects cannot be profiled out by least squares, and that the
boundary geometry changes.

**All three reasons hold, and none of them argues against a special case.** They
argue against folding liability into `ComponentModel`, which this does not do.
Decision 21 already recorded that the non-Gaussian objectives "return by
amendment under decision 20, like anything else" — the route back was designed
in rather than closed off. This is that amendment.

So decision 16 is amended in one respect only: the sentence permitting separate
special cases is now exercised rather than deferred. The general design stays
Gaussian and `ComponentModel` is untouched.

## The model

Every person carries an unobserved liability `L_i = x_i' beta + g_i + e_i` and is
a case when it crosses nought. Only the sign of the liability is ever seen, so
its variance is fixed at one and the covariance is `h2 A + (1 - h2) I`. The
threshold is fixed at nought and the intercept carries it.

The likelihood of a family is the probability of an orthant. One and two people
are exact; above two it is the Mendell-Elston sequential truncation, taking the
rarer class first. That is an approximation and is named as one in the module,
but it is a coherent joint likelihood rather than a pairwise composite, which
would not be a likelihood at all.

## What is in scope, and what is deliberately not

**In**: one binary trait, one relationship matrix, additive plus residual, fixed
effects on the liability scale, maximum likelihood.

**Out, each for a reason**:

- **Proband ascertainment.** GOBS families were not selected on disease, so no
  correction applies to the named analysis. Recorded here so that a later reader
  knows it was considered rather than missed. Ascertainment enters by its own
  amendment, and would need one: it is where the recovered implementation and
  native SOLAR disagree most (h² 0.310 against 0.251).
- **Ordered and multi-level traits.** All four GOBS depression variables are
  binary. Ordered liability exists in the recovered code but has no native
  parity behind it, so adopting it would be adopting the least-supported thing
  available.
- **A household component.** This workspace has just found a real household
  effect on Aβ40, so it is a live question — but household with a liability is a
  research problem, the recovered implementation fails closed on it, and
  bundling it would make this amendment unshippable.
- **Two traits.** A bivariate liability model against a continuous trait is the
  scientifically interesting next step and needs its own amendment. Mixed
  binary/continuous has native parity behind it; binary/binary does not.

## Maximum likelihood, and saying so

Every other model in Asterism defaults to REML. This one cannot have it. REML
removes the fixed effects by projecting the response onto the null space of the
design, and here there is no response to project — the likelihood is the
probability of a region rather than a density at a point.

So the estimator is maximum likelihood and the fit record says `ml`. **A
liability heritability must never be placed beside a REML heritability as though
the two were the same number**, and the record carrying the estimator is what
makes that visible rather than a matter of memory.

## Where the arithmetic is good

The two-person probability is sixteen-point Gauss-Legendre quadrature, and its
accuracy depends on the liability correlation. Measured over a grid: worst error
1.3e-09 below a correlation of 0.5, and 2.3e-04 below 0.99. An additive model
puts sibling and parent-child liability correlations at `h2 / 2`, so an ordinary
pedigree sits in the good range throughout.

A relationship of one does not — monozygotic twins, or one person entered twice,
give a correlation of `h2` rather than `h2 / 2`. `LiabilityModel::build`
therefore refuses an off-diagonal relationship above 0.9 rather than returning a
number quietly worth less than it looks. Removing that guard means replacing the
quadrature first.

## Consequences

- The code is cannibalised from the SOLAR successor at
  `~/Documents/MathiasLab/Studies/Planned/Unaffiliated/Projects/SOLARSuccessor`,
  which Sam intends to delete. No provenance obligation follows it here.
- **The transplant is faithful, and that is not the same as qualified.** The
  likelihood agrees with the SOLAR successor's to ten digits across a grid of
  heritabilities and intercepts on the same four hundred simulated people, and a
  test holds it there now that the source is going. But the region probability,
  the sequential approximation and the ordering were all cannibalised from that
  engine, so this is a transplant checked against its donor rather than an
  independent derivation. Under ADR 0006 agreement proves fidelity only against
  something arrived at separately; native SOLAR is that thing, and it has not
  been run. Nor has the model been calibrated. Until both, it fits and it is
  tested and it is not qualified.
- The intercept sign is this package's own choice and differs from the
  successor's. Here a case is the positive direction, so `Phi(intercept)` reads
  directly as the prevalence a covariate-free model implies. That is worth
  keeping and worth writing down, because a fitted intercept compared across the
  two engines will otherwise look wrong.
- `gobs_mdd_recurr` is not analysed. 88 of its 240 cases are coded with no
  lifetime depression, no current episode and no past episode. That is either a
  coding fault or a variable meaning something other than recurrent major
  depression, and the database carries no codebook to tell which. It is set
  aside pending someone who knows the instrument, not because it is known wrong.
- `gobs_mdd_current` and `gobs_mdd_past` nest exactly inside lifetime and carry
  no independent information, so they are not separate analyses.
