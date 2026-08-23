# 1. How the variance components estimator is parameterised

**Status:** accepted. Nothing has been built.

**Decided by:** Sam, across four rounds of grilling on 10 August 2026 and a
further five on 11 August that settled what Asterism is and is not, rewrote
decision 4 and added decisions 19 to 28. Every entry
below is his answer, not an agent's inference. Where the evidence pointed one way
and Sam chose another, that is recorded. Where a recommendation was made,
accepted, and then overturned by research, that is recorded too.

## Why this document exists

The predecessor attempts failed on scope rather than effort. Asterism produced
107,877 lines in three days and nothing usable; Astrarium produced a working
narrow package in about a day by reusing an existing engine, then sprawled into
twelve unmerged branches. In neither case was the estimator's shape settled
before code was written, so each session decided it again and none of the
decisions agreed.

This file is the fixed point. An agent that disagrees with something here should
say so and change this document with Sam, not work around it.

## What exists today

One model: a single scalar `h2` searched on [0, 1], with the residual variance
profiled out analytically. REML by default, ML available, a 95 per cent
profile-mixture interval, a likelihood ratio test against `h2 = 0` with the
boundary mixture correction, and a standard error from observed information —
interior points only.

It is correct and narrow. It cannot hold a second component, because the analytic
profile, the interval recipe and the standard error all assume the single scalar.

## The evidence the decisions were taken against

- **1,246 SOLAR runs** in the high-frequency heritability project. **None uses
  `house`.** Every one is additive plus residual.
- **86 are bivariate**, testing genetic and environmental correlations; 22 also
  test the phenotypic correlation.
- **284 use `-screen -all`**, which reports covariate significance while keeping
  every covariate. This is reporting, not model selection.
- Covariates are `age_years^1,2#sex` in 1,194 of the 1,246.
- Traits do not cover the same people. In one real bivariate input: 382 subjects,
  but 349, 350 and 361 for three related measures.

## The decisions

### 1. Two jobs, both needed

Taking SOLAR out of the **pedigree gate** and replacing SOLAR for **analysis** are
separate efforts. The gate needs kinship construction and validation, which work
today and involve no estimator at all. Both are wanted; only the gate is small.

### 2. General in both directions: m traits and m components

The evidence said traits only — household has never been run in 1,246 fits, and
`household.rs` sits written and unused. **Sam overruled it:** he intends to run
household models, and numerous later models have three or more components.

### 3. Correctness is independent, not parity with SOLAR

Validate against simulation with known truth. Compare against SOLAR and **record**
the difference rather than treating disagreement as a defect. SOLAR was observed
on 10 August silently inventing a duplicate participant on a clean load, so it is
a comparator, not a definition.

### 4. One product, living in this workspace

**Amended by Sam on 11 August 2026.** As first written this decision said the
engine lands here and *Astrarium depends on it*, leaving Astrarium as the
package. That is no longer the design. **Asterism, at
`staging/studies/existing/safs/projects/asterism`, holds both the engine and the
package. Astrarium and the older Asterism are retired to read-only reference.**

The reason is the count. Three implementations of this one idea already exist
across three generations, and a workspace holding the engine while a separate
repository holds the package keeps that number where it is. Moving both into one
place is the only version of this work that reduces it.

**Amended again by Sam on 11 August 2026, in a second session: this is a fresh
start, not a move.** What stood here said the move was a move with history
rather than a rewrite, because Astrarium's one-trait lane matches native SOLAR
to 8e-09 and passed a 96,000-fit coverage check, so rewriting would be a fourth
generation with extra steps. Measuring the code changed the picture. The part
with numbers behind it is `prepared.rs`, **534 lines**, which imports one
function from `likelihood.rs` and is otherwise self-contained. The crate around
it is 28,274 lines, of which roughly twelve thousand are material this document
puts outside Asterism — 6,337 of them, 41.7 per cent of `likelihood.rs`, being
threshold, ordered, ascertained and Student-t code, plus the record machinery
and the file handling. Copying the crate would import all of that and then
require surgery on it.

**So: a new repository, no imported history, `prepared.rs` and its one helper as
the starting point, and everything else written again.** The fixtures do not
come across either; the dataset checked against is regenerated, because SOLAR
and R can both be reinstalled freely and neither raises a licence obstacle.

What must *not* come across is Astrarium's scope statement — its
`CONTEXT.md` defines the project as replacing SOLAR completely and adds spatial
Matérn kernels, six parent–offspring model families and prospective design
analysis. **The first release is what Astrarium already proved: one trait,
additive plus residual, REML and ML, an interval, and a likelihood ratio test.**
Everything else in this document describes the shape the design must not
foreclose, not the first release.

**A correction to the fact this decision was originally taken against.** It said
`Astrarium/rust/vendor/ml-solar-core/src/` was byte-identical to
`SOLARSuccessor/src/`, so it was a copy rather than a dependency and would drift.
It has since drifted, and the drift is the work worth having: `lib.rs`,
`likelihood.rs` and `python.rs` all differ and `prepared.rs` is new, carrying the
crate's first REML, the sealed prepared model, and four fixes from an adversarial
review. So **the live tree is Astrarium's vendor copy, not SOLARSuccessor's
original**, which is now the historical one of the two, and `prepared.rs` — the
one file the fresh start begins from — is the newest part of it rather than
anything inherited.

**Provenance is settled and does not block the move.** SOLARSuccessor's
`docs/legal_source_constraints.md` established on 16 July 2026 that the published
SOLAR tree is reference-only and not safely reusable, and that the successor is
authored independently from Almasy and Blangero (1998) and documented black-box
behaviour. Every commit in that repository is Sam's. Two smaller things stay
open and neither blocks moving code between Sam's own trees: the successor's own
licence, where the working recommendation is dual MIT/Apache-2.0, and whether the
contributor provenance record that document requires was ever written.

### 5. Cholesky in the interior; nested models at the boundary

**This decision was taken, then overturned by research, and retaken.** The first
version was "optimise a Cholesky factor per component, because the boundary stays
reachable". Reaching it turned out to destroy the inference, for three reasons:

- When a component pins at zero the **whole factor** goes to zero, so m(m+1)/2
  parameters hit the boundary together. The constrained set is the positive
  semi-definite cone, which is **not polyhedral for m ≥ 2**, so the familiar
  binomial mixture weights do not apply. Han & Chang (2008, arXiv:0808.2000) show
  by simulation that for multivariate linkage the weights are neither binomial
  nor chi-squared, and that SOLAR's fixed weights are wrong there.
- The Cholesky map is **unidentified at zero** — column sign flips leave the
  covariance unchanged and the factor is rank-deficient — so the Fisher
  information is singular. This violates the Self & Liang (1987) regularity
  conditions outright, not marginally.
- That singularity also removes the cheap repair. The Silvapulle–Sen Monte Carlo
  weights procedure needs **no model refits at all**, but it needs a non-singular
  information matrix to project against.

Putting the log on the diagonal, as is otherwise standard, fails in the opposite
direction: h² = 0, |rho| = 1 and singular genetic matrices all move to infinity
and become **unreachable**. Sam's own comment on `BIVARIATE_COVARIANCE_STARTS`
already records this.

**So: Cholesky for estimation in the interior, and a component is never left
sitting on the boundary.** Where one wants to pin, refit the nested model with
that component removed and compare the two. Both fits then live at identified
interior points where the information is well behaved.

This converts a degenerate-parameter problem into a model-comparison problem —
which is what gets reported anyway. "Does adding household improve the fit" is
the question a reader wants, not "is this Cholesky element zero".

Ratios — h², rhoG — remain **derived for reporting**, with intervals by profile
likelihood on the derived quantity rather than by delta method. Ratios cannot be
the parameters: for three or more traits, bounding correlations pairwise does not
keep a matrix positive semi-definite.

**The recipe for that interval is `0004`**, which this defers to for one trait —
profile likelihood with the boundary point decided by the Self–Liang mixture,
chosen on 2,000 replicates per cell rather than by argument. The general case
still needs re-deriving, and that work was always implied by decision 2.

**The Cholesky is per component, across traits. It does not constrain how many
components there may be.** Each kernel gets its own factor and they do not
interact:

```
V = Σ_A ⊗ A  +  Σ_C ⊗ H  +  Σ_E ⊗ I
```

| traits | components | Cholesky parameters |
| --- | --- | --- |
| 1 | 2 (A, E) | 2 |
| 1 | 3 (A, C, E) | 3 |
| 2 | 3 | 9 |
| 3 | 2 | 12 |
| 3 | 3 | 18 |

For one trait each `Σ_k` is 1×1, so its factor is the square root of a variance
and the parameterisation is nothing more than a device keeping variances
non-negative. A univariate household model is three numbers with no structure
between them.

**Components multiply freely; traits are what cost.** Both limits in this design
are limits on traits — the ratio parameterisation fails at three or more traits,
and the boundary enumeration of decision 13 is 3^t. Neither is a limit on kernels.

**Computationally it is the other way round, noted on 11 August 2026 and worth
carrying.** The first release owes its speed to diagonalising the relationship
matrix once, per family block, so that every likelihood evaluation is a pass
over a diagonal rather than a factorisation of a dense covariance. More traits
keep that: `Σ_A ⊗ A + Σ_E ⊗ I` retains the Kronecker structure and the same
rotation still applies. **More components destroy it** — the additive and
household matrices share no eigenbasis, so they cannot be diagonalised together,
and each evaluation goes back to factorising a dense covariance.

So the parameter count and the running cost pull in opposite directions, and the
speed measured against SOLAR today — 24 to 808 times, depending on size — is
partly a property of the one-component case. It should not be promised for the
household models decision 2 commits to until it has been measured there.

**One caution about many kernels, statistical rather than parametric.** Three
components require the kernels to be distinguishable *in this pedigree*. Additive
and household are close to collinear where most households hold a single nuclear
family: the data then cannot separate the two variances and the likelihood is
flat along that direction rather than wrong. Report a conditioning diagnostic on
the information matrix. This, rather than the parameterisation, is what will bite
on household models.

### 6. REML by default, ML available, and which one recorded on the result

REML is less biased with fixed effects present, and n is around 352 in the
analysis this was drawn from. Two consequences accepted deliberately: comparisons
against SOLAR will differ **systematically** rather than randomly, and any test
that changes the fixed effects cannot be a REML likelihood ratio.

**Schedule note, not a reason to change the decision:** `likelihood.rs` contains
no REML at all. The bivariate machinery most worth reusing is ML throughout, so
multivariate REML is new work rather than a default flip.

### 7. The model is declared, not built up by calls

A serialisable specification — traits, named components each bound to a matrix,
covariates — passed as one object. A run is reproducible because the model is a
file, and an agent cannot alter a model without the change showing in a diff.

**Decided more precisely in `0003`**, which this defers to: a sealed prepared
model built once, returning a lean frozen record.

### 8. Covariate significance is reported, never acted on

Matching how `-screen -all` is actually used. No automatic selection, ever.
Because of decision 6, covariate significance comes from a **Wald test on the
fixed effect**, not from comparing REML likelihoods.

**Built on 11 August 2026.** Every fixed effect carries a standard error, a z, a
two-sided p-value and a 95 per cent interval, in design-column order. The
variance of the fixed effects is the residual variance times the inverse
weighted cross-product, so this comes off the decomposition the fit already did.
Three things learned in building it, all of which change how the numbers should
be read.

**The inference is available at a boundary fit, unlike the standard error on
h².** Nothing degenerate happens to a fixed effect when a variance component
sits on its bound: the covariance is still positive definite and the estimate is
still asymptotically normal. Withholding these too would have been
over-cautious.

**A covariate p-value depends on how the design is parameterised, and h² does
not.** This was found by comparing against SOLAR and it is the single most
practically important thing here. SOLAR centres age and codes sex as a female
indicator. Centring does not change the design's column span, so h², the
variances and the likelihood are all identical — measured, not argued. But the
*coefficient* on an uncentred age in a model that also carries age squared is
the slope at age zero, while the coefficient on a centred age is the slope at
the mean age. Those are different quantities, so a test of one is not a test of
the other. On the same data the uncentred age gave p = 0.035 and the centred age
gave 2e-11. Reparameterised to match SOLAR exactly, every covariate agreed:
1.98e-11 against SOLAR's 1.01e-10, 0.1101 against 0.1108, 0.00735 against
0.00788, 0.2439 against 0.2445, 0.97001 against 0.97000.

So: **a covariate p-value from Asterism and one from SOLAR are comparable only
if the design is parameterised the same way.** The residual difference above is
Wald against likelihood ratio — SOLAR screens a covariate by refitting without
it, which it can do because it uses ML. Under REML that comparison is not
available, which is the reason this decision specifies Wald in the first place.

**The coverage check scores these intervals too, and they behave.** 0.9485 at
n = 350 and 0.9499 at n = 1400, against a nominal 0.95. The small shortfall at
the smaller size is the Wald approximation being asymptotic — a normal quantile
rather than a t, and a standard error that conditions on estimated variance
components — so it shrinks as the sample grows, which is what the numbers show.
It is not a defect and needs no warning attached to it; a fault in the standard
errors would hold steady or grow with n instead.

What the check is for is exactly that distinction. Keep watching how it moves
with sample size rather than whether it sits below 0.95 at any one size.

### 9. All available data, not complete cases

Subjects keep their measured traits when others are missing. This gives up the
clean Kronecker structure — each subject needs the submatrix matching the traits
they have — and it is built in from the start because retrofitting it is a
rewrite. In a family study, dropping a person also drops the relationships they
carry.

**This is also a live defect, not only a design choice.** SOLAR's default is
`UnbalancedTraits 1`, meaning all available data; the engine's default is
complete-case. They disagree silently whenever traits are unbalanced. The code
path (`union_missingness`) already exists, so the default flip is one line.

Nothing in the JASA manuscript changes: every trait pair it uses has identical
missingness on both traits.

**Report per-trait n alongside the joint n.** SOLAR prints two availability
counts and one analysis size — `Avail 212` and `223`, analysis 223. A table
saying "n = 223" hides that one trait rests on 212 people.

### 10. Per-trait covariates, fixed effects profiled out

A shared covariate set is the common case, but traits may differ, because
pairings across domains are coming. Fixed effects stay profiled out by
generalised least squares at each variance evaluation, as REML does anyway.

### 11. A component is a name and a matrix; builders are extensible

The interface takes a matrix. Builders produce the standard ones — additive from
the pedigree, household from `hhid` — and **more builders will be added over
time**. Marker-based IBD, genomic relationship and dominance matrices then arrive
without the estimator changing at all.

### 12. Three tiers of p-value, and the fallback says so

- **One trait, one component: keep the 50:50 rule.** Already implemented in
  eleven places, and ADR 0003 in Astrarium verified it on 2,000 REML replicates
  at n = 1909 — the mixture was the only candidate inside the binomial band in
  every cell. Two fits per test.
- **The general case: parametric bootstrap.** B = 1000 is 2,000 fits. At n ≈ 350
  that is minutes; the existing harness ran 2,000 fits in about ten seconds at
  n = 1909 by exploiting 202 family blocks rather than one dense matrix.
- **Where that is too dear** (n ≈ 5,400, m = 3, K ≥ 2): calibrate **once per
  design** rather than once per test, as SOLAR already does with `lodadj`. Fifty
  phenotypes on one design cost 40 extra fits each rather than 2,000.

Where no calibration exists, report the naive chi-squared p-value and **mark it
conservative on the record**. Never present a fabricated mixture as exact.

Family block-diagonality in the solver is what decides which tier applies at the
upper end of the scale, so it is built in rather than added later.

### 13. Three traits is the design's ceiling; design for an active set

**Corrected on 11 August 2026.** This decision was headed "three traits in the
first release" and said the first release caps at three. That was written before
decision 4 was amended the same day, and the amendment cut the first release to
**one trait**. Three is the ceiling the *design* must not exceed without an
active set, not what ships. The two statements sat in one document contradicting
each other, which is how an agent comes to build multivariate machinery believing
it was asked for.

`correlation_h2_constraints` enumerates nine boundary states for two traits.
That is 3^t — twenty-seven at three traits, eighty-one at four, before
rank-deficient genetic matrices are considered. Enumeration is therefore the
deliberate exact solution for the admitted two-trait model, not the general
multivariate design. A later general estimator uses an active set before a
fourth trait is admitted; the current nine-state implementation is not copied
upward.

**Two traits is a deferred capability and enters under decision 20**, by an
amendment naming the analysis that needs it. Naming one would be easy: 86 of the
1,246 recorded runs are bivariate. What it takes is not small, and none of it is
a port — the Cholesky parameterisation of decision 5, analytic REML gradients
and the bound-constrained search of decision 14, boundary handling by nested
refit, an interval recipe re-derived for the derived quantities rather than the
scalar one `0004` calibrated, bootstrap p-values under decision 12, and its own
independent check under `0006`. Decision 6's schedule note still holds:
Astrarium's bivariate machinery is ML throughout, so bivariate REML is new work.

One thing that does carry: the eigen-rotation survives more traits, so the speed
would too.

### 14. Bound-constrained quasi-Newton with deterministic starts

**Amended on 11 August 2026 after implementing and measuring the two-trait
search.** The original decision prescribed a full BFGS approximation followed
by a Newton polish. Neither is now part of the convergence claim. The retained
L-BFGS-B search, with six curvature pairs for six parameters, reaches the
independently recomputed projected KKT tolerance; the attempted hand-written
full approximation did not. A Newton step will be assessed when the Hessian is
implemented for standard errors, but it is not required merely to relabel a
point that already meets the stated first-order condition.

The direct `(h², ρ)` coordinates are smooth only while the component exists.
The implementation therefore enumerates the `3² = 9` lower/interior/upper
heritability states required by decision 13. Each exact state optimises only its
identified coordinates: `ρ_G` is absent if either genetic variance is zero and
`ρ_E` is absent if either residual variance is zero. The fully interior state
uses the representable epsilon-open interval, while correlations retain their
exact `±1` bounds. State selection compares the likelihoods and favours the
lower-dimensional state only inside an explicit numerical objective tie. The
comparison is evaluated on one reversible trait-standardised scale and the
reported likelihood transformed back exactly; response units therefore cannot
change what counts as a tie.

**Bounds on the Cholesky diagonal, `diag(L_k) ≥ 0`.** Sam asked for L-BFGS-B and
was right about the bounds, for three reasons — one of which had been missed when
decision 5 was taken:

- It **removes the non-identifiability**. A factor and the same factor with a
  sign-flipped column give the identical covariance; with a non-negative diagonal
  the factorisation is unique.
- It **keeps the boundary reachable**, unlike the log-diagonal trick that is
  otherwise standard and that pushes h² = 0 and singular genetic matrices to
  infinity.
- **The active set is the detector.** A diagonal element pinned at its bound is
  the signal that the component wants to drop out, which is exactly the trigger
  for the nested refit in decision 5. The constraint does not merely keep the fit
  legal; it says which comparison to run.

**The memory is the parameter count.** The original argument rejected limited
memory because a full 18×18 approximation is cheap. That overlooked the hard
part: a reliable bound-aware line search and active set, not storage. At two
traits the retained solver keeps six curvature pairs for six parameters. It is
still L-BFGS-B, but it does not omit a parameter-space direction merely to save
memory, and its adequacy is decided by the independently recomputed KKT measure
rather than by the solver's termination message.

**Analytic REML gradients, not numerical**, at every trait count.

The choice only genuinely arises at three traits and beyond, and that is where
numerical is worst. Univariate uses no gradient at all — a one-dimensional search
over h² with the residual variance profiled out. Bivariate already has analytic
gradients: `transformed_bivariate_objective` returns value and gradient together
from `evaluate_with_constraint`, with no finite differencing in that path.

Cost is the obvious argument and it does scale badly — central differences need
2p likelihood evaluations, where p = K × t(t+1)/2, so twelve evaluations per
gradient at two traits and two components, and thirty-six at three and three.
But two further arguments do not scale away, and they are the ones that decide it:

- **The convergence test is the scaled gradient maximum against a tolerance**,
  7.3e-8 in the G04 sweep. A numerical gradient cannot certify convergence below
  its own noise floor, so the reported tolerance would be a claim the method
  cannot support.
- **Mixing methods by trait count gives two convergence semantics**, and results
  that are not strictly comparable across them — which matters when one paper
  reports a univariate h² and a bivariate rho_G from the same engine.

The trivariate gradient is a generalisation of the bivariate one already derived,
not a fresh derivation, so staying analytic costs much less than it appears to.

**A later Newton assessment uses the Hessian taken as a central difference of
the analytic gradient.** The architecture `bfgs_bivariate` already uses that at
`likelihood.rs:11654`, which carries to t traits unchanged. It belongs with the
standard-error work, where the Hessian is needed in any event; whether a Newton
step materially improves an already KKT-qualified point is measured there
rather than assumed here.

To be unambiguous, because the two are easily confused: the first derivatives are
exact and derived by hand; only the *second* derivatives are differenced, and they
are differenced from something exact rather than from the likelihood. That keeps
roughly half the precision digits of the gradient, which is ample at the tolerance
above. Analytic second derivatives — full Newton throughout — remain the
alternative, and are worth the considerable extra algebra only if the differenced
Hessian turns out to be what limits accuracy.

**Consequence to carry:** standard errors come from that Hessian, so they inherit
its finite-difference error. At interior points this is immaterial. It is a
further reason the boundary is handled by nested comparison rather than by a
standard error, which would be doubly unreliable there — degenerate *and*
differenced.

**Not AI-REML**, despite being what a reviewer from animal breeding would expect.
Its trace cancellation only pays when V is dense and n×n, and the eigen-rotation
already makes exact traces cost O(n·t³) — approximating something computable
exactly.

**Starting values: three deterministic patterns, not a random cloud.** The
original decision prescribed marginal univariate fits as the first start. That
does not cover every design admitted here: a coefficient may be shared across
traits, or a joint design may lose rank when restricted to either marginal.
After reversible trait standardisation, unit total variances and the three fixed
heritability/correlation patterns are meaningful in every admitted design. Each
is projected into every exact heritability state. The retained thirty-seed ML
and REML stress test makes those starts accountable; adding more starts without
a failing case would add cost rather than evidence.

**Dependency note, overtaken on 11 August 2026.** This said the engine had no
optimisation crate and a hand-written BFGS to extend, so extending it would be
cheaper than a dependency. The fresh start of decision 4 brought across only
`prepared.rs`, so there was nothing to extend, and the one-trait fit needs no
optimiser at all — it grids thirty-three points across h² and polishes with
golden section, which is why it never had this problem.

Two traits made an optimiser necessary for the first time. A hand-written
projected BFGS and then `lbfgsb-rs-pure` were measured on the same deterministic
unbalanced problem. The former stalled and the latter abandoned all three
starts with line-search failures, leaving scaled free-coordinate scores between
0.07 and 0.11. Neither is retained. `rcompat-lbfgsb`, a safe-Rust implementation
of the algorithm used by R's `optim(method = "L-BFGS-B")`, reaches Asterism's
independently recomputed `1e-7` projected KKT criterion for both ML and REML on
that problem, including an exact correlation-bound solution. **That dependency
is kept, on this measured comparison.**

A caution worth carrying from how that comparison nearly went wrong. The
hand-written search was first blamed for a failure in the objective callback.
A component covariance at h² = 0 or 1 has an unidentified correlation in the
direct coordinates; at |ρ| = 1 it is merely rank deficient, and the full
observation covariance may remain positive definite. The callback nevertheless
returned a large value with a zero gradient at invalid full-covariance corners,
which tells a quasi-Newton method it has found a stationary point. The retained
callback instead gives an invalid trial a matched inward penalty and gradient,
and convergence is claimed only after independently re-evaluating the analytic
score at a valid returned point.

### 15. One general estimator, with fast paths where volume justifies them

**There is no univariate exception.** One trait with three kernels has two free
ratios after the total variance is profiled out, and needs exactly the machinery
that three traits with two kernels needs. The existing scalar search is not "the
univariate method" — it is the K = 2 special case, and it was mistaken for a
general statement when decision 14 was first written.

**The general estimator is the definition of correct.** A fast path is an
optimisation, never an alternative definition. Three earn their place on volume:

- **One trait, additive and residual.** Roughly 1,160 of the 1,246 recorded SOLAR
  runs. Already written and already calibrated — ADR 0003 in Astrarium verified
  its interval recipe on 2,000 REML replicates at n = 1909.
- **Two traits, additive and residual.** The remaining 86, and `bfgs_bivariate`
  already exists with analytic gradients.
- **Many fits against one prepared model.** The largest win, and structurally
  present already: `PreparedModel` holds the eigenvalues, the rotated design, the
  per-family rotations and `logdet_xtx` — everything independent of the response —
  while `fit` takes the response separately. A parametric bootstrap changes only
  the simulated response, so all 2,000 fits share one preparation. This is what
  decides whether tier 2 of decision 12 is affordable at the upper end of the
  scale, and the per-family rotations are the block-diagonality that makes it so.

**The discipline that keeps this from becoming two engines.** Mixing methods was
argued against in decision 14 for producing two convergence semantics and results
that are not strictly comparable; fast paths reintroduce that risk unless held to
three rules:

1. Every fast path must agree with the general estimator to a stated tolerance,
   and that agreement is **enforced by test**, on simulated and on real data.
2. Where they disagree, the general estimator is right and the fast path has a
   bug. Never the reverse.
3. **The result records which path produced it.** If a fast path is later found
   wrong, the affected results must be identifiable rather than guessed at.

Rule 3 is the one most easily dropped and the most expensive to have dropped.

**These three rules are `0006` arrived at separately**, and `0006` is where the
principle is stated with the measured tolerances behind it. Read them together:
agreement proves fidelity and never correctness, so a fast route agreeing with
the general estimator says only that the shortcut was taken faithfully.

**Consequence to carry:** the profile-mixture interval recipe of ADR 0003 was
derived for the scalar case and needs re-deriving for the general one. That work
was always implied by decision 2; it was hidden behind the word "univariate".

### 16. Gaussian only

The general estimator assumes a Gaussian likelihood. The threshold, ordered-
threshold, ascertained and Student-t objectives already in `likelihood.rs` stay
as **separate one- and two-trait special cases** and are not folded into the
general design.

Everything above assumes Gaussian: REML has no clean meaning for a liability
model, fixed effects cannot be profiled out by least squares, and the boundary
geometry changes again. Generalising liability to m traits and m components is a
research problem rather than a build — and the two Astrarium branches on
liability identifiability suggest it already behaved like one.

Consistent with practice: all 1,246 recorded runs analyse an `inormal_`-
transformed trait. The non-normality is handled before the estimator sees it.

### 17. Profile only what is reported

At three traits and two components the derived quantities number nine — three h²,
three rho_G, three rho_E — and more with a third component. Each profile-likelihood
interval costs twenty to forty constrained fits, so all nine is two to four
hundred fits per model on top of a bootstrap already wanting two thousand.

**Compute intervals only for the quantities being reported, and record which.**
The delta method stays available and any interval built with it is marked as
such; it is not the default, because it is worst exactly where these estimates
live — near boundaries and for correlations near ±1.

### 18. The seed is on the record

A bootstrap p-value depends on its random numbers, and a paper quoting one needs
it to return next year. **A seed is recorded on every result**, derived
deterministically from the model specification.

**The Monte Carlo standard error is reported beside the p-value**, and no more
precision is quoted than it supports. B = 1000 gives roughly 0.007 on a p-value
near 0.03 — enough for "below 0.05", not enough for three decimal places.

## The second session, 11 August 2026

A further round of grilling, on what Asterism is and is not. Decision 4 above was
rewritten by it. These are Sam's answers, and where a recommendation of mine was
overruled or was simply wrong, that is recorded.

### 19. Quantitative genetics software, defined by what it does

Not by SOLAR. Decision 3 says SOLAR is a comparator rather than a definition of
correctness; it is not the definition of the project either. Every predecessor
that defined itself against SOLAR inherited SOLAR's whole feature list as an
obligation, and Astrarium's `CONTEXT.md` is what that looks like written down.

An attempt to sharpen this into a formal definition — covariance as a sum of
*known* matrices, which would have excluded spatial and gene-by-environment
covariance functions by construction — **was rejected.** Asterism will fit those
eventually, and a definition that rules them out would be wrong on the day they
arrive. Nothing is excluded permanently.

### 20. This document is the boundary, and it moves only by amendment

What Asterism is at any moment is what this document describes. A capability
enters when a planned analysis needs it and the amendment naming that analysis
exists — before any code for it does. Deferred capabilities are recorded without
order, priority or dates, because a list with an order is a roadmap.

### 21. The non-Gaussian code does not come across

6,337 lines of `likelihood.rs`, 41.7 per cent of it, are threshold, ordered,
ascertained and Student-t objectives. Under decision 16 these are outside the
general design, so they are not carried into the fresh start. They return by
amendment under decision 20, like anything else.

### 22. One crate named `asterism`

`ml-solar-core` is retired with the rest. One product, one version, one name: a
crate named `asterism` with a Python interface around it. A crate carrying
"solar" in its name reads as a SOLAR port to anyone who looks, including anyone
asking the provenance question.

### 23. Internal, in its own repository, with a remote

An internal tool. It must be citable and reproducible when a paper uses it — a
licence, a version and an archived copy at publication — but it has no users to
support and no interface that must stay stable. It gets its own git repository
with a private remote, which none of the three predecessors has.

### 24. Tables in, a record out

As amended by ADR 0017, Asterism takes tables or arrays already in memory and
returns a fit record; a seed is present only for a stochastic calculation. Its
numerical fitting classes write nothing. The caller-side analysis wrapper may
write the standard receipt beside controlled outputs. Asterism knows nothing of
the input file formats, databases or the tables that go into papers. Builders
that make a standard relationship matrix are a convenience, not the boundary,
and curating SAFS pedigrees stays with the `pedigrees` project.

### 25. The qualification vocabulary is dropped

Gates, ratification, qualification and phases named nothing real: there is one
person here and no committee, and 31 of Astrarium's 53 commits begin "Claim",
"Resolve" or "Record". Two ordinary things survive the cut — comparing against
SOLAR and R on a fixed dataset, and the coverage check, which is what chose the
interval recipe and is the only thing that can show an interval is too narrow.
Any capability added arrives with its own independent check, which is what makes
decision 3 a rule rather than a sentiment.

### 26. The predecessors are not preserved

Astrarium, the older Asterism and SOLARSuccessor are reference for as long as
they last, and nothing depends on their surviving. What matters is this version
running correctly.

### 27. The port crate is a resource, not a route in

`staging/projects/variance-components-rust-ports`, written by Codex on 11 August,
is disposable work to be cannibalised freely. Nothing enters Asterism because it
was ported there; entry is by amendment under decision 20. Its `edge_gaussian`
module is worth reading first: `profile_and_differentiate` is a profiled ML and
REML objective returning an analytic gradient with the fixed effects profiled
out, which is decisions 6, 10 and 14 written fresh and small, and Astrarium has
no REML at all outside `prepared.rs`.

### 28. Where it lives

`staging/studies/existing/safs/projects/asterism`, because SAFS is the immediate
user. It is lab software, and the location says nothing about its scope.

### 29. Two traits are admitted, for the JASA reanalysis

**Sam, 11 August 2026.** The first capability to enter under decision 20, and
the amendment naming the analysis that needs it.

**The analysis.** The JASA high-frequency heritability paper, whose analysis is
being redone on refreshed data once the Acoustic ingest is back. It carries nine
main bivariate runs, twenty-three ear-to-ear ones, and a standard-versus-extended
comparison. Its own summary table says exactly what bivariate has to produce:

- a heritability for each trait, with a standard error;
- the environmental correlation, with a p-value;
- the genetic correlation, with **three** p-values — against zero, against plus
  one and against minus one — and a boundary state;
- the phenotypic correlation.

**The setting is hard, and that is the point.** The recorded run has n = 352 and
reports a genetic correlation of 0.215 with a standard error of 0.588. The
estimate is nowhere near its own precision, the boundary is reachable, and one
row already carries a boundary state. A bivariate estimator that behaves only at
interior points would be useless here.

**What carries over and what does not.** The eigen-rotation survives more traits,
so the speed does. Nothing else is a port: decision 6's schedule note still holds
that Astrarium's bivariate machinery is ML throughout, so bivariate REML is new
work, and `0004`'s interval recipe was calibrated for a scalar and does not
transfer to a derived quantity.

**Five decisions, settled by Sam on 11 August 2026 before any code was
written**, because deciding them afresh in each session is how the three
predecessors failed.

**Both estimators from the start.** Not ML first and REML later. ML gives an
external comparator immediately, since SOLAR's bivariate output is ML and the
paper's existing numbers are ML. REML is the default under decision 6 and R's
`regress` is its comparator. Shipping one without the other would leave half the
package with nothing to check it against.

**Intervals by profile likelihood on each derived quantity, calibrated before
use.** `0004`'s recipe was chosen for a scalar on 2,000 replicates and does not
transfer. The two heritabilities, the genetic correlation, the environmental
correlation and the phenotypic correlation each get a profile interval, and the
coverage check is extended to score them before anything is reported from them.

**Unbalanced traits immediately and by default**, not as a later option. Decision
9 already says retrofitting this is a rewrite, and the recorded inputs force it:
382 subjects but 349, 350 and 361 across three related measures. It breaks the
clean Kronecker structure, so it shapes the code rather than sitting on top of
it.

**The correlations are derived quantities, and the model is reparameterised so
that each can be tested directly.** Sam's instruction, and it is what makes the
tests tractable rather than exotic. Carried as a free parameter, a correlation
supports an ordinary likelihood ratio test against a constrained refit. The
phenotypic correlation is included on the same footing — twenty-two of the
eighty-six recorded runs test it.

**The nulls for the genetic correlation.** Against zero, an ordinary likelihood
ratio against chi-square on one degree of freedom: zero is interior. Against plus
or minus one, the **Self–Liang 50:50 mixture**, because with the correlation
carried as a free parameter and both genetic variances positive, plus or minus
one is a single parameter on a smooth one-sided boundary — the well-behaved
case. Decision 5's objection that the mixture weights fail for m ≥ 2 is about a
whole Cholesky factor collapsing at once, in a cone that is not polyhedral; one
correlation reaching its bound is not that.

**That mixture is verified by simulation before anything is reported from it**,
exactly as `0004` chose the scalar recipe by measurement rather than argument. If
it does not hold at the boundary, the parametric bootstrap of decision 12 is the
fallback, and we will know instead of assuming.

## The five decisions adopted from Astrarium

Astrarium settled five things on 7 August 2026 that several decisions above
re-derived less specifically, which is exactly the drift two numbered series
produce. They are now `0002` to `0006` in this one series, adopted for their
substance with their wording brought into line with `CONTEXT.md`. Where they
overlap with anything above, **they win**: they are more specific and were
settled on measured evidence.

- `0002` — validation happens once on the prepared model, and cannot be switched
  off.
- `0003` — the interface is a sealed prepared model returning a lean frozen
  record. This is decision 7 and the third fast route of decision 15, decided
  more precisely, and it ratifies the REML default that decision 6 re-derived.
- `0004` — the interval recipe. Decision 5's interval clause defers to it for one
  trait; the general case still needs re-deriving.
- `0005` — boundary reporting. This governs how decision 5's nested comparison is
  presented.
- `0006` — agreement proves fidelity, never correctness. This sharpens decision 3,
  and decision 15's three rules for fast routes are the same principle arrived at
  separately.

The two series are now one. That was a prerequisite for building and it is done.

## Provenance of the facts behind these decisions

The SOLAR statements come from the local documentation snapshot at
`Resources/Software/SOLAR-Eclipse/docs/` in the old workspace. **SOLAR's default
starting h² of 0.5 does not** — that comes from a comment in Sam's own code, so
it should not be quoted anywhere without checking. Statements about ASReml,
WOMBAT, BLUPF90 and GCTA were general knowledge rather than anything on this
machine, and none of them is load-bearing for any decision above.

## Where this file lives

**Done on 11 August 2026.** This file was written at the workspace root and moved
here when the repository was made, because decisions describe the estimator
rather than the workspace, and software that has to be citable should carry its
own reasoning. Astrarium's five came with it as `0002` to `0006`, and the
workspace root keeps a signpost.

One thing the glossary settles that this file does not. **Component** is the term
for an estimated variance contribution, and decisions 5, 11 and 12 above also
say "kernel" for the same thing. That is the drift a glossary exists to stop.
The word is left alone rather than edited out, because Asterism will fit
covariance functions carrying their own parameters eventually, and "kernel" is
the natural name for those when they arrive.
