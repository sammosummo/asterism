# Asterism

Quantitative genetics software for the Mathias Lab: one Rust crate named
`asterism` with a Python interface around it, fitting variance components
models. It is lab software, kept here because SAFS is its immediate user.

Read [[staging/studies/existing/safs/projects/asterism/CONTEXT|CONTEXT.md]] for
the vocabulary before writing anything here, and `docs/adr/` for the decisions.

## What it does today

Six families of model, all of them one or two traits on a pedigree, all fitted
by REML or ML, and all calibrated before being used on anything real.

| model | what it adds | reports |
| --- | --- | --- |
| one trait | additive and residual variance | h², interval, test against no additive variance |
| one trait, several components | any number of extra covariance matrices — household, dominance, whatever is handed in | a share and interval for each, a boundary test for each |
| one trait, spatial | a kernel `exp(-λd)` whose range is estimated rather than chosen | the spatial share and interval, the range, a bootstrapped p-value |
| one trait, spatial, range integrated out | the range averaged over rather than maximised over | the share and interval, no range at all |
| two traits | joint fit, unbalanced | both h², genetic, residual and phenotypic correlations, an interval and a test for each |
| one trait, gene by environment | a surface on the genetic and residual covariances, in an environment measured per person | h² at each environment asked about, the genetic correlation between each pair, and two tests |

Which to reach for is a statistical question rather than a menu. The one-trait
model diagonalises the relationship matrix once per family block and is
enormously faster than the rest; the multi-component model gives that up,
because two structured matrices share no eigenbasis; the spatial model gives up
the family blocks entirely, because a distance kernel couples everybody.

**The spatial range is the one thing here that is not really estimated.** Its
interval reaches a bound in 98 per cent of calibration replicates: it covers
because it is wide, which is coverage without information. Report the share and
the p-value; report the range as a point estimate or integrate it out. The
integrated version exists because it is the better answer — with the range
integrated over, the share's uncertainty includes not knowing it, where
otherwise the share is conditional on an estimate that came back at 6 km when
the truth was 35.

**The gene-by-environment model comes in two surfaces, and the choice is not
free.** One makes the variances log-linear in the environment with genetic
effects correlating as `exp(-λ|Δ|)`; the other puts a smooth quadratic on each
covariance, held by its Cholesky factor so it stays a covariance. They report
the same quantities. But neither family contains the other, and a surface that
cannot bend its variance function the way the data does will bend its
correlation instead: on a rank-one genetic surface with no reordering at all,
the exponential form rejected the correlation null on 14 per cent of samples
against a nominal 5. Choose the surface before seeing the answer. Fitting both
and reporting whichever rejects is not a procedure. Only the smooth surface can
represent a crossover, where a genotype that helps in one environment harms in
another: the exponential kernel is positive at every rate, so it reports a
correlation near nought where the truth is near minus one.

**A genetic correlation below one is not on its own evidence of an
interaction.** The estimate cannot exceed one, so under the null every departure
runs downward; a tenth of null samples came back below 0.25 on the exponential
surface. The tests exist for this reason, and unlike the spatial model they need
no bootstrap — at either null every remaining parameter is still identified, so
a mixture of chi-squares is a real reference rather than a hopeful one.

**The class-weighted kinship model needs no code of its own.** It multiplies the
four classes of direct parent–offspring cells by class-specific weights, which
is linear in the weights, so splitting the relationship matrix by class and
handing the pieces to the multi-component model fits it exactly. Report each
class as a share of the total variance and not as a weight: a weight is a ratio
of two estimated variances, and on the GOBS design the ratios come back near 3
when the truth is 1 while the shares are unbiased. An independent qualification
gate failed this model on that bias in 2026 and was right to.

**Real traits have gone through all of it.** Thirteen GOBS traits with one
component, seventy-eight pairs with two, thirteen with a household component,
and a spatial run in progress. The bivariate fits reproduce SOLAR on every one
of the seventy-eight pairs to better than 1e-5.

Each model family is an object built from its matrices and then fitted, and a
fit returns a dictionary with named fields rather than a tuple::

```python
import asterism

# one trait, several components -- household here
model = asterism.ComponentModel([relationship, household], x)
fit = model.fit(y)                    # shares, variances, log-likelihood
model.interval(y, component=1)        # 95 per cent profile interval
model.test(y, component=1)            # against no household variance
fit["fixed_effects"]                  # BLUE, with standard errors
model.predict(y, component=0)         # BLUP, with prediction errors

# two traits
both = asterism.BivariateModel(k, observed, design)
both.fit(y)["rho_g"]
both.test(y, "rho_g", null=0.0)

# one trait with a spatial component
space = asterism.SpatialModel([relationship], distance_km, design)
space.fit(y, integrated=True)         # the range averaged over, not estimated
space.bootstrap(y, replicates=199)    # the only honest p-value here
space.predict(y, component=0)         # BLUP, with the range profiled

# one trait whose genes may act differently across an environment
gxe = asterism.GxeModel(relationship, exposure, design, "random_regression")
fit = gxe.fit(y, grid=[low, middle, high])   # h2 and rho_G at those environments
gxe.test(y, "correlation")            # the same genes throughout?
gxe.test(y, "interaction")            # the environment in the genetics at all?

# the class-weighted kinship split, for ComponentModel
split = asterism.kinship_classes(ids, father, mother, sex, keep=measured)
asterism.ComponentModel(split["matrices"], x).fit(y)["shares"]
```

**Only `prepare` diagonalises.** The rest factorise a covariance on every
evaluation, which is why one trait with one component is enormously faster and
why it stays the right thing to use when one component will do.

```python
import asterism

k, order = asterism.relationship_matrix(ids, father, mother, keep=measured)
model = asterism.prepare(x, k, subject_ids=order)   # validates once
record = model.fit(y)                               # as often as you like
```

`ids`, `father` and `mother` are parallel lists, one entry per person in the
pedigree, in any order — parents are sorted before their children. `keep` names
the people to give rows to, in the order you want them; their ancestors still
carry the relationships without needing rows of their own. The builder returns
that order, and handing it to `prepare` as `subject_ids` commits it: the fit
record echoes a hash of it, so a response lined up wrongly cannot pass quietly.

`x` is the fixed-effect design in that same order, including its own intercept
column if one is wanted. `k` is the relationship matrix. The builder gives you
twice the kinship, which is the usual choice, but `prepare` will take any valid
matrix — a genomic relationship matrix is equally acceptable, because the matrix
is checked as mathematics rather than for where it came from.
`record` is a dictionary held in memory. It carries h² with a 95 per cent
profile interval and a likelihood ratio p-value against no additive variance,
and every fixed effect with a standard error, a z, a two-sided Wald p-value and
a 95 per cent interval, in design-column order. Asterism reads no files and
writes none.

Preparing once and fitting many times is the point rather than an optimisation:
the decomposition is what the cost lives in, and a bootstrap changes only the
response.

## Speed

`uv run --no-project python checks/speed.py`, one heritability analysis from
pedigree to result, six fixed effects, on this machine:

| n | SOLAR | Asterism | ratio | a second trait, same pedigree |
| --- | --- | --- | --- | --- |
| 350 | 2.03 s | 0.003 s | 808× | 0.0013 s |
| 1,400 | 2.90 s | 0.018 s | 161× | 0.0049 s |
| 5,600 | 11.04 s | 0.455 s | 24× | 0.0199 s |

Both columns cover the same work: build the relationship matrix, take the
phenotype, fit. The ratio narrows as the pedigree grows, because SOLAR's cost is
mostly fixed overhead while Asterism's is real arithmetic.

The last column is the one that matters for a batch. A prepared model
decomposes once, and another trait only refits. SOLAR has no equivalent and
repeats everything each time.

**A prepared model fixes the roster**, so that reuse holds only across traits
measured on exactly the same people with the same covariates — a response of a
different length is refused rather than silently accepted. Traits rarely cover
the same people, so in practice you prepare once per distinct roster and not
once per study. At SAFS size this hardly matters, because preparation is under a
millisecond and a whole analysis is about three; at n = 5,600 preparation is
0.44 s against a 0.02 s fit, and there it is worth grouping traits by roster.

That margin is also what makes the parametric bootstrap of decision 12
affordable — 2,000 fits against one prepared model, not 2,000 analyses.

**REML and ML cost the same**: measured at 1.004, 1.001 and 1.001 across those
three sizes. Both share the whole expensive part, and REML adds only the log
determinant of the weighted cross-product, which is p logarithms against n·p²
multiply-adds. Choose the estimator on statistical grounds; the default costs
nothing.

## What it is not

It grows when a planned analysis needs it to and not before, and every
capability it gains is written into `docs/adr/` first. Liability and threshold
models, survival, gene-by-environment, longitudinal, Tobit, signal detection and
prospective design analysis are all deferred. Deferred means not yet,
not never, and the list carries no order. Spatial has left that list because an
analysis needed it.

SOLAR is a comparator. Differences between the two are recorded rather than
treated as defects, and replacing SOLAR is not what defines this.

## Building and testing

A child `uv` environment, separate from the workspace root:

```sh
uv venv --python 3.13
uv pip install 'maturin>=1.10,<2' numpy pytest
uv run --no-project maturin develop --release   # after any change to the Rust
```

Then:

```sh
cargo test --release                       # the estimator, against simulated truth
uv run --no-project pytest tests/ -q       # the Python interface
cargo run --release --bin coverage         # the coverage check, about 20 seconds
uv run --no-project python checks/against_r.py      # REML against R `regress`
uv run --no-project python checks/against_solar.py  # ML against native SOLAR
```

`--release` matters for the Rust tests: one of them is a reduced coverage check
and it takes forty seconds unoptimised against under one optimised.

Both sets of Rust tests check recovery of a heritability the simulation
controls, rather than agreement with another implementation — agreement proves
fidelity and never correctness. They use two pedigrees deliberately. The
estimator tests use sibling pairs, the least informative design in common use
and so the honest one to set tolerances against; the coverage check uses
three-generation families, because an interval has to behave across the range of
relationships a real study carries.

## What came before

Three implementations of this idea already exist, and this is a consolidation
rather than a fourth attempt:

- `~/Astrarium` — a working one-trait package in Python over Rust, checked
  against native SOLAR and R. Its `prepared.rs` is where this started.
- `~/Asterism` — an earlier Python attempt, 107,877 lines in three days.
- SOLARSuccessor — the original engine, now historical, surviving only inside a
  cargo cache.

**This was a fresh start, not a move.** Astrarium's `prepared.rs` and the one
function it imported came across, about six hundred lines, and everything else
is written again. The three predecessors are reference for as long as they last
and nothing depends on their surviving.

`staging/projects/variance-components-rust-ports` is disposable work written by
Codex on 11 August 2026 and may be cannibalised freely. Its `edge_gaussian`
module holds a profiled ML and REML objective with an analytic gradient, which
is the shape the general estimator needs. Nothing enters Asterism because it was
ported there.

## The coverage check

`cargo run --release --bin coverage` simulates datasets whose heritability is
known, fits every one and counts how often the 95 per cent interval contains the
truth. It is the only thing that can tell you an interval is too narrow, and no
amount of reading the code substitutes for it.

It runs with the design the lab actually uses — an intercept, age, age squared,
sex and the two age-by-sex products — because an intercept-only check leaves the
restricted likelihood's determinant term unexercised, which is the part most
likely to be wrong. Pass `intercept` as a third argument to drop back to one
column, which is only useful for isolating whether the covariates caused a
difference.

**All twelve cells pass at n = 1400. Ten of twelve pass at n = 350**, the
exceptions being true heritabilities of 0.05 and 0.07, which over-cover at 0.979
and 0.974.

The reason is worth knowing before you report an interval near zero. The
maximiser is constrained to [0, 1], so when the unconstrained estimate would be
negative the fit stops at zero and the likelihood ratio statistic comes out
smaller than it otherwise would. A statistic that is too small, compared against
a χ²₁ critical value, falls below it too often, and the interval contains the
truth too often. The effect tracks how often the estimate pins: 28.6 per cent of
the time at a true 0.05 with n = 350, 10.1 per cent at n = 1400, and the
conservatism shrinks with it.

**So intervals near zero are wider than they need to be, and the cost is power
rather than validity** — a real but small heritability gets called
non-significant more often than it should. It cannot be repaired by narrowing
the interval, because coverage is already exactly right at interior truths.
`docs/adr/0004` sets out what a proper fix would take.

Results and seeds are in `evidence/`.

## Reading a covariate p-value

Two things about them, both found by comparing against SOLAR and both changing
how the number should be read.

**A covariate p-value depends on how you parameterise the design; h² does not.**
SOLAR centres age and codes sex as a female indicator. That does not change the
design's column span, so h², the variances and the likelihood come out
identical — measured, not argued. But the coefficient on an uncentred age in a
model that also carries age squared is the slope at age zero, while the
coefficient on a centred age is the slope at the mean age. Different quantities,
so different tests. On one dataset the uncentred age gave p = 0.035 and the
centred age 2e-11. Reparameterised to match SOLAR exactly, every covariate
agreed with it. **So a covariate p-value from Asterism and one from SOLAR are
comparable only if the design is parameterised the same way.**

## The real pedigree

Run on 11 August 2026 against the reviewed SAFS pedigree, `latest_reviewed_pedigree.csv`:
5,364 people, 1,920 founders, 283 families, the largest 485 people and the
median family a single person. Twelve identical-twin labels and six inbred
individuals, the highest diagonal 1.0625, which is parents who were first
cousins. Not one record was refused, and nobody in it has one known parent and
not the other.

| roster | build | prepare | fit | another trait |
| --- | --- | --- | --- | --- |
| whole pedigree, n = 5,364 | 0.23 s | 0.30 s | 0.011 s | 0.011 s |
| GOBS-sized, n = 1,909 | 0.03 s | 0.01 s | 0.004 s | 0.004 s |
| JASA-sized, n = 352 | — | — | 0.001 s | 0.001 s |

**The relationship matrix is singular**, which matters. The smallest eigenvalue
is −1.5e-15 over the whole pedigree and −2.0e-16 over a GOBS-sized roster:
numerically zero, so twice the kinship is positive semi-definite but not
positive definite. `prepare` accepts it — that is what the −1e-9 eigenvalue
floor is for — and the fit converges.

The consequence to carry is that **h² = 1 is unreachable on a singular matrix**.
The covariance there has zero entries and is refused, so a fit that wants the
upper bound reports itself as not converged rather than returning a number. That
is the case the upper-bound snap in `prepared.rs` was written for, and this is
the first time it has met real data.

## The comparisons against SOLAR and R

`docs/adr/0006` puts correctness in external comparisons with a fixed division
of labour — SOLAR for ML, R `regress` for REML — because agreement between two
implementations proves fidelity and never correctness. Both are run, both on the
six-column design.

- **REML against R `regress`**: heritability to about 5e-9 relative, total
  variance to 4e-9, every fixed effect to 1e-9. Two traits agree too, against
  `regress` handed the six covariance structures directly.
- **ML against native SOLAR**: heritability to about 1e-8 relative, which is
  every digit SOLAR prints, and standard errors agreeing to five figures despite
  being computed differently. Two traits agree on unbalanced simulated data to
  2e-7, and on all seventy-eight real GOBS trait pairs to better than 1e-5.

**Real data agrees about an order of magnitude less closely than simulated
data, and that is SOLAR.** Its termination on heritability is looser than its
printing — log-likelihoods equal to 1e-8 while heritabilities differ by 8e-7 —
so a tolerance calibrated on simulated data is too tight for a real comparison.
Expect around 1e-6 and treat a tighter demand as a statement about SOLAR's
stopping rule.

**Neither computes a profile interval**, so the intervals have no external
comparator at all. The two-trait ones are checked against an independent Python
implementation instead; the others rest on calibration.

**Their printed log-likelihoods will not match Asterism's, and that is
expected.** Asterism keeps the Gaussian normalising constant; both comparators
drop it. Subtract n/2 · log(2π) to reach SOLAR's convention and (n − p)/2 ·
log(2π) to reach `regress`'s — ML is a density for n observations, REML for
n − p error contrasts. Differences between models fitted in the same program
need no correction, which is why the likelihood ratio tests agree without any of
it. Both checks assert the offset is that named constant rather than merely
stable.

## What is still owed

- **The builder returns a dense matrix**, so it costs n² of memory: about 230 MB
  at the whole SAFS pedigree and 800 MB at 10,000 people. `prepare` then
  exploits the family blocks, but the matrix between them does not. Nothing has
  needed more yet.
- **The spatial model is slow and does not scale like the others.** A distance
  kernel couples everybody, so there are no family blocks and every evaluation
  factorises a dense matrix of the whole roster. At 1,800 people a fit is about
  forty seconds and a 199-replicate bootstrap is hours. The block models are
  three orders of magnitude cheaper.
- **The spatial range is not usefully estimated**, as above. Nothing here fixes
  that; integrating it out sidesteps it.
- **The two-trait profile intervals have one independent implementation and the
  rest have none.** SOLAR and R do not compute profile intervals for these
  models, so every other interval rests on calibration alone. Calibration says
  the recipe covers; it does not say the arithmetic matches a second opinion.
- **The conservatism near zero is unrepaired.** Decision 12's third tier —
  calibrate once per design — is the anticipated fix and is not built.
- **Three traits at once is not possible.** The ratio parameterisation fails at
  three or more, and the boundary enumeration is 3^t.
