# Asterism

Quantitative genetics software for the Mathias Lab: one Rust crate named
`asterism` with a Python interface around it, fitting variance components
models. It is lab software, kept here because SAFS is its immediate user.

Read [[staging/studies/existing/safs/projects/asterism/CONTEXT|CONTEXT.md]] for
the vocabulary before writing anything here, and `docs/adr/` for the decisions.

## What it does today

One trait, additive and residual variance, REML and ML, a 95 per cent interval
and a likelihood ratio test against no additive variance. That is the whole of
it, and it is the whole of the first release.

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

## What it is not

It grows when a planned analysis needs it to and not before, and every
capability it gains is written into `docs/adr/` first. Liability and threshold
models, survival, gene-by-environment, spatial, longitudinal, Tobit, signal
detection, BLUP and prospective design analysis are all deferred. Deferred means
not yet, not never, and the list carries no order.

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

## The comparisons against SOLAR and R

`docs/adr/0006` puts correctness in external comparisons with a fixed division
of labour — SOLAR for ML, R `regress` for REML — because agreement between two
implementations proves fidelity and never correctness. Both are run, both on the
six-column design.

- **REML against R `regress`**: heritability to about 5e-9 relative, total
  variance to 4e-9, every fixed effect to 1e-9.
- **ML against native SOLAR**: heritability to about 1e-8 relative, which is
  every digit SOLAR prints, and standard errors agreeing to five figures despite
  being computed differently.

**Their printed log-likelihoods will not match Asterism's, and that is
expected.** Asterism keeps the Gaussian normalising constant; both comparators
drop it. Subtract n/2 · log(2π) to reach SOLAR's convention and (n − p)/2 ·
log(2π) to reach `regress`'s — ML is a density for n observations, REML for
n − p error contrasts. Differences between models fitted in the same program
need no correction, which is why the likelihood ratio tests agree without any of
it. Both checks assert the offset is that named constant rather than merely
stable.

## What is still owed

- **No real pedigree has ever gone through this.** The builder handles
  inbreeding, identical twins and a pedigree given in any order, and it refuses
  a half-known parent rather than guessing. None of that has met real SAFS data. Every roster here is
  synthetic and block-diagonal, with fourteen-person families and no inbreeding
  loops. A real SAFS pedigree is larger, more tangled, and may make the
  relationship matrix singular, which is the case the upper-bound snap in
  `prepared.rs` exists for and which nothing here exercises.
- **No real phenotype has ever gone through it either.**
- **The conservatism near zero is unrepaired.** Decision 12's third tier —
  calibrate once per design — is the anticipated fix and is not built.
