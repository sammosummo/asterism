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

model = asterism.prepare(x, k)   # validates and decomposes, once
record = model.fit(y)            # as many times as you like
```

`x` is the fixed-effect design in fit order, including its own intercept column
if one is wanted. `k` is the relationship matrix in the same order — twice the
kinship is usual, but a genomic relationship matrix is equally acceptable,
because the matrix is checked as mathematics rather than for where it came from.
`record` is a dictionary held in memory, carrying the estimate, the interval,
the test, the boundary state and the warnings. Asterism reads no files and
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
cargo test                                 # the estimator, against simulated truth
uv run --no-project pytest tests/ -q       # the Python interface
```

The Rust tests simulate sibling pairs, which is the least informative pedigree
in common use and so the honest design to set tolerances against. They check
recovery of a heritability the simulation controls, not agreement with another
implementation — agreement proves fidelity and never correctness.

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

## What is still owed

- The independent check that `docs/adr/0006` requires. There is one
  implementation of this arithmetic and no second one to disagree with it.
- The comparison against native SOLAR and R `regress`, which needs a dataset
  generating first — Astrarium's fixtures deliberately did not come across.
- The coverage check: many simulated datasets at known heritabilities, counting
  how often the interval contains the truth. It is what chose the interval
  recipe in the first place and nothing here has re-earned that yet.
