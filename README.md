# Asterism

Quantitative genetics software for the Mathias Lab: one Rust crate named
`asterism` with a Python interface around it, fitting variance components
models. It is lab software, kept here because SAFS is its immediate user.

Nothing has been built yet. The directory holds this file and `CONTEXT.md`.

Read [[staging/studies/existing/safs/projects/asterism/CONTEXT|CONTEXT.md]] for
the vocabulary before writing anything here.

## What it is, and what it is not

It fits variance components models — currently nothing, and at first release one
trait with additive and residual variance, REML and ML, an interval and a
likelihood ratio test. It grows when a planned analysis needs it to and not
before, and every capability it gains is written into the decision record first.

Liability and threshold models, survival, gene-by-environment, spatial,
longitudinal, Tobit, signal detection, BLUP and prospective design analysis are
all deferred. Deferred means not yet, not never, and the list carries no order.

SOLAR is a comparator. Differences between the two are recorded rather than
treated as defects, and replacing SOLAR is not what defines this.

## What came before

Three implementations of this idea already exist, and this is a consolidation
rather than a fourth attempt:

- `~/Astrarium` — a working one-trait package in Python over Rust, checked
  against native SOLAR and R. Its `prepared.rs` is where the fresh start begins.
- `~/Asterism` — an earlier Python attempt, 107,877 lines in three days.
- SOLARSuccessor — the original engine, now historical, surviving only inside a
  cargo cache.

**This is a fresh start, not a move.** `prepared.rs` and the one function it
imports come across, about six hundred lines, and everything else is written
again. The three predecessors are reference for as long as they last and nothing
depends on their surviving.

`staging/projects/variance-components-rust-ports` is disposable work written by
Codex on 11 August 2026 and may be cannibalised freely. Its `edge_gaussian`
module holds a profiled ML and REML objective with an analytic gradient, which
is the shape this needs. Nothing enters Asterism because it was ported there.

## Where the decisions are

`docs/adr/0001-variance-components-estimator-parameterisation.md` in the
workspace root — twenty-eight decisions taken by Sam on 10 and 11 August 2026.
It moves in here, with Astrarium's five renumbered after it, once this is a
repository of its own.
