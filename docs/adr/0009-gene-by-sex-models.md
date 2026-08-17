# 9. Gene-by-sex models

**Extends ADR 0001 under decision 20. Amends nothing. Departs from the recovered
code in one place, recorded below under ADR 0006.**

**Status:** accepted 14 August 2026. **Renamed 16 August 2026**, decision
unchanged.

> **What this record calls things, and what they are called now.** The model
> was renamed to say what it is: a gene-by-environment model whose environment
> is discrete, rather than one about sex. Sex is now the worked example, not
> the concept, and any binary label serves — an exposure, a cohort, a
> diagnosis. This restores the name the recovered code used, below:
> `discrete_gxe_covariance`.
>
> | this record | now |
> | --- | --- |
> | `GxsModel` | `DiscreteGxeModel` |
> | `src/gxs.rs` | `src/discrete_gxe.rs` |
> | `checks/gxs_calibration.py` | `checks/discrete_gxe_calibration.py` |
> | the `gene_by_sex` null | the `gene_by_environment` null |
>
> Nothing about the five nulls, their reference distributions or the reasoning
> below changed. The model additionally now accepts any two distinct finite
> labels rather than only 1 and 2. The calibration this record cites was rerun
> on 16 August 2026 at 500 replicates per scenario; see
> `evidence/discrete-gxe-calibration-2026-08-16.json`.

## What was asked for

Gene-by-sex: one trait whose genetic effects may differ between men and women.
The recovered code has a method for it, `discrete_gxe_covariance` in
`variance-components-rust-ports`, with sex as a two-valued environment.

## The model is not new, and that had to be established before building

Sex takes two values, so the genetic part of any gene-by-environment surface
collapses to three numbers: a variance in each sex and a covariance between
them. The residual collapses to two. Five quantities in total.

**Every surface Asterism already carries has at least five free parameters over
a two-valued environment, so all of them span the whole model.** This is not a
guess. Fitting the same simulated trait on the real GOBS pedigree three ways —
`GxsModel`, and `GxeModel` with a 0/1 environment on both the random-regression
and the exponential surface — gives the same log likelihood, the same two
heritabilities and the same genetic correlation to every digit reported:

```text
gxs                loglik -2761.747445  h2 [0.7131, 0.1998]  rho 0.5647
random_regression  loglik -2761.747445  h2 [0.7131, 0.1998]  rho 0.5647
exponential        loglik -2761.747445  h2 [0.7131, 0.1998]  rho 0.5647
```

The algebra behind it: the random-regression genetic surface is
`(1, z) Q (1, z')ᵀ` for a positive semidefinite `Q`, and over `z ∈ {0, 1}` the
map to the two-by-two matrix of variances and covariance is `C = M Q Mᵀ` with
`M = [[1,0],[1,1]]`, which is invertible. So `Q` is a covariance exactly when
`C` is, and the two parameterisations reach the same set.

`GxeModel.test(y, "correlation")` on a 0/1 environment therefore returns
p-values **identical** to `GxsModel.test(y, "correlation")`. The gene-by-sex
question proper was already answerable before this module existed.

## So why a separate module

Three reasons, none of them "a model that could not otherwise be fitted".

**The parameterisation is the one the answer is written in.** Genetic standard
deviation per sex, residual standard deviation per sex, correlation between the
sexes, with the correlation bounded in `[-1, 1]` as a parameter rather than
implied by loadings. A reader of the fit does not have to be told what a loading
is.

**The exponential surface cannot represent a crossover and does not say so.**
Over `z ∈ {0, 1}` it correlates the sexes as `exp(-λ)`, which is positive at
every rate. It agreed in the table above because that trait's correlation was
positive. A trait where a genotype raises the value in one sex and lowers it in
the other is unreachable, and the fit will report a correlation near nought
rather than fail. Here the correlation is a bounded parameter and negative
values are reachable by construction.

**Bad sex codes are refused rather than absorbed.** Anything not equal to 1 or 2
is an error, and so is a group of fewer than two people. A model that quietly
sweeps unknown sex into one group is estimating a correlation with a third group
in it, and nothing in the output would show that. Use the pedigree sex.

The honest summary is that this module buys interpretability and safety, not
capability. It was built after the equivalence was established, not before.

## Where the departure from the recovered code is

The recovered inference freezes four references, and this implements all four:

| null | constraints | reference |
| --- | --- | --- |
| genetic standard deviations equal | 1, interior | χ²(1) |
| residual standard deviations equal | 1, interior | χ²(1) |
| correlation at one | 1, on a bound | ½·point mass + ½·χ²(1) |
| overall | 3, one on a bound | ½·χ²(2) + ½·χ²(3) |

**The fourth is not a gene-by-sex test and this module does not present it as
one.** Its null ties the two residual standard deviations, so a trait measured
more noisily in one sex departs from it whether or not anything genetic differs.
In simulation on the GOBS pedigree with identical genetics in both sexes and one
sex measured twice as noisily, it rejected at p = 1e-34 while every genetic test
correctly reported nothing.

That is not a fault in the recovered code, which was answering the question its
null asks. It is a fault in reading it as the headline, which is exactly what a
test named "overall" invites.

So the departure is an **addition**, not a change: a fifth null holding the
genetic standard deviations equal and the correlation at one **while leaving the
two residual standard deviations free**. Two constraints, one of them a simple
upper bound on one parameter, which is a flat face, so the reference is
½·χ²(1) + ½·χ²(2). That is derived here rather than recovered, and is therefore
checked by simulation rather than taken on trust.

The frozen three-constraint null is kept, under the name `any_difference`, with
its own documentation saying what it is. The added one is `gene_by_sex` and is
the default.

## Why the residual standard deviations are free at all

This is the decision the rest hangs on, and it costs a parameter and some power.

A single residual variance forces any sex difference in measurement error into
the only other term that can absorb it, which is the genetic one. The
gene-by-sex test would then reject on a difference in how well the trait was
measured. For the brain and biomarker traits this will be run on, a sex
difference in measurement error is not a remote possibility — it is the default
expectation for anything derived from an image.

The cost is real: two residual parameters instead of one, and less power to
detect a genuine genetic difference. It is worth paying because the alternative
is a test that answers a different question without saying so.

## What is checked, and what that does and does not establish

Under ADR 0006, agreement proves fidelity and never correctness.

**Fidelity.** The covariance is asserted equal to the recovered construction
cell for cell, by exact comparison rather than a tolerance, at four parameter
vectors including one on a bound and one with a negative correlation. The
recovered construction is written out inside the test rather than referred to,
so the two cannot drift apart silently.

**Correctness of the added reference distribution** cannot come from that, and
comes instead from simulation on the real GOBS pedigree with its real and
unbalanced sexes: four scenarios, every test judged as a level wherever the
scenario gives it nothing to find. The row that matters is the headline test
under a sex difference in measurement error alone, which is a level check and
not a power one. See `checks/gxs_calibration.py` and the evidence file it
writes.

**What is not established.** No comparison against SOLAR's own gene-by-sex
procedure has been run, and until one is, the claim here is fidelity to the
recovered Rust port and nothing further about SOLAR.
