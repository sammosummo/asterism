# Validation record

What each model has been compared against, and what simulation has measured
about it.

Four labels distinguish where a figure came from.

| Label | Meaning |
| --- | --- |
| Reproducible | A check in `checks/` regenerates the figure from this repository. |
| Recorded | The figure came from a run that read participant data. It is kept as a record; nothing here regenerates it. |
| Pending | `release.toml` fixes the command, but it has not yet run. |
| Superseded | The run happened, but a later correction means that it no longer qualifies the stated analysis. It remains as history only. |

Coverage and rejection rates carry exact binomial
[Clopper–Pearson](https://doi.org/10.1093/biomet/26.4.404) limits. Every
attempted replicate stays in its denominator, so a refused fit, a nonconverged
fit, or a failed profile point counts as a miss rather than leaving the sample.

## Contents

- [One trait, one variance component](#one-trait-one-variance-component)
- [Several covariance components](#several-covariance-components)
- [Two traits](#two-traits)
- [Binary traits](#binary-traits)
- [Censored traits](#censored-traits)
- [Censored traits, several components](#censored-traits-several-components)
- [Mixed pairs](#mixed-pairs)
- [Gene by environment, measured](#gene-by-environment-measured)
- [Gene by environment, binary](#gene-by-environment-binary)
- [Convergence and polishing](#convergence-and-polishing)
- [Target-design fixtures](#target-design-fixtures)

## One trait, one variance component

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| R `regress` | REML | n = 350, six fixed effects | heritability 8e-9 relative, total variance 4e-9, fixed effects 1e-9 | Reproducible |
| SOLAR | ML | three synthetic pedigrees | heritability 5e-7 | Reproducible |

The SOLAR figure is the precision available from its printed output. The
log-likelihood differences are the constants each program omits: `(n-p)/2 log(2
pi)` for `regress`, `n/2 log(2 pi)` for SOLAR.

### Interval coverage

Development-wheel runs of 21 August 2026: 1,200 replicates per truth on each of
two designs, 14,400 fits per design and 28,800 in total. The standard design is
1,400 people in repeated families; the target design is the 1,909-person
envelope described under [target-design
fixtures](#target-design-fixtures). Both runs recorded no refusals, no
nonconverged free fits and no failed profiles.

| true h² | standard coverage | standard 95% CP | target coverage | target 95% CP |
| ---: | ---: | :--- | ---: | :--- |
| 0.00 | 0.9442 | [0.9296, 0.9565] | 0.9442 | [0.9296, 0.9565] |
| 0.05 | 0.9525 | [0.9389, 0.9638] | 0.9567 | [0.9436, 0.9675] |
| 0.07 | 0.9508 | [0.9370, 0.9624] | 0.9500 | [0.9361, 0.9616] |
| 0.10 | 0.9525 | [0.9389, 0.9638] | 0.9558 | [0.9426, 0.9667] |
| 0.20 | 0.9500 | [0.9361, 0.9616] | 0.9542 | [0.9408, 0.9653] |
| 0.30 | 0.9525 | [0.9389, 0.9638] | 0.9458 | [0.9315, 0.9580] |
| 0.40 | 0.9542 | [0.9408, 0.9653] | 0.9492 | [0.9352, 0.9609] |
| 0.50 | 0.9483 | [0.9343, 0.9602] | 0.9450 | [0.9306, 0.9572] |
| 0.60 | 0.9392 | [0.9241, 0.9520] | 0.9542 | [0.9408, 0.9653] |
| 0.70 | 0.9417 | [0.9269, 0.9543] | 0.9417 | [0.9269, 0.9543] |
| 0.80 | 0.9375 | [0.9223, 0.9505] | 0.9425 | [0.9278, 0.9550] |
| 1.00 | 0.9458 | [0.9315, 0.9580] | 0.9583 | [0.9454, 0.9689] |

A 300-replicate prefix of the same deterministic stream had put the
standard-design 0.70 cell at 0.9100, [0.8718, 0.9399]. Twelve hundred
replicates of that stream gave 0.9417, [0.9269, 0.9543].

An earlier compiled campaign ran 8,000 replicates at each of twelve true
heritabilities under REML (recorded). All twelve coverages were compatible with
0.95 at n = 1400. At n = 350 ten were compatible; truths 0.05 and 0.07 covered
at 0.979 and 0.974, which is conservative rather than anti-conservative
inference near the boundary.

### How replicates are scored

`checks/one_trait_coverage.py` exercises `asterism.prepare` and
`PreparedModel.fit` and nothing else. It takes the replicate count, family
count, worker count, truth grid and `--no-write` policy on the command line,
and writes one JSON document to standard output. At true h² of 0 and 1 it reads
`contains_lower_bound` and `contains_upper_bound` from the fit record rather
than inferring containment from a numerical endpoint.

The acceptance rule was fixed before the target design ran. A cell fails as
anti-conservative only when its two-sided upper limit falls below 0.940. A cell
whose lower limit exceeds 0.960 passes and is labelled conservative;
at truths 0.05 and 0.07 the label is `historically_conservative`.

### Pending

`release.toml` reserves 8,000 replicates per truth, 96,000 fits per design.
Neither 8,000-replicate command has yet run against a fixed release artifact.

## Several covariance components

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| SOLAR | ML | n = 350, six fixed effects, kinship plus a fixed-decay spatial kernel and residual | 5e-7 on all three variances, 4e-7 on the log-likelihood | Reproducible |
| Stopher et al. (2012) | REML | four red deer traits, published animal models | 0.006 on every component of six of the eight fits | Reproducible |

Before the SOLAR comparison this family had no comparison against another
package. The log-likelihood figure is after the `n/2 log(2 pi)` constant SOLAR
omits.

SOLAR cannot be used this way for a matrix whose covariance crosses pedigrees.
It evaluates the likelihood one pedigree at a time, so only the within-pedigree
blocks of a supplied matrix contribute and the rest are discarded without
warning, while `matrix debug` still reports the whole file. Handed the dense
kernel, SOLAR returns its within-pedigree answer bit for bit, and the two
implementations then differ by 0.28 and 0.35 in a variance.
`checks/against_solar.py` asserts this rather than describing it.

### Red deer

[Stopher et al. (2012)](https://doi.org/10.1111/j.1558-5646.2012.01669.x),
*Evolution* 66:2411, fitted animal models to four traits of wild red deer and
then added a matrix of home range overlap, depositing that matrix with its
pedigree and phenotypes (Dryad `doi:10.5061/dryad.jf04r362`).
With `RED_DEER_DATA` set to that unpacked deposit,
`checks/against_red_deer.py` refits all four and reproduces its Table 2. The
checker has no implicit workspace-data fallback.

| Quantity | Asterism | Published |
| --- | ---: | ---: |
| Spring home range heritability, no overlap | 44.02% | 43.67% |
| Spring home range heritability, with overlap | 0.29% | 0.28% |
| Rut home range heritability, no overlap | 31.29% | 31.31% |
| Rut home range heritability, with overlap | 0.00% | 0.11% |
| Likelihood ratio for adding overlap, spring | 1300.4 | 1313.2 |
| Likelihood ratio for adding overlap, rut | 771.3 | 785.8 |

Birth weight is the exception to the 0.006 agreement above. It differs by up to
0.007 without overlap and by 0.038 with it, the largest being the overlap
variance itself at 0.126 against a published 0.088. The paper flags that trait
as unstable, and every difference lies inside one published standard error.

This is the only external check the component family has against matrices it
did not build, on covariance that crosses families throughout.

The check verifies two steps the deposit leaves undocumented. The overlap
matrices carry no identifier file; reading their indices as the pedigree's row
order yields exactly the 948 spring and 766 rut females the paper reports.
Separately, 1,520 of the 4,051 deer have a known mother and no father, so each unknown
father is given its own founder identity, a construction that agrees with a
longhand tabular recursion to `0.000e+00` and recovers 339 inbred animals.

### Coverage, level and power

For additive, household and residual covariance at n = 400: 95% coverage of the
additive and household mean-diagonal proportions was 0.945 and 0.955; a
zero-household test rejected 0.051 at nominal 0.05; power for a household
proportion of 0.2 was 0.90.

These proportions are meaningful in that simulation because all three bases
have positive unit mean diagonal. Off-diagonal kinship-class bases instead have
zero diagonal and are interpreted through coefficients and contrasts. On the
real GOBS pedigree (recorded), the class-equality omnibus rejected 0.035 at
nominal 0.05 under equality and 1.0 when one class coefficient was raised by
0.5. Class contrasts were near nominal under equality; their constrained
parameterisation couples the classes, so they follow the omnibus test.

## Two traits

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| R `regress` | REML | unbalanced synthetic | heritabilities 2.6e-9, correlations 1.9e-8 | Reproducible |
| SOLAR | ML | deliberately unbalanced observations | 2e-7 on all four reported quantities | Reproducible |
| SOLAR | ML | all 78 pairs of thirteen real GOBS traits | 5.9e-6, median 1.5e-7 | Recorded |
| Independent profile endpoints | REML | unbalanced, n = 150 | about 1e-5 | Reproducible |

The independently recomputed projected KKT measure in the `regress` comparison
was 5.5e-9. SOLAR's stopping precision makes roughly 1e-6 a realistic
comparison tolerance on real data. The driver behind the 78-pair figure read
real phenotypes and never lived in this repository; the record is
`evidence/bivariate-against-solar-real-2026-08-12.json`.

### Coverage and level

Three hundred REML simulations at n = 180:

| Quantity | 95% coverage |
| --- | ---: |
| First heritability | 0.957 |
| Second heritability | 0.967 |
| Genetic correlation | 0.960 |
| Residual correlation | 0.953 |
| Phenotypic correlation | 0.977 |

Interior correlation tests were close to nominal. The correlation-one boundary
test was conservative, rejecting 0.033 at nominal 0.05.

At heritability 0.83 (recorded), overall coverage stayed near nominal, but the
residual-correlation interval failed in 23 of 300 replicates and reached a
bound in 94% of those computed. The phenotypic correlation computed in every
replicate and did not reach a bound. `checks/bivariate_calibration.py` fixes
its heritabilities at easier values, so the record is
`evidence/coverage-at-the-real-design-point-2026-08-12.json`.

## Binary traits

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| SOLAR | ML | three synthetic cases | 0.0134 in liability heritability, or 0.086 SOLAR standard errors | Reproducible |

The two-person probability is measured against two independent integrations --
Owen's angular form and the conditional form, required to agree with each other
relative to the answer before either is used -- at 82 points, spanning
correlations from -0.99 to 0.9999 and thresholds from the symmetric centre out
to a log probability of -88. **Worst absolute error 6.9e-12 in the log
probability**, which is the scale that enters a likelihood. Four further points
are reported as having no usable reference, because both forms lose accuracy
there, rather than being passed over.

Below a probability of 1e-3 the region is evaluated on the log scale instead.
The ordinary-scale integral holds an *absolute* tolerance, so its error in the
logarithm is about that tolerance divided by the probability, and at 1e-10 that
is 1.6e-05. The log-scale routine is accurate relative to the answer wherever it
is asked.

A 29 August 2026 censored rerun measured the changed numerical route and was
then recorded as passing six rules. A later audit found that its MCMCglmm,
coverage and target simulations chose censoring limits from each realised
response. Those results, including the reported coverages and 800-attempt
target campaign, are now superseded as qualification evidence. The direct
`censReg` agreement of 9.6e-09 remains current. The immutable historical record
is `evidence/censored-pass-rules-rerun-2026-08-29.json`.

It was not always. Until 29 August 2026 a sixteen-point quadrature stood here,
whose error was 1.3e-9 for a correlation below 0.5 but 2.3e-4 at 0.99 -- and
that second figure held only at thresholds of nought and nought, being about
seventy times larger away from them. Nearly perfectly related pairs were refused
for that reason, and are now accepted. Replacing it moved the liability
model against its SOLAR successor by at most 1.2e-6, which is 2.7e-9 of the
value, and the movement is largest in the tail where the retired arithmetic was
weakest.

### Coverage and level

A 700-replicate campaign on the real GOBS pedigree structure (recorded)
rejected 0.0529 at nominal 0.05 under zero heritability, with 95% interval
coverage of 0.9443 at true heritability 0.25.

`checks/liability_calibration.py` replaces that participant dependency with the
values-free target fixture. It generates each independent pedigree block by
Cholesky factorisation of `h² A + (1 - h²) I`, thresholds at prevalence
K = 0.254, and exercises `LiabilityModel.fit`, `.test` and `.interval` only.
The null truth is h² = 0 and the coverage truth 0.25. The largest family
reaches the same above-two-person sequential Mendell–Elston path whose
approximation needs stress; the test must report `mixture_50_50`.

A bounded development-wheel smoke on 21 August 2026 ran one replicate per
scenario on the complete 1,909-person target. Both attempts returned complete
records with no failures, and the alternative interval covered 0.25 from an
estimate of 0.1906. Two attempts verify the loader, generator, approximation
path, interface and decision; they do not estimate level or coverage.

### How replicates are scored

Every attempted replicate stays in its scenario denominator. Too few cases or
noncases, a refusal, nonconvergence, a missing or malformed test or interval,
and any failed constrained profile point are failures, and the release
allowance is zero. The level cell fails only when the one-sided 95% lower limit
for the rejection rate exceeds the nominal level. Coverage fails when the
corresponding upper limit falls below 0.95. Numerical and profile failures fail
the check separately, so neither calculation can hide a missing result by
changing its denominator.

### Pending

The release command fixes 200 replicates per scenario. That 400-attempt
campaign has not run.

## Censored traits

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| R `censReg` | ML | 4,000 people, 1,131 censored, no relatedness | 9.6e-9 on the log-likelihood and all four reported quantities | Reproducible |
| MCMCglmm 2.36 | Bayesian animal model | 800 sibling pairs, fixed limit, an expected quarter censored (414/1,600 realised) | all three Asterism estimates inside the independent 95% posterior intervals | Reproducible |

The `censReg` row is a direct numerical comparison and remains current. The
MCMCglmm fixture was refreshed with its instrument limit fixed from the
generating mean and variance before the outcome was drawn. Asterism estimated
h² = 0.4846, total variance 3.8299 and intercept 10.0493; all three sit inside
the independent posterior intervals. The h² interval was 0.3487 to 0.6178,
with effective sample size 1,746.2.

### Fixed-instrument recovery and coverage

The corrected recovery campaign fitted 200 replicates at each expected
censoring share 0, 0.10, 0.25, 0.50 and 0.75. Mean h² estimates were 0.4961,
0.4959, 0.4996, 0.4935 and 0.4983 against truth 0.5. At three quarters
censored, replacing the censored values by the limit instead gave 0.3106.
Every cell passed its predeclared recovery rule with no failed fit; see
`evidence/tobit-calibration-2026-08-31.json`.

The fixed-instrument coverage campaign then fitted 1,000 replicates at each of
h² = 0, 0.3 and 0.5, at both 52% and 75% expected censoring. Coverage ranged
from 0.945 to 0.959 across the six cells, every exact interval contained the
nominal 0.95, and there were no refusals. The complete measurement is
`evidence/tobit-coverage-2026-08-31.json`.

### Historical interval coverage — superseded

Three hundred replicates of 300 sibling pairs per cell, scored unconditionally
with no cell dropped and no refusals, produced the following figures under the
outcome-adaptive generator. They do not establish fixed-instrument coverage.

| true h² | 0% censored | 25% | 50% |
| --- | ---: | ---: | ---: |
| 0.0 | 0.937 | 0.950 | 0.960 |
| 0.3 | 0.943 | 0.943 | 0.957 |
| 0.5 | 0.953 | 0.950 | 0.933 |

These figures had replaced a receipt of 18 August 2026 that reported 0.980, 0.977
and 0.983 in the first three cells and passed them under a one-sided rule. That
interval was missing the Self–Liang mixture that [ADR
0004](adr/0004-interval-recipe.md) requires, so a lower end of nought read as
containment whatever the likelihood there said, at a cost ADR 0004 had already
measured as 0.977 against 0.953. The mixture is now on the record as
`contains_lower_bound` and `contains_upper_bound`, the check reads it, and the
boundary allowance was removed. `evidence/tobit-coverage-2026-08-18.json`
predates the mixture and describes a recipe the code no longer implements.
Neither generation of this evidence answers the corrected fixed-instrument
question.

### Target design — fixed-instrument result

`checks/tobit_target_design.py` runs the pedigree-scale route on the
values-free structural fixture: 1,909 synthetic rows in 202 relationship
components, largest component 180, and the six-column participant-free design.
The corrected command fixes a limit from the generating mean, variance and
intended censoring share before simulating outcomes, then records the achieved
share. It calls `asterism.tobit_fit`, `asterism.tobit_interval` and
`asterism.tobit_test` under a boundary truth of h² = 0 and an interior truth of
h² = 0.5.

The corrected release campaign completed all 800 requested attempts with no
refusal, nonconvergence, failed test, failed interval or failed profile point.

| Expected censoring | Null rejection rate | h² = 0.5 coverage | Verdict |
| --- | ---: | ---: | --- |
| 52% | 0.040 | 0.940 | passed |
| 75% | 0.045 | 0.930 | passed |

The coverage figures are read with the predeclared one-sided exact rule: their
upper limits are 0.9650 and 0.9572, so neither is distinguishable from nominal
0.95 in 200 replicates. The null lower limits are 0.0201 and 0.0237, below the
nominal 0.05. The complete result is
`evidence/tobit-target-design-2026-08-31.json`.

### How replicates are scored

Every requested replicate stays in its cell's denominator. An unsuccessful
outcome is assigned exactly one of `invalid_censoring_count`, `refused`,
`nonconverged`, `test_failed`, `interval_failed` or `profile_failed`, and none
is allowed. The null rule fails when the one-sided 95% lower limit for the
rejection rate exceeds 0.05; the interval rule fails when the corresponding
upper limit for coverage falls below 0.95.

The fixed-instrument external comparison, recovery, interval coverage and
target-design campaigns now all pass. Their retained evidence replaces, rather
than alters, the superseded outcome-adaptive records above.

## Censored traits, several components

**Current 0.2 scope, not a released artefact.** Fitting, intervals and the
explicitly asymptotic test are in scope. What follows records what has been
measured rather than claiming that the development checkout is a release. The
model, its equations and its interface are in [the model
page](models/censored-components.md); the decision to generalise the censored
model in place rather than duplicate it is
[ADR 0022](adr/0022-the-censored-model-generalises-in-place.md).

The development API permits fitting and intervals. Its analytic `.test` route
uses the explicitly labelled `asymptotic_mixture_50_50` reference and refuses
when an untested nuisance variance is on its bound. A constrained-null
parametric bootstrap is implemented as `.bootstrap`, but remains experimental
and is not a 0.2 release claim.

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| An independent dense censored likelihood optimised by SciPy | ML | 20 sibling pairs, 2 records each, a quarter censored | 1.4e-3 on the mean-diagonal proportions, 7.0e-2 on the log-likelihood | Reproducible |
| The same, expected half censored | ML | as above | 2.9e-3 on the proportions, 5.5e-2 on the log-likelihood | Reproducible |

The instrument limits were fixed from the generating mean and variance before
outcomes were drawn. The reference shares no likelihood, parameterisation, starting values or
optimiser, and reaches the censored region probability by integrating it where
Asterism reaches it by sequential truncation. **The log-likelihood difference is
that approximation** and not a disagreement about the answer: scoring Asterism's
own fit through the independent likelihood shows the independent optimum is the
better point wherever the two differ, by about 1e-3, which is what a difference
of 3e-3 in a proportion costs on a surface this flat.

### Historical interval coverage — superseded

2,400 people in villages of six, two records each, 0.52 censored, 300
replicates per cell, and an interval on **every** structured component produced
the following figures with a censoring limit selected from each realised
response. Twelve cells covered their truth within a band drawn simultaneously
across them, but that run does not establish fixed-instrument coverage.

| additive / person / household / residual | additive | person | household |
| --- | ---: | ---: | ---: |
| 0.00 / 0.35 / 0.25 / 0.40 | range, see below | 0.9933 | 0.9700 |
| 0.20 / 0.30 / 0.20 / 0.30 | 0.9600 | 0.9867 | 0.9667 |
| 0.35 / 0.25 / 0.20 / 0.20 | 0.9800 | 0.9867 | 0.9700 |
| 0.60 / 0.15 / 0.10 / 0.15 | 0.9667 | 0.9667 | 0.9567 |

**Nominal 0.975, not 0.95**, and it is a lower-end statement: a component
proportion of one leaves a person's own records perfectly correlated, so the
covariance is singular, the profile cannot be evaluated at the upper bound and
the interval covers that end rather than placing it on evidence never gathered.
A two-sided 95 per cent interval puts 0.025 in each tail; closing the upper one
off hands that back.

**The band is simultaneous across the twelve cells.** Drawn at 95 per cent for
each separately, at least one is excluded by chance 46 per cent of the time, so
a per-cell rule would fail about every other run of a model with nothing wrong
with it. The first run of this check demonstrated it, failing on a cell that
over-covered by 0.001.

In the cell whose additive truth is nought, 11 of 300 intervals shut the truth
out and 289 were left undecided, so that cell's coverage is at most 0.963 —
which the band cannot separate from nominal. It is a range rather than a number
because containment splits in two at a bound: an end above nought excludes the
truth whatever rule applies, while an end on nought is the Self–Liang question,
and that verdict is deliberately withheld at several components.

### Historical target design — superseded

1,909 people, 202 relationship components with the largest at 180, six
fixed-effect columns, two records each, families cut into households of three.
The retired outcome-adaptive campaign made 800 attempts, measured all 800, and
had no refusal or failed fit. Its largest censored block was 310, against the
600 the sequential region-probability ladder has climbed. This remains useful
evidence that the numerical route executed at that dimension, not evidence for
coverage or test level under a fixed instrument.

| scenario | censored | rate | exact interval | nominal | |
| --- | ---: | ---: | :---: | ---: | :--- |
| null | 0.52 | 0.055 | [0.028, 0.096] | 0.05 | held |
| null | 0.75 | 0.100 | [0.062, 0.150] | 0.05 | **failed** |
| heritable | 0.52 | 0.985 | [0.957, 0.997] | 0.975 | held |
| heritable | 0.75 | 0.960 | [0.923, 0.983] | 0.975 | held |

The recorded rates, 0.055 at the lower censoring setting and 0.100 at the
higher one, were computed with the asymptotic 50:50 reference. They are
historical defect signals only: the generator selected its limit from each
realised response, and the analytic reference has not been shown to describe
this finite design.

The one-half weight in Self–Liang is an asymptotic tangent-cone result. It is
not a finite-sample requirement that half of fitted null datasets land on the
bound. The observed LRT atom — including shares from about 0.24 to 0.565 in the
retired investigations — is therefore a diagnostic, not a release gate. It
does not identify whether a departure comes from ordinary finite-sample
behaviour, weak identification, the region approximation or numerical fitting.
Earlier causal claims about sample size, block size, the person-level matrix or
matrix collinearity are withdrawn.

The released one-component numerical route is unchanged, but its historical
0.045 rejection rate and 0.530 atom were also obtained with the retired
generator and do not complete the fixed-instrument requalification.

The corrected target check fixes the censoring limits before outcomes and
measures the explicitly labelled asymptotic 50:50-mixture test at the intended
design. The other component variances are positive under the generating null,
and a realised nuisance-boundary fit is refused rather than scored against the
one-boundary reference. The LRT atom is reported beside the rejection level
only as a diagnostic.

The fixed-instrument campaign completed all 800 attempts without a failed fit.
At 52% expected censoring, null rejection was 0.055, its exact interval was
[0.0278, 0.0963], and the operational LRT atom was 0.465. At 75%, rejection was
0.095, its exact interval was [0.0582, 0.1444], and the atom was 0.390. The latter
cell fails because its predeclared one-sided lower bound exceeds 0.05.
Heritable-cell coverage was 0.980 and 0.960, both compatible with nominal
0.975. The largest censored block was 313, inside the ladder measured to 600.
The dry merge therefore wrote no qualifying evidence. The analytic p-value is
anti-conservative for this exact 1,909-person, four-component target at 75%
expected censoring and must not be reported for that analysis without a
design-specific simulated null. This does not establish a universal censoring
threshold or block fitting, intervals or the generally available labelled
asymptotic test.

### Component separation

The corrected release-design sweep fixed two records per person, the paper's
six-column fixed-effect design, 30 or 60 sibling pairs, and expected censoring
shares 0.52 or 0.75 before drawing outcomes. It requested 200 replicates in
each of four cells. There were 799 available fits: the 75%-censored,
30-family cell retained one unavailable fit as 199 rather than silently
changing its requested count.

At 60 families the mean additive and person-level shares were 0.3759 and
0.2996 at 52% expected censoring, and 0.3675 and 0.2936 at 75%, against truths
0.4 and 0.3. Both cells passed the predeclared rule that their bias must be no
larger than 0.03 plus twice the run's Monte Carlo error. Their split costs were
3.55 and 3.05 times the precision of the combined component: the sum is known
more precisely than its allocation between additive and person-level effects.
The complete result is
`evidence/component-separation-2026-09-01.json`. The broader 31 August sweep
remains useful context, but is not the release-design receipt.

### Several-component interval coverage

The fixed-instrument coverage campaign completed 300 replicates in each of
twelve component-by-truth cells at 52% expected censoring, with no refusal.
Coverage in the eleven interior cells ranged from 0.9533 to 0.9933; every
simultaneous exact interval contained the predeclared nominal 0.975. In the
unscoreable lower-bound cell, eleven intervals excluded nought and 289 retained
the deliberately absent boundary verdict. That range did not establish
undercoverage under its predeclared exact rule. The complete measurement is
`evidence/censored-components-coverage-2026-08-31.json`.

### How replicates are scored

Every requested replicate stays in its cell's denominator. The corrected
target-design campaign permits no failed fit, requires the
`asymptotic_mixture_50_50` rule, and scores the resulting rejection rate
rather than on the LRT atom. A p-value equal to 0.05 is a rejection. The
level cell fails only when its one-sided exact lower confidence bound is above
0.05; a conservative result is valid. Coverage analogously fails only when its
one-sided exact upper bound is below nominal. The coverage campaign permits no
refusal.

### Known test limitation and remaining release work

The corrected fixed-instrument component-separation and interval-coverage
campaigns pass.
The target-design campaign completed and identified one exact limitation: the
analytic p-value for the 1,909-person, four-component target at 75% expected
censoring is not reportable without a design-specific simulated null tied to
the model, design, source and seed. Its resumable checkpoint preserved that
result without publishing qualifying evidence.

The optional 999-draw constrained-null bootstrap campaign also completed all
800 requested coordinates. At 52% censoring all 200 null p-values were
available; at 75%, eleven attempts reported
`TOBIT_BOOTSTRAP_REPLICATE_FAILED`, leaving 189. The checker reported rejection
of 0.055 at both shares, and coverage of 0.980 and 0.960, but the no-write merge
failed because the predeclared rule permits no failed attempt. It published no
qualifying evidence, so the bootstrap remains experimental. This finding does
not create a censoring cutoff for another design. The development candidate's
four-platform probe was bit-for-bit identical for every new
censored-component quantity; the exact final wheels must repeat that
comparison. The independent dense-likelihood comparison above remains the
numerical reference for the sequential region approximation.

## Mixed pairs

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| SOLAR | ML | binary with continuous | 0.014 | Reproducible |

The corrected fixed-threshold recovery campaign fitted 100 replicates in each
of the continuous, binary, censored and censored-pair designs. Mean genetic
correlations were 0.3790, 0.3833, 0.3753 and 0.3803 against truth 0.4, and all
four cells passed the predeclared recovery rule with no failed fit. The result
is `evidence/mixed-bivariate-calibration-2026-08-31.json`. The direct SOLAR
comparison remains current.

### Fixed-threshold interval coverage, level and power

The corrected coverage campaign completed 300 replicates in each of twelve
cells: continuous, binary, censored and censored-pair designs at true genetic
correlations 0, 0.4 and 0.7. Every fit was scored. Coverage ranged from 0.930
to 0.957, and every exact interval contained the nominal 0.95. At zero genetic
correlation, rejection rates were 0.050, 0.070, 0.053 and 0.053 respectively.
The retained result is
`evidence/mixed-bivariate-coverage-2026-08-31.json`.

### Historical interval coverage, level and power — superseded

Three hundred replicates of 400 sibling pairs per cell, each cell on its own
stream, scored unconditionally with no refusals.

| Pairing | correlation 0.0 | 0.4 | 0.7 |
| --- | ---: | ---: | ---: |
| Continuous | 0.950 | 0.957 | 0.957 |
| Binary | 0.930 | 0.943 | 0.943 |
| Censored | 0.940 | 0.940 | 0.943 |

The same historical run reported rejection rates of 0.050, 0.070 and 0.060
against a nominal 0.05, with power at correlation 0.7 of 0.997, 0.950 and
0.993. Nought is an interior point of a correlation's range, so the intended
reference is a plain chi-square on one degree of freedom and no boundary
mixture applies.

Testing the genetic correlation against plus or minus one uses the 50:50
mixture instead. Two hundred replicates at a true correlation of 1 rejected
0.010, 0.040 and 0.085 against nominal 0.01, 0.05 and 0.10, with no refusals.

An earlier defect made every correlation interval reach a bound, giving widths
of about 1.5 on a parameter running from minus one to one. The convergence flag
was read from a raw maximum-absolute gradient, neither projected onto the
coordinates the search may move nor divided by the objective; a held profile
fit rests other coordinates on their bounds, so a sound constrained maximum
reported nonconvergence and the profile discarded it. Projecting and scaling
the gradient, as the liability model already did, brought widths to between
0.44 and 0.97 and put the interval and the test in agreement.

### Target design

`checks/mixed_binary_censored_target_design.py` runs the pedigree-scale route
on the values-free fixture with SHA-256
`aed92270dcfafb00a46a24bfaf1f3a2b52950fdea1c34deb3740e5060d6f3645`. The
generating grid fixes prevalence at 0.254, latent-trait heritabilities at 0.5
and 0.5, genetic correlation at 0 or 0.4, residual correlation at 0.15,
complete-hearing variance at 3, and intended right-censoring share at 0.52 or
0.75. Each replicate calls `asterism.mixed_bivariate_fit`,
`asterism.mixed_bivariate_interval` and `asterism.mixed_bivariate_test`.

### Pending

The fixed command requests 300 replicates in each of four correlation-by-censoring
cells, 1,200 attempts. Neither that campaign nor a bounded smoke has completed:
a local one-replicate-per-cell smoke of 21 August 2026 was interrupted during
interval fitting before producing a record. The target design therefore has no
coverage, level, recovery or runtime result.

## Gene by environment, measured

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| Independent target-sized reference, exponential surface | REML | 1,909-person target, 100 null replicates | 5.49e-07 on quantities, 2.68e-11 on the log-likelihood | Reproducible |
| Independent target-sized reference, random regression | REML | 1,909-person target, 100 null replicates | 3.50e-07 on quantities, 3.64e-12 on the log-likelihood | Reproducible |
| Dense-likelihood sentinel | REML | n = 80, both surfaces | passed | Reproducible |

Prewritten tolerances were 0.001 and 0.0001. Both target-sized comparisons
converged on 21 August 2026.

The independent reference assembles each pedigree block as
`V = A * G(z_i, z_j) + R(z_i) on the diagonal`, sums blockwise Gaussian
sufficient statistics, and profiles the six fixed effects globally. For the
exponential surface the genetic term is a product of half-exponentials in each
subject's environment times `exp(-lambda |z_i - z_j|)`, and the residual term
is `exp(alpha + gamma z)`. The random-regression reference uses quadratic forms
in `[1, z]` for both, with each coefficient-covariance matrix formed from
independent Cholesky coordinates. It calls neither Asterism's covariance
assembly nor its optimiser.

### Level

Two thousand simulations per scenario (recorded): the exponential
gene-by-environment correlation test rejected 0.048 at nominal 0.05 under a
flat null, and 0.053 when the true log-linear surface changed genetic scale
without reordering. The random-regression boundary tests were conservative. A
linear-scale rank-one surface fitted with the wrong exponential family produced
a 0.107 rejection rate for a nominal 0.05 no-reordering test, so surface
misspecification can imitate genetic reordering.

The 21 August 2026 run scored all 200 of its surface-replicates under the flat
`V = 0.5 A + 0.5 I` null with no refused, nonconverged, incomplete or
wrong-rule fits. At nominal 0.05, correlation and interaction rejection counts
were 1 and 5 of 100 for the exponential surface, and 1 and 4 of 100 for random
regression.

A cell fails only when its one-sided 95% exact binomial lower bound for the
rejection rate exceeds the nominal level. Conservative cells stay visible
rather than being counted as undercoverage, and any failed or unscored
replicate fails the check separately.

### Interval coverage

Gene-by-environment 95% interval coverage ranged from 0.953 to 0.983 in the
studied settings (recorded). Between 80% and 92% of genetic-correlation
intervals and up to 64% of some heritability intervals reached a bound, and
point estimates were pulled toward correlation one near the boundary.

### Target design

The gate is a pedigree-scale and design-identity stress test, not a
reconstruction of GOBS ages, relationship coefficients, trait-specific
missingness or outcomes. A deterministic synthetic standardised-age coordinate
stays inside the qualified interval [-1.5, 1.5], varies within every
non-singleton component, and combines with synthetic sex in the full-rank
six-column design `1, z, z², s, zs, z²s`. Quantities are judged at
z = -1, 0 and 1 only.

### Pending

The release rule is pinned at 500 replicates per surface and has not run. The
figures above are the completed 100-per-surface reduced run.

## Gene by environment, binary

### Agreement

| Compared against | Estimator | Design | Worst difference | Label |
| --- | --- | --- | --- | --- |
| Independent full-ML sentinel | ML | 120 people, 30 four-sibling families | 7.69e-04 on genetic correlation, 4.92e-05 on the log-likelihood | Reproducible |

Prewritten tolerances were 0.002 and 0.0002; both fits converged on 21 August
2026. `checks/discrete_gxe_against_independent_full_fit.py` parameterises the
genetic covariance by its own Cholesky factor, puts residual variances on log
scales, optimises with derivative-free SciPy Powell, and profiles fixed effects
from a self-contained dense likelihood. It shares neither likelihood code,
parameterisation nor optimiser with Asterism.

### Interval coverage

Four hundred replicates at 600 people in sibships of four (recorded):

| true correlation | coverage | reached a bound | median width |
| ---: | ---: | ---: | ---: |
| 0.3 | 0.958 | 0.138 | 0.682 |
| 0.6 | 0.955 | 0.490 | 0.623 |
| 0.9 | 0.995 | 0.940 | 0.389 |

Coverage is nominal away from the bound and conservative against it. At a true
correlation of 0.9, 94% of intervals reach a bound: they cover, but they do not
pin the value down. An interval that reaches its bound is reported as having
done so.

`checks/discrete_gxe_correlation_interval.py` runs that design participant-free
through `DiscreteGxeModel.fit` and `.correlation_interval` alone, scoring every
replicate as complete, refused, nonconverged, interval-refused or
profile-failed. Undercoverage fails when the one-sided exact upper limit falls
below 0.94; any incomplete denominator fails separately, and a one-sided lower
limit above 0.96 labels conservatism without calling it undercoverage.

### Level and power

Under the complete null on the participant-backed pedigree (recorded), 500
responses per scenario with sex as the environment:

| Test | Rejection at nominal 0.05 |
| --- | ---: |
| `gene_by_environment` | 0.042 |
| Equal genetic effects | 0.044 |
| Equal genetic variances | 0.048 |
| Equal residual variances | 0.036 |
| `any_difference` | 0.040 |

The fitted correlation sat on its upper bound in 52.8% of null fits, which is
why the even mixture is the reference and a plain chi-square would be
conservative.

Where one sex is measured with more error and the genetics are identical, every
genetic test held its level — 0.046 for the principal test, 0.040 for equal
genetic effects, 0.040 for equal genetic variances — while `any_difference` and
the residual test each rejected all 500 replicates. That is what the two free
residual variances exist to support. `any_difference` equates the residual
variances as well as the genetic ones, so what it rejects may be either.

Against genuine alternatives the tests separate: where the genes differ, the
correlation test rejected 0.990 and the equal-variance test stayed at level
(0.062); where only the genetic scale differs, the equal-variance test rejected
0.998 and the correlation test stayed at level (0.042).

### Pending

Neither full campaign has run in this development environment. The
one-replicate-per-truth smoke at n = 600 completed all three attempts in 3.99
seconds with no refusals, nonconvergence, interval refusals or profile
failures, covering all three generating correlations; that is a route check.
The pinned 400-by-three and 500-by-four campaigns were refused by the sandbox
before worker creation, so none of their 1,200 and 2,000 attempts ran. Both
commands stay fixed in `release.toml`, and the measured and configured flags
stay false.

## Convergence and polishing

One threshold, `1e-6` on the projected gradient scaled by the log-likelihood,
applies in every model family. It replaced seven different numbers spanning a
factor of a hundred, none with a recorded reason. Measured, the families do not
differ: across 63 real GOBS fits (recorded) every one reaches about `1e-08`
when allowed to, the reachable floors running from `7.68e-09` to `2.69e-08`.
What separated them in what the flag saw was `factr`, which inflates the
reported gradient nearly thirtyfold in the spatial model and under twice in the
component model.

`1e-6` passes every fit measured here, the worst case being the red deer's rut
home range with overlap at `1.578e-07`. `1e-7` fails that fit outright:
starving the search on purpose produced 165 fits, 92 of them genuinely short of
the optimum, and exactly one landed between `1e-7` and `1e-6`, its estimate
wrong by a millionth of a variance share.

The same measurement says what the number means. The worst error in a variance
share runs about ten times the reported gradient, and that holds across six
orders of magnitude; every starved fit above `1e-5` was caught. A fit reporting
`1e-6` is therefore right to about `1e-5` in any share it quotes, and one
reporting `1e-3` may be wrong in the second decimal place. The latent mediation
model keeps its own `1e-5`, where the number accepts or discards a start rather
than reporting a flag.

Two criteria are in play in the L-BFGS-B families. The search stops on `factr`,
a relative change in the objective, while the flag is decided afterwards on the
projected scaled gradient. In a long flat valley the objective settles long
before the gradient does, so the estimate can be right while the flag says
otherwise. Three of the eight red deer fits reported `converged: false` while
landing within 0.006 of the published values. It is not a size effect: the
largest problem of the eight, spring home range at 4,945 records, converges at
`1.43e-08` in one of its two models. The failures ran `1.16e-07`, `4.62e-07`
and `1.45e-06` while the five successes ran `2.41e-09` to `1.50e-08`.

The fit reports `stop_code` and `stop_message`, the search's own reason for
stopping. All eight deer fits stop the same way and none stops on `pgtol` or
runs out of iterations:

    CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH

Where the gradient test fails, the fit searches once more from the point
already found with `factr` switched off and reports `polished: true`. It fires
nowhere else: the five unpolished deer fits reproduce the unpolished run to
`0.000e+00` on every component.

| Trait | Model | Gradient before | Gradient after | Converged |
| --- | --- | ---: | ---: | --- |
| Rut home range | no spatial | 1.453e-06 | 4.051e-08 | no → yes |
| Rut home range | with overlap | 4.620e-07 | 1.578e-07 | no → no |
| Spring home range | with overlap | 1.162e-07 | 8.864e-08 | no → yes |

All three second searches end the same way:

    ERROR: ABNORMAL_TERMINATION_IN_LNSRCH

which is L-BFGS-B reporting that its line search can no longer make progress in
double precision. None reaches `pgtol`. The remaining failure is not an
exhausted budget: `1.578e-07` is near the numerical floor for that problem. The
returned point is checked rather than trusted — kept only if the objective is
no worse and the projected gradient strictly better — so an abnormal
termination costs the fit nothing. A `polished: true` beside a `stop_code` of
52 is that.

## Target-design fixtures

Four checks run at pedigree scale without retaining participant records. They
read a reviewed, values-free fixture holding aggregate structural metadata from
a predecessor receipt: 1,909 rows in 202 nonzero relationship components, the
exact component-size histogram, and a largest component of 180, with an
assertion that those facts sum back to 1,909 rows and 202 components. The
fixture holds no identifiers, phenotypes, covariates or participant-derived
matrix.

With the generator settings `seed=7`, `sib_mean=3.0`, `marry_in_p=0.35` and
`max_gen=5`, a standard founder-order kinship recursion produces a
structure-matched synthetic pedigree with 27,691 nonzero lower-triangle
relationship pairs. The predecessor aggregate was 27,821, a relative
discrepancy of 0.4673% inside the fixed 0.5% limit, recorded before fitting and
not tuned away. This is aggregate structure matching, and claims neither
coefficient nor eigenvalue identity, nor reconstruction of a pedigree,
relationship matrix or participant row.

A separate `seed=1` stream generates the six fixed-effect columns: intercept,
standardised age, age squared, sex, and both age-by-sex products.

| Fixture | SHA-256 |
| --- | --- |
| Shared structural source | `93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e` |
| Liability | `863eedc4f18063bca27754e782b9ebb5faa341359912b034163a7ea2067c20aa` |
| Mixed binary and censored | `aed92270dcfafb00a46a24bfaf1f3a2b52950fdea1c34deb3740e5060d6f3645` |
| Discrete gene by environment | `2cceae6f2c5ea3f0c3de5946daeaa55cf2f35edd6d365fd1f45499f463d4c1c6` |

The discrete gene-by-environment fixture differs: 1,910 synthetic rows in 203
components, largest component 180, group counts 758 and 1,152, and a full-rank
four-column synthetic age and group design. Its group assignment matches the
retained 13,322 cross-group related pairs and 77 mixed-group components
exactly. It reconstructs neither participants nor their rows, ages, phenotypes,
relationship coefficients, matrices or family-specific group composition.
