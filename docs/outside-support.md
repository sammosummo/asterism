# Public models outside scientific support

These models are importable and usable, but their numbers have not been
established to the standard the [supported analyses](api-support.md) are held
to: no check measures them against a known truth or an independent
implementation, and no release evidence stands behind them.

They are here under [ADR 0013](adr/0013-scientific-support-does-not-remove-public-models.md),
which keeps a public model available for simulations, power calculations and
diagnostic work even when a release makes no claim about it. What they lack is
the evidence: no coverage simulation, no comparison against another
implementation, and nothing in the release standing behind the numbers they
return.

## Spatial covariance 

```python
model = asterism.SpatialModel([relationship], distance_km, design)

profiled = model.fit(y)
integrated = model.fit(y, integrated=True)
p_value = model.bootstrap(y, replicates=999, seed=17)
```

The spatial kernel is `exp(-lambda * distance)`. With `integrated=True`, the
range is averaged over and no single range estimate is returned. The spatial
variance test requires the parametric bootstrap because the range is
unidentified when spatial variance is zero. In simulation, range intervals
usually reached a numerical bound; treat the range mainly as a descriptive
point estimate.

As in `ComponentModel`, `raw_coefficient_proportions` depend on the scaling of
the fixed covariance matrices. When their mean diagonals are positive,
`mean_diagonal_proportions` gives the scale-invariant marginal covariance
decomposition. Component-index intervals profile the raw coefficient
proportion and identify that quantity explicitly in their result.

Prediction methods remain public but are outside 0.1 scientific support.


### The model

The model below is public and its checks are kept, but 0.1 makes no
scientific claim about it. The method is recorded here so a later release
can pick it up unchanged.

### Model, range, and bootstrap

Let $d_{ij}\geq0$ be the supplied distance in kilometres between observations
$i$ and $j$.
Asterism adds the exponential spatial kernel

$$
K_s(i,j;\lambda)=\exp(-\lambda d_{ij}),\qquad\lambda>0,
$$

to $q$ fixed covariance bases:

$$
\mathbf V=\sum_{k=1}^{q}\theta_k\mathbf K_k
+\theta_s\mathbf K_s(\lambda)+\theta_e\mathbf I_n,
\qquad \theta_s\geq0.
$$

Here $\mathbf K_k$ is fixed basis $k$, $\theta_k\geq0$ its coefficient,
$\theta_s$ the spatial coefficient, $\theta_e\geq0$ the residual coefficient,
and $\mathbf I_n$ the identity. ML and REML are available; REML is the default.

Here $\lambda$ is decay per distance unit, $1/\lambda$ is the e-folding range,
and $\log(2)/\lambda$ is the half-correlation distance. Gaussian spatial
covariance ML is developed by [Mardia and Marshall
(1984)](references.bib#mardiaMarshall1984). If $d_{ij}$ is great-circle
distance, the exponential remains positive definite on a sphere by the
completely monotone construction discussed by [Gneiting
(2013)](references.bib#gneiting2013); Asterism still validates the submitted
numeric covariance.

Under $H_0:\theta_s=0$, $\lambda$ disappears. This is the nuisance-only-under-
the-alternative problem studied by [Davies
(1977)](references.bib#davies1977), so neither ordinary $\chi^2$ nor the simple
50:50 component mixture supplies the spatial p-value. Asterism fits the reduced
model, simulates $B$ responses from it, refits the reduced and full models, and
uses

$$
p=\frac{1+\sum_{b=1}^{B}\mathbb 1(T_b\geq T_{\mathrm{obs}})}{B+1},
$$

where $T_{\mathrm{obs}}$ is the observed likelihood-ratio statistic and $T_b$
is the statistic in bootstrap replicate $b$. The bootstrap originates with
[Efron (1979)](references.bib#efron1979); the add-one Monte Carlo correction is
justified by [Phipson and Smyth
(2010)](references.bib#phipsonSmyth2010).

The free fit may profile $\lambda$ or numerically integrate it over the
implemented grid. The same choice is used for the observed and simulated
statistics. Range is descriptive in 0.1. Weak joint identification of variance
and range under fixed-domain asymptotics is established for Matérn models by
[Zhang (2004)](references.bib#zhang2004); the exponential kernel is the
Matérn-$1/2$ case. The theorem motivates caution but does not diagnose any one
submitted layout.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `spatial_presence_test` | `SpatialModel.bootstrap(y)`: `statistic`, `p_value`, `rule`, `exceedances`, `replicates`, `requested`, `seed`, and `smallest_p_value` |

`SpatialModel.fit` supplies the prerequisite `converged`, `estimator`, and
`range_treatment` fields. Its variance decomposition, `decay_per_km`, and
`half_distance_km` are descriptive in the spatial-presence analysis. Raw
coefficient proportions and spatial intervals are diagnostic and are not 0.1
reportable targets.

### Assumptions, measured limitations, and unmet gates

Distances and their units are correct; the exponential kernel is appropriate;
fixed covariance bases are valid and aligned; and the reduced-model generator
represents the null. Monte Carlo resolution is never finer than $1/(B+1)$.
Historical calibration with 199 bootstrap replicates gave rejection 0.050 at
nominal 0.05. Profiled spatial-component and half-distance intervals covered
at 0.984 and 0.976, but 98.4% of half-distance intervals reached a bound.
Independent `spaMM` fits also agreed on a long-range case while both missed its
true decay, illustrating that implementation agreement does not prove
identification. SOLAR discards covariance that crosses pedigrees and is not an
independent reference for that portion of the model. The SOLAR-limited and
`spaMM` adapters have now been refreshed live and frozen. A values-free
target-layout smoke completed its fit and one requested bootstrap replicate;
the code also refuses any incomplete bootstrap denominator. The exact
199-replicate fixed-wheel target command remains unrun, so this is operability
evidence rather than a qualified presence test.

### What has been measured

For 400 null data sets, each with a 199-replicate parametric bootstrap,
rejection rates were 0.003, 0.050, 0.102, 0.255, and 0.547 at nominal 0.01,
0.05, 0.10, 0.25, and 0.50. The likelihood-ratio statistic has an atom at zero,
so bootstrap p-values have an atom at one; validity is judged by rejection
rates rather than continuous uniformity.

The profiled spatial component proportion and half-distance intervals covered
at 0.984 and 0.976 in 250 simulations. However, 98.4% of half-distance
intervals reached a bound, demonstrating poor range identification. Integrating
over the range and then bootstrapping also held its level: rejection was 0.0575
at nominal 0.05 across 400 data sets.

At a fixed decay rate the covariance here is exactly a component model with one
more matrix, so the comparison recorded under "Several components" externally
checks this model's covariance assembly and likelihood. SOLAR cannot reach any
further: it discards covariance between pedigrees, which is precisely what a
spatial kernel is made of.

R's `spaMM` does reach further, and now does. It fits a Matern spatial
random effect beside a supplied correlation matrix by REML, and it estimates
the range itself. At `nu = 0.5` the Matern correlation is `exp(-rho d)`, which
is this kernel exactly, and its `rho` is our decay rate in the same units.
Across five scenarios in `checks/spatial_against_spamm.py` — a range the data
can see, one shorter than the spacing between people, one longer than the map,
one with no additive variance at all, and one whose field is drawn by `geoR`
rather than by us — the two implementations agree on all three variances and
the decay rate to a worst difference of **5.2e-06**, variances taken as shares
of the total and the rate relative to itself.

Read that as evidence about the code and not about identification. The
long-range scenario is one neither implementation recovers: both return a decay
of 0.364 where the truth is 0.002, and they agree on that wrong answer to
2.3e-06. Agreement between two independent fitters says the covariance
assembly, the likelihood and the range estimation are right; it says nothing
about whether a given dataset can pin a range down, and the calibration above —
98.4% of half-distance intervals reaching a bound — says that often it cannot.

The parametric bootstrap has no counterpart in `spaMM` and remains simulation
against this package's own generator.

The target-layout gate in `checks/spatial_target_layout.py` retains no observed
location. Its reviewed fixture contains only aggregate counts and distance
quantiles for the largest tracked spatial analysis: 1,792 analysed rows after
91 trait-or-covariate exclusions from 1,883 people with usable geocodes, 190
relationship components with largest component 165, and 1,140 shared-location
groups producing 1,014 zero-distance pairs. A new planar layout generated from
those aggregates matches all eight frozen nonzero distance quantiles from the
first percentile through the maximum to a worst relative difference of 7.63%.
The points are synthetic coordinates in kilometres, not displaced geocodes;
neither they nor their distance matrix are stored in the fixture or evidence.

A target-sized one-replicate smoke on the development build completed in 315 s.
The free REML fit converged with a scaled projected gradient of `2.30e-8`, and
the bootstrap used its one requested null refit with no refusal. Its add-one
p-value was necessarily 0.5, so this is operability evidence, not detection
evidence. The release command fixes 199 replicates, a 0.005 resolution, and a
five-per-cent target-presence decision; that exact multi-hour command has not
run against a fixed release wheel and is not claimed here.

That smoke also exposed why bootstrap completion must be literal. A bounded
spatial fit can leave a variance at `3.6e-17`: numerically nought, but formerly
treated as interior when projecting its gradient. Using the shared
unit-variance boundary tolerance makes that legitimate null-boundary optimum
converge. The deterministic Rust seed now completes all 12 requested refits.
Separately, the bootstrap rejects any `used != requested`; a failed refit is an
unknown comparison, not a non-exceedance and not a row to drop from the
denominator.

## Marker association 

```python
scan = asterism.AssociationModel(relationship, design_with_pcs, y)

fast = scan.sweep(markers)  # covariance held
staged = scan.sweep(markers, refit_below=1e-3)
full = scan.sweep(markers, variance="refitted")
```

Holding the null covariance makes a scan fast and is conservative for large
marker effects. Selective refitting is usually the useful compromise. The
marker remains inside the relationship matrix; Asterism does not perform a
leave-one-chromosome-out analysis or multiple-testing correction.


### What has been measured

The refitted association mode agreed with native SOLAR to all printed digits on
three synthetic tests, including p-values `3.32e-7` and `6.36e-20`. The held
mode was more conservative for large effects (`4.53e-7` and `1.58e-18` on
those examples), motivating selective refitting.

In 10 million null tests on 50,000 real markers across 200 simulated responses
on the real pedigree, crossings ran at 522 per draw below 0.01 where 500 were
expected, and the genomic inflation factor was 1.036. Between-response variation
is used rather than a binomial band, because linked markers are not independent
and the measured scatter runs two to three times the binomial one. The
experiment does not reach genome-wide `5e-8`, which would need of order ten
billion null tests.

About half that excess is the covariance model rather than the test. Three
things were measured to establish it. It is not driven by rare markers: split by
minor allele frequency the ratio is 0.98 between 0.01 and 0.02 and 1.07 above
0.10, the opposite of the usual cause, so a frequency filter would not remove
it. It is not the held-variance approximation: refitting the components for
every marker gives an inflation factor of 1.0087 against held's 1.0079 on the
same data. What does remove it is permuting each marker across people, which
keeps its allele frequency exactly and destroys any correspondence between a
genotype and how closely two people are actually related.

Over the same 10 million tests, per draw:

| threshold | real markers | permuted | expected |
| --- | --- | --- | --- |
| `0.01` | 522.14 | 510.43 | 500 |
| `0.001` | 53.28 | 51.83 | 50 |
| `1e-4` | 5.44 | 5.13 | 5 |

So of the 22 excess crossings per draw at `0.01`, permuting removes about 12.
That part is the pedigree kinship failing to describe the relatedness the real
genotypes carry, which is why it is largest for common markers: those carry most
of that signal. A genomic relationship matrix, rather than a pedigree one, is
what would close it.

The remaining two per cent is the test's own, from reading a statistic with
estimated variance components against a chi-square. It is marginal at 200 draws
-- 510.43 against a ceiling of 509.9 -- and has gone by `1e-4`, so it does not
threaten a scan at genome-wide thresholds, but it is worth knowing before
quoting a p-value near `0.01`. The check reports both parts separately and
judges only this one.

## Variant sets: genes and pathways 

```python
model = asterism.VariantSetModel([relationship], design, y)

weighted = dosages * variant_weights  # Z = G W, one column per variant
result = model.test(weighted)  # variance-component test
family = model.test_family(weighted)  # across burden-to-variance-component
```

Rare variants tested one at a time find nothing, because each has a handful of
carriers. This asks whether a set carries more trait variance together than
chance allows. The null is fitted once for a whole scan.

Pass `Z = G * w`, not a kernel. Nothing of size `n by n` is formed, so the
memory is one column per variant rather than one per person squared.

`test_family` turns a dial from a variance-component test, which assumes
nothing about direction, to a burden test, which assumes the variants all act
alike. Neither wins in general, so it runs both ends and between, and combines
them by the Cauchy method rather than reporting whichever looked best.

The weighting is yours to choose and is not innocent: a column multiplier `w`
is a variance weight of `w**2`, and the usual rare-focused choice is a
`Beta(1, 25)` density at each minor allele frequency. Under the null that shape
is unidentified, so it cannot be fitted; run a few and combine them instead.


### What has been measured

Against famSKAT — `SKAT_NULL_emmaX` for the kinship-carrying null, then
`SKAT` with Davies' method — on identical data with identical weights, the two
agree to `2e-6` relative across twelve data sets spanning `p = 3.7e-3` to
`p = 0.99`. That is about the accuracy either implementation computes its own
tail to, so it is agreement to the precision available rather than to a
tolerance chosen for comfort.

The reason for building it is calibration. Simulated under the null on a
rare-variant kernel, 2,500 replicates, 300 people:

| kernel | test | rejection at 0.05 | at 0.01 |
| --- | --- | --- | --- |
| linear | likelihood ratio | 0.0436 | 0.0088 |
| linear | score | 0.0484 | 0.0076 |
| burden | likelihood ratio | 0.0200 | 0.0028 |
| burden | score | 0.0512 | 0.0080 |

The burden kernel is rank one, which is where assuming a 50:50 reference goes
furthest wrong: the likelihood ratio rejected 0.020 against a nominal 0.05,
seven binomial standard errors low. The score test, whose reference is computed
from the kernel's own eigenvalues, sits on nominal.

Every member of the correlation family reproduces SKAT's own `r.corr` to
`4e-7`, and the combination sits above its best member, which is the price of
having looked at several.

Across 2,000 null replicates the combined family test rejected 0.0450 at a
nominal 0.05 and 0.0095 at 0.01. Against alternatives, at 600 replicates each,
it recovers most of whichever single test was right:

| truth | variance component | near burden | combined |
| --- | --- | --- | --- |
| all one direction | 0.428 | 0.738 | 0.715 |
| mixed directions | 0.730 | 0.167 | 0.683 |

The median strongest correlation was 0.50 where the variants agreed and 0.00
where they did not, which is the dial finding the truth it was pointed at.

## Latent mediation with continuous and threshold observations 

```python
families = [
    {
        "relationship": relationship,
        "latent_mean": latent_mean,
        "mediator_measurement": mediator_measurement,
        "mediator_measurement_error_variance": measurement_error_variance,
        "mediator_proxy_status": mediator_proxy_status,
        "outcome_status": outcome_status,
        "mediator_threshold": mediator_threshold,
        "outcome_threshold": outcome_threshold,
        "mediator_proxy_sensitivity": sensitivity,
        "mediator_proxy_specificity": specificity,
        "ascertainment": "population_unconditioned",
        "proband_index": None,
    }
]

latent = asterism.LatentMediationModel(families, qmc_points=8192)

point = latent.evaluate(
    a=0.5,
    b=0.3,
    c_prime=0.2,
    d=0.6,
    sigma_m2=0.7,
)
fit = latent.fit()
```

Each family supplies one additive relationship matrix and person-aligned
observations of a latent mediator and outcome. A continuous mediator measurement
has a known positive error variance. The optional binary mediator proxy may be
fallible: its sensitivity and specificity describe its relationship to the
thresholded latent mediator. The binary outcome directly thresholds the latent
outcome. Use `None` for an unobserved measurement or status; scalar thresholds
and proxy accuracies are expanded across a family. `latent_mean` has length
`2*n`, mediator first and outcome second.

`ascertainment="population_unconditioned"` uses the ordinary family likelihood
and requires `proband_index=None`. Alternatively,
`"condition_on_named_proband_case"` requires an observed case at
`proband_index` and divides the family likelihood by that person's marginal
outcome-case probability.

The vertical estimand is `a*b` and the horizontal inherited effect is
`c_prime`. Exact or adaptive Gaussian probabilities are used through dimension
two; higher-dimensional rectangles use deterministic quasi-Monte Carlo.
Evaluation and fit results include convergence and integration diagnostics.
`fit()` needs at least one continuous mediator measurement across the supplied
families to identify the mediator scale; fixed-point `evaluate()` does not.

The current implementation reports likelihoods and point estimates. It does
not yet provide a calibrated p-value or interval for `a*b`, and QMC batch
stability is a numerical diagnostic rather than an inferential error bound.

### What has been measured

Low-dimensional family log likelihoods are independently reconstructed from
the structural covariance and one- or two-dimensional Gaussian rectangle
probabilities. Compiled values agree within `2e-8`; central derivatives for all
five free parameters agree within `3e-6`. A continuous underflow case remains
finite on the log scale.

The normal tail underneath all of this is `erfc(x / sqrt 2) / 2` from FDLIBM.
Against a sixty-digit evaluation it is right to an ulp for ordinary arguments
and never worse than `4e-14` out to 36 deviations. This matters more than it
looks: the tail `statrs` supplies is wrong by about `5e-11`, and its error
wanders from point to point rather than varying smoothly, which to a quadrature
asking for `1e-13` is noise. The conditional integrand inherited it and the
quadrature subdivided to its depth limit chasing rounding, refusing rectangles
as ordinary as `[6, 40] x [5, inf)` at a correlation of 0.7.

Two-person rectangles far into the tail are computed on the log scale by
conditional quadrature rather than by differencing four corner probabilities,
which cancel there. Against an independent high-order Gauss-Legendre value the
log likelihood agrees within `8e-11` at log probabilities of `-25`, `-32`,
`-62`, `-231` and `-512`. The corner-difference route is kept above a
probability of `1e-9`, where it is the more accurate of the two, and the
integration method is reported per family so a reader can tell which was used.

Higher-dimensional quasi-Monte Carlo evaluations are deterministic and are
averaged over eight shifted copies of the Halton sequence. Against an
independently derived trivariate value the 8,192-point estimate agrees within
`4e-5`. Accuracy falls off with the discrete dimension: at 8,192 points the
measured error against scipy was `7e-5` at dimension four, `5e-4` at eight and
`3e-3` at twelve.

This path is on the log scale too, which it was not before. Each sample's
weight is a product of one interval probability per member, so it underflowed
long before any single member's did, and the quantile that places each latent
draw took its interval's width on the ordinary scale and refused anything
beyond about `1e-308` as empty. Both are now carried as logarithms — the
quantile by inverting `log sf` directly, from its asymptote and then by Newton
steps. With independent coordinates the rectangle is exactly the product of its
marginals, and at thresholds of 40, 45, 50 and 55 deviations, a log probability
near `-4600`, the estimate matches that product to `1e-3`.

`maximum_qmc_log_batch_range` is the spread across those eight shifted copies,
as a difference of logarithms and so read relative to the estimate itself. It
is a stability statistic and not an error bound, but shifting is what makes it
informative at all: contiguous blocks of one unshifted sequence share their
bias, so their spread understated the true error by three to four orders of
magnitude at dimension twelve. Reported as an absolute ordinary-scale width, as
it was, it said nothing beside a log probability: the same shakiness read as
`1e-3` at one depth and `1e-200` at another.

The test of `a b = 0` was exercised on 36 people in 12 families of three, with
binary outcomes, across a real mediation and three nulls. Every p-value fell
the right side of 0.05: no null was rejected, and the real mediation was not
claimed either, because `b` could not be established at that size. The point
estimates over the same four runs were +7.79, +0.54, -2459 and -0.09 against
truths of 0.42, 0, 0 and 0 — the test is a likelihood ratio and does not depend
on them. The runaway is confined to the run where `a` is large and identified
while `b` is nought and unconstrained.

That is a demonstration that the construction behaves, not a calibration: four
runs cannot measure a rejection rate. No simulation presently establishes
coverage or type-I error for the vertical estimand `a b`, so the model reports numerical diagnostics and point estimates
without a calibrated interval or p-value.

## The weighted chi-square tail

Checked against Ruben's series expansion, which writes the statistic as a
mixture of ordinary chi-squares and shares no machinery with
characteristic-function inversion.

- Where an exact answer exists -- one weight, or equal weights -- it is taken
  exactly, agreeing with the chi-square tail to `1e-19`.
- Across 83 cases where the series provably converged, the worst relative
  difference was `1.8e-3`, at a tail of `4.9e-8`.

Every published route fails somewhere, and the failures were measured rather
than assumed. Davies' inversion refuses to run at tight accuracy settings and
returns nought at its own default where the answer is `5.7e-7` -- which is the
case a burden kernel produces. Ruben's series stops early when the weights span
a wide range: on eighteen weights spanning eighty to one it returned `1.3e-9`
where a Monte Carlo of eight million draws gives `1.54e-4 +/- 8.6e-6` and
Asterism gives `1.60e-4`.

The series has an exact convergence test, because its coefficients sum to one,
so the check drops it as a reference wherever that mass falls short rather than
comparing against a number that has quietly stopped early.
