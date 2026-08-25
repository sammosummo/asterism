# Public models outside scientific support

These models are importable and usable, but their numbers have not been
established to the standard the [supported analyses](api-support.md) are held
to: no check measures them against a known truth or an independent
implementation, and no release evidence stands behind them.

They are here under [ADR 0013](adr/0013-scientific-support-does-not-remove-public-models.md),
which keeps a public model available for simulations, power calculations and
diagnostic work even when a release makes no claim about it. Use them for that.
Do not report a number from one as though it carried the same weight as a
supported analysis.

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

**Pass `Z = G * w`, not a kernel.** Nothing of size `n by n` is formed, so the
memory is one column per variant rather than one per person squared.

`test_family` turns a dial from a variance-component test, which assumes
nothing about direction, to a burden test, which assumes the variants all act
alike. Neither wins in general, so it runs both ends and between, and combines
them by the Cauchy method rather than reporting whichever looked best.

The weighting is yours to choose and is not innocent: a column multiplier `w`
is a variance weight of `w**2`, and the usual rare-focused choice is a
`Beta(1, 25)` density at each minor allele frequency. Under the null that shape
is unidentified, so it cannot be fitted; run a few and combine them instead.

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
