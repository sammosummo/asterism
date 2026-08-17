# Asterism

Asterism is a Rust/Python package for quantitative-genetic variance-component
models. It accepts numerical arrays already in memory and returns ordinary
Python dictionaries and arrays.

The model equations and interpretation are in
[statistical-methods.md](docs/statistical-methods.md). Empirical comparisons,
coverage simulations, and known numerical limitations are summarised in
[numerical-validation.md](docs/numerical-validation.md).

## Installation

Python 3.13, Rust, and `uv` are required.

```sh
uv venv --python 3.13
uv pip install 'maturin>=1.10,<2' numpy pytest
uv run --no-project maturin develop --release
```

## One trait, one relationship matrix

```python
import asterism

k, order = asterism.relationship_matrix(
    ids,
    father,
    mother,
    keep=analysed_ids,
)

# Align x and y to `order`; alignment is positional.
model = asterism.prepare(x, k)
fit = model.fit(y)              # REML by default
interval = model.interval(y)    # profile interval for h2
test = model.test(y)            # h2 = 0
```

`x` is the fixed-effect design and must include its own intercept column when
one is wanted. `prepare` checks the matrix and design, diagonalises the
relationship matrix, and stores everything independent of the response. Reuse
one prepared model for multiple responses with the same rows and design. Pass
`reml=False` to use ML.

The relationship builder returns twice the kinship coefficient. `prepare`
also accepts any finite, symmetric, positive-semidefinite matrix with the right
dimensions.

## Several covariance components

```python
model = asterism.ComponentModel([relationship, household], x)
fit = model.fit(y)
interval = model.interval(y, component=1)
test = model.test(y, component=1)
prediction = model.predict(y, component=0)
```

`fit["variances"]` contains the raw covariance coefficients, and
`fit["raw_coefficient_proportions"]` divides those coefficients by their sum.
The proportions depend on how each matrix is scaled and are not generic
variance shares. `interval(..., component=...)` likewise
profiles a raw coefficient proportion.

When every structured matrix has a positive mean diagonal, the fit also returns
`mean_diagonal_component_contributions` and `mean_diagonal_proportions`. These
are invariant to positive rescaling of a matrix and its reciprocal coefficient.

For a relationship matrix split into off-diagonal kinship classes, the class
bases have zero diagonal. Report coefficients and class contrasts, not shares:

```python
split = asterism.kinship_classes(ids, father, mother, sex, keep=analysed_ids)
classes = asterism.ComponentModel(split["matrices"], x)
omnibus = classes.equality_test(y)
contrasts = classes.contrasts(y, classes=[1, 2, 3, 4])
```

## Two traits

```python
model = asterism.BivariateModel(relationship, observed, design)
fit = model.fit(y)
fit["h2_first"], fit["h2_second"]
fit["rho_g"], fit["rho_e"], fit["rho_p"]
model.interval(y, "rho_g")
model.test(y, "rho_g", null=0.0)
```

`observed` is one Boolean pair per person, permitting different missingness for
the two traits. `design` and `y` contain only observed person-trait rows, in
person order with trait within person.

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

## Gene by environment, continuous and discrete

```python
gxe = asterism.GxeModel(
    relationship,
    environment,
    design,
    surface="random_regression",
)
fit = gxe.fit(y, grid=[low, middle, high])
gxe.test(y, "correlation")
gxe.test(y, "interaction")

# A binary environment: sex, an exposure, a cohort. Two labels, nothing else.
discrete = asterism.DiscreteGxeModel(relationship, sex, design)
fit = discrete.fit(y)
discrete.test(y, "gene_by_environment")
discrete.test(y, "correlation")
discrete.test(y, "genetic")
discrete.test(y, "residual")
```

Available GxE surfaces are `exponential`, `powered_exponential`, and
`random_regression`. They are different covariance families; choose the family
and any powered-exponential shape independently of the fitted outcome. Only
random regression can represent negative genetic correlations and crossovers.

`GxeModel` takes an environment measured on a range and smooths across it.
`DiscreteGxeModel` takes one measured as a binary label — any two distinct
finite values, with the smaller naming the first group — and smooths nothing:
it carries a genetic and a residual variance per group and one correlation
between them. Sex is the canonical use, and `fit` returns the two `levels` so a
reader can tell which group is which.

The principal `gene_by_environment` test leaves residual variances free in the
two environments; `any_difference` additionally equates them and therefore is
not specifically a genetic test.

## Binary liability

```python
model = asterism.LiabilityModel(relationship, affected, design)
fit = model.fit()
interval = model.interval()
test = model.test()
```

This is maximum likelihood on an unobserved probit liability whose variance is
fixed at one. The reported heritability is on the liability scale, not the
observed binary scale. Family probabilities above dimension two use the
Mendell-Elston sequential approximation.

## Marker association

```python
scan = asterism.AssociationModel(relationship, design_with_pcs, y)

fast = scan.sweep(markers)                    # covariance held
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

weighted = dosages * variant_weights          # Z = G W, one column per variant
result = model.test(weighted)                 # variance-component test
family = model.test_family(weighted)          # across burden-to-variance-component
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

## Testing

```sh
cargo test --release
uv run --no-project pytest tests/ -q
uv run --locked --no-sync python checks/against_famskat.py
```

The full list of simulation and package-comparison commands is in
[numerical-validation.md](docs/numerical-validation.md).
