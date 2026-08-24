# Asterism

Asterism is a Rust/Python package for quantitative-genetic variance-component
models. It accepts numerical arrays already in memory and returns ordinary
Python dictionaries and arrays.

The model equations and interpretation are in
[statistical-methods.md](docs/statistical-methods.md). Empirical comparisons,
coverage simulations, and known numerical limitations are summarised in
[numerical-validation.md](docs/numerical-validation.md).

This checkout is a development build, not an analysis-ready release. Its
embedded `release.toml` deliberately refuses reportable status until every 0.1
scientific pass rule has been rerun on the exact release
artifact. [Python API support status](docs/api-support.md) distinguishes what is
planned for 0.1 from public capabilities outside its scientific support. The
[analysis-receipt guide](docs/analysis-receipts.md) shows the fixed-wheel
release-state check, three outcomes and caller-owned JSON boundary.

## Installation

CPython 3.13 or 3.14, the repository-pinned Rust 1.97.1 toolchain, and `uv` are
required for development. Build isolation, the development environment, and
hosted wheel builds all use the locked Maturin 1.14.1 release.

```sh
uv sync --locked --all-groups
uv run maturin develop --release --locked
```

An analysis that may become reportable installs a checksummed wheel saved by a
fixed release; it never builds from an editable checkout.

## One trait, one relationship matrix

```python
import asterism
import numpy as np

k, order = asterism.relationship_matrix(ids, father, mother, keep=analysed_ids)

# Line the values up with the matrix by identifier, not by hope.
data = asterism.align(k, order, table_ids, y=height, age=age)
order_sha256 = asterism.subject_order_commitment(data["order"])

model = asterism.prepare(
    np.column_stack([np.ones(len(data["order"])), data["age"]]),
    data["relationship"],
    subject_order_sha256=order_sha256,
)
fit = model.fit(data["y"])  # REML by default
fit["h2"], fit["interval"], fit["test"]

ml = model.fit(data["y"], estimator="ml")
```

**Asterism's numerical interface is positional, and that is the one place a
mistake makes no noise.** A relationship matrix whose rows are in a different
order from the response does not fail or warn. On 300 people simulated at a
heritability of 0.6, the aligned fit returns 0.490 with `p = 1.4e-06`; the same
data with the response shuffled returns 0.000, an interval of `[0.000, 0.081]`
and `p = 1`. It does not perturb the answer, it destroys the signal and then
reports no heritability with confidence. `align` takes the matrix with its own
identifiers and the values with theirs, lines them up once, and refuses what it
cannot. It reads a genomic relationship matrix or an estimated kinship computed
elsewhere just as well as a pedigree one — that pairing, a matrix beside a
separate identifier file, is exactly where this goes wrong.

`x` is the fixed-effect design and must include its own intercept column when
one is wanted. `prepare` checks the matrix and design, diagonalises the
relationship matrix, and stores everything independent of the response. Reuse
one prepared model for multiple responses with the same rows and design.
Every supported 0.1 fit record carries immutable build identity and a
`subject_order_sha256` field. Supplying the commitment to its public fit route
echoes the digest without retaining identifiers; the standard analysis runner
also verifies and attaches it before a result can be reportable. Build identity
binds the source commit, release manifest, `Cargo.lock`, and `uv.lock`, so two
builds with the same public version but different source or dependencies remain
distinguishable.

One fit returns everything about that fit: `interval` is the profile interval
for the heritability and `test` is the test against nought, both already inside
the record rather than separate calls that would refit.

The relationship builder returns twice the kinship coefficient. `prepare`
also accepts any finite, symmetric, positive-semidefinite matrix with the right
dimensions.

## Several covariance components

```python
model = asterism.ComponentModel([relationship, household], x)
fit = model.fit(y)
interval = model.interval(y, component=1)
test = model.test(y, component=1)
```

`fit["variances"]` contains the raw covariance coefficients, and
`fit["raw_coefficient_proportions"]` divides those coefficients by their sum.
The proportions depend on how each matrix is scaled and are not generic
variance shares. `interval(..., component=...)` profiles the matching
scale-invariant mean-diagonal proportion by default. The raw coefficient
proportion remains available explicitly as a diagnostic quantity.

When every structured matrix has a positive mean diagonal, the fit also returns
`mean_diagonal_component_contributions` and `mean_diagonal_proportions`. These
are invariant to positive rescaling of a matrix and its reciprocal coefficient.
The numerical model rejects linearly dependent covariance bases, including a
submitted identity matrix that duplicates the implicit residual. Passing that
exact-rank check does not by itself prove that nearly collinear components are
estimated precisely.

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
The 0.1 reportable target is `rho_g` with its interval and test; the other fitted
quantities describe the joint fit but are not additional 0.1 claims.

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
The powered-exponential family remains outside 0.1 scientific support.

`GxeModel` takes an environment measured on a range and smooths across it.
`DiscreteGxeModel` takes one measured as a binary label — any two distinct
finite values, with the smaller naming the first group — and smooths nothing:
it carries a genetic and a residual variance per group and one correlation
between them. Sex is the canonical use, and `fit` returns the two `levels` so a
reader can tell which group is which.
Group-specific heritabilities remain descriptive in 0.1; the genetic
correlation, its interval, and the calibrated genetic tests are the supported
targets.

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

## A trait whose measurement stops at a limit

```python
fit = asterism.tobit_fit(relationship, value, censoring, limit, design)
fit["heritability"], fit["total_variance"], fit["censored_share"]

interval = asterism.tobit_interval(relationship, value, censoring, limit, design)
```

`censoring` is 0 where the value was measured, 1 where it lies at or above its
limit, and 2 where it lies at or below it. `value` is read only where the status
says measured, and `limit` only where it does not.

**The status is given rather than inferred, and the limit belongs to the
observation rather than to the trait.** Extended high-frequency audiometry is
the case this was built for, and there the recorded maximum differs between
frequencies and between sessions, so a censored value can carry the same number
as a genuinely measured one. Only the status tells them apart.

The heritability that comes back is the heritability of the *complete* variable
— the number there would have been had the instrument reached far enough. It is
comparable with an ordinary heritability of an uncensored trait, and **not**
comparable with one fitted to values where the censored ones were replaced by
their limit, which is the usual practice and the thing this exists to replace.
It is maximum likelihood, never REML, because a censored observation has no
response to project onto the null space of the design.
The 16 and 18 kHz uses remain blocked until coverage at their approximately 52
and 75 per cent censoring levels passes on the fixed release artifact.

## Two traits measured differently

```python
first = {"kind": "binary", "value": v1, "censoring": c1, "limit": l1}
second = {"kind": "censored", "value": v2, "censoring": c2, "limit": l2}

fit = asterism.mixed_bivariate_fit(relationship, first, second, design)
fit["genetic_correlation"], fit["residual_correlation"]
```

Either trait may be `continuous`, `binary` or `censored`, and the pair may be
any combination of the three. The covariance is `BivariateModel`'s; what varies
is only how each observation is seen — a density at a point, or the probability
of a region, with a continuous value the degenerate region.

**A binary trait's variance is fixed at one** and comes back as one, because
only the sign of a liability is ever seen. Its heritability is a liability
heritability while a continuous or censored trait's is not, and the two must not
be read as the same quantity. The genetic correlation is unaffected by that
difference, which is what makes a mixed pair worth fitting at all: a correlation
is scale free even where one of its two scales is arbitrary.
Only the prespecified binary psychiatric diagnosis/right-censored hearing pair
is planned for 0.1 support. Its independent SOLAR fixture is now frozen, but
the exact fixed-wheel simulation campaigns remain
unfinished. Other trait-kind combinations remain public but scientifically
unsupported.

## Marker association — outside 0.1 scientific support

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

## Variant sets: genes and pathways — outside 0.1 scientific support

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

## Latent mediation with continuous and threshold observations — outside 0.1 scientific support

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
uv run pytest tests/ -q
uv run --locked python checks/against_famskat.py
```

The full list of simulation and package-comparison commands is in
[numerical-validation.md](docs/numerical-validation.md).

## Citing Asterism

Cite the archived version you actually ran, not the repository. A development
checkout is not a citable version: only a release wheel carries the build
identity a result can be traced back to, and only a release has had its pass
rules configured.

Details are in [CITATION.cff](CITATION.cff). Asterism is archived at
publication rather than before it, so the DOI is added there when the first
release is published — see
[ADR 0018](docs/adr/0018-published-with-the-papers-that-cite-it.md).

## Licence

MIT. See [LICENSE](LICENSE).

Asterism is original work. It depends on third-party packages, named in
`Cargo.lock` and `uv.lock` and used under their own licences, and derives from
nothing else. It has no affiliation with SOLAR or with any other quantitative
genetics package. Where its answers are compared with SOLAR, R, `spaMM`,
MCMCglmm or a published analysis, those are benchmarks: agreement proves
fidelity and never correctness, which is
[ADR 0006](docs/adr/0006-agreement-proves-fidelity.md).
