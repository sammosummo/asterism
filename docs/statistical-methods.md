# Statistical methods

## Gaussian variance components

For one observed trait, Asterism fits

```text
y = X beta + sum_k u_k + e
u_k ~ N(0, theta_k K_k)
e   ~ N(0, theta_e I)
V   = sum_k theta_k K_k + theta_e I.
```

`X` is supplied by the caller and must include an intercept column when one is
wanted. The matrices `K_k` are fixed numerical covariance bases. Their rows and
the rows of `X` and `y` are aligned positionally.

Maximum likelihood (ML) and restricted maximum likelihood (REML) are available;
REML is the default for Gaussian models. Fixed effects are estimated by
generalised least squares at each covariance point. The one-component model
uses `V = sigma2 [h2 K + (1-h2) I]`, profiles `sigma2`, diagonalises `K` once,
and reuses that decomposition for every response fitted to the same design.
Models with several non-commuting matrices factorise `V` at each likelihood
evaluation.

The best linear unbiased prediction for component `k` is

```text
u_hat_k = theta_k K_k V^-1 (y - X beta_hat).
```

Prediction errors treat the fitted covariance coefficients as known, so they
do not include uncertainty from estimating those coefficients.

### Matrix scaling and component interpretation

The fitted `theta_k` is a covariance coefficient. If `K_k` is replaced by
`c K_k`, the same covariance is represented by `theta_k / c`. Consequently,
`theta_k / sum_j theta_j` is scale-dependent. The API calls these
`raw_coefficient_proportions`; they are not generic variance shares.

When every structured matrix has a positive finite mean diagonal, define

```text
m_k = mean(diag(K_k))
c_k = theta_k m_k
p_k = c_k / (sum_j c_j + theta_e).
```

The `c_k` are mean-diagonal covariance contributions and the `p_k` are their
proportions. They are invariant to positive rescaling of any `K_k`. Asterism
reports them only when this construction is defined for all structured
matrices.

Kinship-class bases made by splitting only selected off-diagonal relationships
have zero diagonal. Their coefficients cannot be converted to mean-diagonal
proportions. Report class coefficients, an omnibus equality test, and contrasts
among classes—not shares.

A class *weight* — the ratio of two estimated variances — is badly biased on
the designs studied here. On the GOBS pedigree the weights come back near 3
when the truth is 1, while the shares themselves are unbiased. Report shares
and contrasts; do not report a ratio of two estimated variances as though it
were a parameter.

The several matrices must be distinguishable in the observed design. For an
additive, household, and residual model, for example, `A`, `H`, and `I` must be
linearly independent over the fitted roster. Unrelated co-residents are useful
but not required; varied relatedness among co-resident pairs can also identify
the components.

## Intervals and tests

Reported confidence intervals are profile-likelihood intervals. For a scalar
interior quantity, the 95% endpoints solve

```text
2 [ell(theta_hat) - ell(theta)] = 3.8415.
```

For one heritability exactly at 0 or 1, inclusion of the boundary point uses
the 50:50 point-mass/chi-square reference, whose 95% critical value is 2.7055.
A fit at a boundary is reported at that boundary; a symmetric curvature
standard error is not meaningful there.

Testing one nonnegative component against zero uses a reduced model without
that component and the Self-Liang 50:50 mixture when all other components are
interior. An equality test among `q` split-matrix coefficients compares the
free coefficients with their pooled matrix and uses chi-square on `q-1`
degrees of freedom. Individual class contrasts should be interpreted after the
omnibus equality test.

Near zero, the one-trait profile interval is conservative because the fitted
maximum is constrained to `[0,1]`. This increases coverage and reduces power;
uniformly narrowing the interval would under-cover at interior values.

### Fixed effects are not comparable across parameterisations

A covariate p-value depends on how the design is written. An uncentred age term
gave `p = 0.035` where the same model with age centred gave `2e-11`; both are
correct for the question each design asks. **A covariate p-value from Asterism
and one from another package are comparable only if the design is parameterised
the same way.** Reparameterised to match, every covariate agreed with SOLAR.

### A singular relationship matrix puts one at the boundary out of reach

Twice the kinship is positive semi-definite but, on a real pedigree, not
positive definite: the smallest eigenvalue over the whole SAFS pedigree is
-1.5e-15. `prepare` accepts this — that is what the -1e-9 eigenvalue floor is
for. The consequence to carry is that **a heritability of exactly one is
unreachable on a singular matrix**: the covariance there is refused, so a fit
that wants the upper bound reports itself as not converged rather than
returning a number.

## Two Gaussian traits

`BivariateModel` fits possibly unbalanced observations from

```text
V = kron(Sigma_g, A) + kron(Sigma_e, I),
```

before unobserved trait rows are removed. The reported quantities are the two
heritabilities, genetic correlation `rho_g`, residual correlation `rho_e`, and
the derived phenotypic correlation `rho_p`.

Correlation zero is an interior one-parameter null and uses chi-square on one
degree of freedom. A genetic correlation of plus or minus one is a one-sided
boundary and uses the 50:50 point-mass/chi-square mixture. Profile intervals
are computed directly for the reported derived quantity. At high heritability,
little residual variance remains and `rho_e` can be weakly identified even when
`rho_p` is stable.

## Spatial covariance

`SpatialModel` adds

```text
K_spatial(i,j) = exp(-lambda d_ij)
```

to any fixed covariance matrices. `1/lambda` determines the range and
`log(2)/lambda` is the half-correlation distance. The range may be profiled as
a free parameter or numerically integrated out. Integration propagates
range uncertainty into component and fixed-effect uncertainty but leaves no
single range estimate for reporting or BLUP.

The fitted covariance coefficients and their raw proportions depend on the
scale of the submitted fixed matrices. When every fixed matrix has a positive
mean diagonal, Asterism also reports each coefficient multiplied by that mean
diagonal and the resulting proportions. Those marginal proportions are
unchanged by multiplying a matrix by a positive constant and dividing its
coefficient by the same constant. A component-index profile interval remains
an interval for the raw coefficient proportion and is labelled accordingly.

When the spatial coefficient is zero, `lambda` is unidentified. The likelihood
ratio therefore has no ordinary chi-square or chi-bar-square reference.
`bootstrap` simulates the reduced model and reports the add-one p-value

```text
p = (1 + number of simulated statistics >= observed) / (B + 1).
```

The half-distance is weakly identified in the studied designs; an interval
that repeatedly reaches its search bounds is coverage without precision. Use
the bootstrapped test for the presence of spatial covariance, and treat a
profiled range primarily as a descriptive point estimate.

## Gene by environment

`GxeModel` multiplies pairwise relatedness by a genetic covariance surface over
the two environmental values and permits environment-dependent residual
variance.

- `exponential` uses log-linear genetic and residual variances and genetic
  correlation `exp(-lambda |z_i-z_j|)`.
- `powered_exponential` replaces the distance by `|z_i-z_j|^kappa`, with
  `kappa` fixed at 0.5, 1.0, 1.5, or 2.0.
- `random_regression` uses positive-semidefinite quadratic covariance surfaces
  and can represent negative correlations and crossovers.

These surface families are not nested. Variance-shape misspecification can be
absorbed as apparent correlation decay, so the surface and any powered-
exponential shape must be chosen independently of the fitted result. The four
powered-exponential shapes are barely distinguishable by likelihood while
changing the answer: on one studied problem they spanned 0.31 in log likelihood
while the fitted genetic correlation ran from 0.92 to 0.45. The shape is fixed
rather than fitted for that reason.

The correlation test asks whether the same genetic effect acts throughout the
environment. The broader interaction test also detects changes in genetic
scale. Their boundary references differ by surface: the exponential decay null
is a flat boundary with the usual 50:50 mixture, while the random-regression
null lies on the curved boundary of a positive-semidefinite cone and the
implemented reference is conservative. Profile intervals can cover while
frequently ending at a bound; the tests are generally more informative.

## Gene by discrete environment

`DiscreteGxeModel` is the two-group form of gene-by-environment, parameterised
directly by genetic variance in each environment, residual variance in each
environment, and their genetic correlation. The environment is any binary
labelling — sex is the canonical case — supplied as exactly two distinct
finite values, with the smaller label naming the first group. The principal
test leaves the two residual variances free, preventing unequal measurement
error from being misread as genetic heterogeneity.

The available nulls are:

| null | question | reference |
| --- | --- | --- |
| `gene_by_environment` | equal genetic variances and correlation one | 1/2 chi-square(1) + 1/2 chi-square(2) |
| `correlation` | correlation one | 1/2 point mass + 1/2 chi-square(1) |
| `genetic` | equal genetic variances | chi-square(1) |
| `residual` | equal residual variances | chi-square(1) |
| `any_difference` | genetic and residual quantities all equal | 1/2 chi-square(2) + 1/2 chi-square(3) |

`any_difference` is not specifically a genetic test because unequal residual
variances can reject it.

## Binary liability

`LiabilityModel` assumes

```text
L = X beta + g + e,
Var(L) = h2 A + (1-h2) I,
status = 1[L > 0].
```

The liability variance is fixed to one for identification, and the intercept
carries the threshold. This is an ML model; REML is not defined for the
orthant-probability likelihood. One- and two-person family probabilities are
evaluated directly. Larger families use Mendell-Elston sequential truncation,
which is an approximation to the multivariate-normal orthant probability.
Off-diagonal relationships above 0.9 are refused because the retained
two-person quadrature loses accuracy near perfect liability correlation.

The reported heritability is on the latent liability scale, not the observed
binary scale. Its test against zero uses the 50:50 point-mass/chi-square
reference, supported by simulation for the studied design.

**A liability heritability must never be placed beside a REML heritability as
though the two were the same number.** They are on different scales and come
from different estimators; the fit record carries `ml` so that this is visible
rather than a matter of memory.

The two-person probability is sixteen-point Gauss-Legendre quadrature, and its
accuracy depends on the liability correlation: worst error 1.3e-9 below a
correlation of 0.5, and 2.3e-4 below 0.99. An additive model puts sibling and
parent-child liability correlations at `h2/2`, so an ordinary pedigree sits in
the good range throughout. A relationship of one does not — monozygotic twins,
or one person entered twice, give a correlation of `h2` rather than `h2/2` —
which is why an off-diagonal relationship above 0.9 is refused. Removing that
guard means replacing the quadrature first.

## Marker association

For marker `j`, `AssociationModel` fits

```text
y = X beta + m_j gamma_j + g + e.
```

With `variance="held"`, covariance coefficients are estimated under the null
design and held while each marker is tested by generalised least squares. The
Wald statistic and likelihood-ratio statistic are then algebraically identical.
With `variance="refitted"`, the covariance is re-estimated for every marker;
this is slower and can differ materially for large effects. `refit_below`
supports a conservative held screen followed by selective refitting.

The tested marker remains in the relationship matrix; leave-one-chromosome-out
analysis is not performed. This tends to bias the held test toward the null,
especially for large-effect markers. Multiple-testing correction, allele
coding, missingness, and marker quality are outside the model.

When screening with `refit_below`, set the screen well above the threshold you
will report against. The held statistic is conservative, so a marker can have a
refitted p below your threshold while its held p sits above it. Ten times the
threshold is ample for a gap that never exceeded a factor of 1.1 in the cases
studied.

## Gene and local-IBD matrix construction

For genotype matrix `G` and nonnegative column multipliers `w`, the weighted
linear and burden matrices are

```text
Z_ij = G_ij w_j                 K_linear = Z Z'
b_i  = sum_j G_ij w_j           K_burden = b b'.
```

Thus `w_j^2` is the linear-kernel variance weight. The builders do not centre,
standardise, impute, or normalise. Dosages must be finite and in `[0,2]`.

For two founder-lineage labels per person, let `H_il` count copies carrying
lineage `l`. The local additive relationship matrix is

```text
K_local = H H' / 2.
```

Off-diagonal IBD0, IBD1, and IBD2 are 0, 0.5, and 1. A locally autozygous
person has diagonal 2. For lineage draws `d`, the posterior builder returns the
nonnegative weighted mean `sum_d p_d K_d`. This is a plug-in expected matrix;
it does not integrate lineage uncertainty inside the phenotype likelihood.

All four builders return ordinary dense arrays and require complete numerical
inputs. Their memory cost is quadratic in the number of people, and posterior
local-IBD construction takes one quadratic pass per draw.

### Using these matrices to scan

A builder produces one covariance basis. The test comes from putting it in a
`ComponentModel` beside the genome-wide additive matrix and testing it against
zero. For linkage that is

```text
y = X beta + local + polygenic + e
```

with `local` scaling the local-IBD matrix at one position. Nothing here
corrects for testing many positions or many genes.

**The local and polygenic matrices are separable because the first averages to
the second.** Among full sibs, IBD at a locus is 0, 0.5 or 1 in the ratio
1:2:1, whose mean is the polygenic 0.5. All the linkage information is in the
variation around that mean.

**An estimated IBD matrix costs power, not validity.** Real IBD is a posterior,
and a posterior mean is shrunk towards the polygenic matrix, which takes the
variation with it. In simulation the test held its level at every resolution
studied and grew conservative as resolution fell, while power against a locus
explaining a quarter of the variance fell from 0.89 to 0.10. A shrunken matrix
cannot manufacture a signal; it can only fail to find one, and it does so
silently.

The quantity to compute before scanning is therefore the **standard deviation
of the local IBD values among relatives**. With full sib information and IBD
known exactly it is `sqrt(0.125) = 0.354`. Divided by that, it says how much of
the available information the markers have kept, and power follows it.

### Two ways to get a wrong answer

**Lineage labels are compared for equality and nothing else**, so a missing
lineage must not be encoded as one. Everyone carrying an "unknown" label would
appear to share descent: a pair reaches `K = 2`, past monozygotic twins, and
each reads as autozygous, putting a peak exactly where the genotyping is
poorest.

Labels must therefore be **positive**. Nought and negatives are refused with
`LOCAL_IBD_LINEAGE_NOT_POSITIVE`, on both the Rust and Python surfaces and in
every posterior draw including any the weights would discard, because they are
the missing codes in every pedigree format.

That closes the ordinary case but cannot close every one: a sentinel such as
`999` is a legal label and will build. It betrays itself on the diagonal, since
everyone carrying it becomes autozygous. **Count `(numpy.diag(k) == 2).sum()`
before scanning.** Genuine local autozygosity is real — the reviewed SAFS
pedigree has inbred individuals — so this is a number to judge against the
consanguinity you expect, not a value to refuse.

**A variant set with no carriers is refused rather than returned as zero.**
`gene_linear_matrix` and `gene_burden_matrix` raise
`GENE_MATRIX_WEIGHTED_VALUES_ALL_ZERO` when every weighted dosage is zero,
which in a gene-based sweep means a gene nobody in the sample carries. A scan
should catch that code and record the gene as untestable. Nothing is lost by
doing so: a zero matrix carries no variance, and `ComponentModel` returns
`p = 1` for it.

## Latent mediation with mixed observation types

`LatentMediationModel(families, qmc_points=...)` fits the structural model

```text
M = a U_M + epsilon_M
Y = b M + c_prime U_M + d U_Y + epsilon_Y,
```

where `U_M` and `U_Y` are independent inherited processes with covariance `A`.
The outcome innovation variance is fixed at one; the mediator residual variance
`sigma_m2` is positive and free; `a` and `d` are nonnegative; `b` and
`c_prime` are signed. With `t = a b + c_prime`, the trait covariance is

```text
G = [[a^2, a t], [a t, t^2 + d^2]]
E = [[sigma_m2, b sigma_m2],
     [b sigma_m2, b^2 sigma_m2 + 1]]
Omega = kron(G, A) + kron(E, I).
```

For a family of size `n`, the public mapping contains:

| field | mathematical role |
| --- | --- |
| `relationship` | the finite, symmetric, positive-semidefinite `n x n` additive relationship matrix, with diagonal one |
| `latent_mean` | a finite length-`2n` mean vector, mediator entries followed by outcome entries |
| `mediator_measurement` | optional observations `C_i = M_i + eta_i` |
| `mediator_measurement_error_variance` | known positive `Var(eta_i)` wherever `C_i` is observed |
| `mediator_proxy_status` | optional binary observation of a thresholded but fallibly measured mediator state |
| `outcome_status` | optional binary observation `1[Y_i > outcome_threshold_i]` |
| `mediator_threshold`, `outcome_threshold` | finite person-specific thresholds; a scalar is expanded across the family |
| `mediator_proxy_sensitivity`, `mediator_proxy_specificity` | person-specific values strictly between zero and one; a scalar is expanded |
| `ascertainment`, `proband_index` | either population sampling with no proband, or conditioning on one named observed outcome case |

`None` denotes an unobserved continuous measurement or binary status. A missing
continuous measurement must also have a missing error variance. Families must
be separate connected components and contribute independent likelihood factors.

The likelihood first conditions the latent Gaussian vector on all continuous
mediator measurements. For every observed mediator proxy, it sums over the
unobserved true threshold state `T_i = 1[M_i > mediator_threshold_i]`. Its
measurement factor is

```text
P(R_i=1 | T_i=1) = sensitivity_i
P(R_i=0 | T_i=0) = specificity_i,
```

with the complementary probabilities for false negatives and false positives.
For each joint true-state configuration, that measurement factor multiplies the
conditional Gaussian rectangle probability for the mediator thresholds and the
observed outcome thresholds. Missing observations contribute no term.

With `ascertainment="population_unconditioned"`, this is the family likelihood
directly and `proband_index` must be absent. With
`ascertainment="condition_on_named_proband_case"`, the named person must have
`outcome_status=1`; the numerator is divided by that person's marginal
probability of exceeding their outcome threshold.

The vertical estimand is `a b`, whose null is the union of `a=0` and `b=0`; the
horizontal estimand is `c_prime`. At least one continuous mediator measurement
across all families is required by `fit()` to anchor the mediator scale.
Fixed-point `evaluate()` can still evaluate a fully specified parameter point
without that anchor.

The unrestricted five-parameter maximum is found by deterministic multi-start
L-BFGS-B. One-dimensional normal probabilities are direct; two-dimensional
rectangles use adaptive quadrature; higher-dimensional rectangles use
deterministic quasi-Monte Carlo with the requested `qmc_points`. Evaluation and
fit results expose numerical diagnostics, including convergence, projected
gradient, integration method, work, and QMC batch stability.

The current method supplies likelihood estimates and point estimates, not a
calibrated p-value or confidence interval for `a b`. Higher-dimensional QMC is
deterministic and diagnostically monitored, but its stability statistic is not
an inferential error bound. A causal interpretation additionally depends on the
structural assumptions, not merely on maximising this likelihood.
