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
the designs studied here. Simulated on the GOBS pedigree at 200 replicates with
every true weight 1.0, the four class weights came back at 2.34, 2.38, 2.44 and
2.59, while the coefficient proportions were unbiased to within 0.007 and
covered at 0.925 to 0.945. Report proportions and contrasts; do not report a
ratio of two estimated variances as though it were a parameter.

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

### An interval reports how many of its own evaluations failed

Every profile interval returns `profile_failures` beside its endpoints: the
number of points along the way where the constrained fit could not be made, or
stopped without converging.

The count matters because of what is done with such a point. A failure is
unknown ground, not ground the data ruled out, so the interval is widened over
it. Read the other way -- as an infinite deviance, or as a likelihood that had
fallen below the threshold -- a failure looks exactly like the crossing the
bisection is searching for, and the search stops there. That returns an
interval narrower than the data support, with the same confident endpoints as
a real crossing and nothing to distinguish the two. Widening instead can only
be conservative, and `profile_failures` says how much of the answer rests on
points that did not come back. Nought is the ordinary case and means the
endpoints are crossings throughout.

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

## Censored traits

`tobit_fit` assumes every person carries a complete value

```text
y* = X beta + g + e,
Var(y*) = h2 sigma2 A + (1-h2) sigma2 I,
```

seen exactly where the instrument reached it, and known only to lie beyond a
limit where it did not. The limit belongs to the observation rather than to the
trait, and the censoring status is an input rather than something inferred from
the value — a censored observation can carry the same number as a measured one.

Unlike the liability model, the scale is identified here, because the measured
values arrive on it. So `sigma2` is estimated and the heritability returned is
that of the complete variable. A family splits into its measured and unmeasured
members: the measured contribute a multivariate-normal density, and the
unmeasured the probability that their values lie beyond their limits
**conditional on the measured ones** — not marginally, because within a family
the two are correlated and that correlation is the information a pedigree
carries. The region probability is the one `LiabilityModel` uses: exact to two
people, and Mendell-Elston sequential truncation above that.

This is ML for the same reason the liability model is: a censored observation
has no response to project onto the null space of the design. At least two
measured values are required, below which the scale is not identified.

In calibration at 400 sibling pairs with a true heritability of 0.5, the model
returned 0.496 with nothing censored and 0.494 with half censored, where
replacing the censored values by their limit gave 0.398. Interval coverage was
within nominal at heritabilities of 0 and 0.3 across censoring shares to 0.5.
**All of that evidence was generated on pairs, where the region probability is
exact and the sequential approximation never runs at all.** Its behaviour in
large families, conditional on many measured values, is not established by it.

`mixed_bivariate_fit` places two traits of any kinds — continuous, binary or
censored — in `BivariateModel`'s covariance, each observation entering as a
density or as a region probability according to its kind. A binary trait's
variance is fixed at one; a continuous or censored trait's is free. The genetic
correlation is scale free and is therefore comparable across kinds, while the
heritabilities are not.

## Repeated measures at fixed positions

One trait measured at `T` fixed positions on an ordered continuum, `R` times
over, on a pedigree:

```text
y[p, r, t] = x[p, r]' b[., t] + sum_j u[j, p, t] + e[p, r, t]
```

The person-level effects `u[j, p, .]` are shared by every replicate of that
person, so the covariance is

```text
V = sum_j (K_j (x) J_R (x) S_j) + I (x) I_R (x) S_e
```

with `J_R` the `R x R` matrix of ones. **The replicate is a level within the
person and not a second set of traits.** For an audiogram that is the statement
that a genotype does not know left ear from right: the loadings are shared and
left--right asymmetry gets `S_e` rather than being averaged away.

Each `S_j` is either free, or

```text
S_j = D_j R_j D_j,   R_j(t, u) = c_j + (1 - c_j) exp(-lambda_j d(t, u))
```

-- a free variance at every position and one floor and one rate per component,
where `d` is separation on whatever line the caller supplies. The floor is a
common factor and the exponential is the local decay; a large `lambda` gives a
one-factor model and a `c` of nought a pure distance model, so the fit chooses
between them. **The two are not separately estimable and the correlation they
describe is**, so the fit reports the correlation as a function of separation.

**Estimator: maximum likelihood, never REML**, for the reason every censored
model here gives -- a region has no response to project onto the null space of
the design. Fitted by expectation-conditional-maximisation: the person-level
effects are the missing data, and with a kernel each component's maximisation is
a small bounded search rather than a closed form.

Covariates enter the likelihood rather than being removed beforehand, because
**a censored observation cannot be residualised**: there is no value to subtract
a fitted mean from, so residualising first silently forces the analyst back to
substituting the limit.

Values that were never measured contribute nothing to the likelihood, which is
the ordinary likelihood for data missing at random. They are imputed in the
expectation step regardless, because dropping them would unbalance the data.

**Two approximations, and they are not of the same order.** The region
probability the likelihood reports is exact at one and two censored coordinates
in a family and Mendell-Elston sequential truncation above; the moments the
expectation step uses have only the sequential update. So the fixed point of the
one need not be the maximum of the other, and the fit's gradient reading grows
with the share of the data that is censored rather than measuring the search.
`docs/numerical-validation.md` measures both it and what it costs: five
orders of magnitude in the reading, against 0.015 in a correlation at half
censored and nothing detectable at a tenth.

**There is no interval and no test.** What an interval should be of is an open
question, not an unwritten function.

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

### Which relationship matrix a scan should use

**A pedigree kinship is not enough for an association scan, and the shortfall
has been measured.** A pedigree kinship gives the probability two people share
an allele by descent, averaged over meioses that have already happened. Real
genotypes carry the relatedness that actually resulted, which scatters about
that average. A marker whose genotype tracks the difference is a marker the
covariance model has not accounted for, and the test reads the leftover as
signal.

On the real pedigree with real markers and no marker effect anywhere, this puts
the crossings below `0.01` at 522 per draw where 500 are expected. Permuting each
marker across people -- keeping its allele frequency and removing its relation to
who is related to whom -- brings that to 510, so about half the excess is this
and not the test. It is largest for common markers, which carry the most of the
signal, and a frequency filter therefore does not help: the ratio is 0.98 for
the rarest markers and 1.07 for the commonest. Refitting the variance components
for every marker does not help either.

So use a genomic relationship matrix, or a kinship estimated from the genotypes,
for a scan. `prepare` and `AssociationModel` take any finite symmetric
positive-semidefinite matrix, so this is a choice of input and not a change of
model. A pedigree kinship remains the right thing for heritability, where the
question is about descent rather than about what a particular marker tracks.

When screening with `refit_below`, set the screen well above the threshold you
will report against. The held statistic is conservative, so a marker can have a
refitted p below your threshold while its held p sits above it. Ten times the
threshold is ample for a gap that never exceeded a factor of 1.1 in the cases
studied.

## Testing a whole variant set

Testing rare variants one at a time finds nothing, because each has a handful
of carriers. `VariantSetModel` asks instead whether the variants in a set carry
more trait variance together than chance allows:

```text
y = X beta + Z gamma + g + e,   gamma_j ~ (0, tau),   g ~ N(0, sigma_g^2 A)
```

with `Z = G W` the dosages already multiplied by their column weights, which
the caller forms — a column multiplier `w` means a variance weight of `w^2`,
and nothing here centres, standardises or imputes, and the
test being of `tau = 0`. This is the model famSKAT fits, and Asterism agrees
with famSKAT to `2e-6` relative, which is about the accuracy either computes.

**The test is a score test, not a likelihood ratio, and that is the point.**
The null sits on a boundary, and the usual 50:50 chi-bar-square reference is
right only when the tested matrix spreads across many eigenvalues. A
variant-set kernel does not: a burden kernel has rank one. With `r = P y` for
the residual projection at the fitted null covariance,

```text
Q      = || Z' r ||^2
lambda = eigenvalues of Z' P Z
Q      ~ sum_i lambda_i chi-square(1)   under the null
```

so the reference is computed from the kernel rather than assumed. The null is
fitted once for a whole scan rather than refitted per set.

**The interface takes the root `Z`, not the kernel `Z Z'`.** They carry the
same information, and the root costs one column per variant instead of one
column per person: nothing of size `n by n` is formed, and the eigenvalues come
from a matrix the size of the variant set rather than the roster.

**The weights are the caller's choice and are not innocent.** A column
multiplier `w` is a variance weight of `w^2`. The usual rare-focused choice
evaluates a `Beta(1, 25)` density at each minor allele frequency, but that
encodes a belief about which variants matter and a different belief gives a
different answer. Under the null the weight shape is unidentified, exactly as
the spatial range is when the spatial variance is zero, so it cannot be fitted
as a free parameter without destroying the reference distribution. Running a
small pre-specified set of weightings and combining them is the honest course.

Nothing here corrects for testing many sets, and a set nobody in the roster
carries is refused rather than returned as a statistic of nought.

### One dial from variance component to burden

Two tests bet on different truths about a set. A **burden** test assumes every
variant pushes the trait the same way, adds them into one score, and tests
that: powerful when true, and blind when half the variants raise the trait and
half lower it, because they cancel in the sum. A **variance-component** test
assumes nothing about direction and asks only whether the effects are more
scattered than chance allows: robust to a mixture, weaker when they genuinely
agree.

They are two ends of one dial. Assume the variant effects have correlation
`rho` with each other, so their covariance is `(1-rho) I + rho 11'`. Then

```text
Q_rho  = (1 - rho) s's + rho (1's)^2,      s = Z' P y
lambda = eigenvalues of R^(1/2) (Z' P Z) R^(1/2),   R = (1-rho) I + rho 11'
```

with `rho = 0` the variance-component test and `rho` approaching one the burden
test. Both come from the same `s` and `Z' P Z`, so a whole family costs one
decomposition per value rather than one fit per value.

Neither end dominates. Simulated at 300 people with a twenty-variant set,
power at nominal 0.05:

| truth | variance component | near burden | combined |
| --- | --- | --- | --- |
| nothing (level) | 0.0435 | 0.0490 | 0.0450 |
| all one direction | 0.428 | 0.738 | 0.715 |
| mixed directions | 0.730 | 0.167 | 0.683 |

**Taking whichever test came out best inflates a p-value unless the looking is
paid for.** These tests are strongly dependent, being one score read under
different assumptions, so the members are combined by the Cauchy method, whose
tail is correct whatever the dependence between them. The combined test holds
its level and lands near the winner in both directions, which is the point.

The correlation whose test came out strongest is reported because it describes
the set, but it is a description and not an estimate, and reporting its p-value
alone would be exactly the inflation the combination prevents.

The same mechanism combines across **weightings**, which is the honest response
to a weight being an arbitrary choice: run a few pre-specified ones and combine
them, rather than fitting one, which the null does not identify.

## The tail of a weighted sum of chi-squares

A score test for one variance component gives a statistic distributed as

```text
Q = sum_j lambda_j chi-square(1)
```

where the weights are the eigenvalues of the tested matrix after projection.
There is no closed form in general. A gene scan reads this at around `1e-6`, so
an approximation that behaves at 0.05 is worthless.

Asterism computes it by Gil-Pelaez inversion,

```text
P(Q > q) = 1/2 + (1/pi) Int_0^inf sin(theta(t)) / (t rho(t)) dt
theta(t) = (1/2) sum_j atan(2 lambda_j t) - q t
rho(t)   = exp( (1/4) sum_j log(1 + 4 lambda_j^2 t^2) )
```

cutting the integral at every zero of `sin(theta)` so that successive pieces
alternate in sign and shrink. Truncating there costs about one piece rather
than the whole remaining envelope, which converges far faster.

Two cases have exact answers and are taken directly rather than integrated: a
single weight, and weights that are all equal. The first is what a burden
kernel produces, and it is the case the published inversion handles worst.

**The probability is recovered as `1/2` plus an integral, so a small tail is a
difference of two nearly equal numbers.** About six digits are lost by the time
the tail reaches `1e-6`. That leaves enough in binary64 for a gene scan and not
enough for a genome-wide single-variant threshold, so the result carries the
size of the last term kept and a flag saying whether cancellation has left the
value readable.

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

**The whole likelihood is carried on the log scale**, and the rectangle
probabilities are summed across truth configurations by log-sum-exp rather than
added as ordinary numbers. A family deep in the tail has every configuration
below the smallest double, so adding them plainly would lose the sum to
underflow however carefully each part was computed. Checked against an
independently written evaluation of the same equations, a family log likelihood
of `-4546.421139077654` is reproduced exactly, to the last bit. That is about
`1e-1974`, some 1,650 orders below the smallest number a double can hold.

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
horizontal estimand is `c_prime`.

**The union null is why a single likelihood ratio has no reference here**, and
why the test is an intersection-union: `a b = 0` holds whenever the mediator
carries no inherited signal or the mediator does not reach the outcome, and
those are different models. Rejecting only when both parts are rejected gives
the larger of the two p-values, which is exactly level `alpha` and conservative,
most so near `a = b = 0` where both parts hold at once. `a` sits on a boundary
under its null and takes the even mixture; `b` is interior and takes an ordinary
chi-square on one.

### The reference for `a = 0` depends on where the null fit rests

The loading `a` is non-negative, so its null sits on a bound and takes the even
mixture of a point mass at nought and chi-square on one. That reference assumes
`a` is the only parameter on a bound. Where the outcome loading `d` rests on
its bound as well -- a trait with little inherited outcome variance, which is
not rare -- the fit sits on a corner of the cone rather than a face and the
mixture is the wrong reference. Because the union rule reports whichever part
is larger, a wrong loading p-value becomes the reported one whenever it binds,
so the test refuses with `LATENT_MEDIATION_ANOTHER_LOADING_AT_ZERO` rather than
answering.

Both fits are asked, not just the free one. The reference belongs to the null,
and `d` can be comfortably positive with `a` free and fall to nought once `a`
is held there -- which is precisely the corner the guard exists to catch, and
the free fit alone reports nothing about it.

### The point estimate of `a b` is weaker than the test of it

**The outcome has no continuous form in this interface.** A mediator may be
measured continuously, or observed through a fallible binary proxy, but the
outcome is only ever a status: `1[Y > threshold]`. The outcome *innovation*
variance is fixed at one rather than the outcome variance, so the only thing
constraining the scale of `Y`, and with it `b`, is how often the threshold is
crossed. `a` is pinned by the continuous mediator measurement; `b` is not.

Measured on a deliberately small design — 36 people, 12 families of three,
outcomes binary — the point estimate ran away while the test held:

| truth | `a b` estimated | p(a=0) | p(b=0) | p(a b = 0) |
| --- | --- | --- | --- | --- |
| a=0.7, b=0.6 (a b = 0.42) | +7.79 | 0.018 | 0.742 | 0.742 |
| a=0, b=0.6 (a b = 0) | +0.54 | 0.397 | 0.136 | 0.397 |
| a=0.7, b=0 (a b = 0) | **-2459** | 0.008 | 0.123 | 0.123 |
| a=0, b=0 (a b = 0) | -0.09 | 0.288 | 0.774 | 0.774 |

Every p-value is the right side of 0.05: all three true nulls are not rejected,
and the one real mediation is not claimed either, because `b` could not be
established from 36 binary outcomes.

Notice which row runs away. With both parameters at nought the estimate is
-0.09, near enough right; the damage is done in the third row, where `a` is
large and identified while `b` is nought and unconstrained. That is the shape
the argument above predicts, and it is the shape to watch for. **The test is a likelihood ratio and does
not depend on the estimate**, which is why it survives a point estimate of
minus two thousand.

The practical reading. Report the two parts beside the combined p-value; a
large `a b` next to a `b` that no constrained fit can reject is what a loose
scale looks like. Treat `a b` as a direction rather than a magnitude unless the
outcome is common enough, and the sample large enough, for the threshold to
carry real information about the scale. And note the ceiling this puts on the
design: a continuous outcome would pin `b` directly, and this interface cannot
accept one. At least one continuous mediator measurement
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
