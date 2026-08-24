# Statistical methods and reportability contract

This document is the canonical statistical specification for analyses marked
`supported_in_0_1` by [ADR 0012](adr/0012-small-analysis-ready-releases.md).
It describes the estimand, likelihood, estimator, inference, public record, and
qualification boundary for each supported analysis. The implementation remains
authoritative for computation; this document is authoritative for scientific
interpretation. Author-year links identify entries in the canonical
[BibTeX bibliography](references.bib).

Published theory and an Asterism implementation choice are not the same kind of
evidence. Each section therefore separates the published basis from the exact
construction used here. Numerical agreement, simulation, and a successful fit
are also different claims. A result is reportable only when the release
contract below is satisfied.

## Release contract

A public method can exist without being scientifically supported in 0.1. For a
particular result to be reportable, all of the following must hold:

1. its analysis is `supported_in_0_1` in `release.toml`;
2. all required record fields are finite, the free fit reports
   `converged: true`, and no stable refusal code is present;
3. every null or constrained fit used for inference converges, and a reported
   profile interval has `profile_failures == 0`;
4. every required check named for the analysis has a configured pass rule and
   passes for the released source and artifact; and
5. the analysis record carries the release/build identity and input commitments
   required by the public runner.

`release.toml` is presently a development manifest: `release = false`, its
scientific pass rules are not configured.
Consequently, **no current Asterism output is reportable yet**, even when the
optimizer converges. Historical measurements below explain known behavior and
limitations and are traceable through [Numerical
validation](numerical-validation.md); they do not silently fill those release
gates.

The nine 0.1 analysis families are:

| Analysis ID | Public entry points | Reportable target |
| --- | --- | --- |
| `one_trait_gaussian_heritability` | `prepare`; `PreparedModel.fit` | $h^2$, profile interval, and zero-heritability test |
| `several_covariance_components` | `ComponentModel.fit`, `.interval`, `.test`, `.equality_test`, `.contrasts` | Mean-diagonal contributions/proportions and their intervals; coefficients/contrasts for zero-diagonal bases |
| `bivariate_genetic_correlation` | `BivariateModel.fit`, `.interval`, `.test` | Genetic correlation $\rho_g$, interval, and test |
| `continuous_gene_by_environment` | `GxeModel.fit`, `.interval`, `.test` | Heritability and genetic correlation at prespecified environments, their intervals, and correlation/interaction tests |
| `discrete_gene_by_environment` | `DiscreteGxeModel.fit`, `.correlation_interval`, `.test` | Genetic correlation, its interval, and the three genetic tests |
| `binary_liability_heritability` | `LiabilityModel.fit`, `.interval`, `.test` | Liability-scale $h^2$, interval, and test |
| `one_trait_tobit_audiogram` | `tobit_fit`, `tobit_interval`, `tobit_test` | Complete-trait $h^2$, interval, and test |
| `mixed_binary_censored_genetic_correlation` | `mixed_bivariate_fit`, `mixed_bivariate_interval`, `mixed_bivariate_test` | Genetic correlation for the binary/right-censored pair, interval, and test |
| `spatial_component_presence` | `SpatialModel.fit`, `.bootstrap` | Parametric-bootstrap test of spatial covariance presence |

Quantities called *descriptive* below may accompany a reportable result but are
not themselves a supported inferential claim. Quantities called *diagnostic*
are for checking a fit or reproducing its parameterisation and must not be
promoted to a scientific result.

## Shared notation and Gaussian likelihood

Let $n$ be the number of observed response rows, $p$ the number of linearly
independent fixed-effect columns, $\mathbf y\in\mathbb R^n$ the response,
$\mathbf X\in\mathbb R^{n\times p}$ the caller-supplied design, and
$\boldsymbol\beta\in\mathbb R^p$ its fixed-effect coefficients. Asterism does
not add an intercept; a column of ones must be included in $\mathbf X$ when an
intercept is wanted. Matrices and responses are aligned positionally. The
notation $N(\boldsymbol\mu,\mathbf V)$ denotes a multivariate normal
distribution with mean $\boldsymbol\mu$ and covariance $\mathbf V$.

For a positive-definite covariance $\mathbf V(\boldsymbol\theta)$ and residual
$\mathbf r=\mathbf y-\mathbf X\boldsymbol\beta$, the Gaussian log likelihood is

$$
\ell_{\mathrm{ML}}(\boldsymbol\beta,\boldsymbol\theta)
=-\frac12\left\{n\log(2\pi)+\log|\mathbf V|
+\mathbf r^T\mathbf V^{-1}\mathbf r\right\}.
$$

At fixed $\boldsymbol\theta$, the generalized least-squares estimate is

$$
\widehat{\boldsymbol\beta}(\boldsymbol\theta)
=\left(\mathbf X^T\mathbf V^{-1}\mathbf X\right)^{-1}
\mathbf X^T\mathbf V^{-1}\mathbf y.
$$

Restricted maximum likelihood adds the fixed-effect adjustment and uses
$n-p$ residual degrees of freedom. Up to constants that do not depend on
$\boldsymbol\theta$,

$$
\ell_{\mathrm{REML}}(\boldsymbol\theta)
=-\frac12\left\{\log|\mathbf V|
+\log|\mathbf X^T\mathbf V^{-1}\mathbf X|
+\widehat{\mathbf r}^{\,T}\mathbf V^{-1}\widehat{\mathbf r}\right\},
$$

where $\widehat{\mathbf r}=\mathbf y-
\mathbf X\widehat{\boldsymbol\beta}(\boldsymbol\theta)$. Estimation from
error contrasts is due to [Patterson and Thompson
(1971)](references.bib#pattersonThompson1971); the general ML/REML
variance-component treatment is described by [Harville
(1977)](references.bib#harville1977). Gaussian models default to REML and may be
fit by ML when the caller explicitly requests it. Liability, censored, and
mixed-observation models use ML only because a region observation cannot be
projected into a residual contrast.

### Profile likelihood and the common interval record

For scalar target $\psi$ and nuisance parameters $\boldsymbol\lambda$, define

$$
\ell_p(\psi)=\max_{\boldsymbol\lambda}
\ell(\psi,\boldsymbol\lambda),\qquad
D(\psi)=2\{\ell_p(\widehat\psi)-\ell_p(\psi)\}.
$$

An interior $100(1-\alpha)\%$ profile interval contains values satisfying
$D(\psi)\leq\chi^2_{1,1-\alpha}$, where $\alpha$ is the tail probability and
$\chi^2_{1,1-\alpha}$ is the $1-\alpha$ quantile of chi-square on one degree of
freedom. For 95% intervals the critical value is
$3.841458820694124$. This is the regular large-sample likelihood-ratio result
of [Wilks (1938)](references.bib#wilks1938), operationalized by repeated
constrained maximization as in [Venzon and Moolgavkar
(1988)](references.bib#venzonMoolgavkar1988). It is not a finite-sample theorem
for every pedigree. In particular, [Crainiceanu and Ruppert
(2004)](references.bib#crainiceanuRuppert2004) show why a familiar asymptotic
mixture can be inaccurate for a finite linear mixed model with few independent
blocks.

Every supported profile family returns the same core fields:

| Field | Meaning |
| --- | --- |
| `estimate` | Free-fit estimate of the profiled quantity |
| `lower`, `upper` | Profile endpoints |
| `lower_limited`, `upper_limited` | The endpoint reached the quantity/search boundary rather than a likelihood crossing |
| `level` | Nominal interval level |
| `profile_failures` | Failed or nonconverged constrained evaluations encountered while finding the endpoints |
| `contains_lower_bound`, `contains_upper_bound` | Boundary-inclusion verdict when that verdict has been calibrated; `None` means unmeasured, not false |

`PreparedModel` additionally retains `recipe` for backward compatibility.
Model-specific context remains beside the common shape: component intervals
add `quantity`; spatial intervals add `quantity` and `estimator`;
continuous-G×E intervals add `surface`, `quantity`, `environment`, and
`estimator`; discrete-G×E intervals add `rule` and `estimator`; liability adds
`estimator`; Tobit adds `censored_share` and `estimator`; and mixed bivariate
adds `what` and `estimator`. Boundary containment is currently populated only
for the coverage-scored prepared one-trait and Tobit families; it is `None` in
the other families.

A failed or nonconverged profile evaluation is unknown likelihood, not evidence
that the likelihood crossed its threshold. Asterism widens over that point and
increments `profile_failures`. Any nonzero count blocks reporting. If the free
fit does not converge, the family refuses interval or test inference with a
stable `*_FIT_NOT_CONVERGED` code rather than manufacturing endpoints or a
p-value.

### Likelihood-ratio tests and boundaries

All likelihood-ratio tests report

$$
T=2\max\{0,\ell(\widehat\Theta)-\ell(\widehat\Theta_0)\},
$$

where $\widehat\Theta$ and $\widehat\Theta_0$ are respectively the fitted free
and null parameter spaces. Public test records expose `statistic`, `p_value`,
`rule`, and `null_loglik`; families whose wrapper receives both values also
expose `alternative_loglik`. `rule` is part of the scientific record and names
the distribution used for the tail probability.

For one nonnegative variance component tested at zero, with every other
relevant parameter regular, identified, and interior, the asymptotic reference
is

$$
\tfrac12\chi^2_0+\tfrac12\chi^2_1.
$$

This is the applicable special case of [Self and Liang
(1987)](references.bib#selfLiang1987). It does not authorize the same weights
for an arbitrary covariance cone. [Han and Chang
(2008)](references.bib#hanChang2008) show geometrically why simple mixtures can
fail in multivariate variance-component problems. Their paper is a preprint and
not a derivation of an Asterism test; it is cited as a warning. Model-specific
mixtures below are Asterism parameter-space derivations that still require the
manifest's simulation gates.

## One-trait Gaussian heritability

### Model and estimator

With fixed relationship matrix $\mathbf K$, Asterism fits

$$
\mathbf y\sim N(\mathbf X\boldsymbol\beta,\mathbf V),\qquad
\mathbf V=\sigma^2\{h^2\mathbf K+(1-h^2)\mathbf I_n\},
$$

where $\sigma^2>0$ is total variance, $h^2\in[0,1]$ is the relationship-matrix
heritability, and $\mathbf I_n$ is the $n\times n$ identity. `prepare` validates
the design and covariance basis, partitions the roster into relationship
blocks, eigendecomposes $\mathbf K$ once, and reuses the transformed design.
At each $h^2$, $\boldsymbol\beta$ and $\sigma^2$ are profiled analytically;
$h^2$ is found by a deterministic bounded one-dimensional search.

The zero-heritability test uses the 50:50 boundary mixture. An interval endpoint
at $h^2=0$ or $1$ is not automatically included: the boundary likelihood is
judged using the calibrated mixture recipe and reported in `contains_*`. A
singular $\mathbf K$ can make $h^2=1$ an invalid covariance; the fit then
refuses that point instead of pretending it was evaluated.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `h2` ($h^2$) | `PreparedModel.fit(y)["h2"]` |
| `interval` | `fit["interval"]`, using the common interval fields |
| `test` of $H_0:h^2=0$ | `fit["test"]["null"]`, `["statistic"]`, `["p_value"]`, `["rule"]`, and `["null_loglik"]` |

`fit["estimator"]` distinguishes `reml` from `ml`; `total_variance`, `beta`,
`fixed_effects`, and `standard_errors` are descriptive. `converged`, `boundary`,
`warnings`, `subject_order_sha256`, and `build` are required provenance or fit
diagnostics rather than reportable targets.

### Assumptions, measured limitations, and unmet gates

The response is conditionally Gaussian with covariance exactly proportional to
the supplied $\mathbf K$ plus independent residual variance; rows are aligned;
$\mathbf X$ is full rank; and the fixed effects are correctly specified. A
covariate p-value is comparable across programs only when the design
parameterisation is identical. Interpreting $h^2$ as a marginal phenotypic
variance proportion additionally requires the qualified relationship-matrix
normalisation (normally unit diagonal for twice kinship); rescaling
$\mathbf K$ changes this parameter.

Historical repository simulations found conservative, rather than
anti-conservative, coverage near $h^2=0$: at $n=350$, true values 0.05 and 0.07
covered at 0.979 and 0.974. That evidence is REML- and design-specific. The
release contract therefore requires independent R and SOLAR agreement plus
general and target-design coverage. Their adapters and public-Python campaign
commands are executable, but none becomes release evidence until its exact
prewritten command passes against the fixed release wheel and the supported
pass rules are configured in `release.toml`.

## Several covariance components

### Model, identification, and scale

For $q$ submitted structured covariance bases, Asterism fits

$$
\mathbf V(\boldsymbol\theta)
=\sum_{k=1}^{q}\theta_k\mathbf K_k+\theta_e\mathbf I_n,
\qquad \theta_1,\ldots,\theta_q,\theta_e\geq0,
$$

where $\mathbf K_k$ is the $k$th fixed basis, $\theta_k$ its covariance
coefficient, and $\theta_e$ the residual coefficient. The Gaussian ML/REML
likelihood above is evaluated by dense factorization because the bases need not
commute.

The coefficients are identified only if different allowable vectors
$\boldsymbol\theta$ produce different covariance matrices on the fitted roster.
Linear dependence among $\{\mathbf K_1,\ldots,\mathbf K_q,\mathbf I_n\}$ causes
exact nonidentification; near dependence causes weak identification. Labels
such as “additive” or “household” do not establish it.

Raw coefficient proportions depend on arbitrary matrix scale. Asterism instead
defines

$$
m_k=\frac{1}{n}\operatorname{tr}(\mathbf K_k),\qquad
c_k=\widehat\theta_km_k,qquad
p_k=\frac{c_k}{\sum_{j=1}^{q}c_j+\widehat\theta_e},
$$

where $m_k$ is the mean diagonal, $c_k$ the mean-diagonal contribution, and
$p_k$ its proportion of mean marginal variance. The residual contribution is
$c_e=\widehat\theta_e$ and $p_e=c_e/(\sum_jc_j+c_e)$. Multiplying
$\mathbf K_k$ by a positive constant and dividing $\theta_k$ by it leaves
$c_k$ and $p_k$ unchanged. This invariant is an Asterism reporting definition,
not a conventional result asserted by [Harville
(1977)](references.bib#harville1977).

It is defined only when every structured basis has a positive finite mean
diagonal. Split kinship-class bases have zero diagonal, so their reportable
targets are coefficients and class deviations, never proportions. If $r$
class coefficients are pooled under equality, the omnibus likelihood-ratio
test uses $\chi^2_{r-1}$. Each class contrast is its coefficient's deviation
from the constrained average; deviations sum to zero and use an interior
$\chi^2_1$ reference. Testing a single component against zero uses the 50:50
mixture only when the other fitted components are interior.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `mean_diagonal_component_contributions` ($c_k$) | `ComponentModel.fit(y)["mean_diagonal_component_contributions"][k]` |
| `mean_diagonal_proportions` ($p_k$) | `fit["mean_diagonal_proportions"][k]` |
| `mean_diagonal_proportion_interval` | `ComponentModel.interval(y, k)` with `quantity == "mean_diagonal_proportion"` and the common interval fields |
| `coefficients_for_zero_diagonal_bases` ($\widehat\theta_k$) | `fit["variances"][k]` |
| `contrasts_for_zero_diagonal_bases` | `equality_test(y)` fields `components`, `statistic`, `p_value`, `rule`, `null_loglik`, `estimator`; and each `class`, `deviation`, `lower`, `upper`, `lower_limited`, `upper_limited`, `p_value`, `statistic`, `level` row from `contrasts(y)` |

Component lists follow the submitted structured-basis order and place the
residual last.

`raw_coefficient_proportions`, `raw_coefficient_total`, and an interval requested
with `quantity="raw_coefficient_proportion"` are diagnostic only. Fixed effects,
likelihood, projected/scaled gradient, optimizer stop message, and `polished`
describe the fit but are not component results.

### Assumptions, measured limitations, and unmet gates

All Gaussian assumptions above apply. Each basis must be finite, symmetric, and
aligned to the response, and every evaluated linear combination must be a
positive-definite covariance. A split or contrast basis need not be positive
semidefinite by itself. A scale-free Frobenius Gram-rank check now refuses
linearly dependent bases, including a submitted identity confounded with the
implicit residual. It establishes coefficient identification at numerical rank;
it is not a precision guarantee for bases that merely lie close together.
Historical simulations at $n=400$
gave 0.945 and 0.955 coverage for additive and household mean-diagonal
proportions and 0.051 rejection for a nominal 0.05 zero-household test.
Zero-diagonal class-equality calibration on the GOBS pedigree rejected 0.035
under equality. These observations do not supply a universal identifiability
diagnostic or release envelope. Component calibration and the interval-identity
sentinel now have executable public-API pass rules. The sentinel has passed on
the development build, but the exact calibration command has not yet been
retained from the fixed release wheel and the campaign remains
unmeasured.

## Two Gaussian traits and genetic correlation

### Model and estimator

Stack the two traits within each person. Let $N$ be the number of people,
$\mathbf A$ the $N\times N$ relationship matrix,
$\boldsymbol\Sigma_g$ the $2\times2$ genetic covariance,
$\boldsymbol\Sigma_e$ the $2\times2$ residual covariance, and $\mathbf S$ the
row-selection matrix that removes unobserved person-trait rows. The symbol
$\otimes$ denotes a Kronecker product. Asterism fits

$$
\mathbf V_{\mathrm{obs}}
=\mathbf S\left(
\mathbf A\otimes\boldsymbol\Sigma_g
+\mathbf I_N\otimes\boldsymbol\Sigma_e
\right)\mathbf S^T.
$$

For traits $1$ and $2$,

$$
\rho_g=\frac{\sigma_{g,12}}
{\sqrt{\sigma_{g,11}\sigma_{g,22}}},\qquad
h_t^2=\frac{\sigma_{g,tt}}
{\sigma_{g,tt}+\sigma_{e,tt}}\quad(t\in\{1,2\}),
$$

where $\sigma_{g,st}$ and $\sigma_{e,st}$ are entries of the genetic and
residual covariance matrices. The pedigree covariance and genetic-correlation
lineage is represented by [Almasy, Dyer, and Blangero
(1997)](references.bib#almasyDyerBlangero1997) and the wider SOLAR pedigree
likelihood by [Almasy and Blangero
(1998)](references.bib#almasyBlangero1998). Those linkage papers do not define
Asterism's exact no-linkage wrapper, starts, bounds, or missing-row selection.

ML and REML are available; REML is the default. Testing $H_0:\rho_g=0$ uses
$\chi^2_1$ because zero is interior. A test at $\rho_g=+1$ or $-1$ uses the
implemented one-sided 50:50 mixture. The latter is a local Asterism boundary
choice whose finite-sample calibration remains design-specific.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `rho_g` ($\rho_g$) | `BivariateModel.fit(y)["rho_g"]` |
| `interval` | `BivariateModel.interval(y, "rho_g")`, using the common interval fields |
| `test` | `BivariateModel.test(y, "rho_g", null=...)`: `statistic`, `p_value`, `rule`, `null_loglik`, `alternative_loglik` |

`h2_first`, `h2_second`, `rho_e`, `rho_p`, and `total_variance` are descriptive
in the 0.1 genetic-correlation analysis, not additional reportable targets.

### Assumptions, measured limitations, and unmet gates

The two latent responses are jointly Gaussian with the same supplied
relationship basis and trait-specific residual covariance; missingness is
represented solely by $\mathbf S$. At high heritability, little residual
variance remains and $\rho_e$ can be weakly identified. In one historical
$n=180$ simulation, $\rho_g$ interval coverage was 0.960; the correlation-one
test rejected 0.033 at nominal 0.05. At a higher-heritability design, 23 of 300
residual-correlation intervals failed and most computed intervals reached a
bound. The R and SOLAR adapters have now been refreshed live and their frozen
fixtures verify without either external program. Calibration and independently
computed profile-endpoint commands are also executable. None of those results
is release evidence until the exact manifest commands rerun against the fixed
wheel.

## Continuous gene by environment

### Shared model

Let $z_i$ be person $i$'s prespecified continuous environment, $A_{ij}$ the
corresponding relationship coefficient, $G(z_i,z_j)$ the genetic covariance
surface, and $R(z_i)>0$ the residual variance. Asterism fits

$$
V_{ij}=A_{ij}G(z_i,z_j)+\mathbb 1(i=j)R(z_i),
$$

where $\mathbb 1(i=j)$ equals one for the same observation and zero otherwise.
The environment, reporting grid, and surface family must be chosen without
examining the fitted outcome. ML and REML are available; REML is the default.

For the supported exponential surface,

$$
\begin{aligned}
g(z)&=\exp(\alpha_g+\gamma_gz),\\
R(z)&=\exp(\alpha_e+\gamma_ez),\\
G(z_i,z_j)&=\sqrt{g(z_i)g(z_j)}
\exp\{-\lambda|z_i-z_j|\},\qquad\lambda\geq0,
\end{aligned}
$$

where $g(z)$ is genetic variance, $\alpha_g$ and $\gamma_g$ set its log-linear
level and slope, $\alpha_e$ and $\gamma_e$ do the same for residual variance,
and $\lambda$ is genetic-correlation decay. This construction follows the
genotype-by-age stochastic-process model of [Diego et al.
(2003)](references.bib#diegoEtAl2003), generalized by Asterism from age to a
prespecified continuous environment.

For the supported random-regression surface, let
$\boldsymbol\phi(z)=(1,z)^T$ and let $\mathbf Q_g$ and $\mathbf Q_e$ be
$2\times2$ positive-semidefinite covariance matrices. Then

$$
G(z_i,z_j)=\boldsymbol\phi(z_i)^T\mathbf Q_g\boldsymbol\phi(z_j),
\qquad
R(z)=\boldsymbol\phi(z)^T\mathbf Q_e\boldsymbol\phi(z).
$$

Covariance functions for reaction norms originate with [Kirkpatrick and
Heckman (1989)](references.bib#kirkpatrickHeckman1989), and random-regression
estimation of genetic covariance functions is developed by [Meyer
(1998)](references.bib#meyer1998). Asterism's low-rank factorisation, parameter
bounds, deterministic starts, and tests are local implementation choices.

At a reporting environment $z$, heritability and genetic correlation are

$$
h^2(z)=\frac{G(z,z)}{G(z,z)+R(z)},\qquad
\rho_g(z_a,z_b)=
\frac{G(z_a,z_b)}{\sqrt{G(z_a,z_a)G(z_b,z_b)}}.
$$

Here $z_a$ and $z_b$ are two prespecified environments. Under the correlation
null the genetic effects have correlation one throughout the range while
genetic scale may change. The exponential null $\lambda=0$ uses the 50:50
mixture. The interaction null fixes both genetic shape coordinates and uses
$\tfrac12\chi^2_1+\tfrac12\chi^2_2$. For random regression, the null is on a
curved rank-deficient covariance cone; Asterism retains this reference as a
deliberately conservative rule pending target-design calibration, not as a
universal consequence of Self and Liang.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `heritability` ($h^2(z_r)$) | `GxeModel.fit(y, grid)["heritability"][r]`, paired with `fit["environment"][r]` |
| `genetic_correlation` ($\rho_g(z_r,z_s)$) | `fit["genetic_correlation"][r][s]`, paired with both environment entries |
| `heritability_interval` | `GxeModel.interval(y, quantity="heritability", first=z_r)`, using the common fields |
| `genetic_correlation_interval` | `GxeModel.interval(y, quantity="correlation", first=z_r, second=z_s)`, using the common fields |
| `correlation_test` | `GxeModel.test(y, null="correlation")`: `statistic`, `p_value`, `rule`, `null_loglik`, `alternative_loglik` |
| `interaction_test` | `GxeModel.test(y, null="interaction")`, with the same test fields |

`genetic_variance` and `residual_variance` support interpretation but are
descriptive; `parameters` is diagnostic and does not have a common meaning
across surfaces. `test(null="variance")` is public but is not a 0.1 reportable
quantity. The powered-exponential family remains public but unsupported.

### Assumptions, measured limitations, and unmet gates

The chosen surface must be positive semidefinite and correctly describe how
genetic covariance and residual variance change across the observed
environment. Reportable $h^2(z)$ also assumes the qualified relationship
normalisation, normally unit diagonal. Outcome-dependent family selection changes the estimand. In
historical simulations the exponential correlation test rejected 0.048 at a
nominal 0.05 flat null, but fitting an exponential surface to a misspecified
linear rank-one surface rejected 0.107. Across studied settings, interval
coverage was 0.953–0.983 while 80%–92% of genetic-correlation intervals reached
a bound. Thus coverage did not imply precision. The independent dense sentinel
and both target-sized blockwise fits now agree on the development build. A
reduced target campaign completed 100 replicates per surface with no refusal or
nonconvergence and compatible null rejection, while the manifest fixes 500 per
surface. The exact fixed-wheel interval, calibration, and 500-replicate target
commands remain unrun release gates; the reduced run is qualification evidence,
not permission to report an analysis.

## Gene by discrete environment

### Model and inference

Let $e_i\in\{0,1\}$ be person $i$'s prespecified group, $A_{ij}$ the supplied
relationship coefficient for people $i$ and $j$, $s_{g,r}>0$ and $s_{e,r}>0$
the genetic and residual standard deviations in group $r$, and
$\rho_g\in[-1,1]$ the cross-group genetic correlation. Asterism fits

$$
V_{ij}=A_{ij}s_{g,e_i}s_{g,e_j}
\begin{cases}
1,&e_i=e_j,\\
\rho_g,&e_i\ne e_j,
\end{cases}
+\mathbb 1(i=j)s_{e,e_i}^2.
$$

Treating expression in two environments as two genetically correlated traits
comes from [Falconer (1952)](references.bib#falconer1952). The exact
parameterisation and test catalogue below are Asterism constructions. Separate
residual variances prevent unequal measurement noise from being mislabelled as
a genetic difference. ML and REML are available; REML is the default.

The reportable genetic tests are:

| `null` | Null hypothesis | `rule` |
| --- | --- | --- |
| `gene_by_environment` | $s_{g,0}=s_{g,1}$ and $\rho_g=1$, with residual scales free | `mixture_chi2_1_chi2_2` = $\tfrac12\chi^2_1+\tfrac12\chi^2_2$ |
| `correlation` | $\rho_g=1$ | `mixture_50_50` = $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ |
| `genetic` | $s_{g,0}=s_{g,1}$ | `chi2_1` |

`any_difference` also constrains residual variance and is not a genetic test;
`residual` is a measurement-variance test. The 95% correlation interval uses
the regular interior $\chi^2_1$ profile threshold, not the boundary reference
for testing $\rho_g=1$.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `genetic_correlation` | `DiscreteGxeModel.fit(y)["genetic_correlation"]` |
| `correlation_interval` | `DiscreteGxeModel.correlation_interval(y)`, using the common interval fields |
| `gene_by_environment_test` | `DiscreteGxeModel.test(y, null="gene_by_environment")` |
| `correlation_test` | `DiscreteGxeModel.test(y, null="correlation")` |
| `genetic_variance_test` | `DiscreteGxeModel.test(y, null="genetic")` |

Every test record supplies `null`, `statistic`, `p_value`, `rule`,
`null_loglik`, `alternative_loglik`, and `estimator`. Group-specific
`genetic_variance`, `residual_variance`, `heritability`, `counts`, and `levels`
are descriptive in this 0.1 analysis.

### Assumptions, measured limitations, and unmet gates

The two groups, their order, and fixed-effect design are prespecified; each
group contains enough related observations to identify a genetic covariance;
and the Gaussian covariance model is correct. Historical calibration on the
unbalanced GOBS pedigree found nominal-0.05 rejection rates 0.042, 0.044, and
0.048 for the overall, correlation, and genetic-variance tests. When one group
had twice the residual noise but identical genetics, the three genetic tests
remained near level while `any_difference` rejected every replicate. At true
$\rho_g=0.9$, 0.995 interval coverage accompanied 94% bound-reaching: again,
coverage was not precision. The new independent full fit agrees within
$7.7\times10^{-4}$ on reportable quantities and $4.93\times10^{-5}$ in log
likelihood, and reduced public-API target and interval smokes completed without
fit or profile failures. The exact 2,000-attempt target calibration and
1,200-attempt interval campaigns did not start in the current sandbox because
worker creation was denied before any scientific attempt. Their commands and
acceptance rules are fixed; genuine host execution against the release wheel
remains required.

## Binary liability heritability

### Model and estimator

Let $L_i$ be an unobserved liability and $Y_i$ the observed binary status.
Asterism fixes liability variance to one and fits

$$
\begin{aligned}
L_i&=\mathbf x_i^T\boldsymbol\beta+g_i+e_i,\\
\operatorname{Cov}(\mathbf L)&=h^2\mathbf A+(1-h^2)\mathbf I_n,\\
Y_i&=\mathbb 1(L_i>0),
\end{aligned}
$$

where $\mathbf x_i^T$ is row $i$ of the design, $g_i$ is the additive genetic
liability, $e_i$ is residual liability, $\mathbf A$ is the supplied
relationship matrix, $\mathbf I_n$ is the identity, and $h^2\in[0,1]$ is
liability-scale heritability. Fixing total variance identifies the probit scale. Threshold
heritability originates with [Dempster and Lerner
(1950)](references.bib#dempsterLerner1950), and human-disease liability is
developed by [Falconer (1965)](references.bib#falconer1965).

Each family contributes a multivariate-normal orthant probability. Asterism
uses the likelihood

$$
L(\boldsymbol\beta,h^2)
=\prod_{f=1}^{F}\Pr\{\mathbf D_f\mathbf L_f>\mathbf 0\},
$$

where $F$ is the number of independent family blocks, $\mathbf L_f$ is family
$f$'s liability vector after its fixed-effect mean is applied, and
$\mathbf D_f$ is diagonal with $+1$ for a case and $-1$ for a noncase;
$\mathbf 0$ is the conformable zero vector. It
evaluates dimensions one and two directly and uses sequential Mendell–Elston
truncation above two dimensions. [Mendell and Elston
(1974)](references.bib#mendellElston1974) provides the multifactorial
qualitative-trait approximation lineage; [Tallis
(1961)](references.bib#tallis1961) supplies moments of the truncated normal.
Rare-class-first ordering, probability guards, and the dedicated
low-dimensional path are Asterism choices. Estimation is ML. The test of
$H_0:h^2=0$ uses the implemented 50:50 boundary mixture.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `heritability` | `LiabilityModel.fit()["heritability"]` |
| `interval` | `LiabilityModel.interval()`, using the common interval fields |
| `test` | `LiabilityModel.test()`: `statistic`, `p_value`, `rule`, `null_loglik`, `estimator` |

`scale`, `prevalence`, `fixed_effects`, and `largest_family` are essential
descriptors; `loglik`, `scaled_gradient`, `polished`, and `converged` diagnose
the fit.

### Assumptions, measured limitations, and unmet gates

Status is a deterministic thresholding of a Gaussian liability, the supplied
relationship matrix captures additive covariance, families are independent
blocks, the relationship diagonal is normalized to the unit-liability scale,
and the fixed effects correctly locate liability. The estimand is not
observed-scale heritability. Historical calibration on the GOBS pedigree found
0.0529 rejection at nominal 0.05 and 0.9443 coverage at true $h^2=0.25$; the
two-person quadrature error grew near correlation one. Above two dimensions,
accuracy depends on family size, correlation, imbalance, and truncation order.
The independent SOLAR comparison has been refreshed live and frozen for
portable verification. A values-free 1,909-person target check also exercised
the largest 180-person family: its one null and one alternative smoke attempt
both completed, and the alternative interval covered $h^2=0.25$. That is an
execution seam, not a calibration estimate. The exact 200-replicate-per-scenario
fixed-wheel campaign remains a release
requirements.

## One-trait censored Gaussian model

### Model and likelihood

Let $Y_i^*$ be the complete measurement and let $C_i\in\{M,U,L\}$ denote
measured, upper/right-censored, or lower/left-censored status. Let $a_i$ be the
observation-specific limit. Asterism fits

$$
\mathbf Y^*\sim N\left(
\mathbf X\boldsymbol\beta,
\sigma^2\{h^2\mathbf A+(1-h^2)\mathbf I_n\}
\right).
$$

Here $\mathbf A$ is the relationship matrix, $\mathbf I_n$ the identity,
$\sigma^2>0$ the total variance of the complete measurement, and
$h^2\in[0,1]$ its heritability; $\mathbf X$ and $\boldsymbol\beta$ retain their
shared fixed-effect meanings.

When $C_i=M$, $Y_i^*$ is observed exactly. When $C_i=U$, only
$Y_i^*\geq a_i$ is known; when $C_i=L$, only $Y_i^*\leq a_i$ is known. The
public `censoring` array encodes $M$, $U$, and $L$ as 0, 1, and 2. Status is
supplied rather than inferred from a numeric value. The free
$\sigma^2$ is identified by measured observations, so $h^2$ concerns the
complete trait that would have been observed without the instrument limit.
The public likelihood accepts either censoring direction, but 0.1
reportability is restricted to the right-censored audiogram analysis named in
`release.toml`.

[Tobin (1958)](references.bib#tobin1958) is the foundational censored
latent-Gaussian regression source. Normal mixed models with censoring are
developed by [Hughes (1999)](references.bib#hughes1999). For a family split
into measured indices $M$ and censored indices $C$, Asterism follows the
conditional factorisation

$$
L_f=f(\mathbf y_M;\boldsymbol\mu_M,\mathbf V_{MM})
\Pr\{\mathbf Y_C^*\in\mathcal R_C\mid
\mathbf Y_M^*=\mathbf y_M\},
$$

where $L_f$ is family $f$'s likelihood contribution, $\mathbf y_M$ is the
measured subvector, $\boldsymbol\mu_M$ its fitted mean, $\mathbf V_{MM}$ its
covariance, $f(\cdot)$ is the multivariate-normal density, $\mathcal R_C$ is
the product of the censoring half-lines, and $\Pr$ denotes probability. The
conditional probability uses the usual normal conditional mean and covariance.
This construction is explicit in
[Jacqmin-Gadda et al.
(2000)](references.bib#jacqminGaddaEtAl2000). Asterism directly maximizes this
ML likelihood using the shared region-probability routine. The EM approach of
[Vaida and Liu (2009)](references.bib#vaidaLiu2009) is relevant published
precedent but is not the implemented algorithm.

Testing $H_0:h^2=0$ and deciding whether an interval contains that boundary use
the 50:50 mixture. At least two measured values are required to identify scale.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `heritability` | `tobit_fit(...)["heritability"]` |
| `interval` | `tobit_interval(...)`, using the common interval fields plus `censored_share` and `estimator` |
| `test` | `tobit_test(...)`: `statistic`, `p_value`, `rule`, `null_loglik`, `alternative_loglik`, `estimator` |

`total_variance`, `fixed_effects`, `censored_share`, and `largest_family` are
descriptive; `converged`, `scaled_gradient`, and `loglik` are diagnostics. A
limit-substitution Gaussian heritability is a different, biased estimand.

### Assumptions, measured limitations, and unmet gates

The complete trait is Gaussian with the stated covariance; censoring direction
and each limit are correct; censoring is represented by the stated regions;
the relationship basis has the qualified normalisation; and the measured
portion identifies scale. Historical independent-person
comparison with `censReg` agreed in log likelihood and fitted quantities within
$9.6\times10^{-9}$. In simulations, limit substitution estimated 0.31 at 75%
censoring when true $h^2=0.5$, while the censored model recovered the target.
For 300 sibling-pair replicates per cell through 50% censoring, all nine 95%
coverage cells included 0.95 in their exact binomial intervals. Those pair
designs do not exercise the sequential approximation above two censored family
members. Both independent R comparisons have now been refreshed live and
frozen for portable verification, and the calibration plus explicit 52% and
75% commands are executable with prewritten rules. A target-design smoke with
one replicate in each scenario was route and failure-accounting evidence only:
the 75%-censored null fit returned a finite $h^2=0$ candidate with
`converged=false`, so no interval or test was computed for that attempt and the
command failed. The smoke is not calibration. The 800-attempt exact campaign
has not yet been retained from the fixed release wheel, so the censoring range
remains unmeasured and no audiogram estimate is reportable under 0.1 yet.

## Mixed binary and censored genetic correlation

### Model and likelihood

Let trait 1 be a binary liability and trait 2 a right-censored complete
measurement. Before applying their observation mechanisms, stack both traits
within each of $N$ people. Let $\mathbf A$ be their relationship matrix,
$\mathbf I_N$ the $N\times N$ identity, $\boldsymbol\Sigma_g$ and
$\boldsymbol\Sigma_e$ the two-trait genetic and residual covariance matrices,
and $\otimes$ the Kronecker product. Then use

$$
\mathbf V=
\mathbf A\otimes\boldsymbol\Sigma_g
+\mathbf I_N\otimes\boldsymbol\Sigma_e.
$$

For $t\in\{1,2\}$ let $h_t^2$ be heritability and $\sigma_t^2$ total latent
variance. Asterism parameterises

$$
\boldsymbol\Sigma_g=
\begin{pmatrix}
h_1^2\sigma_1^2 &
\rho_g\sqrt{h_1^2\sigma_1^2h_2^2\sigma_2^2}\\
\rho_g\sqrt{h_1^2\sigma_1^2h_2^2\sigma_2^2} &
h_2^2\sigma_2^2
\end{pmatrix},
$$

and

$$
\boldsymbol\Sigma_e=
\begin{pmatrix}
(1-h_1^2)\sigma_1^2 &
\rho_e\sqrt{(1-h_1^2)\sigma_1^2(1-h_2^2)\sigma_2^2}\\
\rho_e\sqrt{(1-h_1^2)\sigma_1^2(1-h_2^2)\sigma_2^2} &
(1-h_2^2)\sigma_2^2
\end{pmatrix},
$$

where $\rho_g$ and $\rho_e$ are genetic and residual correlations. For the
binary trait, $\sigma_1^2=1$ fixes liability scale; the censored trait retains
a free $\sigma_2^2$. Density observations and threshold/limit regions are
combined family by family and evaluated by ML. In the supported public input,
the first trait has `kind="binary"`, censoring code 1 for a case and 2 for a
noncase, and limit zero; the second has `kind="censored"`, code 0 when measured
and code 1 when right-censored at its supplied limit.

[Catalano and Ryan (1992)](references.bib#catalanoRyan1992) provide a clustered
latent-variable framework for discrete and continuous outcomes. Combined with
the conditional censored-normal likelihood of [Jacqmin-Gadda et al.
(2000)](references.bib#jacqminGaddaEtAl2000), it supports the ingredients but
not the exact Asterism model. The pedigree genetic covariance, one binary
liability, observation-specific censoring, direct block likelihood, and
sequential high-dimensional approximation are an Asterism synthesis. No single
primary publication found in the source review establishes that exact
combination.

The reportable test is $H_0:\rho_g=0$. Zero is interior to $[-1,1]$, so its
reference is $\chi^2_1$, not a boundary mixture. The interval uses the regular
interior profile threshold.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `genetic_correlation` | `mixed_bivariate_fit(...)["genetic_correlation"]` |
| `interval` | `mixed_bivariate_interval(..., coordinate="genetic_correlation")`, using the common interval fields plus `what` and `estimator` |
| `test` of $H_0:\rho_g=0$ | `mixed_bivariate_test(..., coordinate="genetic_correlation")`: `what`, `statistic`, `p_value`, `rule`, `null_loglik`, `alternative_loglik`, `estimator` |

Per-trait `heritability`, `total_variance`, `residual_correlation`,
`fixed_effects`, `largest_family`, and `kinds` are descriptive for the supported
binary/right-censored analysis. Other public trait-kind combinations are not
0.1-supported analyses.

### Assumptions, measured limitations, and unmet gates

Both latent traits follow the joint Gaussian covariance above; the binary scale
and relationship-matrix normalization are fixed correctly; censoring records
and limits are correct; and family blocks are independent. Historical sibling-pair simulations gave genetic-correlation
interval coverage of 0.940, 0.940, and 0.943 at true correlations 0, 0.4, and
0.7 for the censored pairing, and the test rejected 0.060 at a nominal 0.05
null. These runs do not validate larger-family sequential integration by
themselves. The exact binary/right-censored driver now exists, and the SOLAR
adapter has been refreshed live and frozen; calibration and coverage commands
are executable too.

The target driver selects a values-free structural fixture with SHA-256
`aed92270dcfafb00a46a24bfaf1f3a2b52950fdea1c34deb3740e5060d6f3645`,
itself pinned to source-fixture SHA-256
`93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e`.
It matches the reviewed aggregate structure of 1,909 rows, 202 independent
relationship components and a largest component of 180. The predecessor
aggregate records 27,821 nonzero relationship pairs and the generated
structure has 27,691; neither the fixture nor the check claims pedigree,
relationship-matrix or participant-row reconstruction.

The generating design fixes population prevalence at 0.254,
$(h_1^2,h_2^2)=(0.5,0.5)$, $\rho_g\in\{0,0.4\}$,
$\rho_e=0.15$, complete-hearing variance at 3, and intended right-censoring
shares at 0.52 and 0.75. It exercises the public fit, genetic-correlation
interval and zero-correlation test. `release.toml` fixes 300 replicates per
correlation-by-censoring cell, 1,200 attempts in total. Neither a bounded smoke
nor that exact campaign has completed: a local one-replicate-per-cell smoke was
interrupted during public interval fitting (`KeyboardInterrupt`, exit 130)
before producing a completed record. It is not evidence. The target range
therefore remains unmeasured, with no target-design calibration, qualifying
runtime or analysis-readiness claim, and the exact fixed-wheel campaign remains
required before reporting the intended diagnosis/hearing genetic correlation.

## Spatial covariance presence

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
| `spatial_presence_test` | `SpatialModel.bootstrap(y)`: `statistic`, `p_value`, `rule`, `exceedances`, `replicates`, `requested`, `seed`, and `smallest_reportable` |

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

## Numerical implementation

Asterism does not use one optimizer for every model:

| Route | Families | Published algorithmic lineage | Asterism-specific qualification concern |
| --- | --- | --- | --- |
| Bounded golden-section refinement after a fixed grid | Prepared one-kernel Gaussian | [Kiefer (1953)](references.bib#kiefer1953) | Endpoints, singular boundaries, and convergence are checked locally |
| L-BFGS-B with analytic gradients | Multiple components, bivariate Gaussian, continuous/discrete G×E, spatial | [Byrd et al. (1995)](references.bib#byrdEtAl1995); [Zhu et al. (1997)](references.bib#zhuEtAl1997) | Deterministic starts, bounds, polishing, and projected/scaled-gradient acceptance are Asterism choices |
| L-BFGS-B with finite-difference gradients, central when possible and one-sided at bounds | Liability, Tobit, mixed bivariate | [Fornberg (1988)](references.bib#fornberg1988), plus the L-BFGS-B papers | Step size, parameter scale, integration error, and bounds can dominate gradient accuracy |
| Sixteen-point Gauss–Legendre quadrature | Two-dimensional conditional normal probabilities | [Golub and Welsch (1969)](references.bib#golubWelsch1969) | Error increases near singular correlation; larger regions use a different approximation |
| Dense Cholesky factorization | Noncommuting covariance bases and family blocks | Standard linear algebra | Successful factorization does not imply statistical identification |
| Profile bisection | All supported interval families | [Venzon and Moolgavkar (1988)](references.bib#venzonMoolgavkar1988) for the profile construction | Failed/nonconverged points widen the interval and make it nonreportable |

The Rust dependency implements the L-BFGS-B algorithmic family described by
the cited papers. Exact tolerances, bounds, multistart schedules, analytic
derivatives, finite-difference steps, polishing, error codes, and deterministic
bootstrap generator are part of Asterism—not conclusions of those papers.
Every released artifact must therefore be tied to its source commit and must
pass the checks named in `release.toml`.

Dense storage is quadratic in roster size and dense factorization cubic in the
largest unblocked covariance dimension. Spatial covariance crosses families and
therefore loses the pedigree block savings; its bootstrap repeats a costly full
fit. The Gaussian bivariate implementation is deliberately limited to two
traits. Non-Gaussian region probabilities use a univariate normal calculation
in dimension one, fixed sixteen-point quadrature in dimension two, and a
sequential approximation above dimension two.

## Public capabilities outside 0.1 support

The following public objects remain accessible under [ADR
0013](adr/0013-scientific-support-does-not-remove-public-models.md), but their
results are not
scientifically supported by the 0.1 manifest: powered-exponential continuous
G×E, autoregressive covariance, association scans, variant-set tests, weighted
chi-square tail utilities, latent mediation, saturated bivariate benchmarks,
repeated-audiogram decomposition, BLUP/prediction headlines, and mixed
trait-kind combinations other than the binary/right-censored target. Their
continued presence is an API-compatibility decision, not a reportability claim.

## Qualification gaps that remain scientific work

The following gates cannot be closed by improving prose or adding citations:

- execute every exact prewritten scientific command against the clean release
  wheel and retain its fail-closed receipt; this includes the two discrete-G×E
  campaigns that made zero scientific attempts in the current sandbox;
- record what each check measured beside its own evidence, and claim nothing
  beyond it;
- measure target-sized time and peak-memory budgets, macOS/Linux numerical
  tolerances, the Medusa installed-wheel smoke, and all nine standard synthetic
  analysis receipts;
- bound several-component support by demonstrated matrix properties; exact
  basis dependence now refuses, while a qualified near-dependence precision
  rule is still needed before broadening the range beyond the tested bases;
- retain the target liability and mixed-model campaigns needed to bound the
  Mendell–Elston approximation at the intended family sizes, prevalence, and
  censoring pattern; and
- pin every qualification receipt to the released source, dependency locks,
  saved-wheel checksum, and platform artifact.

The primary-source audit behind this specification is retained in
[`research/asterism-0.1-statistical-method-sources.md`](research/asterism-0.1-statistical-method-sources.md).
