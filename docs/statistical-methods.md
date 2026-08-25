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
validation](development.md); they do not silently fill those release
gates.

The eight 0.1 analysis families are:

| Analysis ID | Public entry points | Reportable target |
| --- | --- | --- |
| `one_trait_gaussian_heritability` | `prepare`; `PreparedModel.fit` | $h^2$, profile interval, and zero-heritability test |
| `several_covariance_components` | `ComponentModel.fit`, `.interval`, `.test`, `.equality_test`, `.contrasts` | Mean-diagonal contributions/proportions and their intervals; coefficients/contrasts for zero-diagonal bases |
| `bivariate_genetic_correlation` | `BivariateModel.fit`, `.interval`, `.test` | Genetic correlation $\rho_g$, interval, and test |
| `continuous_gene_by_environment` | `GxeModel.fit`, `.interval`, `.test` | Heritability and genetic correlation at prespecified environments, their intervals, and correlation/interaction tests |
| `discrete_gene_by_environment` | `DiscreteGxeModel.fit`, `.correlation_interval`, `.test` | Genetic correlation, its interval, and the three genetic tests |
| `binary_liability_heritability` | `LiabilityModel.fit`, `.interval`, `.test` | Liability-scale $h^2$, interval, and test |
| `one_trait_tobit_audiogram` | `tobit_fit`, `tobit_interval`, `tobit_test` | Complete-trait $h^2$, interval, and test |
| `mixed_binary_censored_genetic_correlation` | `mixed_bivariate_fit`, `mixed_bivariate_interval`, `mixed_bivariate_test` | Genetic correlation for a mixed pair including a continuous trait, interval, and test |

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

where $\widehat{\mathbf r}=\mathbf y-\mathbf X\widehat{\boldsymbol\beta}(\boldsymbol\theta)$.
Estimation from
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

## What is still outstanding

Every supported analysis has now been measured against the current build, and
the cross-platform agreement and the Medusa installed-wheel smoke with it. What
remains before a release is the standard synthetic receipts, which refuse to
run on anything but a clean released build and are therefore produced during
the release itself.

Two things that were on this list are gone rather than done: the target-sized
time and memory budgets, withdrawn by
[ADR 0019](adr/0019-no-resource-budget-gates-a-release.md), and the supported
design range, withdrawn by
[ADR 0020](adr/0020-no-design-range-gates-a-result.md).

Two remain open as science rather than process. Several-component support is
bounded by demonstrated matrix properties: exact basis dependence refuses, and
a qualified near-dependence precision rule is still wanted before the tested
bases are broadened. And the Mendell–Elston approximation is still to be
bounded at the intended family sizes, prevalence and censoring pattern for the
liability and mixed models.

The primary-source audit behind this specification is retained in
[`research/asterism-0.1-statistical-method-sources.md`](research/asterism-0.1-statistical-method-sources.md).
