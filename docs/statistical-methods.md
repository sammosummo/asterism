# Statistical methods

This document is the canonical statistical specification for analyses whose
versioned `supported_in_*` declaration is active in `release.toml`, following
[ADR 0012](adr/0012-small-analysis-ready-releases.md).
It describes the estimand, likelihood, estimator, inference, public record, and
qualification boundary for each supported analysis. The implementation remains
authoritative for computation; this document is authoritative for scientific
interpretation. Author-year links identify entries in the canonical
[BibTeX bibliography](references.bib).

Published theory and an Asterism implementation choice are not the same kind of
evidence. Each section therefore separates the published basis from the exact
construction used here. Numerical agreement, simulation, and a successful fit
are also different claims.

## What a release requires

A public method can exist without being one of the nine 0.2 supported analyses.
Before a release is made, each supported analysis must meet three conditions:

1. its versioned `supported_in_*` declaration is active in `release.toml`;
2. every check named for it has a configured pass rule, and that rule passes
   for the released source and artifact;
3. its standard synthetic receipt records a fit with every fact true — finite
   values, a converged free fit, converged constrained fits for inference, and
   no failed profile evaluation.

The `0.2.0` release includes the corrected one-component censored and
mixed-pair checks. The several-component censored fit, intervals and explicitly
asymptotic test are supported; the exact failed 75%-expected-censoring target is
retained as a narrow known limitation rather than a global release gate, and
the optional bootstrap remains experimental.
The figures in the [validation record](validation.md) provide broader
scientific context; the release manifest and its recorded evidence remain the
authority for a particular artefact.

The nine 0.2 analysis families are:

| Analysis ID | Public entry points | Supported quantities |
| --- | --- | --- |
| `one_trait_gaussian_heritability` | `prepare`; `PreparedModel.fit` | $h^2$, profile interval, and zero-heritability test |
| `several_covariance_components` | `ComponentModel.fit`, `.interval`, `.test`, `.equality_test`, `.contrasts` | Mean-diagonal contributions/proportions and their intervals; coefficients/contrasts for zero-diagonal bases |
| `bivariate_genetic_correlation` | `BivariateModel.fit`, `.interval`, `.test` | Genetic correlation $\rho_g$, interval, and test |
| `continuous_gene_by_environment` | `GxeModel.fit`, `.interval`, `.test` | Heritability and genetic correlation at prespecified environments, their intervals, and correlation/interaction tests |
| `discrete_gene_by_environment` | `DiscreteGxeModel.fit`, `.correlation_interval`, `.test` | Genetic correlation, its interval, and the three genetic tests |
| `binary_liability_heritability` | `LiabilityModel.fit`, `.interval`, `.test` | Liability-scale $h^2$, interval, and test |
| `one_trait_censored` | `tobit_fit`, `tobit_interval`, `tobit_test` | Complete-trait $h^2$, interval, and test |
| `mixed_binary_censored_genetic_correlation` | `mixed_bivariate_fit`, `mixed_bivariate_interval`, `mixed_bivariate_test` | Genetic correlation for a mixed pair including a continuous trait, interval, and test |
| `one_trait_censored_components` | `CensoredComponentModel.fit`, `.interval`, `.test` | Coefficients, mean-diagonal proportions, `coefficient_interval`, `mean_diagonal_proportion_interval`, and `asymptotic_test` |

Quantities marked *descriptive* have no check standing behind them; they are
returned because the fit computes them. Quantities marked *diagnostic* exist to
inspect a fit or reproduce its parameterisation, and their values depend on how
the problem was set up rather than only on the data.

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
adds `what` and `estimator`. The implementation populates boundary containment
only for the prepared one-trait and Tobit families; it is `None` in the other
families. The existing Tobit fields do not by themselves qualify the interval
recipe; the separate fixed-instrument coverage and boundary-truth checks now
pass and are retained in the validation record.

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

The one-half weight is an asymptotic tangent-cone probability, not a
finite-sample requirement that half of fitted null datasets have
$T=0$. [Crainiceanu and Ruppert
(2004)](references.bib#crainiceanuRuppert2004) derive finite-sample null laws
for Gaussian mixed models whose atom and positive part depend on the design.
The share of null fits in Asterism's settled LRT atom is therefore reported as
a diagnostic only. Neither agreement with one half nor departure from it is a
stand-alone test of calibration.

### Several-component censored tests

For a censored model with several identifiable covariance components, `.test`
uses the same one-boundary Self–Liang approximation but labels it
`asymptotic_mixture_50_50`. It refuses if any untested variance or the residual
rests on its bound: the resulting tangent cone is no longer the one-boundary
problem. A fixed-instrument target campaign records the approximation's
rejection level for that exact design, without treating the realised LRT atom
as a requirement or calling the reference finite-sample exact. Such a
measurement does not define a censoring threshold for another design.

The exact 1,909-person, four-component fixed-instrument target failed at 75%
expected censoring. Its analytic p-value must not be reported without a
design-specific simulated null tied to the model, design, source and seed. This
named limitation does not withdraw the generally available asymptotic test or
affect fitting, intervals or the released one-component test.

The experimental `.bootstrap` route instead fits the tested coefficient at exactly zero,
generates latent responses from that fitted constrained null while preserving
the supplied design and fixed censoring limits, and refits both the null and
alternative for every bootstrap dataset. If $T_{\mathrm{obs}}$ is the observed
settled LRT and $T_b^\ast$ is bootstrap statistic $b$, it reports

$$
\widehat p
=\frac{1+\sum_{b=1}^{B}\mathbb{1}(T_b^\ast\geq T_{\mathrm{obs}})}{B+1}.
$$

The returned rule is `parametric_bootstrap_add_one`. The caller chooses $B$;
the add-one count follows [Phipson and Smyth
(2010)](references.bib#phipsonSmyth2010), and every requested bootstrap fit
must complete for a p-value to be returned. Inner datasets use deterministic
coordinate-derived random streams and are refitted in parallel, so thread
schedule and thread count do not change the result. If any inner refit fails,
the full request still refuses rather than shortening its denominator. The
error now names the lowest failed zero-based coordinate and preserves the
underlying fit code. `CensoredComponentModel.bootstrap_replay` regenerates that
exact seed-coordinate pair as a diagnostic record with no p-value; it does not
discard, replace or reinterpret the failed draw.

This route is deliberately narrower than a general boundary bootstrap. It
refuses when an untested nuisance variance is fitted on its bound. Ordinary
plug-in bootstrap can be inconsistent at nuisance boundaries, as [Andrews
(2000)](https://doi.org/10.1111/1468-0262.00114) demonstrates; the shrunk
boundary-aware procedure of [Guédon, Baey and Kuhn
(2024)](https://doi.org/10.1093/biomet/asae025) would require a separate
implementation and qualification. Nuisance boundaries are assessed on
mean-diagonal proportions rather than raw coefficients, so rescaling a
covariance matrix cannot by itself change this refusal. Asterism's
constrained-null bootstrap is implemented but not scientifically qualified as a
finite-sample release procedure. Its optional outer calibration is separate
from the 0.2 gate for the fit, interval and labelled asymptotic test.

## Numerical implementation

Asterism does not use one optimizer for every model:

| Route | Families | Published algorithmic lineage | Asterism-specific qualification concern |
| --- | --- | --- | --- |
| Bounded golden-section refinement after a fixed grid | Prepared one-kernel Gaussian | [Kiefer (1953)](references.bib#kiefer1953) | Endpoints, singular boundaries, and convergence are checked locally |
| L-BFGS-B with analytic gradients | Multiple components, bivariate Gaussian, continuous/discrete G×E, spatial | [Byrd et al. (1995)](references.bib#byrdEtAl1995); [Zhu et al. (1997)](references.bib#zhuEtAl1997) | Deterministic starts, bounds, polishing, and projected/scaled-gradient acceptance are Asterism choices |
| L-BFGS-B with finite-difference gradients, central when possible and one-sided at bounds | Liability, Tobit, mixed bivariate | [Fornberg (1988)](references.bib#fornberg1988), plus the L-BFGS-B papers | Step size, parameter scale, integration error, and bounds can dominate gradient accuracy |
| Owen's angular form, integrated adaptively to a tolerance | Two-dimensional conditional normal probabilities | [Owen (1956)](references.bib#owen1956) | Measured against an independent reference to under 1e-9 across the whole correlation range and away from the symmetric centre; larger regions use a different approximation |
| Fixed-order Gauss–Legendre quadrature | Banded inversion of the weighted chi-square tail | [Golub and Welsch (1969)](references.bib#golubWelsch1969) | Fixed-order accuracy deteriorates where the integrand oscillates fastest, which the banding is chosen to bound |
| Dense Cholesky factorization | Noncommuting covariance bases and family blocks | Standard linear algebra | Successful factorization does not imply statistical identification |
| Profile bisection | All supported interval families | [Venzon and Moolgavkar (1988)](references.bib#venzonMoolgavkar1988) for the profile construction | Failed or nonconverged points widen the interval, and the fit record counts them |

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
in dimension one, an adaptively integrated Owen angular form in dimension two,
and a sequential approximation above dimension two.

## What is still outstanding

The one-component censored numerical code is unchanged. Its earlier
outcome-adaptive records are historical, and its corrected fixed-instrument
MCMCglmm comparison, recovery, interval coverage and target-design level now
pass. Corrected mixed-pair recovery and coverage checks also pass for the
continuous, binary, censored and censored-pair cells.

Two things that were on this list are gone rather than done: the target-sized
time and memory budgets, withdrawn by
[ADR 0019](adr/0019-no-resource-budget-gates-a-release.md), and the supported
design range, withdrawn by
[ADR 0020](adr/0020-no-design-range-gates-a-result.md).

Several-component censored support is part of the 0.2 release. Its independent
dense-likelihood comparison measures the sequential region approximation, and
its fixed-instrument
component-recovery and interval-coverage checks pass. The analytic test remains
available with the explicit `asymptotic_mixture_50_50` label when nuisance
variances are interior. The exact 1,909-person, four-component target at 75%
expected censoring is a known exception whose p-value requires a design-specific
simulated null. The optional bootstrap target failed its zero-refusal rule
because eleven outer bootstrap requests in the 75% null cell each encountered
at least one inner-fit failure. It remains experimental. None of this creates a
universal censoring threshold.

Two further matters remain open as science rather than process. General
several-component support is bounded by demonstrated matrix properties: exact
basis dependence refuses, and a qualified near-dependence precision rule is
still wanted before the tested bases are broadened. The Mendell–Elston
approximation is also still to be bounded at the intended family sizes,
prevalence and censoring pattern for the liability and mixed models.

The primary-source audit behind this specification is retained in
[`research/asterism-0.1-statistical-method-sources.md`](research/asterism-0.1-statistical-method-sources.md).
