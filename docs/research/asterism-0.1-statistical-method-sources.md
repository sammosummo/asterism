# Primary-source map for Asterism 0.1 statistical methods

Research note, 2026-08-21. This note is a source dossier for the 0.1
statistical-methods specification. It is not the specification itself, is not
qualification evidence, and does not change the support boundary in
[ADR 0012](../adr/0012-small-analysis-ready-releases.md). It contains no
participant data.

## How to read the evidence

Three kinds of statement must remain separate in the eventual methods
document:

- Published theory defines a statistical model, likelihood, asymptotic
  result, or numerical algorithm.
- **Asterism choice** identifies a parameterisation, search rule, reporting
  quantity, failure policy, or combination of published ideas implemented in
  this repository.
- Asterism evidence is a simulation, independent implementation comparison,
  or target-design check. A paper does not validate this implementation, and a
  simulation on one design does not validate another.

The implementation links below describe the current working tree. The eventual
canonical methods document should cite a released source commit as well as the
papers.

## Coverage summary

| Supported 0.1 analysis | Closest primary foundation | Exactness of the match | Main remaining evidence/source issue |
| --- | --- | --- | --- |
| One-trait Gaussian heritability, ML/REML and profile interval | Patterson and Thompson (1971); Harville (1977); Venzon and Moolgavkar (1988) | Strong | Boundary and finite-sample calibration remain design dependent. |
| Several covariance components | Harville (1977) | Strong for the likelihood; Asterism-specific for mean-diagonal reporting | No primary source was found for Asterism's reporting invariant or its exact identifiability preflight. |
| Boundary variance-component tests | Self and Liang (1987); Crainiceanu and Ruppert (2004) | Strong only for the regular one-boundary setting | Cone geometry and finite-sample designs cannot inherit a universal 50:50 rule. |
| Bivariate genetic correlation | Almasy, Dyer, and Blangero (1997); Almasy and Blangero (1998) | Strong for pedigree covariance modelling | The Asterism missingness layout, parameterisation, and boundary interval behavior require their own checks. |
| Continuous GxE, exponential surface | Diego et al. (2003) | Very close model lineage | Full-fit independent agreement and target-design calibration are still release gates. |
| Continuous GxE, random regression | Kirkpatrick and Heckman (1989); Meyer (1998) | Strong covariance-function foundation | Asterism's rank-one null and conservative reference are local choices requiring empirical calibration. |
| Discrete GxE | Falconer (1952) | Strong two-environment conceptual foundation | The exact five Asterism nulls and mixture weights were not found verbatim in a primary source. |
| Binary liability | Dempster and Lerner (1950); Falconer (1965); Mendell and Elston (1974) | Strong model and approximation lineage | Accuracy of the sequential approximation in the supported family-size range needs explicit qualification. |
| One-trait censored/Tobit mixed model | Tobin (1958); Hughes (1999); Jacqmin-Gadda et al. (2000) | Strong latent-censoring and conditional-likelihood foundation | Asterism's pedigree covariance plus per-observation limits is an implementation synthesis. |
| Binary-diagnosis/censored-hearing genetic correlation | Catalano and Ryan (1992), combined with censored-mixed-model sources | Partial: the ingredients are published, the exact combination is not | The exact trait-kind combination needs independent and simulation checks before support. |
| Spatial-component presence test | Mardia and Marshall (1984); Davies (1977); Efron (1979) | Strong model and nonidentified-nuisance rationale | Bootstrap level and power must be checked on the target layout; range stays descriptive. |
| Numerical optimisation and integration | Byrd et al. (1995); Zhu et al. (1997); Kiefer (1953); Golub and Welsch (1969); Fornberg (1988) | Strong algorithmic provenance | Algorithm citations do not establish convergence of any fit; Asterism's convergence contract remains empirical. |

Powered-exponential GxE, prediction, association, variant-set discovery,
inferential latent mediation, autoregressive models, and the repeated-audiogram
decomposition are outside the 0.1 support set and are not researched here as
supported methods.

## 1. Gaussian variance components, REML, and profile likelihood

### Published foundation

For one trait, the common Gaussian variance-component model is

$$
\mathbf y \sim N(\mathbf X\boldsymbol\beta,\mathbf V),
\qquad
\mathbf V(\boldsymbol\theta)
=\sum_{k=1}^{q}\theta_k\mathbf K_k+\theta_e\mathbf I,
$$

where the covariance bases $\mathbf K_k$ are known and the nonnegative
coefficients are estimated. [Patterson and Thompson
(1971)](https://doi.org/10.1093/biomet/58.3.545) introduced estimation from
error contrasts, the construction now called restricted maximum likelihood.
[Harville (1977)](https://doi.org/10.1080/01621459.1977.10480998) gives the
general ML/REML variance-component treatment, including generalized least
squares for fixed effects and iterative covariance estimation.

For a scalar parameter of interest $\psi$, profile likelihood is

$$
\ell_p(\psi)=\max_{\boldsymbol\lambda}
\ell(\psi,\boldsymbol\lambda),
$$

where $\boldsymbol\lambda$ denotes all nuisance parameters. An interior 95%
likelihood-ratio interval uses values satisfying
$2\{\ell_p(\widehat\psi)-\ell_p(\psi)\}\leq\chi^2_{1,0.95}$.
[Wilks (1938)](https://doi.org/10.1214/aoms/1177732360) is the primary
large-sample likelihood-ratio reference; its chi-square conclusion requires a
regular interior parameter. It is not a boundary theorem.
[Venzon and Moolgavkar
(1988)](https://doi.org/10.2307/2347496) is a primary computational reference
for repeatedly maximizing over nuisance parameters to obtain profile endpoints.

### Asterism construction

The one-kernel model writes

$$
\mathbf V=\sigma^2\{h^2\mathbf K+(1-h^2)\mathbf I\},
$$

profiles $\sigma^2$, diagonalises $\mathbf K$ once, and uses a bounded
one-dimensional search over $h^2$. Models with multiple noncommuting bases
factorise $\mathbf V$ at each likelihood evaluation. These are implementation
choices in [`prepared.rs`](../../src/prepared.rs) and
[`components.rs`](../../src/components.rs), not claims made by the cited
papers. The exact bisection behavior, boundary inclusion rule, and
`profile_failures` widening policy are Asterism choices centralized in
[`interval.rs`](../../src/interval.rs).

The eigendecomposition reuse has later precedent in efficient mixed-model
work such as [Kang et al. (2008)](https://doi.org/10.1534/genetics.107.080101),
but that paper is not the source of Asterism's implementation and agreement
with it would not by itself qualify the code.

## 2. Boundary likelihood-ratio tests

### What published asymptotics support

When a single nonnegative variance component is zero and the remaining
parameters are regular and interior, the likelihood-ratio statistic has the
familiar limiting reference

$$
\tfrac12\chi^2_0+\tfrac12\chi^2_1.
$$

This is a special case of the boundary theory in [Self and Liang
(1987)](https://doi.org/10.1080/01621459.1987.10478472). It supports the form
of Asterism's one-component 50:50 rule, not its finite-sample accuracy in every
pedigree or censoring design.

[Crainiceanu and Ruppert
(2004)](https://doi.org/10.1111/j.1467-9868.2004.00438.x) derive finite-sample
and asymptotic distributions for a one-variance-component linear mixed model
and show that the usual large-sample mixture can be poor when the observations
cannot be partitioned into many independent, identically distributed blocks.
That warning is directly relevant to a modest number of nonidentical families.

For multivariate covariance cones, mixture weights are governed by local
parameter-space geometry, not by counting constrained coordinates. [Han and
Chang (2008)](https://doi.org/10.48550/arXiv.0808.2000) demonstrate that simple
binomial chi-square mixtures can be wrong in multivariate variance-component
linkage tests. The paper is a preprint and its linkage model is not Asterism's
GxE model, but its geometric counterexample is a strong warning against
transferring a 50:50 reference to a curved positive-semidefinite boundary.

### Asterism consequences

- One nonnegative component tested against zero may use the 50:50 rule only
  when all other relevant parameters are interior and identified.
- Correlation zero is an interior null and ordinarily uses $\chi^2_1$.
- Correlation $+1$ or $-1$ is a one-sided covariance-boundary null, but the
  exact reference still depends on the local model geometry.
- The random-regression GxE null lies on a curved rank-deficient covariance
  boundary. Asterism deliberately labels its current reference conservative;
  that is an empirically assessed project choice, not an application of a
  universal theorem.
- A failed constrained fit is unknown ground. Widening an interval over it is
  Asterism's conservative reporting policy, not a result in Self and Liang.

## 3. Multiple covariance components and identifiability

The Gaussian likelihood above directly accommodates multiple structured
covariance bases. A necessary algebraic warning is immediate: if the submitted
bases are linearly dependent on the observed roster, distinct coefficient
vectors can produce the same $\mathbf V$, so the individual coefficients are
not identified. Near dependence produces weak rather than exact
nonidentification. Harville's likelihood framework covers estimation once a
model is identified, but no primary source was found that validates Asterism's
particular preflight or a universal numerical condition-number cutoff.

Asterism's reportable scale-invariant summaries are defined by

$$
m_k=\frac{1}{n}\mathrm{tr}(\mathbf K_k),\qquad
c_k=\widehat\theta_k m_k,\qquad
p_k=\frac{c_k}{\sum_j c_j+\widehat\theta_e}.
$$

The invariance follows directly: replacing $\mathbf K_k$ by
$a\mathbf K_k$ and $\theta_k$ by $\theta_k/a$ leaves $c_k$ unchanged. This is
an Asterism-defined reporting invariant, not a conventional variance share
established by Harville. It is defined only when every structured basis has a
positive finite mean diagonal. A zero-diagonal kinship-class basis therefore
has reportable coefficients and contrasts, but no mean-diagonal proportion.
Raw coefficient ratios also depend on arbitrary matrix scaling and remain
diagnostic. The implementation is in
[`components.rs`](../../src/components.rs).

The release methods should state that identification is a property of the
submitted matrices, roster, missingness, and design—not a property of labels
such as “additive” or “household.” A primary reference for a practical,
model-agnostic identifiability diagnostic remains a research gap.

## 4. Two Gaussian traits and genetic correlation

Asterism stacks traits person by person and uses

$$
\mathbf V=\mathbf A\otimes\boldsymbol\Sigma_g
+\mathbf I\otimes\boldsymbol\Sigma_e
$$

before selecting the observed person-trait rows. The genetic correlation is

$$
\rho_g=\frac{\sigma_{g,12}}
{\sqrt{\sigma_{g,11}\sigma_{g,22}}}.
$$

[Almasy, Dyer, and Blangero
(1997)](https://doi.org/10.1002/%28SICI%291098-2272%281997%2914%3A6%3C953%3A%3AAID-GEPI65%3E3.0.CO%3B2-K)
is a primary pedigree application of bivariate quantitative-trait covariance
modelling and genetic correlation; [Almasy and Blangero
(1998)](https://doi.org/10.1086/301844) describes the broader pedigree
variance-component likelihood used by SOLAR. Both papers concern linkage as
well as polygenic covariance, so they provide methodological lineage rather
than an exact specification of Asterism's no-linkage bivariate model.

Trait-standardized parameters, preservation of exact correlation boundaries,
analytic gradients, deterministic starts, missing-observation selection, and
direct profiles for the reported correlations are Asterism choices in
[`bivariate.rs`](../../src/bivariate.rs). For 0.1, the reportable target is
$\rho_g$; weakly identified residual correlation must not be promoted merely
because the optimizer returns it.

## 5. Continuous gene by environment

### Exponential covariance surface

For environments $z_i,z_j$, Asterism uses

$$
\mathrm{Cov}(g_i,g_j)
=A_{ij}\sqrt{g(z_i)g(z_j)}
\exp\{-\lambda|z_i-z_j|\},
$$

with log-linear genetic and residual variances. [Diego et al.
(2003)](https://doi.org/10.1186/1471-2156-4-S1-S34) is the closest primary
source: it develops a genotype-by-age null model as a Gaussian stationary
stochastic process with log-linear variance and exponentially decaying genetic
correlation. Asterism generalizes the environmental coordinate beyond age and
fixes its own parameter bounds, starts, gradients, and test API.

The correlation null $\lambda=0$ is a flat lower boundary, so the 50:50
reference has the right asymptotic form under the Self-Liang conditions.
Whether it has the advertised level in the intended pedigrees and environment
distribution is an Asterism simulation question.

### Random-regression covariance surface

With $\boldsymbol\phi(z)=(1,z)^T$, the genetic surface is

$$
G(z_i,z_j)=\boldsymbol\phi(z_i)^T
\mathbf Q\boldsymbol\phi(z_j),\qquad \mathbf Q\succeq0.
$$

[Kirkpatrick and Heckman
(1989)](https://doi.org/10.1007/BF00290638) introduced covariance functions
for reaction norms and other function-valued traits. [Meyer
(1998)](https://doi.org/10.1186/1297-9686-30-3-221) gives a direct random-
regression route to estimating covariance functions from longitudinal genetic
data. These sources support the covariance-function and random-coefficient
construction, including correlations that can change sign; they do not define
Asterism's particular quadratic factorisation.

Asterism parameterises $\mathbf Q$ so positive semidefiniteness is preserved by
construction. Its no-correlation-change null is rank deficient and lies on a
curved cone boundary. The currently conservative reference and its observed
type-I behavior are Asterism evidence, not a result imported from Meyer. The
implemented surfaces and tests are in [`gxe.rs`](../../src/gxe.rs).

The powered-exponential family remains public but is outside 0.1 support. Its
shape cannot be selected after looking at the outcome without changing the
estimand and selection procedure.

## 6. Discrete gene by environment

[Falconer (1952)](https://doi.org/10.1086/281736) supplies the key primary
idea: treat expression of a trait in two environments as two genetically
correlated traits. Asterism's covariance is

$$
V_{ij}=A_{ij}s_{g,e_i}s_{g,e_j}
\begin{cases}
1,&e_i=e_j,\\
\rho_g,&e_i\ne e_j,
\end{cases}
+\mathbb 1(i=j)s_{e,e_i}^2,
$$

with a free genetic and residual scale in each environment. Free residual
scales are essential to keep unequal measurement noise from being forced into
the genetic contrast.

The parameterisation in [`discrete_gxe.rs`](../../src/discrete_gxe.rs) is an
Asterism implementation of Falconer's two-trait idea. No primary paper was
found that states Asterism's exact set of five tests and references verbatim:

- equal genetic variances and $\rho_g=1$:
  $\tfrac12\chi^2_1+\tfrac12\chi^2_2$;
- $\rho_g=1$ alone: $\tfrac12\chi^2_0+\tfrac12\chi^2_1$;
- equal genetic variances alone: $\chi^2_1$;
- equal residual variances alone: $\chi^2_1$;
- equality of genetic and residual quantities:
  $\tfrac12\chi^2_2+\tfrac12\chi^2_3$.

Those references should therefore be documented as Asterism's constrained-
parameter derivation plus calibration, with Self and Liang as general boundary
theory—not cited as if Falconer supplied the mixture weights. In 0.1 the
genetic-correlation interval and calibrated genetic tests are reportable;
group-specific heritabilities remain descriptive.

## 7. Liability-threshold model

The model is

$$
L_i=\mathbf x_i^T\boldsymbol\beta+g_i+e_i,
\qquad
\mathrm{Var}(\mathbf L)=h^2\mathbf A+(1-h^2)\mathbf I,
\qquad
Y_i=\mathbb 1(L_i>0).
$$

Fixing the total liability variance to one identifies the probit scale.
[Dempster and Lerner
(1950)](https://doi.org/10.1093/genetics/35.2.212) is the primary threshold-
heritability source, and [Falconer
(1965)](https://doi.org/10.1111/j.1469-1809.1965.tb00500.x) develops the
liability model for human disease recurrence.

The likelihood contribution of a family is a multivariate-normal orthant
probability. [Mendell and Elston
(1974)](https://pubmed.ncbi.nlm.nih.gov/4813384/) develops genetic analysis of
multifactorial qualitative traits and the sequential approximation lineage
used here. [Tallis (1961)](https://doi.org/10.1111/j.2517-6161.1961.tb00408.x)
is a primary source for moments of truncated multivariate normals underlying
sequential moment matching.

Asterism evaluates one- and two-dimensional probabilities directly and uses
Mendell-Elston sequential truncation above two dimensions; the two-dimensional
conditional integral uses 16-point Gauss-Legendre quadrature. Rare-class-first
ordering, correlation guards, finite-difference gradients, deterministic
starts, and failure codes are implementation choices in
[`liability.rs`](../../src/liability.rs). The liability-scale estimand is not
an observed-scale heritability and ML here is not interchangeable with Gaussian
REML.

The original papers establish the model and approximation, not its accuracy in
Asterism's target family structures. Large-family approximation error,
ordering sensitivity, prevalence imbalance, and boundary-test calibration
remain explicit qualification targets.

## 8. Censored Gaussian/Tobit mixed model

Let $Y_i^*$ be the complete latent measurement and let the observed record be
either its exact value or a statement that it lies above or below a
person-specific limit. Asterism uses

$$
\mathbf Y^*\sim N(\mathbf X\boldsymbol\beta,
\sigma^2\{h^2\mathbf A+(1-h^2)\mathbf I\}).
$$

[Tobin (1958)](https://doi.org/10.2307/1907382) is the foundational censored
latent-Gaussian regression reference, though its observations are independent
and its application is economic. [Hughes
(1999)](https://doi.org/10.1111/j.0006-341X.1999.00625.x) extends normal mixed
models to left and right censoring. Most directly, [Jacqmin-Gadda et al.
(2000)](https://doi.org/10.1093/biostatistics/1.4.355) write each correlated
censored-data likelihood contribution as the density of measured values times
the conditional distribution function of censored values given the measured
ones. That is the likelihood factorisation Asterism follows.

The free $\sigma^2$ is identified by measured values, so the target is the
heritability of the complete latent measurement—not the heritability of values
with limits substituted. Because a censored observation is a region rather
than a realized response, Asterism uses ML, not REML.

Per-observation upper and lower limits, pedigree covariance, direct
maximisation, shared orthant routines, and profile/failure behavior are
Asterism choices in [`tobit.rs`](../../src/tobit.rs). [Vaida and Liu
(2009)](https://doi.org/10.1198/jcgs.2009.07130) is relevant to efficient
censored normal mixed models, but its EM implementation is not the algorithm
used by Asterism.

The 0.1 support claim for extended-frequency audiograms requires interval
simulation at the intended censoring shares. Pair-family validation cannot
establish accuracy of the Mendell-Elston approximation in larger families.

## 9. Mixed binary and censored bivariate model

[Catalano and Ryan
(1992)](https://doi.org/10.1080/01621459.1992.10475264) derive a joint latent-
variable model for clustered discrete and continuous outcomes. Combined with
the conditional censored-normal likelihood of Jacqmin-Gadda et al. (2000), it
supports the ingredients of a binary/censored bivariate model. It does not
describe Asterism's exact model: Catalano and Ryan use a quasi-likelihood route,
while Asterism directly maximises a pedigree-block likelihood with exact or
approximated Gaussian region probabilities.

Asterism places the two latent traits in

$$
\mathbf V=\boldsymbol\Sigma_g\otimes\mathbf A
+\boldsymbol\Sigma_e\otimes\mathbf I,
$$

fixes a binary trait's total variance to one, leaves a continuous or censored
trait's scale free, and reports the scale-free genetic correlation. Observation
types alter whether a row contributes a density or a region probability. The
implementation is in [`mixed_bivariate.rs`](../../src/mixed_bivariate.rs).

No primary source was found for the exact synthesis of pedigree genetic
covariance, one binary liability, one observation-specific censored trait, and
Mendell-Elston family integration. Accordingly, the binary psychiatric-
diagnosis/censored-hearing combination must be described as an Asterism model
built from published components. Its reportable $\rho_g$ requires the exact-
combination simulation and independent full-fit check required by ADR 0012;
the other fitted quantities remain descriptive.

## 10. Spatial covariance and parametric bootstrap

Asterism adds

$$
K_s(i,j;\lambda)=\exp(-\lambda d_{ij}),
\qquad \lambda>0,
$$

with coefficient $\theta_s\geq0$ to the fixed covariance components.
[Mardia and Marshall
(1984)](https://doi.org/10.1093/biomet/71.1.135) is a primary source for ML
estimation in Gaussian spatial regression with a parametric residual covariance
model.

For great-circle distances, validity of the exponential kernel is not merely a
Euclidean assumption. [Gneiting
(2013)](https://doi.org/10.3150/12-BEJSP06) shows that completely monotone
functions, including the exponential family, are positive definite on spheres
of any dimension. Asterism should still validate the submitted numerical
matrix because duplicates, rounding, and additional component combinations can
create singular or ill-conditioned covariance matrices.

Under $H_0:\theta_s=0$, $\lambda$ disappears from the model. [Davies
(1977)](https://doi.org/10.1093/biomet/64.2.247) establishes why ordinary
likelihood-ratio asymptotics do not directly apply when a nuisance parameter is
present only under the alternative. This is stronger justification for
Asterism's parametric bootstrap than treating the problem as an ordinary
single variance-component boundary.

The bootstrap simulates from the fitted reduced model, refits reduced and full
models, and uses

$$
p=\frac{1+\sum_{b=1}^{B}\mathbb 1(T_b\geq T_{obs})}{B+1}.
$$

[Efron (1979)](https://doi.org/10.1214/aos/1176344552) is the foundational
bootstrap source. [Phipson and Smyth
(2010)](https://doi.org/10.2202/1544-6115.1585) specifically explains the
add-one correction for Monte Carlo tests, preventing a reported zero p-value.

The range $1/\lambda$ or half-correlation distance $\log(2)/\lambda$ is
descriptive in 0.1. [Zhang
(2004)](https://doi.org/10.1198/016214504000000241) proves that variance and
range parameters in the Matérn class cannot all be consistently estimated
under fixed-domain asymptotics; the exponential model is a Matérn special case.
That theorem is not a diagnosis of every Asterism layout, but it supplies a
strong theoretical reason to expect weak range identification and to avoid
turning a boundary-hitting range profile into a precise scientific claim.

Profiling versus numerically integrating over range, the integration grid,
analytic gradients, deterministic starts, bootstrap generator, and fit-failure
policy are Asterism choices in [`spatial.rs`](../../src/spatial.rs). The target-
layout bootstrap check remains necessary; the cited theory cannot supply its
empirical level or power.

## 11. Numerical and optimisation methods actually used

The current implementation uses several different numerical routes; the final
methods document should not imply that every model is optimized identically.

| Implementation path | Current Asterism use | Primary algorithmic source | Important limitation |
| --- | --- | --- | --- |
| Bounded one-dimensional golden search | Profiled one-kernel Gaussian fit | [Kiefer (1953)](https://doi.org/10.1090/S0002-9939-1953-0055639-3) | Requires an effectively unimodal bounded objective; Asterism still verifies endpoints and convergence. |
| L-BFGS-B with analytic gradients | Multiple Gaussian components, bivariate Gaussian, continuous/discrete GxE, spatial | [Byrd et al. (1995)](https://doi.org/10.1137/0916069); [Zhu et al. (1997)](https://doi.org/10.1145/279232.279236) | An optimizer return code is not sufficient evidence that the statistical fit is reportable. |
| L-BFGS-B with central finite-difference gradients, clamped to one-sided differences at bounds | Liability, one-trait Tobit, mixed bivariate | [Fornberg (1988)](https://doi.org/10.1090/S0025-5718-1988-0935077-0) for finite-difference formulas, plus the L-BFGS-B sources | Step size, parameter scale, bounds, and integration noise can dominate gradient accuracy. |
| Dense Cholesky or eigendecomposition | Gaussian block likelihoods and covariance checks | Standard numerical linear algebra; implementation is in the linked Rust modules | A successful factorisation does not establish parameter identifiability. |
| 16-point Gauss-Legendre quadrature | Two-dimensional conditional Gaussian probabilities in non-Gaussian models | [Golub and Welsch (1969)](https://doi.org/10.1090/S0025-5718-69-99647-1) | Fixed-order quadrature accuracy deteriorates near extreme correlation; Asterism's guard and error study are local evidence. |
| Bisection over profile deviance | Reported profile endpoints | Venzon and Moolgavkar (1988) for profile-likelihood computation; bisection is Asterism's endpoint solver | A failed profile evaluation must not be treated as likelihood below threshold. |

The bound-constrained implementation actually linked by the Rust modules is
`rcompat-lbfgsb`, recorded in [`Cargo.toml`](../../Cargo.toml). The cited
L-BFGS-B papers identify the algorithmic family; exact behavior, tolerances,
starts, polishing, scaled-gradient checks, and convergence classification come
from Asterism's source and tests. The comparisons and simulations behind them
are in the [validation record](../validation.md).

## 12. Gaps to close before canonical methods documentation

1. Exact discrete-GxE test derivation. Write and independently check the
   tangent-cone argument for all five nulls. Cite Self and Liang as general
   theory, but label the resulting weights as an Asterism derivation unless an
   exact primary match is found.
2. Random-regression null reference. Retain the conservative label and
   target-design simulation. If a calibrated test is wanted, a parametric
   bootstrap is more defensible than asserting a universal chi-bar-square
   weight.
3. Multiple-component identifiability. Specify a preflight tied to the
   observed roster and covariance bases. No source found here justifies a
   universal condition-number threshold.
4. Mendell-Elston accuracy envelope. Document approximation error by family
   size, correlation, outcome imbalance, censoring pattern, and ordering. The
   foundational paper does not establish the target envelope automatically.
5. Mixed binary/censored synthesis. State plainly that the exact model was
   not found in one primary paper. The intended diagnosis/hearing pairing needs
   its own independent full-fit and simulation evidence.
6. Spatial bootstrap. Fix the bootstrap generator, number of replicates,
   failure handling, add-one calculation, and target-layout pass rule before
   looking at actual outcomes.
7. Numerical provenance. Pin paper references to the released dependency
   versions and source commit. Algorithm names alone do not demonstrate that
   Asterism used them correctly.
8. Reference metadata (closed in the canonical-documentation pass). The
   verified references below were transferred into `docs/references.bib` with
   DOI plus PMID/PMCID or a stable archive where available, and automated
   citation-key validation was added.

## Verified source list transferred to the canonical bibliography

These entries are retained here as research provenance. The canonical keys and
machine-checked metadata now live in `docs/references.bib`.

- Almasy L, Blangero J. 1998. Multipoint quantitative-trait linkage analysis
  in general pedigrees. *American Journal of Human Genetics* 62:1198–1211.
  [doi:10.1086/301844](https://doi.org/10.1086/301844),
  [PMID 9545414](https://pubmed.ncbi.nlm.nih.gov/9545414/).
- Almasy L, Dyer TD, Blangero J. 1997. Bivariate quantitative trait linkage
  analysis: pleiotropy versus coincident linkages. *Genetic Epidemiology*
  14:953–958.
  [doi:10.1002/(SICI)1098-2272(1997)14:6<953::AID-GEPI65>3.0.CO;2-K](https://doi.org/10.1002/%28SICI%291098-2272%281997%2914%3A6%3C953%3A%3AAID-GEPI65%3E3.0.CO%3B2-K),
  [PMID 9433606](https://pubmed.ncbi.nlm.nih.gov/9433606/).
- Byrd RH, Lu P, Nocedal J, Zhu C. 1995. A limited memory algorithm for bound
  constrained optimization. *SIAM Journal on Scientific Computing*
  16:1190–1208. [doi:10.1137/0916069](https://doi.org/10.1137/0916069).
- Catalano PJ, Ryan LM. 1992. Bivariate latent variable models for clustered
  discrete and continuous outcomes. *Journal of the American Statistical
  Association* 87:651–658.
  [doi:10.1080/01621459.1992.10475264](https://doi.org/10.1080/01621459.1992.10475264).
- Crainiceanu CM, Ruppert D. 2004. Likelihood ratio tests in linear mixed
  models with one variance component. *Journal of the Royal Statistical
  Society B* 66:165–185.
  [doi:10.1111/j.1467-9868.2004.00438.x](https://doi.org/10.1111/j.1467-9868.2004.00438.x).
- Davies RB. 1977. Hypothesis testing when a nuisance parameter is present
  only under the alternative. *Biometrika* 64:247–254.
  [doi:10.1093/biomet/64.2.247](https://doi.org/10.1093/biomet/64.2.247).
- Dempster ER, Lerner IM. 1950. Heritability of threshold characters.
  *Genetics* 35:212–236.
  [doi:10.1093/genetics/35.2.212](https://doi.org/10.1093/genetics/35.2.212),
  [PMID 17247344](https://pubmed.ncbi.nlm.nih.gov/17247344/).
- Diego VP, Almasy L, Dyer TD, Soler JMP, Blangero J. 2003. Strategy and model
  building in the fourth dimension: a null model for genotype by age
  interaction as a Gaussian stationary stochastic process. *BMC Genetics*
  4(Suppl 1):S34.
  [doi:10.1186/1471-2156-4-S1-S34](https://doi.org/10.1186/1471-2156-4-S1-S34).
- Efron B. 1979. Bootstrap methods: another look at the jackknife. *Annals of
  Statistics* 7:1–26.
  [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552).
- Falconer DS. 1952. The problem of environment and selection. *American
  Naturalist* 86:293–298.
  [doi:10.1086/281736](https://doi.org/10.1086/281736).
- Falconer DS. 1965. The inheritance of liability to certain diseases,
  estimated from the incidence among relatives. *Annals of Human Genetics*
  29:51–76.
  [doi:10.1111/j.1469-1809.1965.tb00500.x](https://doi.org/10.1111/j.1469-1809.1965.tb00500.x).
- Fornberg B. 1988. Generation of finite difference formulas on arbitrarily
  spaced grids. *Mathematics of Computation* 51:699–706.
  [doi:10.1090/S0025-5718-1988-0935077-0](https://doi.org/10.1090/S0025-5718-1988-0935077-0).
- Gneiting T. 2013. Strictly and non-strictly positive definite functions on
  spheres. *Bernoulli* 19:1327–1349.
  [doi:10.3150/12-BEJSP06](https://doi.org/10.3150/12-BEJSP06).
- Golub GH, Welsch JH. 1969. Calculation of Gauss quadrature rules.
  *Mathematics of Computation* 23:221–230.
  [doi:10.1090/S0025-5718-69-99647-1](https://doi.org/10.1090/S0025-5718-69-99647-1).
- Han SS, Chang JT. 2008. Reconsidering the asymptotic null distribution of
  likelihood ratio tests for genetic linkage in multivariate variance
  components models. arXiv:0808.2000.
  [doi:10.48550/arXiv.0808.2000](https://doi.org/10.48550/arXiv.0808.2000).
- Harville DA. 1977. Maximum likelihood approaches to variance component
  estimation and to related problems. *Journal of the American Statistical
  Association* 72:320–338.
  [doi:10.1080/01621459.1977.10480998](https://doi.org/10.1080/01621459.1977.10480998).
- Hughes JP. 1999. Mixed effects models with censored data with application to
  HIV RNA levels. *Biometrics* 55:625–629.
  [doi:10.1111/j.0006-341X.1999.00625.x](https://doi.org/10.1111/j.0006-341X.1999.00625.x),
  [PMID 11318225](https://pubmed.ncbi.nlm.nih.gov/11318225/).
- Jacqmin-Gadda H, Thiébaut R, Chêne G, Commenges D. 2000. Analysis of
  left-censored longitudinal data with application to viral load in HIV
  infection. *Biostatistics* 1:355–368.
  [doi:10.1093/biostatistics/1.4.355](https://doi.org/10.1093/biostatistics/1.4.355),
  [PMID 12933561](https://pubmed.ncbi.nlm.nih.gov/12933561/).
- Kang HM, Zaitlen NA, Wade CM, Kirby A, Heckerman D, Daly MJ, Eskin E. 2008.
  Efficient control of population structure in model organism association
  mapping. *Genetics* 178:1709–1723.
  [doi:10.1534/genetics.107.080101](https://doi.org/10.1534/genetics.107.080101),
  [PMID 18385116](https://pubmed.ncbi.nlm.nih.gov/18385116/).
- Kiefer J. 1953. Sequential minimax search for a maximum. *Proceedings of the
  American Mathematical Society* 4:502–506.
  [doi:10.1090/S0002-9939-1953-0055639-3](https://doi.org/10.1090/S0002-9939-1953-0055639-3).
- Kirkpatrick M, Heckman N. 1989. A quantitative genetic model for growth,
  shape, reaction norms, and other infinite-dimensional characters. *Journal
  of Mathematical Biology* 27:429–450.
  [doi:10.1007/BF00290638](https://doi.org/10.1007/BF00290638),
  [PMID 2769086](https://pubmed.ncbi.nlm.nih.gov/2769086/).
- Mardia KV, Marshall RJ. 1984. Maximum likelihood estimation of models for
  residual covariance in spatial regression. *Biometrika* 71:135–146.
  [doi:10.1093/biomet/71.1.135](https://doi.org/10.1093/biomet/71.1.135).
- Mendell NR, Elston RC. 1974. Multifactorial qualitative traits: genetic
  analysis and prediction of recurrence risks. *Biometrics* 30:41–57.
  [doi:10.2307/2529616](https://doi.org/10.2307/2529616),
  [PMID 4813384](https://pubmed.ncbi.nlm.nih.gov/4813384/).
- Meyer K. 1998. Estimating covariance functions for longitudinal data using a
  random regression model. *Genetics Selection Evolution* 30:221–240.
  [doi:10.1186/1297-9686-30-3-221](https://doi.org/10.1186/1297-9686-30-3-221).
- Patterson HD, Thompson R. 1971. Recovery of inter-block information when
  block sizes are unequal. *Biometrika* 58:545–554.
  [doi:10.1093/biomet/58.3.545](https://doi.org/10.1093/biomet/58.3.545).
- Phipson B, Smyth GK. 2010. Permutation p-values should never be zero:
  calculating exact p-values when permutations are randomly drawn.
  *Statistical Applications in Genetics and Molecular Biology* 9:Article 39.
  [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585),
  [PMID 21044043](https://pubmed.ncbi.nlm.nih.gov/21044043/).
- Self SG, Liang K-Y. 1987. Asymptotic properties of maximum likelihood
  estimators and likelihood ratio tests under nonstandard conditions.
  *Journal of the American Statistical Association* 82:605–610.
  [doi:10.1080/01621459.1987.10478472](https://doi.org/10.1080/01621459.1987.10478472).
- Tallis GM. 1961. The moment generating function of the truncated multi-normal
  distribution. *Journal of the Royal Statistical Society B* 23:223–229.
  [doi:10.1111/j.2517-6161.1961.tb00408.x](https://doi.org/10.1111/j.2517-6161.1961.tb00408.x).
- Tobin J. 1958. Estimation of relationships for limited dependent variables.
  *Econometrica* 26:24–36.
  [doi:10.2307/1907382](https://doi.org/10.2307/1907382).
- Vaida F, Liu L. 2009. Fast implementation for normal mixed effects models
  with censored response. *Journal of Computational and Graphical Statistics*
  18:797–817.
  [doi:10.1198/jcgs.2009.07130](https://doi.org/10.1198/jcgs.2009.07130).
- Venzon DJ, Moolgavkar SH. 1988. A method for computing profile-likelihood-
  based confidence intervals. *Journal of the Royal Statistical Society C*
  37:87–94. [doi:10.2307/2347496](https://doi.org/10.2307/2347496).
- Wilks SS. 1938. The large-sample distribution of the likelihood ratio for
  testing composite hypotheses. *Annals of Mathematical Statistics* 9:60–62.
  [doi:10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360).
- Zhang H. 2004. Inconsistent estimation and asymptotically equal
  interpolations in model-based geostatistics. *Journal of the American
  Statistical Association* 99:250–261.
  [doi:10.1198/016214504000000241](https://doi.org/10.1198/016214504000000241).
- Zhu C, Byrd RH, Lu P, Nocedal J. 1997. Algorithm 778: L-BFGS-B: Fortran
  subroutines for large-scale bound-constrained optimization. *ACM
  Transactions on Mathematical Software* 23:550–560.
  [doi:10.1145/279232.279236](https://doi.org/10.1145/279232.279236).
