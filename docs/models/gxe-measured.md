# Gene by environment, measured

Analysis id `continuous_gene_by_environment`. Runnable example in [`examples/`](../../examples/).

## Using it

A measured environment, with the runnable version in
[`examples/gxe.py`](../../examples/gxe.py):

```sh
python examples/gxe.py
```

```
people:                    2000
environment -1: h2 0.639, genetic variance 1.848
environment +0: h2 0.502, genetic variance 0.998
environment +1: h2 0.684, genetic variance 2.043
rho_g between the ends:    0.026
converged:                 True
p (no interaction):        0.0001

Genetic variance is smallest at the centre and grows towards either end,
which is what a random-regression slope does. Genes at opposite ends of
the environment are almost unrelated, and the test finds it. Detecting
interaction takes far more data than estimating a heritability does.
```

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
```

`GxeModel` takes an environment measured on a range and smooths across it. For
an environment that is a label rather than a range, see
[gxe-binary.md](gxe-binary.md).

Available surfaces are `exponential` and `random_regression`. They are
different covariance families; choose the family independently of the fitted
outcome. Only random regression can represent negative genetic correlations and
crossovers. The powered-exponential family is public but unsupported; see
[outside-support.md](../outside-support.md).

## The model

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
(2003)](../references.bib#diegoEtAl2003), generalized by Asterism from age to a
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
Heckman (1989)](../references.bib#kirkpatrickHeckman1989), and random-regression
estimation of genetic covariance functions is developed by [Meyer
(1998)](../references.bib#meyer1998). Asterism's low-rank factorisation, parameter
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

## What stands behind it

Across 2,000 simulations per scenario, the exponential gene-by-environment
correlation test rejected 0.048 at nominal 0.05 under a flat null and 0.053
when the true log-linear surface changed genetic scale without reordering. The
random-regression boundary tests were conservative. A linear-scale rank-one
surface fitted with the wrong exponential family produced a 0.107 rejection
rate for a nominal 0.05 no-reordering test, showing that surface
misspecification can imitate genetic reordering.

Gene-by-environment 95% interval coverage ranged from 0.953 to 0.983 in the
studied settings, but 80% to 92% of genetic-correlation intervals and up to 64%
of some heritability intervals reached a bound. Point estimates were also
pulled toward correlation one near the boundary.

The participant-free target-design gate is a pedigree-scale and design-identity
stress test, not a reconstruction of GOBS ages, relationship coefficients,
trait-specific missingness or outcomes. Its reviewed aggregate envelope has
1,909 rows in 202 pedigree components, with largest component 180. The
structure-matched synthetic pedigree has 27,691 nonzero lower-triangle
relationships, 0.47% fewer than the predecessor aggregate receipt's 27,821;
that discrepancy was recorded before fitting and was not tuned away. A
deterministic synthetic standardised-age coordinate stays in the already
qualified interval [-1.5, 1.5], varies within every non-singleton component,
and combines with synthetic sex in the full-rank six-column design
`1, z, z^2, s, zs, z^2s`. Quantities are judged only at z = -1, 0 and 1.

For either supported surface, the independent reference assembles each pedigree
block from

\[
V_{ij}=A_{ij}G(z_i,z_j)+\mathbf{1}_{i=j}R(z_i),
\]

then sums blockwise Gaussian sufficient statistics and profiles the six fixed
effects globally. For the exponential surface,
\(G(z_i,z_j)=\exp\{[\alpha_g+\gamma_g z_i]/2\}
\exp\{[\alpha_g+\gamma_g z_j]/2\}
\exp\{-\lambda_g|z_i-z_j|\}\) and
\(R(z)=\exp(\alpha_e+\gamma_e z)\). The random-regression reference instead
uses \(G(z_i,z_j)=[1,z_i]\Sigma_g[1,z_j]^\mathsf{T}\) and
\(R(z)=[1,z]\Sigma_e[1,z]^\mathsf{T}\), with both coefficient-covariance
matrices formed from independent Cholesky coordinates. Thus the target-sized
reference does not call Asterism's covariance assembly or optimiser.

On the reduced qualification run of 100 null replicates per surface on 21
August 2026, both target-sized REML comparisons converged. The worst common
quantity and log-likelihood differences were respectively **5.49e-07** and
**2.68e-11** for the exponential surface, and **3.50e-07** and **3.64e-12**
for random regression, against prewritten tolerances of 0.001 and 0.0001. The
separate n = 80 dense-likelihood sentinel also passed both surfaces.

The same run scored every one of its 200 surface-replicates under the flat
\(V=0.5A+0.5I\) null, with no refused, nonconverged, incomplete or wrong-rule
fits. At nominal 0.05, correlation and interaction rejection counts were 1 and
5 of 100 for the exponential surface, and 1 and 4 of 100 for random regression.
For \(r\) rejections among \(m\) attempted replicates, the refusal rule is

\[
L=\mathrm{Beta}^{-1}(0.05;r,m-r+1)>0.05,
\]

with \(L=0\) when \(r=0\): a cell fails only when its one-sided 95% exact
binomial lower bound establishes anti-conservative rejection. Conservative
cells remain visible rather than being misclassified as undercoverage, and any
failed or unscored replicate fails the gate separately. The exact release rule
is pinned at 500 replicates per surface but has not yet been executed; the
evidence above is the completed 100-per-surface reduced qualification.

The historical discrete-model campaign simulated 500 responses per scenario on
the participant-backed, unbalanced GOBS pedigree, with sex as the environment.
Those results remain informative but are not portable release evidence. The
replacement gate uses a reviewed, values-free structural fixture with 1,910
synthetic rows in 203 components, largest component 180, group counts 758 and
1,152, and a full-rank four-column synthetic age/group design. The deterministic
pedigree has 27,691 nonzero lower-triangle relationships, 0.4673% fewer than the
retained aggregate of 27,821; its group assignment exactly matches the retained
13,322 cross-group related pairs and 77 mixed-group components. It reconstructs
neither participants nor participant rows, ages, phenotypes, relationship
coefficients, matrices, or family-specific group composition. The exact fixture
SHA-256 is
`2cceae6f2c5ea3f0c3de5946daeaa55cf2f35edd6d365fd1f45499f463d4c1c6`.

`checks/discrete_gxe_against_independent_full_fit.py` supplies a separate
participant-free full-ML sentinel at 120 people in 30 four-sibling families.
It parameterises the genetic covariance by its own Cholesky factor, residual
variances on log scales, optimises with derivative-free SciPy Powell, and
profiles fixed effects from a self-contained dense likelihood,

\[
-2\ell=n\log(2\pi)+\log|V|+
(y-X\widehat\beta)^\mathsf{T}V^{-1}(y-X\widehat\beta),\qquad
\widehat\beta=(X^\mathsf{T}V^{-1}X)^{-1}X^\mathsf{T}V^{-1}y.
\]

It therefore shares neither likelihood code, parameterisation nor optimiser
with Asterism. Both fits converged on 21 August 2026. The worst public-quantity
difference was **7.69e-04** for genetic correlation and the log-likelihood
difference was **4.92e-05**, inside prewritten tolerances 0.002 and 0.0002.

The portable calibration scores every fit as complete, refused, nonconverged,
or test-refused, and retains all attempts in the denominator. For \(r\)
rejections in \(m\) attempts at level \(\alpha\), a level cell fails only when

\[
\mathrm{Beta}^{-1}(0.05;r,m-r+1)>\alpha,
\]

with zero allowed unsuccessful attempts; alternative-scenario power is
descriptive rather than post-hoc gated. A one-replicate-per-scenario public-API
smoke on the full values-free target envelope completed all four attempts in
14.04 seconds, with zero refusals, nonconvergence, or test refusals. This is a
route check, not Monte Carlo calibration. The pinned 500-by-four campaign has
not run in this development environment: its 12-worker launch was refused by
the sandbox before worker creation, so **0 of 2,000 scientific attempts** ran.

In the historical participant-backed campaign, under the complete null,
rejection at nominal 0.05 was 0.042 for the principal
`gene_by_environment` test, 0.044 for equal genetic effects, 0.048 for equal
genetic variances, 0.036 for equal residual variances, and 0.040 for
`any_difference`. The fitted correlation sat on its upper bound in 52.8% of
null fits, which is why the even mixture is the right reference and a plain
chi-square would be conservative.

The scenario that matters is one sex measured with more error and identical
genetics. **Every genetic test held its level there** — 0.046 for the principal
test, 0.040 for equal genetic effects and 0.040 for equal genetic variances —
while `any_difference` and the residual test each rejected all 500 replicates.
That is the claim the two free residual variances exist to support, and it is
why `any_difference` must not be reported as a genetic finding.

Against genuine alternatives the tests separate as they should: where the genes
differ, the correlation test rejected 0.990 and the equal-variance test stayed
at level (0.062); where only the genetic scale differs, the equal-variance test
rejected 0.998 and the correlation test stayed at level (0.042).

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `GxeModel.fit(self, y: 'Any', grid: 'Any' = (-1.0, 0.0, 1.0), reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, and report the surface at the environments in ``grid``.

``grid`` is in the environment's own units, so it should be chosen from
the data — quantiles of the observed environment usually. The genetic
correlations come back as a square list of lists in the grid's order.

### `GxeModel.interval(self, y: 'Any', quantity: 'str' = 'heritability', first: 'float' = 0.0, second: 'float' = 0.0, reml: 'bool' = True) -> 'dict[str, Any]'`

A 95 per cent profile-likelihood interval for a reported quantity.

``"heritability"`` uses ``first`` as the environment; ``"correlation"``
uses both, and is the genetic correlation between them.

The quantity is held by solving one coordinate of the surface for it and
re-maximising over the rest, so the interval is a likelihood one and not
a Wald one — it is not symmetric about the estimate and does not have to
be. ``lower_limited`` or ``upper_limited`` says the endpoint ran to
the edge of what the quantity can be rather than to a likelihood
crossing, which is a limit of the model rather than a measurement.

### `GxeModel.test(self, y: 'Any', null: 'str' = 'correlation', reml: 'bool' = True) -> 'dict[str, Any]'`

Test one of the two genotype-by-environment nulls.

- ``"correlation"``: the genetic effects at any two environments are the
  same effects. The genetic variance may still change; what is ruled out
  is a change in *which* genes matter. This is the narrower claim and
  usually the interesting one — a heritability that rises with an
  environment can follow from a change of scale in the measurement, and a
  correlation below one cannot.
- ``"interaction"``: the genetic covariance does not involve the
  environment at all. Rejecting says something about genes and
  environment together, but not what.
- ``"variance"``: the genetic variance does not change with the
  environment. This is the recovered SOLAR model's own ``gamma_G = 0``
  null, and on the exponential surface it is one interior coordinate
  referred to chi-square on one degree of freedom. On the random-regression
  surface it is not a separate test — a quadratic genetic variance is
  constant only when both its shape coordinates are nought, which is
  the interaction null — so it returns that instead of a differently
  named copy.

The residual surface is free under both nulls, so a residual variance
that changes with the environment is not mistaken for a genetic one.

