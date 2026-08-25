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
across surfaces. `test(null="variance")` is public but is not one of the
analysis's supported quantities. The powered-exponential family remains public but unsupported.

### Assumptions and limits

The chosen surface must be positive semidefinite and correctly describe how
genetic covariance and residual variance change across the observed
environment. Reading $h^2(z)$ as a variance proportion also assumes the
qualified relationship normalisation, normally unit diagonal.
Outcome-dependent family selection changes the estimand.

Surface misspecification can imitate genetic reordering: an exponential surface
fitted to a linear rank-one truth rejected 0.107 at a nominal 0.05
no-reordering test. Coverage does not imply precision here, either. Between 80%
and 92% of genetic-correlation intervals reached a bound in the settings
studied, and point estimates were pulled toward correlation one near the
boundary.

## Validation

Both surfaces agree with an independent target-sized REML reference to
`5.5e-07` or better on reported quantities and `2.7e-11` on the
log-likelihood. Under a flat null the correlation test rejected 0.048 at
nominal 0.05 across 2,000 simulations per scenario. Between 80% and 92% of
genetic-correlation intervals reach a bound in the settings studied.

The designs behind those figures, which of them can be reproduced from this
repository, and what has still to run, are in the
[validation record](../validation.md#gene-by-environment-measured).

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

