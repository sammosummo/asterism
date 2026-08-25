# Two traits

Analysis id `bivariate_genetic_correlation`. Runnable example in [`examples/`](../../examples/).

## Using it

Two continuous traits and the genetic correlation between them. `observed`
is a boolean pair per person saying which traits they have; the values and
the design are stacked, trait within person. The runnable version is
[`examples/bivariate.py`](../../examples/bivariate.py):

```sh
python examples/bivariate.py
```

```
people:                600
h2 first  (true 0.5):  0.431
h2 second (true 0.4):  0.357
rho_g     (true 0.6):  0.512
converged:             True
95% interval:          [0.224, 0.768]
p (rho_g = 0):         7.52e-04
```

```python
model = asterism.BivariateModel(relationship, observed, design)
fit = model.fit(values)
fit["h2_first"], fit["h2_second"]
fit["rho_g"], fit["rho_e"], fit["rho_p"]
model.interval(values, "rho_g")
model.test(values, "rho_g", null=0.0)
```

`observed` is one Boolean pair per person, permitting different missingness for
the two traits. `design` and the values contain only observed person-trait rows,
in person order with trait within person.

The 0.1 reportable target is `rho_g` with its interval and test; the other
fitted quantities describe the joint fit but are not additional 0.1 claims.

`test` takes a `null`, and both interesting values are supported. Against
nought it asks whether the traits share any genes at all, and nought is an
interior point so a plain chi-square on one degree of freedom applies. Against
plus or minus one it asks whether they are governed by the *same* genes, and
that value sits on the boundary of a correlation's range, so the reference is
the Self–Liang mixture instead.

```python
model.test(values, "rho_g", null=0.0)   # any shared genes?
model.test(values, "rho_g", null=1.0)   # the same genes?
```

## The model

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
(1997)](../references.bib#almasyDyerBlangero1997) and the wider SOLAR pedigree
likelihood by [Almasy and Blangero
(1998)](../references.bib#almasyBlangero1998). Those linkage papers do not define
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

## What stands behind it

- Against R `regress` under REML on an unbalanced synthetic problem,
  heritabilities agreed within `2.6e-9`, correlations within `1.9e-8`, and the
  independently recomputed projected KKT measure was `5.5e-9`.
- Against native SOLAR under ML with deliberately unbalanced observations, all
  four directly reported quantities agreed within `2e-7`.
- Across all 78 pairs of thirteen real GOBS traits, the largest discrepancy
  from SOLAR was `5.9e-6` and the median `1.5e-7`. SOLAR's stopping precision
  makes roughly `1e-6` a realistic real-data comparison tolerance. Recorded
  rather than reproducible: the driver that produced it read real phenotypes
  and never lived in this repository, so the record is
  `evidence/bivariate-against-solar-real-2026-08-12.json` and nothing here
  regenerates it.
- In 300 REML simulations at `n=180`, 95% coverage for the two
  heritabilities, genetic correlation, residual correlation, and phenotypic
  correlation was 0.957, 0.967, 0.960, 0.953, and 0.977. Interior correlation
  tests were close to nominal; the correlation-one boundary test was
  conservative (0.033 rejection at nominal 0.05).
- Independently implemented profile endpoints agreed with the compiled
  endpoints to about `1e-5` on an unbalanced `n=150` problem.
- At heritability 0.83, overall coverage remained near nominal, but the
  residual-correlation interval failed in 23 of 300 replicates and reached a
  bound in 94% of those computed. The phenotypic correlation computed in all
  replicates and did not reach a bound. Recorded rather than reproducible:
  `checks/bivariate_calibration.py` fixes its heritabilities at easier values,
  so the record is `evidence/coverage-at-the-real-design-point-2026-08-12.json`
  and rerunning the check does not return to this design point.

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `BivariateModel.fit(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, and return both heritabilities and all three correlations.

The phenotypic correlation is derived from the others rather than
estimated, which is why it has no variance of its own.

### `BivariateModel.interval(self, y: 'Any', quantity: 'str', reml: 'bool' = True) -> 'dict[str, Any]'`

A 95 per cent profile interval for one reported quantity.

### `BivariateModel.test(self, y: 'Any', quantity: 'str', null: 'float' = 0.0, reml: 'bool' = True) -> 'dict[str, Any]'`

Test one correlation against a fixed value.

Against nought the value is interior and a plain chi-squared applies;
against plus or minus one it sits on a bound and the Self–Liang mixture
does. Heritabilities are not testable this way.

