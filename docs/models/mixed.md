# Mixed pairs

Analysis id `mixed_binary_censored_genetic_correlation`. Runnable example in [`examples/`](../../examples/).

## Using it

One trait censored, one not. This is the pairing the extended high-frequency
work needs. The runnable version is
[`examples/mixed.py`](../../examples/mixed.py):

```sh
python examples/mixed.py
```

```
people:                800
censored:              320 of 800
rho_g     (true 0.6):  0.538
converged:             True
95% interval:          [0.342, 0.722]
```

```python
first = {"kind": "censored", "value": v1, "censoring": c1, "limit": l1}
second = {"kind": "continuous", "value": v2, "censoring": c2, "limit": l2}
fit = asterism.mixed_bivariate_fit(relationship, first, second, design)
fit["genetic_correlation"], fit["residual_correlation"]
```

Either trait may be `continuous`, `binary` or `censored`. The covariance is
`BivariateModel`'s; what varies is only how each observation is seen — a density
at a point, or the probability of a region, with a continuous value the
degenerate region.

A binary trait's variance is fixed at one and comes back as one, because
only the sign of a liability is ever seen. Its heritability is a liability
heritability while a continuous or censored trait's is not, and the two must not
be read as the same quantity. The genetic correlation is unaffected by that
difference, which is what makes a mixed pair worth fitting at all: a correlation
is scale free even where one of its two scales is arbitrary.

`mixed_bivariate_test` takes a `null`, and both interesting values work the
same way they do for a continuous pair. Against nought it asks whether the two
traits share any genes; against plus or minus one, whether the same genes
govern both. Nought is interior so a plain chi-square applies, while one sits
on the edge of a correlation's range and takes the Self-Liang mixture, reported
as `mixture_50_50`.

```python
asterism.mixed_bivariate_test(relationship, first, second, design, null=0.0)
asterism.mixed_bivariate_test(relationship, first, second, design, null=1.0)
```

Simulated at a true correlation of one, the boundary test rejects at 0.010,
0.040 and 0.085 against nominal levels of 0.01, 0.05 and 0.10 over two hundred
replicates, none refused — correctly sized or mildly conservative.

0.1 supports the pairings that include a continuous trait: continuous with
continuous, binary with continuous, and censored with continuous, along with
censored with censored. The binary-with-censored pair is deferred — it stays
public and its checks are kept, but 0.1 makes no scientific claim about it.

## The model

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

[Catalano and Ryan (1992)](../references.bib#catalanoRyan1992) provide a clustered
latent-variable framework for discrete and continuous outcomes. Combined with
the conditional censored-normal likelihood of [Jacqmin-Gadda et al.
(2000)](../references.bib#jacqminGaddaEtAl2000), it supports the ingredients but
not the exact Asterism model. The pedigree genetic covariance, one binary
liability, observation-specific censoring, direct block likelihood, and
sequential high-dimensional approximation are an Asterism synthesis. No single
primary publication found in the source review establishes that exact
combination.

The default test is $H_0:\rho_g=0$. Zero is interior to $[-1,1]$, so its
reference is $\chi^2_1$, not a boundary mixture, and the interval uses the
regular interior profile threshold. A null of $\pm 1$ sits on the boundary and
uses the even mixture instead.

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

### Assumptions and limits

Both latent traits follow the joint Gaussian covariance above; the binary scale
and relationship-matrix normalisation are fixed correctly; censoring records
and limits are correct; and family blocks are independent.

Coverage and level have been measured on sibling pairs, which do not exercise
the sequential integration used for larger families.

## Validation

The worst difference from SOLAR on the binary-with-continuous pair is 0.014.
Genetic-correlation interval coverage on 300 replicates of 400 sibling pairs
per cell ran from 0.930 to 0.957 across three pairings and three true
correlations, and the tests against nought and against one both held their
level.

The designs behind those figures, the rules they are scored against, and
what has still to run, are in the
[validation record](../validation.md#mixed-pairs).

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `mixed_bivariate_fit(relationship: 'Any', first: 'dict[str, Any]', second: 'dict[str, Any]', design: 'Any', *, subject_order_sha256: 'str | None' = None) -> 'dict[str, Any]'`

Fit two traits whose measurements need not be of the same kind.

Each trait is a dictionary with ``kind`` (``"continuous"``, ``"binary"`` or
``"censored"``), ``value``, ``censoring`` and ``limit``. For a binary trait
a censoring code of 1 is a case, and the limit is nought because the
threshold is carried by the intercept.

A binary trait's variance is fixed at one and comes back as one, because
only the sign of a liability is ever seen. Its heritability is therefore a
liability heritability, while a continuous or censored trait's is a
heritability of the observed scale. They are different quantities. The genetic correlation is
unaffected, which is what makes a mixed pair worth fitting.

``subject_order_sha256`` optionally carries the lowercase SHA-256 from
``subject_order_commitment`` for the exact fitted row order. The fit record
echoes it without retaining identifiers.

### `mixed_bivariate_interval(relationship: 'Any', first: 'dict[str, Any]', second: 'dict[str, Any]', design: 'Any', coordinate: 'str' = 'genetic_correlation') -> 'dict[str, Any]'`

A 95 per cent profile-likelihood interval for one bivariate coordinate.

``coordinate`` is ``"heritability_one"``, ``"heritability_two"``,
``"genetic_correlation"`` or ``"residual_correlation"``.

The variances have no interval on purpose. A binary trait's is fixed at
one because a liability has no scale of its own, so an interval on it would
describe that assumption rather than the data.

``lower_limited`` and ``upper_limited`` say whether an end sits on the
coordinate's own bound — nought or one for a heritability, minus one or one
for a correlation — rather than where the profile fell away. An end on a
bound means the data did not rule that end out, which is a different
statement from the interval stopping there.

### `mixed_bivariate_test(relationship: 'Any', first: 'dict[str, Any]', second: 'dict[str, Any]', design: 'Any', coordinate: 'str' = 'genetic_correlation', null: 'float' = 0.0) -> 'dict[str, Any]'`

Test one correlation of the mixed bivariate model against a fixed value.

``coordinate`` is ``genetic_correlation`` or ``residual_correlation``, the
same names the interval takes.

``null`` is the value tested against, and the two worth asking are the ends
of the question. Against nought: do these traits share any genes at all? An
estimate with an interval does not answer that. Against one: are they the
same genes?

The reference distribution follows from which. Nought is interior to a
correlation's range, so a plain chi-square on one degree of freedom applies
and no boundary mixture is needed. Plus or minus one is the edge of that
range, so the null rests on a bound and takes the Self-Liang even mixture,
which is reported as ``mixture_50_50`` rather than ``chi2_1``.

