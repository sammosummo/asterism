# Censored traits

Analysis id `one_trait_censored`. Runnable example in [`examples/`](../../examples/).

## Using it

The heritability returned is of the *complete* trait: what you would have
had if the instrument reached far enough. It is maximum likelihood, never
REML, so do not place it beside a REML heritability. The runnable version is
[`examples/censored.py`](../../examples/censored.py):

```sh
python examples/censored.py
```

```
people:             800
censored:           400 of 800
h2 (true 0.5):      0.450
converged:          True
95% interval:       [0.282, 0.615]
```

```python
fit = asterism.tobit_fit(relationship, value, censoring, limit, design)
fit["heritability"], fit["total_variance"], fit["censored_share"]

interval = asterism.tobit_interval(relationship, value, censoring, limit, design)
```

`censoring` is 0 where the value was measured, 1 where it lies at or above its
limit, and 2 where it lies at or below it. `value` is read only where the status
says measured, and `limit` only where it does not.

The status is given rather than inferred, and the limit belongs to the
observation rather than to the trait. Extended high-frequency audiometry is
the case this was built for, and there the recorded maximum differs between
frequencies and between sessions, so a censored value can carry the same number
as a genuinely measured one. Only the status tells them apart.

The heritability that comes back is that of the *complete* variable — the number
there would have been had the instrument reached far enough. It is comparable
with an ordinary heritability of an uncensored trait, and **not** comparable
with one fitted to values where the censored ones were replaced by their limit,
which is the usual practice and the thing this exists to replace. It is maximum
likelihood, never REML, because a censored observation has no response to
project onto the null space of the design.

## The model

### Model and likelihood

Let $Y_i^\ast$ be the complete measurement and let $C_i\in\{M,U,L\}$ denote
measured, upper/right-censored, or lower/left-censored status. Let $a_i$ be the
observation-specific limit. Asterism fits

$$
\mathbf Y^\ast\sim N\left(
\mathbf X\boldsymbol\beta,
\sigma^2\left\{\sum_c p_c\mathbf K_c+\Bigl(1-\sum_c p_c\Bigr)\mathbf I_n\right\}
\right).
$$

Here $\mathbf A$ is the relationship matrix, $\mathbf I_n$ the identity,
$\sigma^2>0$ the total variance of the complete measurement, and $p_c\in[0,1]$
the coefficient of component $\mathbf K_c$; $\mathbf X$ and $\boldsymbol\beta$
retain their shared fixed-effect meanings.

**Any number of components may be supplied.** The residual takes
$1-\sum_c p_c$ and is never passed, which is what puts the coefficients on a
common scale. The search works in stick-breaking coordinates, each in
$[0,1]$ and each taking a share of what the earlier ones left, so a set summing
past one is unreachable rather than refused: the feasible set is a box and the
bounded optimiser meets no cliff. A small residual is therefore reachable, which
matters, because additive, person-level and household terms together can leave
little behind.
They are proportions of $\sigma^2$ exactly where every $\mathbf K_c$ carries a
unit diagonal, which additive kinship, a person-level matrix and a household
kernel all do.

**One component is the case this model began as.** With a single kinship matrix
$\mathbf K_1=\mathbf A$, $p_1$ is $h^2$, the residual is $1-h^2$, and the fit is
what it always was, to the last decimal.

Two records per person want a person-level component beside the genetic one, or
the resemblance between a person's own two ears has nowhere to go but the
heritability.

When $C_i=M$, $Y_i^\ast$ is observed exactly. When $C_i=U$, only
$Y_i^\ast\geq a_i$ is known; when $C_i=L$, only $Y_i^\ast\leq a_i$ is known. The
public `censoring` array encodes $M$, $U$, and $L$ as 0, 1, and 2. Status is
supplied rather than inferred from a numeric value. The free
$\sigma^2$ is identified by measured observations, so $h^2$ concerns the
complete trait that would have been observed without the instrument limit.
The public likelihood accepts either censoring direction. The release's checks
are run right-censored, so that is the direction they cover.

[Tobin (1958)](../references.bib#tobin1958) is the foundational censored
latent-Gaussian regression source. Normal mixed models with censoring are
developed by [Hughes (1999)](../references.bib#hughes1999). For a family split
into measured indices $M$ and censored indices $C$, Asterism follows the
conditional factorisation

$$
L_f=f(\mathbf y_M;\boldsymbol\mu_M,\mathbf V_{MM})
\Pr\{\mathbf Y_C^\ast\in\mathcal R_C\mid
\mathbf Y_M^\ast=\mathbf y_M\},
$$

where $L_f$ is family $f$'s likelihood contribution, $\mathbf y_M$ is the
measured subvector, $\boldsymbol\mu_M$ its fitted mean, $\mathbf V_{MM}$ its
covariance, $f(\cdot)$ is the multivariate-normal density, $\mathcal R_C$ is
the product of the censoring half-lines, and $\Pr$ denotes probability. The
conditional probability uses the usual normal conditional mean and covariance.
This construction is explicit in
[Jacqmin-Gadda et al.
(2000)](../references.bib#jacqminGaddaEtAl2000). Asterism directly maximizes this
ML likelihood using the shared region-probability routine. The EM approach of
[Vaida and Liu (2009)](../references.bib#vaidaLiu2009) is relevant published
precedent but is not the implemented algorithm.

Testing $H_0:h^2=0$ and deciding whether an interval contains that boundary use
the 50:50 mixture. At least two measured values are required to identify scale.

**Both are offered at one component only.** The coverage simulation that scored
the boundary rule ran there; with several components more than one coefficient
can rest on nought at once, which is not the case it scored. So the test refuses
and the interval is returned with its boundary verdict absent rather than filled
in from a measurement of a different model. Scoring them at several components
is issue 38.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `heritability` | `tobit_fit(...)["heritability"]` |
| `interval` | `tobit_interval(...)`, using the common interval fields plus `censored_share` and `estimator` |
| `test` | `tobit_test(...)`: `statistic`, `p_value`, `rule`, `null_loglik`, `alternative_loglik`, `estimator` |

`total_variance`, `fixed_effects`, `censored_share`, and `largest_family` are
descriptive; `converged`, `scaled_gradient`, and `loglik` are diagnostics. A
limit-substitution Gaussian heritability is a different, biased estimand.

### Assumptions and limits

The complete trait is Gaussian with the stated covariance; the censoring
direction and each limit are correct; censoring is represented by the stated
regions; the relationship basis has the qualified normalisation; and the
measured portion identifies scale.

Coverage has been measured on sibling pairs, which do not exercise the
sequential approximation above two censored family members. Censoring beyond
three quarters has not been measured.

## Validation

The log-likelihood and all four reported quantities agree with R `censReg` to
`9.6e-9`. On 300 replicates of 300 sibling pairs per cell, all nine cells of a
heritability-by-censoring grid contain 0.95. Substituting the limit returns
0.31 where the truth is 0.5 and three quarters are censored; the censored
model recovers 0.5.

The designs behind those figures, which of them can be reproduced from this
repository, and what has still to run, are in the
[validation record](../validation.md#censored-traits).

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `tobit_fit(relationship: 'Any', value: 'Any', censoring: 'Any', limit: 'Any', design: 'Any', *, subject_order_sha256: 'str | None' = None) -> 'dict[str, Any]'`

Fit one trait whose measurement stops at a limit.

``censoring`` is 0 where the value was measured, 1 where it lies at or
above its limit, and 2 where it lies at or below it. The status is given
rather than inferred, because a censored value can carry the same number
as a measured one — extended high-frequency audiometry records several
limits within one frequency, and measured values coincide with them.

``value`` is read only where the status says measured, and ``limit`` only
where it does not.

The heritability that comes back is the heritability of the *complete*
variable — the number you would have had if the instrument reached far
enough. It is comparable with an ordinary heritability of an uncensored
trait, and not with one fitted to values where the censored ones were
replaced by the limit. It is maximum likelihood, never REML, so it must not
be placed beside a REML heritability as though the two were the same.

``subject_order_sha256`` optionally carries the lowercase SHA-256 from
``subject_order_commitment`` for the exact fitted row order. The fit record
echoes it without retaining identifiers.

### `tobit_interval(relationship: 'Any', value: 'Any', censoring: 'Any', limit: 'Any', design: 'Any') -> 'dict[str, Any]'`

A 95 per cent profile-likelihood interval for the censored heritability.

``lower_limited`` and ``upper_limited`` say whether an end sits on the
parameter's own bound rather than where the profile fell away. An end on a
bound means the data did not rule that end out, which is a different
statement from the interval stopping there.

``profile_failures`` counts fits along the profile that failed or did not
converge. Each one widened the interval rather than narrowing it, which is
the safe direction, but a large count means the interval rests on fewer
points than its width suggests.

Read ``censored_share`` beside the answer. On simulated data the model
recovers the truth to three quarters censored; on real extended
high-frequency thresholds it degrades past about half, where too little of
the upper tail is left to estimate a variance from.

### `tobit_test(relationship: 'Any', value: 'Any', censoring: 'Any', limit: 'Any', design: 'Any') -> 'dict[str, Any]'`

Test the censored heritability against nought.

The null holds the heritability at nought, which is its own bound, so the
reference is the Self-Liang 50:50 mixture of chi-square on nought and one
degrees of freedom rather than a plain chi-square. ``rule`` says which was
used, as data rather than as a promise.

