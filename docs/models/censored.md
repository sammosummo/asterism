# Censored traits

Analysis id `one_trait_tobit_audiogram`. Runnable example in [`examples/`](../../examples/).

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

**The status is given rather than inferred, and the limit belongs to the
observation rather than to the trait.** Extended high-frequency audiometry is
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

[Tobin (1958)](../references.bib#tobin1958) is the foundational censored
latent-Gaussian regression source. Normal mixed models with censoring are
developed by [Hughes (1999)](../references.bib#hughes1999). For a family split
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
(2000)](../references.bib#jacqminGaddaEtAl2000). Asterism directly maximizes this
ML likelihood using the shared region-probability routine. The EM approach of
[Vaida and Liu (2009)](../references.bib#vaidaLiu2009) is relevant published
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

## What stands behind it

Against R's `censReg` with no relatedness, the log likelihood and all four
reported quantities agreed to `9.6e-9` at 4,000 people with 1,131 censored.
With relatedness, the maximum-likelihood estimates sit inside MCMCglmm's
posterior. Substituting the limit -- the usual practice -- returns 0.46 at a
quarter censored and 0.31 at three quarters where the truth is 0.5; the
censored model recovers 0.5 at every rate to three quarters.

**Interval coverage, on 300 replicates of 300 sibling pairs per cell**, scored
unconditionally with no cell dropped and no refusals:

| true h² | 0% censored | 25% | 50% |
| --- | --- | --- | --- |
| 0.0 | 0.937 | 0.950 | 0.960 |
| 0.3 | 0.943 | 0.943 | 0.957 |
| 0.5 | 0.953 | 0.950 | 0.933 |

All nine cells contain the nominal 0.95 in their Clopper-Pearson intervals,
**two-sided, including the cells at nought**.

Correcting the receipt of 18 August 2026, which reported 0.980, 0.977 and 0.983
in those three cells and passed them under a one-sided rule: the interval was
missing the Self-Liang mixture that ADR 0004 requires, so a lower end of nought
was read as containment whatever the likelihood there said. ADR 0004 had
already measured what that costs -- 0.977 against 0.953 -- and the censored
model reproduced it. The mixture is now on the record as
`contains_lower_bound` and `contains_upper_bound`, the check reads it, and the
boundary allowance that hid the fault is gone. `evidence/tobit-coverage-2026-08-18.json`
predates the mixture and describes a recipe the code no longer implements.

`checks/tobit_target_design.py` adds the missing pedigree-scale route without
retaining participant records. It uses the reviewed values-free structural
fixture to generate 1,909 synthetic rows in 202 relationship components, with a
largest component of 180 and the same six-column participant-free fixed design.
At the nearest attainable fractions to 52% and 75% right censoring (993/1,909 =
0.52017 and 1,432/1,909 = 0.75013), it runs the public `asterism.tobit_fit`,
`asterism.tobit_interval`, and `asterism.tobit_test` functions under a boundary
truth of $h^2=0$ and an interior truth of $h^2=0.5$. The fixed release campaign
requests 200 replicates in each of those four cells. Every requested replicate
stays in its cell's denominator; unsuccessful outcomes are assigned exactly one
of `invalid_censoring_count`, `refused`, `nonconverged`, `test_failed`,
`interval_failed`, or `profile_failed`, and none is allowed. The null rule fails
when the one-sided 95% Clopper--Pearson lower limit for the rejection rate is
above 0.05. The interval rule fails when the corresponding upper limit for
coverage is below 0.95.

A bounded development smoke on 21 August 2026 ran one replicate per scenario at
each censoring level: four attempted in 62.87 seconds, three complete, and one
`nonconverged`. At 52% censoring, the null estimate was 0.0142 with $p=0.3403$
and interval $[0,0.0949]$; the interior estimate was 0.4973 with interval
$[0.3943,0.5989]$. At 75%, the interior estimate was 0.4762 with interval
$[0.3390,0.6163]$, while the null fit returned a finite boundary candidate
($h^2=0$, total variance 3.9412) with `converged=false`; the check therefore
recorded `nonconverged` and did not compute an interval or test for that attempt.
The command exited 1. This is route and failure-accounting evidence only, not a
coverage or type-I-error calibration. The 800-attempt release campaign has not
run; `pass_rules_configured`
remains false.

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `tobit_fit(relationship: 'Any', value: 'Any', censoring: 'Any', limit: 'Any', design: 'Any', *, subject_order_sha256: 'str | None' = None) -> 'dict[str, Any]'`

Fit one trait whose measurement stops at a limit.

``censoring`` is 0 where the value was measured, 1 where it lies at or
above its limit, and 2 where it lies at or below it. **The status is given
rather than inferred**, because a censored value can carry the same number
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
bound means **the data did not rule that end out**, which is a different
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

