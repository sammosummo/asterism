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
censored:           406 of 800
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

Testing $H_0:h^2=0$ uses the asymptotic Self–Liang 50:50 mixture. The
one-component numerical path is unchanged from the released API, but its
fixed-instrument recovery, coverage and target checks now pass. At least two
measured values are required to identify scale.

**The interval's boundary verdict is filled at one component only.** At several
components it remains absent rather than borrowing a one-boundary conclusion
for a model whose finite-sample null law is design-dependent.

At several components `test` instead labels the reference
`asymptotic_mixture_50_50` and refuses if a nuisance variance rests on its
bound. This explicitly asymptotic test remains generally available. Its p-value
must not be reported for the exact failed 1,909-person, four-component target at
75% expected censoring without a design-specific simulated null. That is not a
universal censoring threshold. The constrained-null `bootstrap` route remains
experimental after its target failed the zero-refusal rule. Fit, interval and
the released one-component test are unaffected.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `heritability` | `tobit_fit(...)["heritability"]` |
| `interval` | `tobit_interval(...)`, using the common interval fields plus `censored_share` and `estimator` |
| `test` | `tobit_test(...)`: `statistic`, `p_value`, `rule`, `null_loglik`, `alternative_loglik`, `estimator` |

`total_variance`, `fixed_effects`, `censored_share`, and `largest_family` are
descriptive; `converged`, `scaled_gradient`, and `loglik` are diagnostics. A
limit-substitution Gaussian heritability is a different, biased estimand.

### Several components, from Python

`CensoredComponentModel` is shaped like `ComponentModel`: the components and the
design at construction, the censored data at each call.

```sh
python examples/censored_components.py
```

```
records:              1200 (600 people, two each)
censored:             501 of 1200
additive (true 0.4):  0.458
person   (true 0.3):  0.284
residual (true 0.3):  0.258
converged:            True
person 95% interval:  [0.158, 0.424]
upper on its bound:   False
profile failures:     0
additive test rule:   asymptotic_mixture_50_50
additive p-value:     5.199e-11
```

An upper end can rest on its bound because the profile could not be evaluated
there rather than because the likelihood never fell away: at a proportion of
one the person-level matrix is the whole covariance and is singular, so there
is no residual left to make it invertible. `upper_limited` alone does not tell
that apart from a genuine bound, which is why the failure count is printed
beside it.

```python
listener = asterism.grouping_matrix([f"listener-{row // 2}" for row in range(rows)])
model = asterism.CensoredComponentModel([expanded, listener], design)

fit = model.fit(value, censoring, limit)
fit["mean_diagonal_proportions"]  # the residual's share last

model.interval(value, censoring, limit, component=1)  # on the proportion
model.test(value, censoring, limit, component=1)  # labelled asymptotic
model.bootstrap(
    value,
    censoring,
    complete_limit,
    censoring_direction,
    component=1,
    replicates=999,
    seed=4117,
)
```

`interval` gives the mean-diagonal proportion by default, which is the
comparable quantity; pass `quantity="coefficient"` for the raw one. At several
components `test` reports `asymptotic_mixture_50_50` and refuses an observed
nuisance boundary. `bootstrap` needs the instrument limit and direction for
every row, including measured rows, and is an experimental alternative. The
interval leaves its boundary verdict absent at several components.

### What separates one component from another

A component is identified by resemblance the other components do not already
explain, and which pairs of rows carry that information differs by component.
It is worth knowing which, because it decides what a given sample can support.

**Additive against person-level: a matter of precision.** Two
records of one person resemble each other through both, so a person's own rows
say nothing about the split; only the correlation between relatives carries the
additive term alone. The information therefore scales with **related pairs**,
not with records, and giving everybody a second record sharpens their sum while
doing nothing for the split.

An early exploratory simulation used sibling pairs at a true 0.40 and 0.30,
over **eight replicates per size**. It selected censoring limits from realised
outcomes, so the figures are investigation history rather than fixed-instrument
qualification evidence:

| people | additive (sd) | person-level (sd) |
| ---: | ---: | ---: |
| 80 | 0.398 (0.21) | 0.279 (0.20) |
| 400 | 0.404 (0.11) | 0.296 (0.11) |
| 2400 | 0.370 (0.036) | 0.330 (0.036) |

The corrected fixed-instrument recovery campaign now passes its predeclared
consistency rule; it supports the measured recovery and precision trend, not an
unbiasedness claim.

**Household against additive: this one can fail outright, and not measured.**
Where a home holds exactly one relationship class, the household matrix is a
linear combination of the others and nothing can separate them. For homes that
are exactly full-sibling pairs the identity is exact:

$$
\mathbf H = 2\mathbf A - \mathbf I.
$$

`ComponentModel` refuses such a set as `COMPONENTS_COVARIANCE_BASES_RANK_DEFICIENT`,
which is the right answer: it is not a hard estimate but no estimate.

What breaks the tie is homes holding people of **different** relatedness --
spouses, an unrelated carer, a lodger -- so the separating information is the
number of such pairs rather than the size of the sample. With unrelated parents
present it fits: household 0.209 and additive 0.385 at 240 people, though with
standard errors of 0.114 and 0.204 those are wide.

**So the two bullets are not the same kind of statement**, and the difference
matters more than either number. The first is measured and is about precision.
The second is reasoned and is about whether the question has an answer at all in
a given pedigree. Fit the model and read the intervals; where the components are
collinear there will be nothing to read.

### Reporting several components

Every component can be reported, not only adjusted for -- a shared-environment
term that cannot be quoted is a term the model allowed for without saying what
it was.

| What | Where |
| --- | --- |
| Each component's coefficient | `coefficients`, in the order the components were given, the residual not among them |
| The comparable quantity | `mean_diagonal_proportions`, the residual's last, summing to one |
| An interval for a component's coefficient | `coefficient_interval(index)` |
| An interval for its proportion, which is the one to report | `mean_diagonal_interval(index)` |
| The asymptotic component test | `test(..., component=index)`; rule `asymptotic_mixture_50_50` |
| An experimental finite-sample candidate | `bootstrap(..., component=index, replicates=B, seed=...)` |

**Compare components by the mean-diagonal proportions, not by the
coefficients.** A coefficient is comparable across matrices only where their
diagonals agree; the proportion is the coefficient's variance scaled by its
matrix's mean diagonal and normalised, which is what `ComponentModel` reports
and for the same reason. Where every component carries a unit diagonal -- true
of additive kinship, a person-level matrix and a household kernel -- the two
agree exactly.

The difference is easiest to see in what a rescaling does. Multiplying a
component's matrix by four describes the same model, the coefficient simply
absorbing the constant: the proportion's interval does not move and the
coefficient's does. An interval that shifts when nothing about the model has is
measuring the parameterisation as much as the data.

There is no separate test on a proportion, because a proportion is nought
exactly when its coefficient is.

**An interval on a component may rest on its bound for two different
reasons.** A component whose matrix is singular at its full share -- a
person-level matrix is ones within a person, so with two records each it has
half the rank of its size -- has no evaluable likelihood at a coefficient of
one, because there is no residual left to make the covariance invertible. The
interval widens to the bound, which is the safe direction, and
`profile_failures` is what separates that from a bound the likelihood genuinely
never left. Read the two together; `upper_limited` alone does not tell them
apart.

**Two things the record says about itself.** An interval's boundary verdict is
absent at several components because it has not been qualified there. Analytic
`test` declares its asymptotic rule and refuses when another coefficient or the
residual rests on nought in the observed fit. `bootstrap` makes the same
nuisance-boundary refusal and remains experimental.

### Assumptions and limits

The complete trait is Gaussian with the stated covariance; the censoring
direction and each limit are correct; censoring is represented by the stated
regions; the relationship basis has the qualified normalisation; and the
measured portion identifies scale.

The corrected one-component fixed-instrument recovery, coverage and target
campaigns pass. For several components, recovery and interval coverage pass,
and the analytic test remains available with its explicit asymptotic label. Its
p-value is a known failure on the exact 1,909-person, four-component target at
75% expected censoring and requires a design-specific simulated null there. The
bootstrap remains experimental after its target failed its declared rule.

## Validation

The log-likelihood and all four reported quantities agree with R `censReg` to
`9.6e-9` on their retained conditional comparison. Earlier recovery and
coverage campaigns selected each censoring limit from the response they then
analysed. Those runs are retained as diagnostic history. The corrected
one-component campaigns now pass. For several components, the exact target
limitation above is kept separate from the generally available asymptotic test;
it is not a censoring threshold or a limitation on fitting and intervals.

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

Read ``censored_share`` beside the answer. The released numerical path is
unchanged, and the corrected fixed-instrument coverage campaign passes at
the intended 52% and 75% expected censoring shares.

### `tobit_test(relationship: 'Any', value: 'Any', censoring: 'Any', limit: 'Any', design: 'Any') -> 'dict[str, Any]'`

Test the censored heritability against nought.

The null holds the heritability at nought, which is its own bound, so the
analytic reference is the Self-Liang 50:50 mixture of chi-square on nought
and one degrees of freedom rather than a plain chi-square. ``rule`` says
which was used, as data rather than as a promise. Its corrected
fixed-instrument target check passes at the intended 52% and 75% expected
censoring shares.

