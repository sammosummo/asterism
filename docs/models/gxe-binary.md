# Gene by environment, binary

Analysis id `discrete_gene_by_environment`. Runnable example in [`examples/`](../../examples/).

## Using it

The runnable version is
[`examples/discrete_gxe.py`](../../examples/discrete_gxe.py):

```sh
python examples/discrete_gxe.py
```

```
people:                 1000
in environment 1:       500 of 1000
rho_g     (true 0.6):   0.614
converged:              True
95% interval:           [0.297, 0.982]
p (no interaction):     0.083
```

```python
discrete = asterism.DiscreteGxeModel(relationship, sex, design)
fit = discrete.fit(y)
discrete.test(y, "gene_by_environment")
discrete.test(y, "correlation")
discrete.test(y, "genetic")
discrete.test(y, "residual")
```

`DiscreteGxeModel` takes an environment measured as a binary label — any two
distinct finite values, with the smaller naming the first group — and smooths
nothing: it carries a genetic and a residual variance per group and one
correlation between them. Sex is the canonical use, and `fit` returns the two
`levels` so a reader can tell which group is which.

Group-specific heritabilities remain descriptive in 0.1; the genetic
correlation, its interval, and the calibrated genetic tests are the supported
targets.

The principal `gene_by_environment` test leaves residual variances free in the
two environments; `any_difference` additionally equates them and therefore is
not specifically a genetic test.

## The model

### Model and inference

Let $e_i\in\{0,1\}$ be person $i$'s prespecified group, $A_{ij}$ the supplied
relationship coefficient for people $i$ and $j$, $s_{g,r}>0$ and $s_{e,r}>0$
the genetic and residual standard deviations in group $r$, and
$\rho_g\in[-1,1]$ the cross-group genetic correlation. Asterism fits

$$
V_{ij}=A_{ij}s_{g,e_i}s_{g,e_j}
\begin{cases}
1,&e_i=e_j,\\
\rho_g,&e_i\ne e_j,
\end{cases}
+\mathbb 1(i=j)s_{e,e_i}^2.
$$

Treating expression in two environments as two genetically correlated traits
comes from [Falconer (1952)](../references.bib#falconer1952). The exact
parameterisation and test catalogue below are Asterism constructions. Separate
residual variances prevent unequal measurement noise from being mislabelled as
a genetic difference. ML and REML are available; REML is the default.

The reportable genetic tests are:

| `null` | Null hypothesis | `rule` |
| --- | --- | --- |
| `gene_by_environment` | $s_{g,0}=s_{g,1}$ and $\rho_g=1$, with residual scales free | `mixture_chi2_1_chi2_2` = $\tfrac12\chi^2_1+\tfrac12\chi^2_2$ |
| `correlation` | $\rho_g=1$ | `mixture_50_50` = $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ |
| `genetic` | $s_{g,0}=s_{g,1}$ | `chi2_1` |

`any_difference` also constrains residual variance and is not a genetic test;
`residual` is a measurement-variance test. The 95% correlation interval uses
the regular interior $\chi^2_1$ profile threshold, not the boundary reference
for testing $\rho_g=1$.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `genetic_correlation` | `DiscreteGxeModel.fit(y)["genetic_correlation"]` |
| `correlation_interval` | `DiscreteGxeModel.correlation_interval(y)`, using the common interval fields |
| `gene_by_environment_test` | `DiscreteGxeModel.test(y, null="gene_by_environment")` |
| `correlation_test` | `DiscreteGxeModel.test(y, null="correlation")` |
| `genetic_variance_test` | `DiscreteGxeModel.test(y, null="genetic")` |

Every test record supplies `null`, `statistic`, `p_value`, `rule`,
`null_loglik`, `alternative_loglik`, and `estimator`. Group-specific
`genetic_variance`, `residual_variance`, `heritability`, `counts`, and `levels`
are descriptive in this 0.1 analysis.

### Assumptions, measured limitations, and unmet gates

The two groups, their order, and fixed-effect design are prespecified; each
group contains enough related observations to identify a genetic covariance;
and the Gaussian covariance model is correct. Historical calibration on the
unbalanced GOBS pedigree found nominal-0.05 rejection rates 0.042, 0.044, and
0.048 for the overall, correlation, and genetic-variance tests. When one group
had twice the residual noise but identical genetics, the three genetic tests
remained near level while `any_difference` rejected every replicate. At true
$\rho_g=0.9$, 0.995 interval coverage accompanied 94% bound-reaching: again,
coverage was not precision. The new independent full fit agrees within
$7.7\times10^{-4}$ on reportable quantities and $4.93\times10^{-5}$ in log
likelihood, and reduced public-API target and interval smokes completed without
fit or profile failures. The exact 2,000-attempt target calibration and
1,200-attempt interval campaigns did not start in the current sandbox because
worker creation was denied before any scientific attempt. Their commands and
acceptance rules are fixed; genuine host execution against the release wheel
remains required.

## What stands behind it

The prior campaign used 400 replicates at 600 people in sibships of four. Its
95 per cent profile intervals for the genetic correlation were:

| true correlation | coverage | reached a bound | median width |
| --- | --- | --- | --- |
| 0.3 | 0.958 | 0.138 | 0.682 |
| 0.6 | 0.955 | 0.490 | 0.623 |
| 0.9 | 0.995 | 0.940 | 0.389 |

Coverage is nominal away from the bound and conservative against it. At a true
correlation of 0.9, 94 per cent of intervals reach one: they cover, but they do
not pin the value down, and an interval that reaches its bound is reported as
having done so for that reason.

`checks/discrete_gxe_correlation_interval.py` now makes that documented design
participant-free and executable solely through
`DiscreteGxeModel.fit` and `.correlation_interval`. Every replicate is scored
as complete, refused, nonconverged, interval-refused, or profile-failed. With
\(c\) covering intervals among \(m\) complete attempts, the prewritten
undercoverage rule fails when the one-sided exact upper limit

\[
\mathrm{Beta}^{-1}(0.95;c+1,m-c)<0.94;
\]

any incomplete denominator fails separately, while a one-sided lower limit
above 0.96 labels conservatism without calling it undercoverage. The n = 600
one-replicate-per-truth smoke completed all three attempts in 3.99 seconds,
with zero refusals, nonconvergence, interval refusals, or profile failures; all
three generating correlations were covered. This does not estimate coverage.
The pinned 400-by-three development campaign likewise reached the sandbox
process boundary before worker creation, so **0 of 1,200 scientific attempts**
ran. Both full campaign commands remain fixed in `release.toml`; the global and
analysis-level measured/configured flags remain false until genuine full and
fixed-wheel evidence exists.

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `DiscreteGxeModel.fit(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, with everything free.

``counts`` comes back with the answer because a correlation estimated
across a group of thirty is not the same claim as one across a thousand,
and the fit itself cannot tell you which you have.

### `DiscreteGxeModel.correlation_interval(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

A 95 per cent profile interval for the genetic correlation.

The correlation is a parameter here rather than a function of one, so
the interval comes from pinning it and refitting everything else, with
endpoints where twice the drop in log likelihood reaches 3.8415.

**The reference is the ordinary chi-square on one degree of freedom,
not the mixture** ``test(y, "correlation")`` **uses.** That test asks
about a correlation of exactly one, which is the edge of the parameter
space; an interval is a statement about interior values and takes the
interior reference. Borrowing the test's mixture would give a narrower
interval than the coverage it claims.

``lower_limited`` and ``upper_limited`` say whether an endpoint sat at
the edge of what a correlation may be rather than where the likelihood
fell away. An interval reaching a bound is coverage without precision,
and that is worth knowing before it is quoted.

### `DiscreteGxeModel.test(self, y: 'Any', null: 'str' = 'gene_by_environment', reml: 'bool' = True) -> 'dict[str, Any]'`

Test one of the five nulls.

- ``"gene_by_environment"``: no genetic difference of any kind — the
  same variance and the same genes in both environments — with the two
  residual variances left free. Two constraints, one of which sits on a
  bound, so the reference is an even mixture of chi-square on one and
  on two degrees of freedom. **Read this one first.**
- ``"any_difference"``: nothing differs between the environments at
  all, residual included. Three constraints on an even mixture of
  chi-square on two and on three. It is **not** a genetic test: a
  noisier environment rejects it.
- ``"correlation"``: the same genes act in both environments. This is
  the gene-by-environment question proper. The null puts the
  correlation at the edge of what it may be, so the reference is the
  even mixture of a point mass at nought with chi-square on one degree
  of freedom. A plain chi-square would roughly double the p-value.
- ``"genetic"``: the same genetic variance in both environments.
  Interior, so chi-square on one degree of freedom.
- ``"residual"``: the same residual variance in both environments.
  Report it beside the others as a measurement fact, not as a genetic
  finding.

``rule`` names the reference distribution the p-value is a tail of, so a
reader need not take it on trust.

