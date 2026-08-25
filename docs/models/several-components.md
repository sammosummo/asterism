# Several covariance components

Analysis id `several_covariance_components`. Runnable example in [`examples/`](../../examples/).

## Using it

Genes and a shared household, fitted together. The complete example is
[`examples/components.py`](../../examples/components.py):

```sh
python examples/components.py
```

```
people:               600
genetic   (true 0.4): 0.440
household (true 0.2): 0.210
converged:            True
p (household = 0):    0.003

A household that is exactly one family is nearly the pedigree itself, so
the two variances are hard to tell apart: unbiased, but wide. Households
holding people who share no genes are what separate them.
```

`mean_diagonal_proportions` gives each component's share of the *average
person's* variance. If half your sample belongs to no household, a true
household variance of 0.2 is reported as 0.1: the average person carries half
as much of it.

```python
model = asterism.ComponentModel([relationship, household], design)
fit = model.fit(y)
interval = model.interval(y, component=1)
test = model.test(y, component=1)
```

## The model

### Model, identification, and scale

For $q$ submitted structured covariance bases, Asterism fits

$$
\mathbf V(\boldsymbol\theta)
=\sum_{k=1}^{q}\theta_k\mathbf K_k+\theta_e\mathbf I_n,
\qquad \theta_1,\ldots,\theta_q,\theta_e\geq0,
$$

where $\mathbf K_k$ is the $k$th fixed basis, $\theta_k$ its covariance
coefficient, and $\theta_e$ the residual coefficient. The Gaussian ML/REML
likelihood above is evaluated by dense factorization because the bases need not
commute.

The coefficients are identified only if different allowable vectors
$\boldsymbol\theta$ produce different covariance matrices on the fitted roster.
Linear dependence among $\{\mathbf K_1,\ldots,\mathbf K_q,\mathbf I_n\}$ causes
exact nonidentification; near dependence causes weak identification. Labels
such as “additive” or “household” do not establish it.

Raw coefficient proportions depend on arbitrary matrix scale. Asterism instead
defines

$$
m_k=\frac{1}{n}\mathrm{tr}(\mathbf K_k),\qquad
c_k=\widehat\theta_km_k,\qquad
p_k=\frac{c_k}{\sum_{j=1}^{q}c_j+\widehat\theta_e},
$$

where $m_k$ is the mean diagonal, $c_k$ the mean-diagonal contribution, and
$p_k$ its proportion of mean marginal variance. The residual contribution is
$c_e=\widehat\theta_e$ and $p_e=c_e/(\sum_jc_j+c_e)$. Multiplying
$\mathbf K_k$ by a positive constant and dividing $\theta_k$ by it leaves
$c_k$ and $p_k$ unchanged. This invariant is an Asterism reporting definition,
not a conventional result asserted by [Harville
(1977)](../references.bib#harville1977).

It is defined only when every structured basis has a positive finite mean
diagonal. Split kinship-class bases have zero diagonal, so they are read
through coefficients and class deviations, never proportions. If $r$
class coefficients are pooled under equality, the omnibus likelihood-ratio
test uses $\chi^2_{r-1}$. Each class contrast is its coefficient's deviation
from the constrained average; deviations sum to zero and use an interior
$\chi^2_1$ reference. Testing a single component against zero uses the 50:50
mixture only when the other fitted components are interior.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `mean_diagonal_component_contributions` ($c_k$) | `ComponentModel.fit(y)["mean_diagonal_component_contributions"][k]` |
| `mean_diagonal_proportions` ($p_k$) | `fit["mean_diagonal_proportions"][k]` |
| `mean_diagonal_proportion_interval` | `ComponentModel.interval(y, k)` with `quantity == "mean_diagonal_proportion"` and the common interval fields |
| `coefficients_for_zero_diagonal_bases` ($\widehat\theta_k$) | `fit["variances"][k]` |
| `contrasts_for_zero_diagonal_bases` | `equality_test(y)` fields `components`, `statistic`, `p_value`, `rule`, `null_loglik`, `estimator`; and each `class`, `deviation`, `lower`, `upper`, `lower_limited`, `upper_limited`, `p_value`, `statistic`, `level` row from `contrasts(y)` |

Component lists follow the submitted structured-basis order and place the
residual last.

`raw_coefficient_proportions`, `raw_coefficient_total`, and an interval requested
with `quantity="raw_coefficient_proportion"` are diagnostic only. Fixed effects,
likelihood, projected/scaled gradient, optimizer stop message, and `polished`
describe the fit but are not component results.

### Assumptions and limits

All Gaussian assumptions above apply. Each basis must be finite, symmetric and
aligned to the response, and every evaluated linear combination must be a
positive-definite covariance. A split or contrast basis need not be positive
semidefinite by itself.

A scale-free Frobenius Gram-rank check refuses linearly dependent bases,
including a submitted identity confounded with the implicit residual. It
establishes coefficient identification at numerical rank. It is not a precision
guarantee for bases that merely lie close together, and it is not a general
identifiability diagnostic.

Mean-diagonal proportions are meaningful only where every basis has positive
unit mean diagonal. Off-diagonal class bases have zero diagonal and are read
through coefficients and contrasts instead.

## Validation

The three variances of a model with kinship, a second supplied matrix and
residual agree with SOLAR under ML to about `5e-7`. Refitting the four red
deer traits of Stopher et al. (2012) reproduces its Table 2, with every
component of six of the eight fits inside 0.006 of the published value. At
n = 400, coverage of the additive and household mean-diagonal proportions was
0.945 and 0.955.

The designs behind those figures, the rules they are scored against, and
what has still to run, are in the
[validation record](../validation.md#several-covariance-components).

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `ComponentModel.fit(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, and return variance coefficients and their proportions.

``raw_coefficient_proportions`` depend on matrix scale. They are useful
for inspecting the optimiser parameterisation but are not generic
variance shares. When all structured matrices have positive finite mean
diagonals, ``mean_diagonal_proportions`` reports the corresponding
scale-invariant marginal contributions.

Matrix, design, and response alignment is positional and is the
caller's responsibility.

### `ComponentModel.interval(self, y: 'Any', component: 'int', reml: 'bool' = True, quantity: 'str' = 'mean_diagonal_proportion') -> 'dict[str, Any]'`

A 95 per cent profile interval for one component proportion.

The default profiles the scale-invariant ``mean_diagonal_proportion``
reported by :meth:`fit`. The diagnostic
``raw_coefficient_proportion`` remains available explicitly. It changes
when a relationship matrix is rescaled, so it describes this
parameterisation rather than the data alone.

### `ComponentModel.test(self, y: 'Any', component: 'int', reml: 'bool' = True) -> 'dict[str, Any]'`

Test one component against having no variance at all.

The null sits on the edge of the parameter space, so the reference is
the Self–Liang half-and-half mixture and not a plain chi-squared.

It refuses where another component has itself gone to nought, because
the mixture assumes only one is on the boundary; a number there would be
a p-value for a question nobody asked.

### `ComponentModel.equality_test(self, y: 'Any', components: 'list[int] | None' = None, reml: 'bool' = True) -> 'dict[str, Any]'`

Test whether several components share one variance.

This is the question a split matrix asks, and `test` is not it.
Splitting a relationship matrix by class of parent–offspring tie gives
four classes, and every one carries variance if the trait is heritable
at all — so testing each against nought returns a p-value near nought
for anything heritable and answers nothing. Whether a mother resembles
her son by as much as a father resembles his daughter is the classes
being equal to *one another*.

The null pools the named components by adding their matrices, which is
exact: the pieces came from splitting a matrix, so their sum is that
matrix. Pool all of them and the null is the ordinary additive model.
Defaults to every structured component.

### `ComponentModel.contrasts(self, y: 'Any', classes: 'list[int] | None' = None, reml: 'bool' = True) -> 'list[dict[str, Any]]'`

Every class's deviation from the average class, with an interval.

This is what `equality_test` cannot give. The omnibus says the
classes are not all alike and stops; a contrast against the others
*pooled* couples them, because raising one class raises the pool the
rest are measured against — in calibration a single lifted class made a
second reject half the time.

The deviations are constrained to sum to nought, so the baseline is the
average class variance and each deviation is a departure from it. The
obvious alternative, a free baseline plus free differences, is rank
deficient: a constant moved from the baseline into every difference
changes nothing.

``classes`` must name only things that are classes of one split.
Anything not named keeps its own variance, which is what should happen
to a remainder component — pooling "everything that is not a
parent–child tie" into the baseline would compare siblings with parents.

With only two classes this says less than it appears to: the deviations
must be mirror images, so "the first is above average" and "the second
is below" are one statement.

