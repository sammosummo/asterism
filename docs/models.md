# Overview

Asterism currently supports eight models for different analytic use cases, with more planned. What each model is for, what it returns, and what it does not claim. The
[README](../README.md) is the overview; this is the detail behind it.

Every model here is one of the eight Asterism supports. The ones it does not
support are in [outside-support.md](outside-support.md). Signatures for
everything public are in the generated [API reference](api-reference.md), the
equations are in [statistical-methods.md](statistical-methods.md), and what
stands behind the numbers is in
[numerical-validation.md](numerical-validation.md).

Each section links to a complete runnable program in
[`examples/`](../examples/). Those live in the repository, not in the installed
package, so clone it if you want to run them.

## Building the inputs

The models take covariance matrices. Where those come from is your business:
every model here accepts any finite, symmetric, positive-semidefinite matrix of
the right size, whether it came from a pedigree, from genotypes, or from
somewhere else entirely. Asterism ships two builders because pedigrees are
tedious to turn into matrices, not because it insists on them.

**`relationship_matrix`** builds the additive relationship matrix from a
pedigree. Identifiers are strings and a founder's parents are `None`. It
returns the matrix and the row order it is in, which will not be the order you
supplied.

```python
relationship, order = asterism.relationship_matrix(ids, father, mother)
```

It returns twice the kinship coefficient, so full siblings are 0.5 and a parent
and child are 0.5.

**`kinship_classes`** splits a pedigree into separate zero-diagonal bases — one
per relationship class — so their coefficients can be estimated and contrasted
rather than assumed. Use it with `ComponentModel`.

**`align`** matters whether or not you used a builder, because Asterism's
numerical interface is positional — **and that is the one place a mistake makes
no noise.** A relationship matrix whose rows are in a different
order from the response does not fail or warn. On 300 people simulated at a
heritability of 0.6, the aligned fit returns 0.490 with `p = 1.4e-06`; the same
data with the response shuffled returns 0.000, an interval of `[0.000, 0.081]`
and `p = 1`. It does not perturb the answer, it destroys the signal and then
reports no heritability with confidence. `align` takes the matrix with its own
identifiers and the values with theirs, lines them up once, and refuses what it
cannot. It reads a genomic relationship matrix or an estimated kinship computed
elsewhere just as well as a pedigree one — that pairing, a matrix beside a
separate identifier file, is exactly where this goes wrong.

```python
data = asterism.align(k, order, table_ids, y=height, age=age)
```

**`subject_order_commitment`** takes the aligned order and returns a digest of
it. Pass that to a model and it is echoed back on the fit record, so a result
can be tied to the exact rows it came from without ever storing an identifier.

## One trait, one variance component

```python
import asterism
import numpy as np

k, order = asterism.relationship_matrix(ids, father, mother, keep=analysed_ids)

# Line the values up with the matrix by identifier, not by hope.
data = asterism.align(k, order, table_ids, y=height, age=age)
order_sha256 = asterism.subject_order_commitment(data["order"])

model = asterism.prepare(
    np.column_stack([np.ones(len(data["order"])), data["age"]]),
    data["relationship"],
    subject_order_sha256=order_sha256,
)
fit = model.fit(data["y"])  # REML by default
fit["h2"], fit["interval"], fit["test"]

ml = model.fit(data["y"], estimator="ml")
```

`x` is the fixed-effect design and must include its own intercept column when
one is wanted. `prepare` checks the matrix and design, diagonalises the
covariance matrix, and stores everything independent of the response — so one
prepared model fits many responses with the same rows and design, at the cost
of one decomposition. Every fit record carries the identity of the build that
produced it.

One fit returns everything about that fit: `interval` is the profile interval
for the heritability and `test` is the test against nought, both already inside
the record rather than separate calls that would refit.

## Several covariance components

Genes and a shared household, fitted together. The complete example is
[`examples/components.py`](examples/components.py):

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
household variance of 0.2 is reported as 0.1, and that is right: the average
person carries half as much of it.

```python
model = asterism.ComponentModel([relationship, household], design)
fit = model.fit(y)
interval = model.interval(y, component=1)
test = model.test(y, component=1)
```

## Two traits

Two continuous traits and the genetic correlation between them. `observed`
is a boolean pair per person saying which traits they have; the values and
the design are stacked, trait within person. The runnable version is
[`examples/bivariate.py`](examples/bivariate.py):

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


## Gene by environment, continuous and discrete

A measured environment, with the runnable version in
[`examples/gxe.py`](examples/gxe.py):

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

A binary environment, with the runnable version in
[`examples/discrete_gxe.py`](examples/discrete_gxe.py):

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
gxe = asterism.GxeModel(
    relationship,
    environment,
    design,
    surface="random_regression",
)
fit = gxe.fit(y, grid=[low, middle, high])
gxe.test(y, "correlation")
gxe.test(y, "interaction")

# A binary environment: sex, an exposure, a cohort. Two labels, nothing else.
discrete = asterism.DiscreteGxeModel(relationship, sex, design)
fit = discrete.fit(y)
discrete.test(y, "gene_by_environment")
discrete.test(y, "correlation")
discrete.test(y, "genetic")
discrete.test(y, "residual")
```

Available GxE surfaces are `exponential`, `powered_exponential`, and
`random_regression`. They are different covariance families; choose the family
and any powered-exponential shape independently of the fitted outcome. Only
random regression can represent negative genetic correlations and crossovers.
The powered-exponential family is not among them; see
[outside-support.md](docs/outside-support.md).

`GxeModel` takes an environment measured on a range and smooths across it.
`DiscreteGxeModel` takes one measured as a binary label — any two distinct
finite values, with the smaller naming the first group — and smooths nothing:
it carries a genetic and a residual variance per group and one correlation
between them. Sex is the canonical use, and `fit` returns the two `levels` so a
reader can tell which group is which.
Group-specific heritabilities remain descriptive in 0.1; the genetic
correlation, its interval, and the calibrated genetic tests are the supported
targets.

The principal `gene_by_environment` test leaves residual variances free in the
two environments; `any_difference` additionally equates them and therefore is
not specifically a genetic test.

## Binary liability

The runnable version is
[`examples/liability.py`](examples/liability.py):

```sh
python examples/liability.py
```

```
people:                 1000
cases:                  200 of 1000
h2 liability (true 0.5): 0.641
converged:              True
95% interval:           [0.402, 0.861]
```

```python
model = asterism.LiabilityModel(relationship, affected, design)
fit = model.fit()
interval = model.interval()
test = model.test()
```

This is maximum likelihood on an unobserved probit liability whose variance is
fixed at one. The reported heritability is on the liability scale, not the
observed binary scale. Family probabilities above dimension two use the
Mendell-Elston sequential approximation.

## A trait whose measurement stops at a limit

The heritability returned is of the *complete* trait: what you would have
had if the instrument reached far enough. It is maximum likelihood, never
REML, so do not place it beside a REML heritability. The runnable version is
[`examples/censored.py`](examples/censored.py):

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

## Two traits measured differently

One trait censored, one not. This is the pairing the extended high-frequency
work needs. The runnable version is
[`examples/mixed.py`](examples/mixed.py):

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

**A binary trait's variance is fixed at one** and comes back as one, because
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
