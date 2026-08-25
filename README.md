# Asterism ⁂

**Asterism** is statistical software for fitting the kinds of variance-component models typically found in quantitative genetics. It comprises a compiled numerical core written in Rust and a Python API, which takes NumPy arrays and returns ordinary Python objects.

Two important things to note before using Asterism in your own research. First, almost every Asterism capability can be replicated in other software by design—great care has been taken to ensure the numbers you get from Asterism closely match those from SOLAR or R wherever they have the same capabilities. Second, Asterism code is **100% AI authored**. I (Sam Mathias) took great care in planning and overseeing development and I stand by the results, but I did not write the code myself. If this bothers you, use other software instead.

## Installation

```sh
pip install asterism
```

Wheels are published for macOS on Apple silicon and for Linux on x86-64, and
carry a compiled binary, so nothing is built on your machine and no Rust
toolchain is needed. Python 3.13 or 3.14. The only runtime dependency is NumPy.

To work on Asterism itself, see [development.md](docs/development.md).

## Quickstart

Paste this. It builds a pedigree of eighty nuclear families, simulates one
trait with a true heritability of 0.5, and fits it — no data required.

```python
import numpy as np
import asterism

FAMILIES, CHILDREN = 80, 4

# Two founders and four full siblings per family.
parents = [(f"{f}-dad", f"{f}-mum") for f in range(FAMILIES)]
ids = [name for pair in parents for name in pair] + [
    f"{f}-child{c}" for f in range(FAMILIES) for c in range(CHILDREN)
]
father = [None] * (2 * FAMILIES) + [
    parents[f][0] for f in range(FAMILIES) for _ in range(CHILDREN)
]
mother = [None] * (2 * FAMILIES) + [
    parents[f][1] for f in range(FAMILIES) for _ in range(CHILDREN)
]

relationship, order = asterism.relationship_matrix(ids, father, mother)
people = len(order)

# Simulate a trait that really is 50% heritable.
covariance = 0.5 * relationship + 0.5 * np.eye(people)
trait = np.linalg.cholesky(covariance) @ np.random.default_rng(0).normal(size=people)

model = asterism.prepare(np.ones((people, 1)), relationship)
fit = model.fit(trait)          # REML by default

print(f"h2 (true 0.5): {fit['h2']:.3f}")
print(f"95% interval:  [{fit['interval']['lower']:.3f}, {fit['interval']['upper']:.3f}]")
print(f"p (h2 = 0):    {fit['test']['p_value']:.2e}")
```

```
h2 (true 0.5): 0.605
95% interval:  [0.455, 0.745]
p (h2 = 0):    1.11e-19
```

Every analysis below has a complete runnable program behind it in
[`examples/`](examples/), which lives in the repository rather than in the
installed package.

## What is supported

Eight analyses are supported, meaning each one has checks that measure it
against a known truth or an independent implementation, and those checks pass
on the exact wheel you installed. They are listed in
[api-support.md](docs/api-support.md), which also names the public objects that
fall *outside* that support: models you can import and use, but whose numbers
have not been established to the same standard. Those are documented separately,
in [outside-support.md](docs/outside-support.md), and are deliberately not in
this file.

What stands behind the supported ones is in
[numerical-validation.md](docs/numerical-validation.md) — coverage simulations,
comparisons against R, SOLAR, censReg and MCMCglmm, and agreement between
platforms to within five parts in a thousand million.

Every fit carries the identity of the build that produced it, so a number can
be traced back to an exact version. Results that did not converge are returned
as diagnostic rather than quietly reported.

## One trait, one relationship matrix

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

**Asterism's numerical interface is positional, and that is the one place a
mistake makes no noise.** A relationship matrix whose rows are in a different
order from the response does not fail or warn. On 300 people simulated at a
heritability of 0.6, the aligned fit returns 0.490 with `p = 1.4e-06`; the same
data with the response shuffled returns 0.000, an interval of `[0.000, 0.081]`
and `p = 1`. It does not perturb the answer, it destroys the signal and then
reports no heritability with confidence. `align` takes the matrix with its own
identifiers and the values with theirs, lines them up once, and refuses what it
cannot. It reads a genomic relationship matrix or an estimated kinship computed
elsewhere just as well as a pedigree one — that pairing, a matrix beside a
separate identifier file, is exactly where this goes wrong.

`x` is the fixed-effect design and must include its own intercept column when
one is wanted. `prepare` checks the matrix and design, diagonalises the
relationship matrix, and stores everything independent of the response. Reuse
one prepared model for multiple responses with the same rows and design.
Every fit record carries the identity of the build that produced it, and an
optional `subject_order_sha256`. That second one is a digest of the row order
you fitted: pass it in and it is echoed back on the record, so a result can be
tied to the exact rows it came from without ever storing an identifier.

One fit returns everything about that fit: `interval` is the profile interval
for the heritability and `test` is the test against nought, both already inside
the record rather than separate calls that would refit.

The relationship builder returns twice the kinship coefficient. `prepare`
also accepts any finite, symmetric, positive-semidefinite matrix with the right
dimensions.

## Several covariance components

Genes and a shared household, fitted together. The complete example is
[`examples/components.py`](examples/components.py):

```sh
python examples/components.py
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

0.1 supports the pairings that include a continuous trait: continuous with
continuous, binary with continuous, and censored with continuous, along with
censored with censored. The binary-with-censored pair is deferred — it stays
public and its checks are kept, but 0.1 makes no scientific claim about it.




## Citing Asterism

Cite the archived version you actually ran, not the repository. Each release is
archived with its own DOI, and that is what makes a number traceable — the
repository moves, a release does not.

Details are in [CITATION.cff](CITATION.cff). Asterism is archived at
publication rather than before it, so the DOI is added there when the first
release is published — see
[ADR 0018](docs/adr/0018-published-with-the-papers-that-cite-it.md).

## Licence

MIT. See [LICENSE](LICENSE).

Asterism is original work. It depends on third-party packages, named in
`Cargo.lock` and `uv.lock` and used under their own licences, and derives from
nothing else. It has no affiliation with SOLAR or with any other quantitative
genetics package. Where its answers are compared with SOLAR, R, `spaMM`,
MCMCglmm or a published analysis, those are benchmarks: agreement proves
fidelity and never correctness, which is
[ADR 0006](docs/adr/0006-agreement-proves-fidelity.md).
