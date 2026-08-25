# Asterism ⁂

Asterism** is statistical software for fitting the kinds of variance-component models typically found in quantitative genetics. It comprises a compiled numerical core written in Rust and a Python API, which takes NumPy arrays and returns ordinary Python objects.

Two important things to note before using Asterism in your own research. First, almost every Asterism capability can be replicated in other software by design — great care has been taken to ensure the numbers you get from Asterism closely match those from SOLAR or R wherever they have the same capabilities. Second, Asterism code is 100% AI authored**. I (Sam Mathias) took great care in planning and overseeing development and I stand by the results, but I did not write the code myself. If this bothers you, use other software instead.

## Installation

Wheels are published for macOS on Apple silicon and for Linux on x86-64, and carry a compiled binary, so nothing is built on your machine and no Rust toolchain is needed. Python 3.13 or 3.14. The only runtime dependency is NumPy. Install in the usual way:

```sh
pip install asterism
```

To work on Asterism itself, see [development.md](docs/development.md).

## Quickstart

The following builds a pedigree, simulates one trait with a true heritability of 0.5, and fits it — no data required.

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

Every capability has a complete runnable program behind it in [`examples/`](examples/), which lives in the repository rather than in the installed package.

## Capabilities

Eight models are currently supported. Each one has checks that measure it against a known truth or an independent implementation. They are listed in [api-support.md](docs/api-support.md), which also names the public objects that fall outside current support (models whose numbers have not been established to the same standard yet; see [outside-support.md](docs/outside-support.md)).

| Capability                    | Runnable example                              |
| ----------------------------- | --------------------------------------------- |
| One trait, one component      | [`quickstart.py`](examples/quickstart.py)     |
| Several components            | [`components.py`](examples/components.py)     |
| Two traits                    | [`bivariate.py`](examples/bivariate.py)       |
| Binary traits                 | [`liability.py`](examples/liability.py)       |
| Censored traits               | [`censored.py`](examples/censored.py)         |
| Mixed pairs                   | [`mixed.py`](examples/mixed.py)               |
| Gene by environment, measured | [`gxe.py`](examples/gxe.py)                   |
| Gene by environment, binary   | [`discrete_gxe.py`](examples/discrete_gxe.py) |

Matrices can come from anywhere. `relationship_matrix` builds one from a pedigree and `kinship_classes` splits a pedigree into separate bases, but every model accepts any finite, symmetric, positive-semidefinite matrix — a genomic relationship matrix, an estimated kinship, whatever you have. `align` lines a matrix up with your values by identifier, which matters because the numerical interface is positional and a misaligned fit does not warn, it silently destroys the signal.

Each model has [a page of its own](docs/models/) carrying everything it needs: how to use it, its equations, what has been measured about it, and its interface. What they share is in [statistical-methods.md](docs/statistical-methods.md), and the whole public surface in the [API reference](docs/api-reference.md).

## Citing Asterism

{{to complete when published/minted}}

## Licence

MIT. See [LICENSE](LICENSE).

Asterism is original work that depends on third-party packages used under their own licences. It has no affiliation with any other quantitative genetics package. Where its answers are compared with other software packages or a published analysis, those are benchmarks not dependencies.
