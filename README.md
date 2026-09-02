# Asterism ⁂

**Asterism** is statistical software for fitting the kinds of variance-component models typically found in quantitative genetics. It comprises a compiled numerical core written in Rust and a Python API, which takes NumPy arrays and returns ordinary Python objects.

Two important things to note before using Asterism in your own research. First, almost every Asterism capability can be replicated in other software by design — care has been taken to ensure the numbers you get from Asterism closely match those from SOLAR or R wherever they have the same capabilities. Second, Asterism code is **100% AI authored**. I (Sam Mathias) planned development carefully and I stand by the results, but I did not write the code myself. If this bothers you, feel free to use the code provided here to replicate analysis in other software.

## Installation

Wheels are published on the [GitHub release page](https://github.com/sammosummo/asterism/releases/tag/v0.2.0) for macOS on Apple silicon and for Linux on x86-64. They carry a compiled binary, so nothing is built on your machine and no Rust toolchain is needed. Python 3.13 or 3.14. The only runtime dependency is NumPy. Download the wheel for your platform, then install that exact file:

```sh
# macOS on Apple silicon
python -m pip install ./asterism-0.2.0-cp313-abi3-macosx_11_0_arm64.whl

# Linux on x86-64
python -m pip install ./asterism-0.2.0-cp313-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Do not install the distribution named `asterism` from PyPI: that name belongs
to an unrelated project. The import from Asterism's release wheel is still
`import asterism`.

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
fit = model.fit(trait)  # REML by default

print(f"h2 (true 0.5): {fit['h2']:.3f}")
print(
    f"95% interval:  [{fit['interval']['lower']:.3f}, {fit['interval']['upper']:.3f}]"
)
print(f"p (h2 = 0):    {fit['test']['p_value']:.2e}")
```

```
h2 (true 0.5): 0.605
95% interval:  [0.455, 0.745]
p (h2 = 0):    1.11e-19
```

Every capability has a complete runnable program behind it in [`examples/`](examples/), which lives in the repository rather than in the installed package.

## Capabilities

The 0.2.0 release preserves the eight 0.1.1 analyses and adds the
several-component censored model: fitting, intervals and an explicitly labelled
asymptotic test are supported, while the bootstrap remains experimental. The
corrected fixed-instrument one-component censored and mixed-pair checks pass.
The several-component analytic p-value failed on the
exact 1,909-person, four-component target at 75% expected censoring and must not
be reported for that analysis without a design-specific simulated null. That is
not a universal censoring threshold, and it does not affect fitting, intervals
or the released one-component test. Bootstrap refusals retain the exact inner
coordinate and original fit code; `CensoredComponentModel.bootstrap_replay`
reproduces that draw for diagnosis without returning a p-value or weakening the
all-or-nothing denominator. The exact published statuses are listed in
[api-support.md](docs/api-support.md), alongside public objects outside the 0.2
scientific support set (see
[outside-support.md](docs/outside-support.md)).

| Capability                    | Runnable example                              |
| ----------------------------- | --------------------------------------------- |
| One trait, one component      | [`quickstart.py`](examples/quickstart.py)     |
| Several components            | [`components.py`](examples/components.py)     |
| Two traits                    | [`bivariate.py`](examples/bivariate.py)       |
| Binary traits                 | [`liability.py`](examples/liability.py)       |
| Censored traits               | [`censored.py`](examples/censored.py)         |
| Censored traits, components   | [`censored_components.py`](examples/censored_components.py) |
| Mixed pairs                   | [`mixed.py`](examples/mixed.py)               |
| Gene by environment, measured | [`gxe.py`](examples/gxe.py)                   |
| Gene by environment, binary   | [`discrete_gxe.py`](examples/discrete_gxe.py) |

Matrices can come from anywhere. `relationship_matrix` builds one from a pedigree and `kinship_classes` splits a pedigree into separate bases, but every model accepts any finite, symmetric, positive-semidefinite matrix — a genomic relationship matrix, an estimated kinship, whatever you have. `align` lines a matrix up with your values by identifier, which matters because the numerical interface is positional and a misaligned fit does not warn, it silently destroys the signal.

Each model has [a page of its own](docs/models/) carrying everything it needs: how to use it, its equations, a summary of what has been measured about it, and its interface. What they share is in [statistical-methods.md](docs/statistical-methods.md), the comparisons and simulations in full are in the [validation record](docs/validation.md), and the whole public surface in the [API reference](docs/api-reference.md).

For an analysis whose provenance has to be recorded — which wheel, which commit, which inputs — `run_analysis` returns a receipt alongside the fit. See [analysis-receipts.md](docs/analysis-receipts.md).

## Citing Asterism

Cite the version you actually ran. A development checkout is not a citable version: only a release wheel carries the build identity a result can be traced to.

- Version 0.1.1: [10.5281/zenodo.22116963](https://doi.org/10.5281/zenodo.22116963)
- Version 0.1.0: [10.5281/zenodo.22113715](https://doi.org/10.5281/zenodo.22113715)
- All versions: [10.5281/zenodo.22113714](https://doi.org/10.5281/zenodo.22113714)

`CITATION.cff` carries the second, so GitHub's "Cite this repository" and most reference managers will pick it up.

## Licence

MIT. See [LICENSE](LICENSE).

Asterism is original work that depends on third-party packages used under their own licences. It has no affiliation with any other quantitative genetics package. Where its answers are compared with other software packages or a published analysis, those are benchmarks not dependencies.
