# One trait, one variance component

Analysis id `one_trait_gaussian_heritability`. Runnable example in [`examples/`](../../examples/).

## Using it

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

## The model

### Model and estimator

With fixed relationship matrix $\mathbf K$, Asterism fits

$$
\mathbf y\sim N(\mathbf X\boldsymbol\beta,\mathbf V),\qquad
\mathbf V=\sigma^2\{h^2\mathbf K+(1-h^2)\mathbf I_n\},
$$

where $\sigma^2>0$ is total variance, $h^2\in[0,1]$ is the relationship-matrix
heritability, and $\mathbf I_n$ is the $n\times n$ identity. `prepare` validates
the design and covariance basis, partitions the roster into relationship
blocks, eigendecomposes $\mathbf K$ once, and reuses the transformed design.
At each $h^2$, $\boldsymbol\beta$ and $\sigma^2$ are profiled analytically;
$h^2$ is found by a deterministic bounded one-dimensional search.

The zero-heritability test uses the 50:50 boundary mixture. An interval endpoint
at $h^2=0$ or $1$ is not automatically included: the boundary likelihood is
judged using the calibrated mixture recipe and reported in `contains_*`. A
singular $\mathbf K$ can make $h^2=1$ an invalid covariance; the fit then
refuses that point instead of pretending it was evaluated.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `h2` ($h^2$) | `PreparedModel.fit(y)["h2"]` |
| `interval` | `fit["interval"]`, using the common interval fields |
| `test` of $H_0:h^2=0$ | `fit["test"]["null"]`, `["statistic"]`, `["p_value"]`, `["rule"]`, and `["null_loglik"]` |

`fit["estimator"]` distinguishes `reml` from `ml`; `total_variance`, `beta`,
`fixed_effects`, and `standard_errors` are descriptive. `converged`, `boundary`,
`warnings`, `subject_order_sha256`, and `build` are required provenance or fit
diagnostics rather than reportable targets.

### Assumptions, measured limitations, and unmet gates

The response is conditionally Gaussian with covariance exactly proportional to
the supplied $\mathbf K$ plus independent residual variance; rows are aligned;
$\mathbf X$ is full rank; and the fixed effects are correctly specified. A
covariate p-value is comparable across programs only when the design
parameterisation is identical. Interpreting $h^2$ as a marginal phenotypic
variance proportion additionally requires the qualified relationship-matrix
normalisation (normally unit diagonal for twice kinship); rescaling
$\mathbf K$ changes this parameter.

Historical repository simulations found conservative, rather than
anti-conservative, coverage near $h^2=0$: at $n=350$, true values 0.05 and 0.07
covered at 0.979 and 0.974. That evidence is REML- and design-specific. The
release contract therefore requires independent R and SOLAR agreement plus
general and target-design coverage. Their adapters and public-Python campaign
commands are executable, but none becomes release evidence until its exact
prewritten command passes against the fixed release wheel and the supported
pass rules are configured in `release.toml`.

## What stands behind it

- Against R `regress` under REML at `n=350` with six fixed effects,
  heritability agreed within `8e-9` relative, total variance to `4e-9`, and
  fixed effects to `1e-9`. The log-likelihood difference was exactly the
  `(n-p)/2 log(2 pi)` constant omitted by `regress`.
- Against native SOLAR under ML on three synthetic pedigrees, heritability
  agreed within `5e-7`, the precision available from SOLAR's printed output.
  Its log-likelihood omits `n/2 log(2 pi)`.
- A REML coverage simulation used 8,000 replicates at each of twelve true
  heritabilities. All twelve 95% interval coverages were compatible with 0.95
  at `n=1400`. At `n=350`, ten were compatible; truths 0.05 and 0.07
  over-covered at 0.979 and 0.974. This is conservative near-boundary
  inference, not under-coverage.

`checks/one_trait_coverage.py` is the participant-free, reproducible successor
to that historical compiled campaign. It exercises only `asterism.prepare` and
`PreparedModel.fit`, requires the replicate count, family count, worker count,
truth grid, and `--no-write` policy explicitly on the command line, and emits
one JSON document to standard output. Every requested replicate remains in the
coverage denominator: a refused fit, a nonconverged free fit, or any failed
constrained profile evaluation is a coverage miss. At true $h^2=0$ and $h^2=1$
the check reads the fit record's `contains_lower_bound` and
`contains_upper_bound` fields, respectively, rather than inferring containment
from a numerical endpoint.

For $k$ covered replicates out of $m$, the check reports the two-sided 95%
Clopper--Pearson interval

$$
L = B^{-1}_{0.025}(k,m-k+1), \qquad
U = B^{-1}_{0.975}(k+1,m-k),
$$

with $L=0$ when $k=0$ and $U=1$ when $k=m$. The acceptance rule was fixed
before running the target design: a cell fails as anti-conservative only when
$U<0.940$. A cell with $L>0.960$ passes but is labelled conservative; at the
historically conservative truths 0.05 and 0.07 the more specific label is
`historically_conservative`. Thus extra coverage is reported as lost precision,
not misclassified as under-coverage.

The reviewed target-design fixture is values-free aggregate structural
metadata from a predecessor receipt. It fixes 1,909 rows in 202 nonzero
relationship components, the exact component-size histogram, and a largest
component of 180, and asserts that those facts sum back to 1,909 rows and 202
components. It contains no identifiers, phenotypes, or participant-derived
matrix. With the historical generator settings `seed=7`, `sib_mean=3.0`,
`marry_in_p=0.35`, and `max_gen=5`, a standard founder-order kinship recursion
produces a structure-matched synthetic pedigree with 27,691 nonzero
lower-triangle relationship pairs. The predecessor aggregate count was 27,821,
so the prewritten relative discrepancy is 0.4673%, inside the fixed 0.5% limit;
this is not a reconstruction or a claim of coefficient- or eigenvalue-level
identity. A separate `seed=1` stream generates the six columns intercept,
standardised age, age squared, sex, and both age-by-sex products. The exact
fixture bytes are selected by SHA-256
`93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e`.

On 21 August 2026, development-wheel smoke runs completed 1,200 replicates per
truth on both the 1,400-person repeated-family design and the 1,909-person
target envelope: 14,400 fits per design, 28,800 in total. All cells were
compatible with the prewritten rule and both runs recorded zero refusals, zero
nonconverged free fits, and zero profile-failed replicates:

| true $h^2$ | standard coverage | standard 95% CP | target coverage | target 95% CP |
| ---: | ---: | :--- | ---: | :--- |
| 0.00 | 0.9442 | [0.9296, 0.9565] | 0.9442 | [0.9296, 0.9565] |
| 0.05 | 0.9525 | [0.9389, 0.9638] | 0.9567 | [0.9436, 0.9675] |
| 0.07 | 0.9508 | [0.9370, 0.9624] | 0.9500 | [0.9361, 0.9616] |
| 0.10 | 0.9525 | [0.9389, 0.9638] | 0.9558 | [0.9426, 0.9667] |
| 0.20 | 0.9500 | [0.9361, 0.9616] | 0.9542 | [0.9408, 0.9653] |
| 0.30 | 0.9525 | [0.9389, 0.9638] | 0.9458 | [0.9315, 0.9580] |
| 0.40 | 0.9542 | [0.9408, 0.9653] | 0.9492 | [0.9352, 0.9609] |
| 0.50 | 0.9483 | [0.9343, 0.9602] | 0.9450 | [0.9306, 0.9572] |
| 0.60 | 0.9392 | [0.9241, 0.9520] | 0.9542 | [0.9408, 0.9653] |
| 0.70 | 0.9417 | [0.9269, 0.9543] | 0.9417 | [0.9269, 0.9543] |
| 0.80 | 0.9375 | [0.9223, 0.9505] | 0.9425 | [0.9278, 0.9550] |
| 1.00 | 0.9458 | [0.9315, 0.9580] | 0.9583 | [0.9454, 0.9689] |

A 300-replicate prefix had first placed the standard-design $h^2=0.70$ cell at
0.9100 with interval [0.8718, 0.93985], just below the lower safety edge. The
same deterministic stream at 1,200 replicates gave 0.9417 [0.9269, 0.9543].
That observed sampling fluctuation is why these development runs qualify the
portable command but are not release evidence. `release.toml` reserves 8,000
replicates per truth, or 96,000 fits per design; neither exact 8,000-replicate
public-Python command has yet run against a fixed release artifact. The earlier
8,000-replicate result above came from the predecessor compiled campaign.

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `prepare(x: 'Any', k: 'Any', *, subject_order_sha256: 'str | None' = None) -> 'PreparedModel'`

Validate and decompose a fixed-effect design and a relationship matrix.

Parameters
----------
x
    The fixed-effect design, one row per subject in fit order. It must
    include its own intercept column if one is wanted; nothing is added.
k
    The relationship matrix, in the same subject order as ``x``. Twice the
    kinship is the usual choice, but nothing here requires it: the matrix
    is checked numerically, so a genomic relationship matrix is equally
    acceptable. Row alignment among ``x``, ``k``, and the response is
    positional and is the caller's responsibility.
subject_order_sha256
    Lowercase SHA-256 from :func:`subject_order_commitment` for the exact
    row order. The prepared model echoes it on every fit record without
    retaining participant identifiers.

Raises
------
ValueError
    With a stable code naming what was wrong: the matrix not square, not
    symmetric, not positive semi-definite, shapes disagreeing, values not
    finite, the design rank-deficient, or no residual degrees of freedom.

### `PreparedModel.fit(self, /, y, estimator='reml')`

Fit the ordinary Gaussian model. The estimator is data on the record;
the objective never depends on the response's values.

