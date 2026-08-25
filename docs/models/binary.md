# Binary traits

Analysis id `binary_liability_heritability`. Runnable example in [`examples/`](../../examples/).

## Using it

The runnable version is
[`examples/liability.py`](../../examples/liability.py):

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

## The model

### Model and estimator

Let $L_i$ be an unobserved liability and $Y_i$ the observed binary status.
Asterism fixes liability variance to one and fits

$$
\begin{aligned}
L_i&=\mathbf x_i^T\boldsymbol\beta+g_i+e_i,\\
\mathrm{Cov}(\mathbf L)&=h^2\mathbf A+(1-h^2)\mathbf I_n,\\
Y_i&=\mathbb 1(L_i>0),
\end{aligned}
$$

where $\mathbf x_i^T$ is row $i$ of the design, $g_i$ is the additive genetic
liability, $e_i$ is residual liability, $\mathbf A$ is the supplied
relationship matrix, $\mathbf I_n$ is the identity, and $h^2\in[0,1]$ is
liability-scale heritability. Fixing total variance identifies the probit scale. Threshold
heritability originates with [Dempster and Lerner
(1950)](../references.bib#dempsterLerner1950), and human-disease liability is
developed by [Falconer (1965)](../references.bib#falconer1965).

Each family contributes a multivariate-normal orthant probability. Asterism
uses the likelihood

$$
L(\boldsymbol\beta,h^2)
=\prod_{f=1}^{F}\Pr\{\mathbf D_f\mathbf L_f>\mathbf 0\},
$$

where $F$ is the number of independent family blocks, $\mathbf L_f$ is family
$f$'s liability vector after its fixed-effect mean is applied, and
$\mathbf D_f$ is diagonal with $+1$ for a case and $-1$ for a noncase;
$\mathbf 0$ is the conformable zero vector. It
evaluates dimensions one and two directly and uses sequential Mendell–Elston
truncation above two dimensions. [Mendell and Elston
(1974)](../references.bib#mendellElston1974) provides the multifactorial
qualitative-trait approximation lineage; [Tallis
(1961)](../references.bib#tallis1961) supplies moments of the truncated normal.
Rare-class-first ordering, probability guards, and the dedicated
low-dimensional path are Asterism choices. Estimation is ML. The test of
$H_0:h^2=0$ uses the implemented 50:50 boundary mixture.

### Public record mapping

| Manifest quantity | Public field |
| --- | --- |
| `heritability` | `LiabilityModel.fit()["heritability"]` |
| `interval` | `LiabilityModel.interval()`, using the common interval fields |
| `test` | `LiabilityModel.test()`: `statistic`, `p_value`, `rule`, `null_loglik`, `estimator` |

`scale`, `prevalence`, `fixed_effects`, and `largest_family` are essential
descriptors; `loglik`, `scaled_gradient`, `polished`, and `converged` diagnose
the fit.

### Assumptions, measured limitations, and unmet gates

Status is a deterministic thresholding of a Gaussian liability, the supplied
relationship matrix captures additive covariance, families are independent
blocks, the relationship diagonal is normalized to the unit-liability scale,
and the fixed effects correctly locate liability. The estimand is not
observed-scale heritability. Historical calibration on the GOBS pedigree found
0.0529 rejection at nominal 0.05 and 0.9443 coverage at true $h^2=0.25$; the
two-person quadrature error grew near correlation one. Above two dimensions,
accuracy depends on family size, correlation, imbalance, and truncation order.
The independent SOLAR comparison has been refreshed live and frozen for
portable verification. A values-free 1,909-person target check also exercised
the largest 180-person family: its one null and one alternative smoke attempt
both completed, and the alternative interval covered $h^2=0.25$. That is an
execution seam, not a calibration estimate. The exact 200-replicate-per-scenario
fixed-wheel campaign remains a release
requirements.

## What stands behind it

Against native SOLAR, liability heritability differed by at most 0.0134, or
0.086 SOLAR standard errors, across three synthetic cases. The historical
700-replicate campaign used the real GOBS pedigree structure: the
zero-heritability test rejected 0.0529 at nominal 0.05 and 95% interval
coverage was 0.9443 at true heritability 0.25. Those figures remain a recorded
result, not reproducible release evidence, because the portable gate no longer
reads participant material.

`checks/liability_calibration.py` replaces that dependency with a reviewed,
values-free target fixture. The fixture pins 1,909 rows in 202 relationship
components, the exact component-size histogram, largest component 180, and the
predecessor aggregate of 27,821 nonzero lower-triangle relationship pairs. The
same deterministic synthetic generator used by the Gaussian target gate
produces 27,691 such pairs, a prewritten discrepancy of 0.4673% inside the 0.5%
limit. This is aggregate structure matching, not pedigree reconstruction: the
fixture retains no identifiers, participant rows, phenotypes, covariates, or
participant-derived relationship matrix, and claims neither coefficient nor
eigenvalue identity. The liability-specific fixture SHA-256 is
`863eedc4f18063bca27754e782b9ebb5faa341359912b034163a7ea2067c20aa`; it in
turn binds the shared structural fixture SHA-256
`93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e`.

For component $f$, the command generates

$$
\begin{aligned}
\mathbf z_f &= \mathbf C_f\boldsymbol\varepsilon_f,
&\boldsymbol\varepsilon_f&\sim N(\mathbf 0,\mathbf I),\\
\mathbf C_f\mathbf C_f^{\mathsf T}
  &=h^2\mathbf A_f+(1-h^2)\mathbf I,
&Y_i&=\mathbb 1\!\left\{z_i>\Phi^{-1}(1-K)\right\},
\end{aligned}
$$

with prevalence $K=0.254$, null truth $h^2=0$, and coverage truth
$h^2=0.25$. It factors each independent synthetic pedigree block separately,
then exercises only `asterism.LiabilityModel.fit`, `.test`, and `.interval`.
The largest family therefore reaches the same above-two-person sequential
Mendell--Elston path whose approximation needs stress; [Mendell and Elston
(1974)](../references.bib#mendellElston1974) is the method lineage. The test must
report `mixture_50_50`; this boundary reference is an implemented assumption to
be calibrated, not a conclusion inherited automatically from [Self and Liang
(1987)](../references.bib#selfLiang1987).

Every attempted replicate remains in its scenario denominator. Too few cases
or noncases, a public refusal, nonconvergence, a missing or malformed test or
interval, and any failed constrained profile evaluation are failures; the
release allowance is zero. Let $m_0$ be all attempted null replicates and
$r_\alpha$ the complete public p-values no greater than level $\alpha$. At
one-sided confidence $\gamma=0.95$, the exact [Clopper--Pearson
(1934)](https://doi.org/10.1093/biomet/26.4.404) lower limit is

$$
L_\alpha=
\begin{cases}
0, & r_\alpha=0,\\
B^{-1}_{1-\gamma}(r_\alpha,m_0-r_\alpha+1), & r_\alpha>0.
\end{cases}
$$

The level cell fails only when $L_\alpha>\alpha$. Let $c$ be complete,
failure-free intervals covering $h^2=0.25$ among all $m_1$ attempted
alternative replicates. The corresponding exact upper limit is

$$
U=
\begin{cases}
1, & c=m_1,\\
B^{-1}_{\gamma}(c+1,m_1-c), & c<m_1,
\end{cases}
$$

and coverage fails when $U<0.95$. Numerical and profile failures fail the gate
separately even when these exact-binomial bounds remain compatible, so neither
calculation can hide a missing result by changing its denominator.

On 21 August 2026, a bounded development-wheel smoke ran one replicate per
scenario on the complete 1,909-person target. Both attempts returned complete
public records with zero failures. The alternative interval covered 0.25 and
its estimate was 0.1906. This two-attempt run verifies the loader, generator,
largest-family approximation path, public interface, and executable decision;
it does not estimate level or coverage. The release command fixes 200
replicates per scenario, but that full 400-attempt campaign has not yet run and
the manifest's scientific configured and measured flags remain false.

The two-person quadrature error was at most `1.3e-9` for liability correlation
below 0.5 and `2.3e-4` below 0.99. This motivates refusing nearly perfectly
related pairs in the current numerical method.

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `LiabilityModel.fit(self) -> 'dict[str, Any]'`

Fit, by maximum likelihood because nothing else is available.

### `LiabilityModel.interval(self) -> 'dict[str, Any]'`

A 95 per cent profile interval for the liability heritability.

### `LiabilityModel.test(self) -> 'dict[str, Any]'`

Test the liability heritability against nought.

A heritability of nought sits on a bound, so the reference is the even
mixture of a point mass and chi-square on one degree of freedom. That
was assumed from the Gaussian case rather than derived for a liability,
and then measured: on the real pedigree it rejects 0.055 of the time at
a nominal 0.05.

