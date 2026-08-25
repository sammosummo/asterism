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
household variance of 0.2 is reported as 0.1, and that is right: the average
person carries half as much of it.

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
diagonal. Split kinship-class bases have zero diagonal, so their reportable
targets are coefficients and class deviations, never proportions. If $r$
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

### Assumptions, measured limitations, and unmet gates

All Gaussian assumptions above apply. Each basis must be finite, symmetric, and
aligned to the response, and every evaluated linear combination must be a
positive-definite covariance. A split or contrast basis need not be positive
semidefinite by itself. A scale-free Frobenius Gram-rank check now refuses
linearly dependent bases, including a submitted identity confounded with the
implicit residual. It establishes coefficient identification at numerical rank;
it is not a precision guarantee for bases that merely lie close together.
Historical simulations at $n=400$
gave 0.945 and 0.955 coverage for additive and household mean-diagonal
proportions and 0.051 rejection for a nominal 0.05 zero-household test.
Zero-diagonal class-equality calibration on the GOBS pedigree rejected 0.035
under equality. These observations do not supply a universal identifiability
diagnostic or release envelope. Component calibration and the interval-identity
sentinel now have executable public-API pass rules. The sentinel has passed on
the development build, but the exact calibration command has not yet been
retained from the fixed release wheel and the campaign remains
unmeasured.

## What stands behind it

Against native SOLAR under ML at `n=350` with six fixed effects, a model with
kinship, a second supplied matrix, and residual agreed on all three variances
to about `5e-7` and on the log-likelihood to `4e-7` after the `n/2 log(2 pi)`
constant SOLAR omits. Until this check the component family had no comparison
against another package at all; its second matrix is a spatial kernel with the
decay rate held fixed, which makes it an ordinary known matrix.

**SOLAR cannot be used this way for a matrix whose covariance crosses
pedigrees.** It evaluates the likelihood one pedigree at a time, so only the
within-pedigree blocks of a supplied matrix contribute, and the rest are
discarded without warning — `matrix debug` still reports the whole file. Handed
the dense kernel, SOLAR returns its within-pedigree answer bit for bit, while
the two implementations then differ by 0.28 and 0.35 in a variance. The check
asserts this rather than describing it, so the limit stays measured.

**Against a published analysis of real data.** Stopher et al. (2012, *Evolution*
66:2411) fitted animal models to four traits of wild red deer and then added a
matrix of home range overlap, and it deposits that matrix along with its
pedigree and phenotypes (Dryad `doi:10.5061/dryad.jf04r362`). Refitting all
four in `checks/against_red_deer.py` reproduces its Table 2: spring home range
heritability of 44.02% against a published 43.67, falling to 0.29% against 0.28
once overlap is in the model, and the rut's 31.29% against 31.31 falling to
0.00% against 0.11. The likelihood ratios for adding overlap come to 1300.4 and
771.3 against 1313.2 and 785.8.

**Six of the eight fits agree to within 0.006 on every variance component. Birth
weight is the exception and it is worth naming rather than averaging away**: it
differs by up to 0.007 without overlap and by 0.038 with it, the largest being
the overlap variance itself at 0.126 against a published 0.088. That trait is
the one the paper itself flags as unstable, and every difference is inside one
published standard error, but "no component differs by more than 0.006" is a
claim about the other three traits and this document used to make it about all
four. This is the only external check the component family has against
matrices it did not build, on covariance that crosses families throughout —
which is exactly what the SOLAR comparison above cannot reach.

The check verifies its own two undocumented steps. The deposited overlap
matrices carry no identifier file; reading their indices as the pedigree's row
order yields exactly the 948 spring and 766 rut females the paper reports. And
1,520 of the 4,051 deer have a known mother and no father, so each unknown
father is given its own founder identity — a construction that agrees with a
longhand tabular recursion to `0.000e+00` and recovers 339 inbred animals.

**The reported convergence flag is not trustworthy on an ill-conditioned
problem, and that comparison is how we know.** Three of its eight fits report
`converged: false` while landing within 0.006 of the published values. The
cause is that two different criteria are in play: the search stops on `factr`,
a relative change in the objective, while the flag is decided afterwards on the
projected gradient scaled by the log-likelihood against a fixed `1e-7`. In a
long flat valley the objective settles long before the gradient does, so the
estimate is right and the flag is wrong. It is not a size effect — the largest
problem of the eight, spring home range at 4,945 records, converges at
`1.43e-08` in one of its two models. The failures run `1.16e-07`, `4.62e-07`
and `1.45e-06`; the five successes run `2.41e-09` to `1.50e-08`, so the two
groups are cleanly separated rather than straddling the threshold. This is now reconciled, at one threshold shared by every
model family, and `converged` can be read again.

The fit now also reports `stop_code` and `stop_message`, which are the search's
own reason for stopping rather than ours, and they turn that account from an
inference into a measurement. **All eight fits stop the same way — `factr`
fires — and not one stops on `pgtol` or runs out of iterations:**

    CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH

So the gradient the flag tests is never a criterion the search pursues, in any
of the eight. The five that pass do so because their gradient happened already
to be small when the objective settled, not because anything drove it there.
That is worth stating plainly: on this family the flag reports a property of the
valley's shape rather than a property of the search. It also identifies the
remedy exactly, since `factr` is the only thing stopping the three short.

**One threshold, `1e-6`, in every family.** It was seven different numbers
spanning a factor of a hundred, none with a recorded reason. Measured, the
families do not differ: across 63 real GOBS fits every one reaches about
`1e-08` when allowed to, the reachable floors spanning `7.68e-09` to
`2.69e-08`. What separated them in what the flag saw was `factr`, which
inflates the reported gradient nearly thirtyfold in the spatial model and under
twice in the component model.

`1e-6` passes every fit measured here, the worst case anywhere being the deer's
rut home range with overlap at `1.578e-07`. `1e-7` fails that fit outright and
buys nothing: starving the search on purpose produced 165 fits, 92 of them
genuinely short of the optimum, and exactly one landed between `1e-7` and
`1e-6` — its estimate wrong by a millionth of a variance share.

**That starving measurement also says what the number means.** The worst error
in a variance share runs about ten times the reported gradient, and it holds
across six orders of magnitude; every starved fit above `1e-5` was caught. So a
fit reporting `1e-6` is right to about `1e-5` in any share it quotes, and one
reporting `1e-3` may be wrong in the second decimal place. The latent mediation
model keeps its own `1e-5` deliberately, because there the number accepts or
discards a start rather than reporting a flag.

**So where the gradient test fails, the fit now searches once more from the
point already found with `factr` switched off, and reports `polished: true`.**
It fires nowhere else, so every fit that passed before is untouched to the last
bit — measured, not asserted: the five unpolished deer fits reproduce the
unpolished run to `0.000e+00` on every component.

| trait | model | gradient before | gradient after | converged |
| --- | --- | --- | --- | --- |
| rhr | no spatial | 1.453e-06 | 4.051e-08 | no → **yes** |
| rhr | with overlap | 4.620e-07 | 1.578e-07 | no → no |
| shr | with overlap | 1.162e-07 | 8.864e-08 | no → **yes** |

**And the second search says something the first could not.** All three end the
same way:

    ERROR: ABNORMAL_TERMINATION_IN_LNSRCH

which is L-BFGS-B reporting that its line search can no longer make progress in
double precision. Not one reaches `pgtol`. So the remaining failure is not a
budget that ran out or a search that gave up early — 1.578e-07 is near the
numerical floor for that problem, and a `1e-7` threshold is asking for slightly
more precision than the likelihood affords there. Because the returned point is
checked rather than trusted — kept only if the objective is no worse and the
projected gradient strictly better — an abnormal termination costs the fit
nothing; it either improves or is discarded. A `polished: true` beside a
`stop_code` of 52 is that, and is expected rather than alarming.

For additive, household, and residual covariance at `n=400`, 95% coverage of
the additive and household mean-diagonal proportions was 0.945 and 0.955. A
zero-household test rejected 0.051 at nominal 0.05, and power for a household
proportion of 0.2 was 0.90.

These proportions are meaningful in that simulation because all three bases
have positive unit mean diagonal. The off-diagonal kinship-class bases instead
have zero diagonal and must be interpreted through coefficients and contrasts.
On the real GOBS pedigree, the class-equality omnibus rejected 0.035 at nominal
0.05 under equality and 1.0 when one class coefficient was increased by 0.5.
Class contrasts were near nominal under equality; they should be read after the
omnibus test because their constrained parameterisation couples the classes.

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

**This is the question a split matrix asks, and `test` is not it.**
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

**This is what `equality_test` cannot give.** The omnibus says the
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

