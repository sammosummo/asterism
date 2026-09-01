# Censored model, several components

Analysis id `one_trait_censored_components`. Runnable example in [`examples/`](../../examples/).

**Current 0.2 scope, not a released artefact.** Fitting, intervals and the
explicitly asymptotic test are in scope; the constrained-null bootstrap remains
experimental. The analytic p-value must not be reported for the exact failed
1,909-person, four-component fixed-instrument target at 75% expected censoring
without a design-specific simulated null. This is not a universal censoring
threshold, and fit, interval and the released one-component test are unaffected.
Conditional implementation comparisons and superseded investigation results
are separated below.

## Using it

The censored model with more than one variance component. The complete example
is [`examples/censored_components.py`](../../examples/censored_components.py):

```sh
python examples/censored_components.py
```

```python
model = asterism.CensoredComponentModel([relationship, listener], design)
fit = model.fit(value, censoring, limit)
interval = model.interval(value, censoring, limit, component=0)
test = model.test(value, censoring, limit, component=1)

# Optional experimental finite-sample reference:
censoring_direction = np.ones_like(censoring)  # right censoring for every row
bootstrap = model.bootstrap(
    value,
    censoring,
    limit,
    censoring_direction,
    component=1,
    replicates=999,
    seed=4117,
)
```

The components are an ordinary list, so the one-component case is this model
with a list of length one, not a separate model. Each frequency is fitted on its
own; nothing here combines them.

`test` labels its reference `asymptotic_mixture_50_50` and refuses if another
variance rests on its bound. `bootstrap` needs a finite instrument limit and
censoring direction for every row, including rows measured in this sample,
because a null draw may cross any of them. The analytic route is explicitly
asymptotic rather than a finite-sample guarantee, and the bootstrap remains
experimental; see [At the design it will be used on](#at-the-design-it-will-be-used-on).

## The model

For $q$ submitted structured components, the latent trait has covariance

$$
\mathbf V(\boldsymbol\theta)
=\sum_{k=1}^{q}\theta_k\mathbf K_k+\theta_e\mathbf I_n,
\qquad \theta_1,\ldots,\theta_q,\theta_e\geq0,
$$

the same sum the [Gaussian several-component model](several-components.md)
fits. What differs is the response. An instrument that stops at a limit gives
$y_i=\min(z_i,\ell_i)$, so a measured record contributes its density and a
censored one contributes the probability that the latent value lies at or
beyond its limit. The likelihood factorises over blocks of $\mathbf V$, not of
any single component, because it is the sum that must be block diagonal.

The residual is added for you and is never passed. The raw coefficients are
shares of the total only when every component has unit mean diagonal; otherwise
use the reported mean-diagonal proportions to compare variance contributions.
The residual is the last proportion and takes what the structured components
leave.

**Maximum likelihood, never REML.** A censored record has no residual to
project onto the null space of the design, so the REML correction is not
available here. A heritability from this model must not be placed beside a REML
one as though the two were the same quantity.

**The components decide the blocks, so dropping one changes the likelihood by
more than the component it dropped.** The likelihood factorises over blocks of
the covariance, and those blocks are computed from the components you submit.
Take the additive matrix out of `[additive, person, household]` and nothing
links relatives any more, so the remaining pair factorises over households where
the three factorised over families. The region probability is reached per block
by sequential truncation, so a coarser or finer partition carries a different
approximation: fitting `[person, household]` and fitting
`[additive, person, household]` with the additive coefficient held at nought are
the same model on paper and differed here by 0.05 log units. Neither is wrong.
They are the same likelihood approximated over different blocks, and the
difference is not a sign that either search failed.

Identification is a property of the submitted components on the fitted roster,
not of their names. Two components that are linearly dependent there cannot be
told apart however they are labelled: a household that is exactly a sibling pair
is $2\mathbf A-\mathbf I$, which is the pedigree, and no amount of data
separates it from the additive component. The several-component builder refuses
such a rank-deficient basis set, including an explicit identity matrix that
duplicates the residual already added by the model.

## Validation

An additive genetic component and a person-level component are both 1 within a
person and differ only through relatives, so whether they can be told apart is a
property of the design rather than of the code. The fixed-instrument campaign
completed 200 replicates in each of eight cells: two or three records per person
and 30, 60, 120 or 240 sibling pairs. Every cell fitted. At 240 pairs:

| records | mean additive | mean person-level | truths |
| ---: | ---: | ---: | :--- |
| 2 | 0.3984 | 0.3028 | 0.4, 0.3 |
| 3 | 0.3858 | 0.3132 | 0.4, 0.3 |

Both largest-design cells passed the predeclared 0.03 bias rule. The two
estimates remain strongly negatively correlated, but their precision and
separation improve as the design grows.

The check is [`checks/component_separation.py`](../../checks/component_separation.py).

### Against an independent fit

SciPy optimises a censored likelihood assembled separately from its definition,
sharing no likelihood, parameterisation, starting values or optimiser with this
model, and reaching the censored region probability by integrating it rather
than by sequential truncation. Twenty sibling pairs at two records each, twelve
replicates at each censoring share:

| censored | proportions agree to | log likelihoods agree to |
| ---: | ---: | ---: |
| 0.25 | 1.4e-03 | 7.0e-02 |
| 0.50 | 2.9e-03 | 5.5e-02 |

The instrument limits in this comparison were fixed before outcomes were
drawn. The proportions are what the science reads. The log likelihoods agree less
closely and are not expected to: **the gap between them is the
sequential-truncation approximation**, measured rather than assumed. The check
also scores this model's own answer through the independent likelihood, and
wherever the two disagree the independent optimum is the better point — so the
reference is not stopping short, and what separates them is the approximation
moving the optimum. It wins by about 1e-03, because the likelihood is nearly
flat along the split between components.

The check is
[`checks/censored_components_against_independent_full_fit.py`](../../checks/censored_components_against_independent_full_fit.py).

### Interval coverage

The historical campaign used 2,400 people in villages of six, two records each,
and forced 0.52 censoring by selecting the limit from each realised outcome.
Its recorded coverage band of 0.9567 to 0.9933 against a nominal 0.975 is
superseded. The corrected fixed-instrument campaign completed 300 replicates in
each of twelve component-by-truth cells with no refusal. Coverage in the eleven
interior cells ranged from 0.9533 to 0.9933, and every simultaneous exact
interval contained 0.975. The lower-bound cell retained the deliberately absent
boundary verdict in 289 cases and excluded nought in eleven; its predeclared
rule did not establish undercoverage.

**The nominal is 0.975 and not 0.95.** A component proportion of one puts every
other component and the residual at nought, leaving a covariance in which a
person's two records are perfectly correlated. It is singular, the profile
cannot be evaluated at the upper bound, and the interval covers that end rather
than placing it on evidence never gathered. So the upper end is the bound every
time — measured at one, two and three components alike, so it is the repeated
records and not the component count. A two-sided 95 per cent interval puts 0.025
in each tail, and closing the upper one off hands that back.

A truth sitting exactly on a component's bound is reported as a range rather
than a number, because containment splits in two there and the Self–Liang
verdict is deliberately withheld at several components; see
[ADR 0022](../adr/0022-the-censored-model-generalises-in-place.md).

The check is
[`checks/censored_components_coverage.py`](../../checks/censored_components_coverage.py).

### At the design it will be used on

1,909 people, 202 relationship components with the largest at 180, six
fixed-effect columns, two records each and families cut into households of
three. The original 800-attempt campaign completed without a refused fit, but
it selected each limit from the outcome it then analysed. The table is retained
as investigation history and does not qualify point estimates, intervals or
analytic p-values for a fixed instrument:

| scenario | censored | rate | exact interval | nominal | |
| --- | ---: | ---: | :---: | ---: | :--- |
| null | 0.52 | 0.055 | [0.028, 0.096] | 0.05 | held |
| null | 0.75 | **0.100** | [0.062, 0.150] | 0.05 | **failed** |
| heritable | 0.52 | 0.985 | [0.957, 0.997] | 0.975 | held |
| heritable | 0.75 | 0.960 | [0.923, 0.983] | 0.975 | held |

The reproducible point mass in the null likelihood-ratio statistic is useful
diagnostically, but **one half is not a finite-sample acceptance target**.
Self–Liang's 50:50 law is an asymptotic tangent-cone result; exact results for
Gaussian mixed models show design-dependent finite-sample atoms, and the
positive part of the distribution can move as well. One retained cell made the
distinction concrete: 0.325 of its statistics were in the operational atom while
its rejection rate was exactly 0.050. Gating on the atom would reject a
calibrated test.

The investigation also ruled out the code change in `e4bd51f`: on the exact 40
affected target seeds, the old and new held-fit searches selected the same 12
point-mass statistics and the same 7 rejections. Their log likelihoods differed
by at most 1.85e-09. Choosing each simulation limit from its realised outcome
was a separate harness error; replacing it with a limit fixed before the draw
left the family-four rejection rate and atom essentially unchanged, so it was
not the source of the p-value failure.

The public several-component `test` reports
`asymptotic_mixture_50_50`, distinguishing the asymptotic Self–Liang
approximation from the released one-component compatibility label. It refuses
if another component or the residual rests on its bound. The fixed-instrument
target completed all 800 attempts. Null rejection was 0.055 at 52% censoring
and 0.095 at 75%; the latter cell failed because its one-sided exact lower bound
was 0.0631, above 0.05. The descriptive two-sided interval began at 0.0582.
Coverage was 0.980 and 0.960, and the operational LRT atoms were 0.465 and
0.390. The failed cell is the exact 1,909-person, four-component target at 75%
expected censoring. Its analytic p-value must not be reported without a
design-specific simulated null tied to the model, design, source and seed. It
does not define a censoring threshold for another design.

The optional 999-draw constrained-null bootstrap target also completed all 800
requested coordinates, but eleven attempts in the 75% null cell refused after
an inner bootstrap fit failed. The other cell rates passed their statistical
rules, but the campaign permits no failed attempt, so its no-write merge failed
and no evidence was published. The bootstrap therefore remains experimental;
it has not qualified a replacement p-value. Fit and interval evidence, the
generally available labelled asymptotic test, and the released one-component
numerical and analytic routes are unchanged.

The check is
[`checks/censored_components_target_design.py`](../../checks/censored_components_target_design.py).

### The largest censored region that has been measured

The likelihood factorises over blocks of the covariance, and within a block the
probability that the censored rows lie beyond their limits is reached by
sequential truncation. How far that approximation can be pushed is measured by
[`checks/sequential_against_ghk.py`](../../checks/sequential_against_ghk.py)
against a GHK reference, climbed under this model's own design — one frequency
at a time, two ears per person, a household kernel beside the genetic term.

**600 censored rows in one block is the largest rung climbed, and it is not a
limit that was found**: the ladder ran out before the approximation did. At 600
the error is 0.053 of what a heritability step of 0.1 does to the same
quantity, against an allowance of 0.25.

Blocks above 600 have not been measured by this comparison. That is a warning
about the available evidence, not a support boundary or a runtime refusal. The
corrected fixed-instrument target campaigns produced a largest censored block of
313, inside the measured ladder. A roster's block is its pedigree component,
not its size: 1,909 people make structural blocks of at most 360 rows because
the largest family has 180 people.

<!-- API: generated by tools/build_model_pages.py -->

## Interface

Generated from the package's own docstrings. The complete public surface is in the [API reference](../api-reference.md).

### `CensoredComponentModel.fit(self, value: 'Any', censoring: 'Any', limit: 'Any') -> 'dict[str, Any]'`

Fit, and return each component's coefficient and its proportion.

``censoring`` is 0 where the value was measured, 1 where it lies at or
above its limit, and 2 where it lies at or below it. The status is given
rather than inferred, because a censored value can carry the same number
as a measured one. ``value`` is read only where the status says
measured, and ``limit`` only where it does not.

``coefficients`` are shares of the total variance where every component
carries a unit diagonal, and raw coefficients otherwise.
``mean_diagonal_proportions`` is the comparable quantity, the residual's
last, and is absent where a component's mean diagonal is not positive
and finite.

**There is no ``heritability`` key**, and its absence is deliberate.
With one component it would be ``coefficients[0]``; with several it is
the first component's coefficient whatever that component happens to be,
and it moves when a matrix is rescaled while the heritability does not —
multiplying the first matrix by four returned 0.141 where the
heritability was 0.396. Read ``mean_diagonal_proportions[0]`` when the
first component is additive kinship, knowing that you have decided it
is. :class:`ComponentModel` carries no such key either.

### `CensoredComponentModel.interval(self, value: 'Any', censoring: 'Any', limit: 'Any', component: 'int', quantity: 'str' = 'mean_diagonal_proportion') -> 'dict[str, Any]'`

A profile-likelihood interval for one component.

``quantity`` is ``"mean_diagonal_proportion"`` by default, which is the
comparable one and the one to report: rescaling a component's matrix
describes the same model and leaves it alone. ``"coefficient"`` gives
the interval on the raw coefficient, which that rescaling moves.

``contains_lower_bound`` and ``contains_upper_bound`` are absent at
several components. No valid fixed-instrument campaign has qualified a
boundary-containment rule there, so an absent verdict means nobody has
measured it.

### `CensoredComponentModel.test(self, value: 'Any', censoring: 'Any', limit: 'Any', component: 'int') -> 'dict[str, Any]'`

Test that one component's coefficient is nought.

A proportion is nought exactly when its coefficient is, so this answers
both questions.

With one component this is the released ``mixture_50_50`` calculation,
on its unchanged numerical path. With several components the requested
component is tested after moving it to the held search coordinate, and
``rule`` is ``"asymptotic_mixture_50_50"``. That reference assumes
every nuisance component and the residual are interior; the method
refuses as ``TOBIT_COMPONENT_TEST_NUISANCE_AT_BOUND`` otherwise.

The several-component reference is explicitly asymptotic rather than a
finite-sample guarantee. Its p-value must not be reported for the exact
failed 1,909-person, four-component target at 75% expected censoring
without a design-specific simulated null. This does not create a
universal censoring threshold. :meth:`bootstrap` supplies a simulated
constrained-null reference, but its target calibration failed the
zero-refusal rule and it remains experimental. Fit, interval and the
released one-component test are unaffected.

