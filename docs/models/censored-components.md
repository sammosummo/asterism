# Censored model, several components

Analysis id `one_trait_censored_components`. Runnable example in [`examples/`](../../examples/).

**Not qualified.** The entry in `release.toml` is marked `planned_for_0_2`, and
`run_analysis` refuses it. The checks behind that qualification are being
written; what is measured so far is below.

## Using it

The censored model with more than one variance component. The complete example
is [`examples/censored_components.py`](../../examples/censored_components.py):

```sh
python examples/censored_components.py
```

```python
model = asterism.CensoredComponentModel([relationship, listener], design, limit)
fit = model.fit(y)
interval = model.interval(y, component=0)
test = model.test(y, component=1)
```

The components are an ordinary list, so the one-component case is this model
with a list of length one, not a separate model. Each frequency is fitted on its
own; nothing here combines them.

`test` is a supported quantity, and its measured level is recorded beside it in
`release.toml` rather than left to be assumed: 0.055 against a nominal 0.05 at
half the rows censored, and 0.100 at three quarters. See [At the design it will
be used on](#at-the-design-it-will-be-used-on) for what that second number is
and is not.

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

The residual is added for you and is never passed, so the coefficients are
shares of the total and the residual takes what they leave.

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
separates it from the additive component.

## Validation

An additive genetic component and a person-level component are both 1 within a
person and differ only through relatives, so whether they can be told apart is a
property of the design rather than of the code. Sweeping families and records
per person, with truths of 0.4 additive and 0.3 person-level:

| families | records | additive | person-level | correlation | split costs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 2 | 0.358 (0.251) | 0.327 (0.248) | −0.935 | 2.8x |
| 240 | 2 | 0.398 (0.115) | 0.303 (0.115) | −0.979 | 4.8x |
| 240 | 3 | 0.385 (0.107) | 0.313 (0.104) | −0.980 | 5.0x |

They do separate: each is recovered, and its spread falls as the design grows.
The cost is that splitting the two is about five times less precise than their
sum, so a design sized on the total variance is not sized on the split. The
estimates correlate near −1 at the larger designs, which is the sum becoming
certain while the split stays open, not the fit degrading.

At 30 and 60 families the estimates are still biased towards each other. That is
what a maximum-likelihood variance component does in a small sample, not a
fault; the bias is gone by the largest design swept.

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
| 0.50 | 6.3e-03 | 5.8e-02 |

The proportions are what the science reads. The log likelihoods agree less
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

2,400 people in villages of six, two records each, 0.52 censored, 300
replicates per cell, with an interval taken for **every** structured component
rather than only the first. Every scoreable cell covered its truth within a band
drawn simultaneously across the twelve, from 0.9567 to 0.9933 against a nominal
0.975.

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
three. Eight hundred attempts, eight hundred measured, no refusal and no failed
fit.

| scenario | censored | rate | exact interval | nominal | |
| --- | ---: | ---: | :---: | ---: | :--- |
| null | 0.52 | 0.055 | [0.028, 0.096] | 0.05 | held |
| null | 0.75 | **0.100** | [0.062, 0.150] | 0.05 | **failed** |
| heritable | 0.52 | 0.985 | [0.957, 0.997] | 0.975 | held |
| heritable | 0.75 | 0.960 | [0.923, 0.983] | 0.975 | held |

**The boundary test's level is 0.100 at three quarters censored, against a
nominal 0.05.** It holds at 0.052 censored. Point estimates and intervals held
at both shares.

**Three explanations were tested. Two are refuted and the third is measured.**

*Not a small sample.* Two copies of the pedigree side by side — 3,818 people,
7,636 rows — give 0.105 with an exact interval of [0.066, 0.156]. Doubling the
data moved nothing, where a finite-sample failure of an asymptotic reference
would have shrunk.

*Not the approximated region.* At a design whose censored blocks are small
enough to integrate exactly — 100 families of four, blocks of eight rows, three
components, the same three quarters censored — the level is 0.075 from
Asterism's own test, 0.065 from its statistic, and 0.060 from the same statistic
recomputed with the region **integrated** rather than approximated. All three
contain the nominal. The approximation error does fail to cancel between the two
fits, by 0.069 log units, but it makes the statistic *smaller*: 0.338 against
0.475. It is conservative here, so it cannot be what inflates the level.

*Not the size of the censored block.* Holding the roster at 1,920 people and
sweeping family size from 4 to 128 — censored blocks from 8 rows to 227 — the
level does not trend: 0.085, 0.090, 0.100, 0.050, 0.050, 0.060.

*What is measured.* The share of null fits whose estimate lands exactly on the
bound. The 50:50 mixture is the right reference only if that share is a half. It
is 0.565 in the released one-component configuration, which is calibrated. It is
0.24 to 0.40 across every several-component cell above, and 0.310 at the target
design. Self–Liang's half-and-half holds for one variance component at a bound
with interior, uncorrelated nuisance components; the additive and person-level
components here are correlated at about −0.98, which is what
[`checks/component_separation.py`](../../checks/component_separation.py)
measures from the other side. A boundary mass that is not a half makes a p-value
computed as though it were a half too small.

*Not the optimiser stopping short.* The three-component model nests the
two-component one, so a free fit can never score worse than the null fit unless
the search failed. Across 200 null replicates at the target design that happened
once, by 0.009 log units. Of the 126 replicates whose estimate came back
positive, every one bought real likelihood; none was a positive number sitting
at no gain. So the boundary mass is a property of the surface being fitted and
not a numerical fault.

**What remains is the reference, and it is not settled.** A 50:50 mixture is
right when one variance component sits at a bound and the rest are interior,
which is the situation here — so a measured mass of 0.37 says one of the
conditions behind that result does not hold at this design. The likeliest
candidate is the information for the tested component going nearly singular
under heavy censoring, since the additive matrix differs from the person-level
one only between relatives, and at three quarters censored few pairs of
relatives are both measured. That is a hypothesis. It has not been tested, and
the previous three hypotheses in this section were each refuted by the
measurement that followed them.

What would settle the level regardless of which reading is right is a reference
that assumes nothing about the mass — fit under the null, simulate from that
fit, and read the p-value from the statistic's own distribution. At this design
that costs hours for a single test, which is why it is a decision about the
software and not a patch.

Read a p-value here knowing its level at your censoring, which is what
`measured_levels` in `release.toml` records. [ADR
0020](../adr/0020-no-design-range-gates-a-result.md) settled that no design
range gates a result; a measured level is a fact to report beside the number,
not a gate in front of it.

The check is
[`checks/censored_components_target_design.py`](../../checks/censored_components_target_design.py).

### The largest censored region that has been qualified

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

**Fits whose largest censored block exceeds 600 are not supported.** Nothing
says they are wrong; nothing has measured them. The target-design campaign above
produced 310 at its heaviest, so the design this model was built for sits inside
the evidence with room to spare. A roster's block is its pedigree component, not
its size: 1,909 people made blocks of at most 360 rows because the largest
family has 180 people.

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
several components. The coverage simulation that scored the boundary
rule ran at one, and an absent verdict means nobody has measured it.

### `CensoredComponentModel.test(self, value: 'Any', censoring: 'Any', limit: 'Any', component: 'int') -> 'dict[str, Any]'`

Test that one component's coefficient is nought.

A proportion is nought exactly when its coefficient is, so this answers
both questions.

``nuisance_at_bound`` says whether another component or the residual
also rested on nought. Where it is true, ``rule`` is not the reference
the p-value should be read against: the fifty-fifty mixture answers for
one parameter on one bound with the rest inside.

