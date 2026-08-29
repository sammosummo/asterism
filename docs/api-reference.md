# API reference

Generated from the package's own docstrings by
`tools/build_api_reference.py`. Do not edit it by hand: a test requires this
file to match what that script produces.

Support status for each object is in [api-support.md](api-support.md). Runnable
programs using them are in [`examples/`](../examples/).


## Models

### `AssociationModel`

Many markers, one at a time, each a fixed effect in a polygenic model.

The model is ``y = X0 b + m_j c + g + e``, where ``X0`` carries the
intercept, the ancestry components and any other covariate. The polygenic
term is what makes it worth doing: relatedness and population structure
inflate an association test, and the relationship matrix absorbs both.

The variance components are fitted once under the null and then held,
which is what makes a scan take minutes rather than hours — each marker
becomes a weighted least squares on rotated data. That is an assumption, not
a trick: it is good when no single marker explains much of the variance,
which is the situation a scan is in and exactly not the situation for a
marker of large effect. Pass ``variance="refitted"`` to refit under every
marker, which is around a hundred times slower.

Wald and the likelihood ratio are the same number when held. The profile
log likelihood in the fixed effects is exactly quadratic when the covariance
is known, so the likelihood ratio is the Wald statistic squared. Both come
back because both are asked for; they differ only under ``refitted``.

The marker under test is inside the relationship matrix. Leaving its
chromosome out is the usual answer and is not done, so every test is biased
towards the null.

#### `AssociationModel.sweep(self, markers: 'Any', variance: 'str' = 'held', refit_below: 'float | None' = None) -> 'dict[str, Any]'`

Test every column of ``markers``.

With ``refit_below`` set, the sweep runs held and then refits only the
markers whose held p-value falls under it — which is how a scan should
be run. Refitting everything costs about twenty times as much and the
two modes agree to two decimal places on the markers nobody cares about.

Set ``refit_below`` some way above the threshold you will report
against. Held is conservative — measured across four hundred real
markers it was never smaller than refitted, with the gap growing from a
ratio of 1.00 above p = 0.01 to 1.10 below 1e-06 — so a marker can have
a refitted p under your threshold while its held p sits above it.
Screening at exactly the threshold would miss it; ten times the
threshold is ample and costs almost nothing.

Each marker says whether it was ``refitted``, because a two-stage result
that does not is one nobody can check.

A marker that cannot be tested — one with no variation, or one leaving
the design rank deficient — comes back with a stable code in place of
its numbers rather than stopping the sweep.

### `AutoregressiveModel`

One trait with a separable first-order autoregressive effect on a grid.

Two people in cells ``(r_i, c_i)`` and ``(r_j, c_j)`` share
``sigma_s^2 * rho_row^|r_i-r_j| * rho_col^|c_i-c_j|``. This is the model
Stopher and colleagues fitted to the red deer of Rum, and the standard one
for field trials laid out in rows and columns.

It is not :class:`SpatialModel`. That one estimates the range of an
isotropic kernel in true distance; this one takes the cell size as given
and estimates only how fast correlation falls off per cell. The scale is
therefore a choice the caller makes, and the two rates mean nothing without
the cell size that produced them — so sweep the cell size rather than
reporting one grid's answer.

At both rates nought the kernel is the same-cell indicator, because
``00`` is one and ``0k`` is nought. So this model contains a plain
shared-cell random effect as a special case, and asking whether the rates
are nought asks whether smooth decay across neighbouring cells buys
anything over shared membership.

Parameters
----------
fixed
    The matrices whose variances are estimated but whose shape is given —
    a relationship matrix, a household matrix.
row, column
    Cell coordinates per person, as whole numbers of cells.
design
    The fixed-effect design.

#### `AutoregressiveModel.fit(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, and report the variances and the two rates.

### `BivariateModel`

Two traits fitted jointly, on possibly unbalanced data.

Parameters
----------
k
    The relationship matrix, one row per person.
observed
    One pair of booleans per person, saying which of the two traits they
    have. People missing either trait are the ordinary case.
design
    The fixed-effect design over the observed person–trait rows, in person
    order with trait within person.
subject_order_sha256
    Optional lowercase SHA-256 from ``subject_order_commitment`` for the
    exact person order. It is echoed on every fit record.

#### `BivariateModel.fit(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, and return both heritabilities and all three correlations.

The phenotypic correlation is derived from the others rather than
estimated, which is why it has no variance of its own.

#### `BivariateModel.interval(self, y: 'Any', quantity: 'str', reml: 'bool' = True) -> 'dict[str, Any]'`

A 95 per cent profile interval for one reported quantity.

#### `BivariateModel.test(self, y: 'Any', quantity: 'str', null: 'float' = 0.0, reml: 'bool' = True) -> 'dict[str, Any]'`

Test one correlation against a fixed value.

Against nought the value is interior and a plain chi-squared applies;
against plus or minus one it sits on a bound and the Self–Liang mixture
does. Heritabilities are not testable this way.

### `ComponentModel`

One trait with any number of variance components.

The residual is added for you and is always last, so a model built with a
relationship matrix and a household matrix reports three raw coefficient
proportions: additive, household, residual. When every structured matrix
has a positive finite mean diagonal, the fit also reports each coefficient
times its matrix's mean diagonal. These mean-diagonal contributions and
their proportions do not change if a matrix is multiplied by a positive
constant and its fitted coefficient changes reciprocally.

Parameters
----------
matrices
    The structured components, in the order you want them reported. Each
    is copied at construction, so later caller mutation cannot change the
    fitted model. Row alignment is positional and is the caller's
    responsibility.
x
    The fixed-effect design, one row per person, including its own
    intercept column if one is wanted. Its values are copied at
    construction.
subject_order_sha256
    Optional lowercase SHA-256 from ``subject_order_commitment`` for the
    exact fitted row order. It is echoed on every fit record.

#### `ComponentModel.contrasts(self, y: 'Any', classes: 'list[int] | None' = None, reml: 'bool' = True) -> 'list[dict[str, Any]]'`

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

#### `ComponentModel.equality_test(self, y: 'Any', components: 'list[int] | None' = None, reml: 'bool' = True) -> 'dict[str, Any]'`

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

#### `ComponentModel.fit(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, and return variance coefficients and their proportions.

``raw_coefficient_proportions`` depend on matrix scale. They are useful
for inspecting the optimiser parameterisation but are not generic
variance shares. When all structured matrices have positive finite mean
diagonals, ``mean_diagonal_proportions`` reports the corresponding
scale-invariant marginal contributions.

Matrix, design, and response alignment is positional and is the
caller's responsibility.

#### `ComponentModel.interval(self, y: 'Any', component: 'int', reml: 'bool' = True, quantity: 'str' = 'mean_diagonal_proportion') -> 'dict[str, Any]'`

A 95 per cent profile interval for one component proportion.

The default profiles the scale-invariant ``mean_diagonal_proportion``
reported by :meth:`fit`. The diagnostic
``raw_coefficient_proportion`` remains available explicitly. It changes
when a relationship matrix is rescaled, so it describes this
parameterisation rather than the data alone.

#### `ComponentModel.predict(self, y: 'Any', component: 'int', reml: 'bool' = True) -> 'dict[str, Any]'`

Predict the random effects of one component.

The best linear unbiased prediction, one value per person, with the
standard error of prediction beside it.

The error is how far the prediction may be from the effect, not the
spread of the predictions. A prediction is shrunk toward nought, so its
own spread is smaller than the effect's; the question worth answering is
how wrong it might be.

The variance components are treated as known, though they were estimated
from the same data, so the errors are a little optimistic. That is the
usual approximation and the same one the fixed effects make.

#### `ComponentModel.test(self, y: 'Any', component: 'int', reml: 'bool' = True) -> 'dict[str, Any]'`

Test one component against having no variance at all.

The null sits on the edge of the parameter space, so the reference is
the Self–Liang half-and-half mixture and not a plain chi-squared.

It refuses where another component has itself gone to nought, because
the mixture assumes only one is on the boundary; a number there would be
a p-value for a question nobody asked.

### `DiscreteGxeModel`

One trait whose genes may act differently in two environments.

This is the discrete case of genotype-by-environment: the environment is a
binary label rather than a measured range, so nothing is smoothed and no
surface has to be chosen. The model carries one genetic standard deviation
per environment, one residual standard deviation per environment, and one
genetic correlation between them. Sex is the canonical environment; any
other binary label — an exposure, a cohort, a diagnosis — works the same
way.

``environment`` is one label per person and must take exactly two distinct
finite values, compared exactly. Name them through ``levels`` if you can
— the column is then checked against what you expected, so a third value or
a missing code where a group should be is refused rather than fitted. Left
unnamed the two are inferred, and a negative one is refused, because ``-9``
is the missing code in every pedigree format and far likelier a sentinel
than a group. Nought is left alone, since a 0/1 exposure is ordinary; a
caller who genuinely means -1 and +1 says so through ``levels``. The people carrying the smaller label form
the first group everywhere in the results. A missing or unknown label
must be resolved or removed before building, because a model that quietly
puts the unknowns together is estimating a correlation with a third group
in it.

Two findings live here and they are not the same.

*The heritability differs between the environments.* The genetic variance
is larger in one than the other. That is a difference of scale, and a
difference of scale can come from the measurement rather than the genetics
— men are larger, so a volume in millimetres varies more in men whether or
not the genes differ. ``test(y, "genetic")``.

*The genes differ between the environments.* The genetic correlation across
the environments is below one, so the genes that matter in one are not
exactly those that matter in the other. No change of units can produce
this, and it is usually the interesting claim. ``test(y, "correlation")``.

Unlike the kernel surfaces in :class:`GxeModel`, the correlation here is a
parameter rather than a function of distance, so it is free to be negative:
a genotype raising a trait in one environment and lowering it in the other
is reachable.

The two residual standard deviations are free, and they should be. A
trait simply noisier in one environment would otherwise push its extra
variance into the genetic term, and the genetic tests would then reject
because of measurement rather than because of genes.

Read the headline test first. ``test(y, "gene_by_environment")`` puts
the genetic constraints back at once — same variance, same genes — while
leaving the two residual variances free. It is what stops several tests on
one trait being read as several findings.

Do not use ``test(y, "any_difference")`` as the headline. It ties
the residual variances too, so a trait merely measured more noisily in one
environment rejects it hard with nothing genetic happening. In simulation
on the GOBS pedigree a sex difference in measurement error alone rejected
it at p = 1e-34 while every genetic test correctly reported nothing.

``subject_order_sha256`` optionally carries the lowercase SHA-256 from
``subject_order_commitment`` for the exact fitted row order. Every fit
record echoes it without retaining identifiers.

#### `DiscreteGxeModel.correlation_interval(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

A 95 per cent profile interval for the genetic correlation.

The correlation is a parameter here rather than a function of one, so
the interval comes from pinning it and refitting everything else, with
endpoints where twice the drop in log likelihood reaches 3.8415.

The reference is the ordinary chi-square on one degree of freedom,
not the mixture ``test(y, "correlation")`` uses. That test asks
about a correlation of exactly one, which is the edge of the parameter
space; an interval is a statement about interior values and takes the
interior reference. Borrowing the test's mixture would give a narrower
interval than the coverage it claims.

``lower_limited`` and ``upper_limited`` say whether an endpoint sat at
the edge of what a correlation may be rather than where the likelihood
fell away. An interval reaching a bound covers without pinning the
value down.

#### `DiscreteGxeModel.fit(self, y: 'Any', reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, with everything free.

``counts`` comes back with the answer because a correlation estimated
across a group of thirty is not the same claim as one across a thousand,
and the fit itself cannot tell you which you have.

#### `DiscreteGxeModel.test(self, y: 'Any', null: 'str' = 'gene_by_environment', reml: 'bool' = True) -> 'dict[str, Any]'`

Test one of the five nulls.

- ``"gene_by_environment"``: no genetic difference of any kind — the
  same variance and the same genes in both environments — with the two
  residual variances left free. Two constraints, one of which sits on a
  bound, so the reference is an even mixture of chi-square on one and
  on two degrees of freedom. Read this one first.
- ``"any_difference"``: nothing differs between the environments at
  all, residual included. Three constraints on an even mixture of
  chi-square on two and on three. It is not a genetic test: a
  noisier environment rejects it.
- ``"correlation"``: the same genes act in both environments. This is
  the gene-by-environment question proper. The null puts the
  correlation at the edge of what it may be, so the reference is the
  even mixture of a point mass at nought with chi-square on one degree
  of freedom. A plain chi-square would roughly double the p-value.
- ``"genetic"``: the same genetic variance in both environments.
  Interior, so chi-square on one degree of freedom.
- ``"residual"``: the same residual variance in both environments.
  Report it beside the others as a measurement fact, not as a genetic
  finding.

``rule`` names the reference distribution the p-value is a tail of, so a
reader need not take it on trust.

### `GxeModel`

One trait whose genetic effects may act differently across an environment.

The genetic covariance between two people becomes their relationship times a
surface in their two environments, and the residual variance a surface in
one. Two surfaces are available and they are given the same data and asked
the same questions:

- ``"exponential"``: the genetic and residual variances are log-linear in
  the environment, and genetic effects a distance apart in the environment
  correlate as ``exp(-λ|Δ|)``. Five parameters.
- ``"random_regression"``: a smooth quadratic surface on each covariance,
  held by loadings so it stays a covariance. Six parameters.
- ``"powered_exponential"``: the exponential with the decay taken to a
  fixed power, ``exp(-λ|Δ|^κ)``. The same five free parameters, with
  ``shape`` chosen from 0.5, 1.0, 1.5 or 2.0 — chosen and not fitted,
  because a shape and a decay rate trade off against each other and a search
  over both wanders. At ``shape=1.0`` it is the exponential surface exactly.

The shape is barely identified and strongly changes the answer. On one
simulated set the four shapes spanned 0.31 in log likelihood — a deviance of
0.62, which is nothing — while the genetic correlation they reported ran
from 0.92 to 0.45. A shape chosen to suit the answer would be invisible in
the fit, so choose it for a reason outside the data and report which one was
used beside the result.

Nothing is reported in either surface's own coordinates. What comes back is
the heritability at each environment you ask about and the genetic
correlation between each pair — quantities that mean the same thing whichever
surface produced them.

Only the random-regression surface can represent a crossover — a genotype that
helps in one environment and harms in another, so the genetic correlation
falls below nought rather than merely below one. The exponential surface
correlates two environments as ``exp(-λ|Δ|)``, which is positive at every
rate, so it will report a correlation near nought where the truth is near
minus one. If a crossover is on the table, the choice is already made.

Choose the surface before looking at the answer, and know what it costs to
choose wrongly. Neither family contains the other, and a rank-one genetic
surface from one is misspecified for the other. The misfit goes into the one
parameter that is free under the alternative and pinned under the null, so a
surface that cannot bend its variance function the way the data does will
bend its correlation instead. In calibration over 2000 samples with a linear rank-one genetic
surface and no reordering at all, the exponential surface rejected
``test(y, "correlation")`` on 10.7 per cent of them against a nominal 5. The
random-regression surface was conservative rather than anti-conservative in the mirror
case, which is why it is the default.
Fitting both and reporting whichever rejects is not a defensible procedure.

A correlation below one is not by itself evidence of an interaction. The
estimate cannot exceed one, so under the null every departure runs downward:
on the exponential surface a tenth of null samples came back below 0.25. Use
``test``.

``subject_order_sha256`` optionally carries the lowercase SHA-256 from
``subject_order_commitment`` for the exact fitted row order. Every fit
record echoes it without retaining identifiers.

#### `GxeModel.fit(self, y: 'Any', grid: 'Any' = (-1.0, 0.0, 1.0), reml: 'bool' = True) -> 'dict[str, Any]'`

Fit, and report the surface at the environments in ``grid``.

``grid`` is in the environment's own units, so it should be chosen from
the data — quantiles of the observed environment usually. The genetic
correlations come back as a square list of lists in the grid's order.

#### `GxeModel.interval(self, y: 'Any', quantity: 'str' = 'heritability', first: 'float' = 0.0, second: 'float' = 0.0, reml: 'bool' = True) -> 'dict[str, Any]'`

A 95 per cent profile-likelihood interval for a reported quantity.

``"heritability"`` uses ``first`` as the environment; ``"correlation"``
uses both, and is the genetic correlation between them.

The quantity is held by solving one coordinate of the surface for it and
re-maximising over the rest, so the interval is a likelihood one and not
a Wald one — it is not symmetric about the estimate and does not have to
be. ``lower_limited`` or ``upper_limited`` says the endpoint ran to
the edge of what the quantity can be rather than to a likelihood
crossing, which is a limit of the model rather than a measurement.

#### `GxeModel.test(self, y: 'Any', null: 'str' = 'correlation', reml: 'bool' = True) -> 'dict[str, Any]'`

Test one of the two genotype-by-environment nulls.

- ``"correlation"``: the genetic effects at any two environments are the
  same effects. The genetic variance may still change; what is ruled out
  is a change in *which* genes matter. This is the narrower claim and
  usually the interesting one — a heritability that rises with an
  environment can follow from a change of scale in the measurement, and a
  correlation below one cannot.
- ``"interaction"``: the genetic covariance does not involve the
  environment at all. Rejecting says something about genes and
  environment together, but not what.
- ``"variance"``: the genetic variance does not change with the
  environment. This is the recovered SOLAR model's own ``gamma_G = 0``
  null, and on the exponential surface it is one interior coordinate
  referred to chi-square on one degree of freedom. On the random-regression
  surface it is not a separate test — a quadratic genetic variance is
  constant only when both its shape coordinates are nought, which is
  the interaction null — so it returns that instead of a differently
  named copy.

The residual surface is free under both nulls, so a residual variance
that changes with the environment is not mistaken for a genetic one.

### `LatentMediationModel`

A collection of independent families for evaluation and fitting.

#### `LatentMediationModel.evaluate(self, *, a: 'float', b: 'float', c_prime: 'float', d: 'float', sigma_m2: 'float') -> 'dict[str, Any]'`

Evaluate the Rust likelihood at one structural-parameter point.

#### `LatentMediationModel.fit(self) -> 'dict[str, Any]'`

Fit the five structural parameters with a fixed numerical recipe.

Rust owns parameter scaling, bounds, deterministic starts, optimisation,
convergence checks, integration diagnostics, and record construction.

#### `LatentMediationModel.horizontal_set(self, *, searched_to: 'float' = 4.0) -> 'dict[str, Any]'`

A 97.5 per cent confidence set for the horizontal estimand.

Built by inverting the same likelihood ratio :meth:`test_horizontal`
computes, at chi-square on one at 0.975 because that is what a
two-sided set at the Bonferroni .025 gives.

An end that did not close is ``None``, not a number. Reporting the
edge of the search would be a statement about how far the search went
rather than about the data. ``unbounded`` is true when neither end
closed, and that is the identification diagnostic in its most useful
form: a profile that falls away in neither direction is what a direct
path the data cannot locate looks like from the data's side. It sees
the near-boundary case that :meth:`test_horizontal` cannot, because
that refuses only when the loading rests exactly on its bound.

``searched_to`` is the distance either side of the estimate that is
searched. Widening it costs fits and can only close an end that a
narrower search left open.

#### `LatentMediationModel.test_horizontal(self) -> 'dict[str, Any]'`

Test the horizontal estimand ``c_prime`` against nought.

The null is a point, not a union, which is what makes this the
simpler of the two tests. The direct inherited effect is a single
signed coordinate -- an effect outside measured hearing may run either
way -- so it is interior, and the ordinary chi-square on one degree of
freedom is the whole of the reference. There is no boundary mixture to
choose and no simulated reference, so this costs two fits rather than
the few hundred the vertical test can cost.

It refuses where the mediator loading is at nought. There the
inherited covariance carries the direct path and the outcome loading
only as ``c'^2 + d^2``, the two rotate freely against each other, and a
p-value would report which of the pair the optimiser happened to pick.
Both the free and the held fit are checked, because the loading can be
positive when free and fall to its bound once the direct path is held.

A refusal here is the answer, not a gap: a horizontal component that
fails its identification diagnostics is not separately estimable, which
is a different statement from its being nought.

#### `LatentMediationModel.test_vertical(self, *, bootstrap_replicates: 'int' = 200) -> 'dict[str, Any]'`

Test the vertical estimand ``a * b`` against nought.

The null is a union, not a point. ``a * b = 0`` holds whenever the
mediator carries no inherited signal (``a = 0``) *or* the mediator does
not reach the outcome (``b = 0``), and those are different models. One
likelihood ratio has no reference distribution across a union, which is
why this model reported point estimates and no p-value until now.

The construction is the intersection-union test: reject the union only
when both parts are rejected, so the p-value is the larger of the two.
That is exactly level ``alpha``, at the cost of being conservative --
most so near ``a = b = 0``, where both parts are true at once.

Each part carries its own reference. ``a`` is bounded below at nought,
so its null sits on a boundary and takes the even mixture of a point
mass with chi-square on one; ``b`` is signed and interior, so it takes
an ordinary chi-square on one. Both are reported beside the combined
p-value, because which of them binds says what the data could not
establish.

This tests the mediated path, not whether mediation is the right
account. A trait and a mediator sharing inherited causes will reject
this null with nothing being mediated, and no likelihood separates
those two stories.

#### `LatentMediationModel.vertical_set(self, *, searched_to: 'float' = 1.0, bootstrap_replicates: 'int' = 200) -> 'dict[str, Any]'`

A 97.5 per cent confidence set for the vertical estimand.

Nought is decided differently from everywhere else, and has to be.
Away from nought, holding the estimand is one constraint on a curve --
the estimand is a product, so a held value fixes the path at the value
over the loading -- and the likelihood ratio has an ordinary
chi-square reference. At nought the null is a union, the loading is
nought or the path is, and no single ratio spans it, so membership
there comes from the intersection-union test instead.

Read ``disjoint`` before treating ``lower`` and ``upper`` as an
interval. The profile can admit values either side of nought while
the union test excludes nought itself, and then the set is genuinely
two pieces and the values between the ends are not all in it. The
application requires such a set to be retained rather than reported as
the interval that covers both.

This is the expensive one. Holding a product means scanning the loading
and taking the best of two dozen fits per value examined, where the
horizontal set needs one.

### `LiabilityModel`

One binary trait on a pedigree, through a liability threshold.

Every person carries an unobserved liability and is a case when it crosses
a threshold. Only the sign is ever seen, so the liability's variance is
fixed at one and the threshold at nought, with the intercept carrying it.

The heritability is of the liability, not of the observed status, which
is what anybody means by the heritability of a disease. It is not comparable
with the REML heritabilities the rest of this package reports, and the fit
record says ``estimator: ml`` so the difference is visible rather than
remembered — there is no REML here, because there is no response to project
onto the null space of the design.

A family's likelihood is the probability of an orthant: exact at one and two
people, and the Mendell–Elston sequential truncation above that, taking the
rarer class first. Agreement with native SOLAR is exact where no
approximation is used and within a tenth of a standard error where it is.

A relationship of one off the diagonal is accepted. Monozygotic twins and
one person entered twice push the liability correlation to ``h2`` rather
than ``h2 / 2``; ``build`` used to refuse them because the quadrature behind
the two-person probability lost digits there, and since 29 August 2026 the
integral is accurate across the whole range and the refusal has gone.

``subject_order_sha256`` optionally carries the lowercase SHA-256 from
``subject_order_commitment`` for the exact fitted row order. Every fit
record echoes it without retaining identifiers.

#### `LiabilityModel.fit(self) -> 'dict[str, Any]'`

Fit, by maximum likelihood because nothing else is available.

#### `LiabilityModel.interval(self) -> 'dict[str, Any]'`

A 95 per cent profile interval for the liability heritability.

#### `LiabilityModel.test(self) -> 'dict[str, Any]'`

Test the liability heritability against nought.

A heritability of nought sits on a bound, so the reference is the even
mixture of a point mass and chi-square on one degree of freedom. That
was assumed from the Gaussian case rather than derived for a liability,
and then measured: on the real pedigree it rejects 0.055 of the time at
a nominal 0.05.

### `PreparedModel`

A validated, eigendecomposed design and relationship matrix; the only way in.

#### `PreparedModel.fit(self, /, y, estimator='reml')`

Fit the ordinary Gaussian model. The estimator is data on the record;
the objective never depends on the response's values.

### `SpatialModel`

One trait with a spatial component whose range is estimated.

The kernel is ``exp(-λd)`` with distances in kilometres. Any fixed
components — a relationship matrix, a household matrix — are passed
alongside and are reported before the spatial component, which is followed
by the residual. ``raw_coefficient_proportions`` divide their fitted
covariance coefficients by their sum, so they depend on fixed-component
matrix scaling. When every fixed component has a positive finite mean
diagonal, ``mean_diagonal_proportions`` reports scale-invariant marginal
covariance contributions instead.

Two cautions that the numbers do not carry themselves. The range is
barely estimated: its interval reaches a bound in 98 per cent of calibration
replicates, so report it as a point estimate or use ``integrated=True`` and
be rid of it. And an interval for the spatial raw coefficient proportion
reaching nought is not a test of whether there is a spatial effect — under
that null the range is unidentified, which is why the test is a bootstrap.

``subject_order_sha256`` optionally carries the lowercase SHA-256 from
``subject_order_commitment`` for the exact fitted row order. Every fit
record echoes it without retaining identifiers.

#### `SpatialModel.bootstrap(self, y: 'Any', replicates: 'int' = 199, seed: 'int' = 1, reml: 'bool' = True, integrated: 'bool' = False) -> 'dict[str, Any]'`

The parametric bootstrap p-value for no spatial variance.

The p-value adds one to both counts, so it can never be nought however
extreme the statistic; the smallest it can report is
``1 / (replicates + 1)``.

#### `SpatialModel.fit(self, y: 'Any', reml: 'bool' = True, integrated: 'bool' = False) -> 'dict[str, Any]'`

Fit, taking the range as a free parameter or integrating it out.

``raw_coefficient_proportions`` and ``raw_coefficient_total`` depend
on fixed-component matrix scaling. When the fixed component diagonals
are positive and finite, ``mean_diagonal_component_contributions`` and
``mean_diagonal_proportions`` instead report the scale-invariant
marginal covariance decomposition. The spatial kernel and residual
identity both have unit diagonals.

With ``integrated=True`` the range is averaged over rather than
maximised over, and comes back as ``None``: there is nothing estimated
to report, which is the point.

#### `SpatialModel.interval(self, y: 'Any', quantity: 'str', reml: 'bool' = True, integrated: 'bool' = False) -> 'dict[str, Any]'`

An interval for a raw coefficient proportion, or for the range.

``quantity`` is a component index as a string, or ``"lambda"``. Asking
for the range when it has been integrated out is refused rather than
answered.

A component index profiles the raw proportion, which is not the
headline the fit reports. ``fit`` leads with
``mean_diagonal_proportions`` where those are defined, because they do
not move when a matrix is rescaled; the raw proportion does, and the two
can differ by orders of magnitude for the same component of the same
fit. The estimate these endpoints are actually around is returned beside
them as ``estimate``, so the pair reads together and cannot be
mismatched by taking the headline from one dictionary and the endpoints
from the other.

#### `SpatialModel.predict(self, y: 'Any', component: 'int', reml: 'bool' = True, integrated: 'bool' = False) -> 'dict[str, Any]'`

Predict the random effects of one component.

`component` indexes the fixed components first and then the spatial one.
With the range integrated out this is refused: there is no single kernel
to predict from, and averaging predictions across the grid is a
different quantity that has not been calibrated.

#### `SpatialModel.statistic(self, y: 'Any', reml: 'bool' = True, integrated: 'bool' = False) -> 'float'`

The likelihood ratio against no spatial variance.

A statistic and not a p-value: with the range unidentified under that
null there is no closed-form reference, so a p-value takes `bootstrap`.

### `VariantSetModel`

Score a whole set of variants at once, in the famSKAT form.

Testing rare variants one at a time finds nothing, because each has a
handful of carriers. This asks instead whether the variants in a set — a
gene, a pathway — carry more trait variance together than chance allows,
without committing to which of them matters or which way each pushes.

``backgrounds`` are the covariance bases carrying everything that is not
the set under test, and ``design`` must include its own intercept.

Pass ``Z = G * w``, not the kernel. One row per person, one column per
variant, with the column weights already applied. Nothing is lost — the
kernel is ``Z Z'`` — and nothing ``n by n`` is ever formed, so the memory
is one column per variant rather than one per person squared and the
eigenvalues come from a matrix the size of the set rather than the roster.

The weights are your choice and they are not innocent. Squaring is
implicit: a column multiplier ``w`` is a variance weight of ``w2``. The
usual rare-focused choice is ``Beta(1, 25)`` evaluated at each minor allele
frequency, but it encodes a belief about which variants matter, and a
different belief gives a different answer. Running a small pre-specified
set of weightings and combining them is more honest than picking one.

Why a score test and not a likelihood ratio.** The null sits on a
boundary, and Asterism's usual 50:50 reference is right only when the
tested matrix spreads across many eigenvalues. A variant-set kernel does
not — a burden kernel has rank one. Measured under the null on a
rare-variant kernel, the likelihood ratio rejected 0.020 against a nominal
0.05, where this reached 0.0467. The null is also fitted once for a whole
scan rather than refitted per set.

Nothing here corrects for testing many sets.

#### `VariantSetModel.scan(self, roots: 'Sequence[Any]') -> 'list[dict[str, Any]]'`

Score every set, fitting the null once.

Each record carries the statistic, the p-value, the chi-square mixture
weights it was read against, and ``trustworthy``, which is false where
the tail is small enough that cancellation has eaten the digits. A set
nobody carries returns ``code`` of ``VARIANT_SET_NO_CARRIERS`` rather
than a statistic of nought dressed up as a result.

#### `VariantSetModel.scan_family(self, roots: 'Sequence[Any]', correlations: 'Sequence[float]' = (0.0, 0.01, 0.04, 0.09, 0.25, 0.5, 0.9)) -> 'list[dict[str, Any]]'`

Score every set across a family of assumptions, and combine them.

Two tests bet on different truths about a set. A burden test
assumes every variant pushes the trait the same way and adds them into
one score: powerful when true, blind when half raise the trait and half
lower it, because they cancel. A variance-component test assumes
nothing about direction and asks only whether the effects are more
scattered than chance allows: robust to a mixture, weaker when they
genuinely agree.

They are two ends of one dial, and ``correlations`` is that dial — the
assumed correlation between variant effects, nought giving the
variance-component test and approaching one giving burden.

Taking the best of several tests inflates a p-value unless the
looking is paid for. These tests are strongly dependent, being one
score read under different assumptions, so the combination is the
Cauchy method, whose tail is right whatever the dependence.
``strongest_correlation`` comes back because it says something about
the set, but reporting its p-value alone would be exactly the inflation
this exists to avoid: report ``p_value``.

The same mechanism combines across weightings, which is the honest
answer to a weight being an arbitrary choice: run several and combine,
rather than fitting one, which the null does not identify.

#### `VariantSetModel.test(self, root: 'Any') -> 'dict[str, Any]'`

Score one set.

#### `VariantSetModel.test_family(self, root: 'Any', correlations: 'Sequence[float]' = (0.0, 0.01, 0.04, 0.09, 0.25, 0.5, 0.9)) -> 'dict[str, Any]'`

Score one set across the family.

## Functions and values

### `__version__`

`str`. Its value belongs to the build.

### `align(relationship: 'Any', relationship_ids: 'list[str]', ids: 'list[str]', *, keep: 'list[str] | None' = None, allow_missing: 'bool' = False, **columns: 'Any') -> 'dict[str, Any]'`

Line a relationship matrix up with per-person values, by identifier.

Asterism's numerical interface is positional, and that is the one place a
mistake makes no noise. A relationship matrix whose rows are in a
different order from the response does not fail, or warn, or look wrong: it
returns a heritability, an interval and a p-value, all of them plausible and
all of them for a pedigree nobody has. This does the alignment by
identifier, once, and refuses whatever it cannot line up.

The cost of getting it wrong is not a small bias. On 300 people in 60
families of five, simulated at a heritability of 0.6, the aligned fit
returns 0.490 with a p-value of `1.4e-06`; the same data with the response
in the wrong order returns 0.000, an interval of `[0.000, 0.081]`, and a
p-value of 1. Shuffling does not perturb the answer, it destroys the
signal, and then reports no heritability with complete confidence.

Returns a dictionary with ``relationship`` and ``order``, one array per
keyword column in that order, ``observed`` saying who has a complete set of
values, and ``dropped`` naming anybody the matrix has and the values do not.
Nothing is returned positionally, so no pair of outputs can be swapped.

Read ``dropped``. A handful of names there is people without
measurements. A great many is an identifier mismatch — one side writing
``001`` where the other writes ``1``, or a table that was filtered
already — and the analysis would otherwise proceed, quietly, on whoever
happened to survive it.

Parameters
----------
relationship
    A square, symmetric matrix — twice the kinship from
    :func:`relationship_matrix`, or a genomic relationship or estimated
    kinship computed elsewhere. It is subset and reordered, not rebuilt.
relationship_ids
    Who each of its rows is, in its own order. For a matrix from
    :func:`relationship_matrix` this is the second value it returned; for
    one computed elsewhere it is that tool's own identifier file, which is
    exactly the pairing that goes wrong.
ids
    Who each row of the ``columns`` is. Any order, and it may cover people
    the matrix does not.
keep
    The people to analyse, in the order wanted. Omit it to use everybody
    the matrix and the values have in common, in the matrix's own order.
allow_missing
    By default a person with no value for some column is refused. Set this
    to keep them, with ``nan`` where a value is missing and ``observed``
    marking who is complete — which is what the unbalanced models want.

Raises
------
ValueError
    With a stable code: a matrix that is not square or not symmetric, a
    duplicate identifier on either side, somebody in ``keep`` that one side
    does not have, or a missing value where ``allow_missing`` is not set.

### `build_identity() -> 'dict[str, Any]'`

Return immutable identity compiled into this Asterism build.

Returns:
    Public and Cargo versions, source commit, clean/dirty state, whether
    this is a fixed release build, and immutable release/dependency-lock
    commitments.

Raises:
    RuntimeError: If the extension lacks required source identity.

### `grouping_matrix(groups: 'list[str | None]') -> 'Any'`

Build a grouping matrix: one where two rows share a group.

**One builder, several components.** Pass household identifiers and it is a
household matrix. Pass the person each row belongs to and it is the
person-level matrix — the listener kernel, where a listener contributes two
ears — because sharing a person is the same relation as sharing a home.
Pass a testing session and it is a session effect. The matrix does not know
which it is, and neither does the model: what it means is what you grouped
by.

Parameters
----------
groups
    One entry per row, in the row order the fit will use. ``None`` or an
    empty string is a group nobody knows: that row shares with nobody and
    keeps its diagonal. It is not the absence of a group — the person does
    have a home, and what is missing is which — so their effect cannot be
    told apart from their residual and they inform the component only by
    not sharing.

Returns
-------
A square matrix, rows in the order given. One on the diagonal throughout, so
a coefficient fitted against it is a proportion of the total variance.

Notes
-----
It joins only rows that share a group, so it cannot enlarge a likelihood
block beyond the groups that straddle two families. A kernel over distances
would join every pair arithmetically, which is a different thing.

### `kinship_classes(ids: 'list[str]', father: 'list[str | None]', mother: 'list[str | None]', sex: 'list[str | None]', keep: 'list[str] | None' = None) -> 'dict[str, Any]'`

Split the relationship matrix by the kind of parent–offspring tie.

Returns the five matrices the class-weighted kinship model wants —
everything else, then mother–son, mother–daughter, father–son,
father–daughter — with the order their rows are in, the class names, and how
many pairs fell in each class.

Hand ``matrices`` to `ComponentModel` and report the fitted covariance
coefficients, the omnibus equality test, and class contrasts. The class
matrices have zero diagonals, so neither raw coefficient proportions nor
mean-diagonal proportions are interpretable as shares of phenotypic
variance.

The pair counts are worth reading before the answer is. They are rarely
balanced, and a class with few pairs has the least precise coefficient and
contrasts involving it.

### `mixed_bivariate_fit(relationship: 'Any', first: 'dict[str, Any]', second: 'dict[str, Any]', design: 'Any', *, subject_order_sha256: 'str | None' = None) -> 'dict[str, Any]'`

Fit two traits whose measurements need not be of the same kind.

Each trait is a dictionary with ``kind`` (``"continuous"``, ``"binary"`` or
``"censored"``), ``value``, ``censoring`` and ``limit``. For a binary trait
a censoring code of 1 is a case, and the limit is nought because the
threshold is carried by the intercept.

A binary trait's variance is fixed at one and comes back as one, because
only the sign of a liability is ever seen. Its heritability is therefore a
liability heritability, while a continuous or censored trait's is a
heritability of the observed scale. They are different quantities. The genetic correlation is
unaffected, which is what makes a mixed pair worth fitting.

``subject_order_sha256`` optionally carries the lowercase SHA-256 from
``subject_order_commitment`` for the exact fitted row order. The fit record
echoes it without retaining identifiers.

### `mixed_bivariate_interval(relationship: 'Any', first: 'dict[str, Any]', second: 'dict[str, Any]', design: 'Any', coordinate: 'str' = 'genetic_correlation') -> 'dict[str, Any]'`

A 95 per cent profile-likelihood interval for one bivariate coordinate.

``coordinate`` is ``"heritability_one"``, ``"heritability_two"``,
``"genetic_correlation"`` or ``"residual_correlation"``.

The variances have no interval on purpose. A binary trait's is fixed at
one because a liability has no scale of its own, so an interval on it would
describe that assumption rather than the data.

``lower_limited`` and ``upper_limited`` say whether an end sits on the
coordinate's own bound — nought or one for a heritability, minus one or one
for a correlation — rather than where the profile fell away. An end on a
bound means the data did not rule that end out, which is a different
statement from the interval stopping there.

### `mixed_bivariate_test(relationship: 'Any', first: 'dict[str, Any]', second: 'dict[str, Any]', design: 'Any', coordinate: 'str' = 'genetic_correlation', null: 'float' = 0.0) -> 'dict[str, Any]'`

Test one correlation of the mixed bivariate model against a fixed value.

``coordinate`` is ``genetic_correlation`` or ``residual_correlation``, the
same names the interval takes.

``null`` is the value tested against, and the two worth asking are the ends
of the question. Against nought: do these traits share any genes at all? An
estimate with an interval does not answer that. Against one: are they the
same genes?

The reference distribution follows from which. Nought is interior to a
correlation's range, so a plain chi-square on one degree of freedom applies
and no boundary mixture is needed. Plus or minus one is the edge of that
range, so the null rests on a bound and takes the Self-Liang even mixture,
which is reported as ``mixture_50_50`` rather than ``chi2_1``.

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

### `region_log_probability(mean: 'Any', sign: 'Any', covariance: 'Any') -> 'float'`

The conditional region log-probability the censored models rest on.

Exposed so that it can be checked, not so that it can be used. This uses
the univariate normal calculation at one coordinate, an adaptively
integrated Owen angular form at two, and Mendell-Elston sequential
truncation above two. Every
censored heritability in the package rests on it. All the evidence for those
models was generated on pairs, where the sequential branch never runs at
all, so the only way to learn how it behaves in a large family is to call it
beside an independent reference.
``checks/sequential_against_ghk.py`` is that reference.

``mean`` is each coordinate's mean already centred on its own limit, and
``sign`` is 1.0 where the value lies above that limit and -1.0 where it
lies below — the convention the censored model builds. The region is
therefore about nought, and the probability returned is that of the whole
orthant, jointly and not coordinate by coordinate.

### `relationship_matrix(ids: 'list[str]', father: 'list[str | None]', mother: 'list[str | None]', *, mz_twin: 'list[str | None] | None' = None, keep: 'list[str] | None' = None) -> 'tuple[Any, list[str]]'`

Build the additive relationship matrix from a pedigree.

Returns the matrix and the identifiers its rows are in. Align the response
and design to that order before fitting; Asterism's numerical interface is
positional.

Parameters
----------
ids, father, mother
    Parallel lists, one entry per person, in any order — parents are sorted
    before their children. A person with no parents recorded is a founder;
    use ``None`` or an empty string for both. One parent known and the other
    not is refused rather than guessed at.
mz_twin
    An optional group label per person. People sharing a label are treated
    as genetically identical.
keep
    The people to give rows to, in the order wanted. Their ancestors still
    contribute to the relationships without getting rows of their own, so a
    large pedigree costs only what the analysis roster needs. Omit it to
    keep everybody.

Raises
------
ValueError
    With a stable code naming the person at fault: a duplicate identifier,
    one known parent, a parent with no record, somebody who is their own
    parent, a loop in the pedigree, or an identifier in ``keep`` that the
    pedigree does not contain.

### `release_manifest() -> 'dict[str, Any]'`

Return the release manifest compiled into this Asterism build.

Returns:
    A newly parsed manifest. Mutating it cannot change the build's embedded
    contract.

Raises:
    RuntimeError: If the extension was built without release metadata.
    ValueError: If the embedded metadata is not a valid version-1 manifest.

### `run_analysis(analysis: 'str', design: 'Mapping[str, Any]', fit: 'Callable[[], Mapping[str, Any]]', *, model: 'Mapping[str, Any]', provenance: 'Mapping[str, Any]', subject_order: 'Sequence[str]') -> 'dict[str, Any]'`

Preflight one analysis and return its serialisable receipt data.

Args:
    analysis: Stable supported-analysis identifier.
    design: Non-identifying design summary checked before fitting.
    fit: Zero-argument callback that performs the numerical fit only after
        the release-state check succeeds.
    model: Caller-owned model and estimator settings.
    provenance: Caller-owned artifact, dependency, consumer and input
        commitments.
    subject_order: Identifiers in exact fitted row order. Only their
        commitment enters the returned receipt.

Returns:
    Version-1 receipt data. This function performs no file I/O; the caller
    writes the returned mapping beside its controlled outputs.

### `subject_order_commitment(subject_order: 'Sequence[str]') -> 'str'`

Commit to exact subject order without retaining identifiers.

Args:
    subject_order: Identifiers in the same row order as every fitted array.

Returns:
    Lowercase hexadecimal SHA-256 of the newline-separated identifiers,
    matching Asterism's existing empirical-matrix receipt convention.

Raises:
    ValueError: If the order is empty, duplicates an identifier, contains a
        non-string identifier or contains a newline.

### `tobit_fit(relationship: 'Any', value: 'Any', censoring: 'Any', limit: 'Any', design: 'Any', *, subject_order_sha256: 'str | None' = None) -> 'dict[str, Any]'`

Fit one trait whose measurement stops at a limit.

``censoring`` is 0 where the value was measured, 1 where it lies at or
above its limit, and 2 where it lies at or below it. The status is given
rather than inferred, because a censored value can carry the same number
as a measured one — extended high-frequency audiometry records several
limits within one frequency, and measured values coincide with them.

``value`` is read only where the status says measured, and ``limit`` only
where it does not.

The heritability that comes back is the heritability of the *complete*
variable — the number you would have had if the instrument reached far
enough. It is comparable with an ordinary heritability of an uncensored
trait, and not with one fitted to values where the censored ones were
replaced by the limit. It is maximum likelihood, never REML, so it must not
be placed beside a REML heritability as though the two were the same.

``subject_order_sha256`` optionally carries the lowercase SHA-256 from
``subject_order_commitment`` for the exact fitted row order. The fit record
echoes it without retaining identifiers.

### `tobit_interval(relationship: 'Any', value: 'Any', censoring: 'Any', limit: 'Any', design: 'Any') -> 'dict[str, Any]'`

A 95 per cent profile-likelihood interval for the censored heritability.

``lower_limited`` and ``upper_limited`` say whether an end sits on the
parameter's own bound rather than where the profile fell away. An end on a
bound means the data did not rule that end out, which is a different
statement from the interval stopping there.

``profile_failures`` counts fits along the profile that failed or did not
converge. Each one widened the interval rather than narrowing it, which is
the safe direction, but a large count means the interval rests on fewer
points than its width suggests.

Read ``censored_share`` beside the answer. On simulated data the model
recovers the truth to three quarters censored; on real extended
high-frequency thresholds it degrades past about half, where too little of
the upper tail is left to estimate a variance from.

### `tobit_test(relationship: 'Any', value: 'Any', censoring: 'Any', limit: 'Any', design: 'Any') -> 'dict[str, Any]'`

Test the censored heritability against nought.

The null holds the heritability at nought, which is its own bound, so the
reference is the Self-Liang 50:50 mixture of chi-square on nought and one
degrees of freedom rather than a plain chi-square. ``rule`` says which was
used, as data rather than as a promise.

### `weighted_chi2_upper_tail(q: 'float', weights: 'Any') -> 'dict[str, Any]'`

The upper tail of a weighted sum of chi-squares on one degree of freedom.

``P(sum_j weights_j * chisq_1 > q)``, which is what a variance-component
score test reads. ``weights`` are the eigenvalues of the tested matrix
after projection, so they are nonnegative.

A gene scan reads this at around 1e-6, where an approximation calibrated at
0.05 is worthless, so the result carries its own diagnostics. ``method``
says which recipe answered: a single weight, or weights that are all equal,
have exact chi-square answers and are taken directly rather than
integrated. ``settled_to`` is the size of the last term kept as a share of
the answer. ``trustworthy`` is false where the probability is small enough
that cancellation has eaten the digits, because the tail is recovered as
``1/2 + integral`` and a small answer is a difference of two nearly equal
numbers.
