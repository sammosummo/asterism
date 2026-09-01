# Developing Asterism

For working on Asterism itself. If you only want to *use* it, the
[README](../README.md) is the whole story and none of this applies.

## Building from source

CPython 3.13 or 3.14, the repository-pinned Rust 1.97.1 toolchain, and `uv`.
Build isolation, the development environment and wheel builds all use the
locked Maturin 1.14.1 release.

```sh
uv sync --locked --all-groups
uv run maturin develop --release --locked
```

That gives an editable build. An editable build is deliberately not a
releasable one: `run_analysis` refuses before fitting, because a receipt is
tied to a checksummed wheel saved by a fixed release. See
[ADR 0012](adr/0012-small-analysis-ready-releases.md).

## The gate

```sh
sh tools/check_locally.sh
```

That runs everything: Rust formatting, Clippy with warnings refused, the Rust
suite, the locked environment, the Python formatter and linter, Asterism's own
Python style rules, the generated API reference's staleness check, the release
metadata check, and the Python suite.

It runs here rather than on GitHub because Asterism's hosted automation is
switched off — the account does not pay for hosted minutes, so every automatic
run failed before it started. The workflows are kept and can still be started
by hand, but nothing starts them.

## The scientific checks

The gate above is about the codebase. The checks that measure whether the
*numbers* are right live in `checks/`, and each supported analysis names the
ones it depends on in `release.toml`.

```sh
uv run --locked python checks/against_famskat.py
```

The complete list, with what each one establishes, is in
[development.md](development.md).

## The generated reference

`docs/api-reference.md` is built from the package's own docstrings and must not
be edited by hand:

```sh
uv run python tools/build_api_reference.py
```

A test fails when the file on disk stops matching what the generator produces,
so a changed signature cannot ship beside a stale reference.

## Examples

Every example in `examples/` is a complete program that generates its own data
from a fixed seed. Tests require each one to run, to keep recovering whatever
truth it claims to recover, and — where the README quotes its output — to match
that quotation exactly.

## Running the scientific checks

Several checks compare against native SOLAR, whose input files quantise the
numbers written into them. What that means for reproducing a fixture is in
[solar-reference-fixtures.md](solar-reference-fixtures.md).

Every script in `checks/` is listed here. A script left off the list is never
run, and so is never seen to break.

The tests, and the equation checks that need only NumPy:

```sh
cargo test --release
uv run --no-project pytest tests/ -q
uv run --no-project python checks/against_latent_mediation.py
uv run --no-project python checks/against_mixture_tail.py
uv run --no-project python checks/bivariate_reference.py
uv run --no-project python checks/bivariate_intervals_against_reference.py
uv run --no-project python checks/component_interval_identity.py
uv run --no-project python checks/gxe_against_independent_full_fit.py
uv run --no-project python checks/discrete_gxe_against_independent_full_fit.py
uv run python checks/gxe_target_design.py \
  --replicates 500 \
  --workers 12 \
  --design-fixture checks/design_fixtures/continuous_gxe_target_design.json \
  --fixture-sha256 f5facfdf09b9690de3db4f35961e8f9e805bc1374eccf8b1a52bbec51c99283f \
  --dense-sentinel checks/gxe_against_independent_full_fit.py \
  --no-write
```

The continuous-GxE full-fit comparison uses the independent dense likelihood
and SciPy optimizer in `checks/gaussian_full_fit_reference.py`; that helper is
not itself a scientific pass command.

Comparisons against another package. These need native SOLAR, or R with
`regress`, `spaMM`, `geoR` or `SKAT` 2.2.5, and fail rather than skip when it
is absent:

```sh
uv run --no-project python checks/against_r.py
uv run --no-project python checks/bivariate_against_r.py
uv run --no-project python checks/against_solar.py
uv run --no-project python checks/spatial_against_solar.py
uv run --no-project python checks/spatial_against_spamm.py
uv run --no-project python checks/bivariate_against_solar.py
uv run --no-project python checks/liability_against_solar.py
uv run --no-project python checks/association_against_solar.py
uv run --no-project python checks/mixed_bivariate_against_solar.py
uv run --locked --no-sync python checks/against_famskat.py
```

The censored model against R. `censReg` gives the fit with no relatives in it,
which is the only case the two models share; `MCMCglmm` carries the relatedness
and is compared by posterior interval rather than by point estimate, because it
is a different estimator and agreement to a decimal would be the surprising
outcome:

```sh
uv run --no-project python checks/tobit_against_censreg.py
uv run --no-project python checks/tobit_against_mcmcglmm.py
```

`checks/censoring_design.py` is the shared, non-runnable helper that solves a
fixed instrument limit from pre-outcome means, marginal variances and an
expected censoring share.

The sequential region approximation against a reference that does not come
from this crate. Every censored heritability rests on one conditional region
probability, exact to two coordinates and Mendell-Elston sequential truncation
above. The early recovery evidence used pairs, where the approximate branch
never runs; the later GHK ladders exercise larger regions directly. This climbs
a ladder of 5, 20, 50, 100 and 221 censored dimensions against a GHK simulator,
which is unbiased and carries its own standard error. The crate's own
quasi-Monte Carlo rectangle cannot serve as the reference here: it refuses
above 25 dimensions.

```sh
uv run --no-project python checks/sequential_against_ghk.py
```

The two-person probability against two independent integrations. Below three
coordinates the region probability is a bivariate normal integral rather than
the sequential update, and that integral has its own reference: Owen's angular
form and the conditional form, computed here and required to agree with each
other before either is believed. It exists because `src/normal_integrals.rs`
carries a table of reference values, and a table nothing can recompute is not a
check:

```sh
uv run --no-project python checks/bivariate_against_owen.py
```

On 29 August 2026: 53 points, worst absolute error 1.72e-11 in the log
probability. The grid moves the rectangle as well as the correlation, because
the sixteen-point quadrature this replaced was accurate at thresholds of nought
and nought and some seventy times worse away from them.

What it found, on 19 August 2026. Conditioned on the rest of the family --
which is what the models actually evaluate -- the error at 221 dimensions is
0.062 log units, about 0.09 of what a heritability step of 0.1 does to the same
quantity. The approximation is safe there.

It is safe because of the conditioning, not because the routine is
accurate. Conditioning on a nearly complete audiogram leaves a mean absolute
correlation of 0.009 between the censored residuals, and sequential truncation
is exact when coordinates are independent. On the same number of coordinates
with nothing conditioned away, where the correlations are the audiogram's own,
the error is 84 times larger and the answer moves by 3.3 log units depending on
the order the coordinates are given in -- an order that is arbitrary, because
when every censored value lies the same side of its limit there is no rarer
class to sort by.

The result is conditional on the design. Fewer frequencies, heavier
censoring, smaller families, or a person whose audiogram is mostly unmeasurable
all reduce how much is conditioned on and move back towards the regime where it
is not safe. Run it again when the design changes.

**The design did change.** The several-component censored model fits one
frequency at a time, not seventeen, and carries a household component beside the
genetic and person-level ones. A censored threshold is therefore conditioned on
the other ear and on whoever in the family was measurable at that one frequency,
rather than on a nearly complete audiogram. So the same ladder is climbed again
under that design:

```sh
uv run --no-project python checks/sequential_against_ghk.py \
  --design components --families 6 --replicates 8 --draws 400000
```

The reference, the yardstick and the conditional-versus-marginal contrast are the
same ones described above; only the covariance being climbed is different. It is
one script and one GHK implementation deliberately: a second ladder would be a
second answer to the same question.

What it found, on 29 August 2026. Six families, 480 people, 960 rows at 18 kHz,
two thirds of ears censored, households of three. **Nothing failed.**

| censored dimension | 5 | 20 | 50 | 100 | 200 | 400 | 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| error as a share of a heritability step | 0.000 | 0.003 | 0.007 | 0.006 | 0.014 | 0.036 | 0.053 |

Against an allowance of 0.25. **600 is the largest rung climbed and not a limit
that was found**: the ladder ran out before the approximation did. The reference
resolves all of it -- the error is 55 times the GHK standard error at 200 and 8
times it at 600.

**The "fewer frequencies" worry above turned out to be the wrong one.** The
paragraph predicted that conditioning on less would leave more correlation among
the censored residuals and move the approximation back towards failing. It does
not. The leftover mean absolute correlation is 0.003 fitting one frequency and
0.009 fitting seventeen.

**The one number worth watching is the ordering gap.** Sequential truncation
conditions on coordinates in an order, and sorts the rarer class first; when
every censored value lies the same side of its limit there is no rarer class, so
the order is arbitrary. At 600 the answer moves by 0.55 log units with that
arbitrary order. Against a heritability step of 4.74 that is twelve per cent, so
it does not threaten the estimate here, but it is the quantity that would fail
first if the design moved.

**A component that is not block diagonal is a different problem, and an earlier
run measured it.** Spatial kernels were dropped from this model on 29 August; a
distance kernel is non-zero for every pair, so a block rule testing entries
against exact nought would put a whole roster in one block. Climbed with such a
kernel -- 480 people in a 30 km square, where nothing is negligible -- the same
ladder failed at 400, at 0.57 of a heritability step, and the qualified dimension
was 200. That is not this model, and it is recorded only to say what a dense
component costs: roughly a factor of ten in the error ratio at every rung above
100. A household kernel joins only people who share a home, so blocks stay the
size the pedigree implies and the question does not arise.

**The conditioning still earns most of it, as it did for the audiogram design.**
Conditioning leaves a mean absolute correlation of 0.003, flat across every rung.
The marginal rungs, where nothing is conditioned away and the correlation is 0.36
at five coordinates, are worse at the low rungs; by 600 the two regimes have
converged, which they did not under the dense kernel.

The run takes a few hours. Reduce `--replicates` and `--draws` for an exploratory
pass, but quote the command with any number taken from it, because the recorded
evidence did not use the defaults.

Whether an additive component and a person-level one can be told apart. Both are
one within a person, so what separates them is that additive kinship is also
non-zero between relatives: the information comes from **related pairs** rather
than from records, and a person's second record sharpens their sum while saying
nothing about the split. This sweeps the design and reports where a pedigree is
enough:

```sh
uv run --no-project python checks/component_separation.py
```

**Its pass rule is consistency, not unbiasedness**, and that is deliberate.
Maximum likelihood variance components are biased in finite samples -- it is
what REML exists to remedy, and this model cannot use REML because a censored
observation has no residual to project. The historical numerical results used a
limit selected from each realised response and are superseded. The corrected
fixed-instrument campaign completed 200 replicates in each of eight cells with
no failed fit. At 240 sibling pairs the mean additive and person-level shares
were 0.3984 and 0.3028 for two records per person, and 0.3858 and 0.3132 for
three, against truths 0.4 and 0.3. Both largest-design cells passed the
predeclared 0.03 bias rule.

The other number it reports is **how much less precisely each component is known
than their sum**, which ran from 2.9 to 5.0 times across the corrected design
grid. The correlation between the two estimates is reported too but is not the
verdict: it goes to minus one as a design grows, because the sum becomes certain
while the split stays open, so it describes the shape of the uncertainty and not
whether a design can answer the question.

The same model against something that is not itself. SciPy optimises a censored
likelihood assembled separately from its definition, sharing no likelihood,
parameterisation, starting values or optimiser with Asterism:

```sh
uv run --no-project python checks/censored_components_against_independent_full_fit.py --no-write
```

That independent likelihood lives in `checks/censored_full_fit_reference.py`;
the helper is not itself a scientific pass command. It reaches the censored
region probability by integrating it -- `log_ndtr` for one censored row, SciPy's
Genz implementation for more -- where Asterism reaches it by sequential
truncation.

**Two tolerances, because the two quantities disagree for different reasons**,
and both are declared in `release.toml` beside the pass rule rather than left in
the script. The mean-diagonal proportions are what the science reads. The log
likelihoods agree less closely, and are not expected to: the gap between them
**is** the sequential-truncation approximation, measured here rather than
assumed.

Measured with fixed pre-outcome limits on 31 August 2026, twenty sibling pairs
at two records each, twelve replicates at an expected quarter censored and
twelve at an expected half: the proportions agree to 1.4e-03 at the lighter
censoring and 2.9e-03 at the heavier, and the log likelihoods to about 0.07
throughout.

**The check also says which side the disagreement comes from**, which is the
part worth reading. It scores Asterism's own answer through the independent
likelihood and compares that with the independent optimum. Wherever the two
disagree the independent optimum wins, so SciPy is not merely stopping short of
what Asterism found -- it reaches a better point, and what separates them is the
approximation moving the optimum. It wins by very little, around 1e-03 in log
units, and where the two agree the comparison can go a millionth the other way,
which is two numbers tying rather than a verdict. The reason it is so close is
that the likelihood is nearly flat along the split between the two components.
That flatness is the same thing
[`checks/component_separation.py`](../checks/component_separation.py) measures
from the other side, where it shows up as the split being four to five times
less precisely known than the sum: a surface that flat turns a small error in
the likelihood into a comparatively large move in the proportions.

The same model at the design it will actually be used on. The published
one-component target rule and the old several-component campaign both used
outcome-adaptive limits. The corrected several-component command is:

```sh
uv run --no-project python checks/censored_components_target_design.py \
  --test-reference asymptotic --replicates 200 --workers 8 \
  --checkpoint .scientific-checkpoints/censored-components-target.json \
  --no-write
```

It reuses the reviewed aggregate the censored rule already uses -- 1,909
analysed people, 202 relationship components with the largest at 180, six
fixed-effect columns -- and makes two changes. Each person contributes **two
records**, which is what wants a person-level component, and each family is cut
into **households of three**, giving a shared-environment kernel beside the
genetic one. Both are the construction the region ladder climbs, so the ladder's
measurement covers this design rather than a different one. The completed
fixed-instrument runs recorded a largest censored block of 313, within the
ladder measured to 600 dimensions. That measurement is evidence, not a runtime
support boundary.

**Why the person-level component is not optional.** On one replicate of this
design, fitting the additive component alone put the genetic share at 0.81
against a generating truth of 0.35. Adding the person-level component brought it
to 0.40 and adding the household to 0.34. With two records per person and
nowhere else for their resemblance to go, it goes into the heritability.

**The upper end of a component's interval cannot be reached here, and the
nominal coverage is 0.975 rather than 0.95 because of it.** A proportion of one
puts every other component and the residual at nought, leaving the additive
matrix alone; spread over records that matrix gives a person's own records a
correlation of exactly one, so it is singular. The profile cannot be evaluated
there, and `src/interval.rs` covers an end it could not evaluate rather than
placing it on evidence never gathered. So the upper end is the bound every time
-- measured at one, two and three components alike, so it is the repeated
records and not the component count. A two-sided 95 per cent interval puts 0.025
in each tail; closing the upper one off hands that back, which makes 0.975 what
the lower end can be scored against. The one profile failure this produces is
recorded and not counted as a failed fit, because nothing went wrong in it. A
failure anywhere else in the bracket is counted.

**Superseded investigation result from 30 August 2026.** Eight hundred
attempts, eight hundred measured, no refusal and no failed fit. The largest
censored block was 310. Because the limit was selected from each outcome, the
table does not qualify the fixed-instrument design.

| scenario | censored | rate | exact interval | nominal | |
| --- | ---: | ---: | :---: | ---: | :--- |
| null | 0.52 | 0.055 | [0.028, 0.096] | 0.05 | held |
| null | 0.75 | **0.100** | [0.062, 0.150] | 0.05 | **failed** |
| heritable | 0.52 | 0.985 | [0.957, 0.997] | 0.975 | held |
| heritable | 0.75 | 0.960 | [0.923, 0.983] | 0.975 | held |

**The superseded campaign raised a real concern about the analytic boundary
reference.** Its three-quarter-censored rate excluded nominal. A fresh run
moved the rate from 0.100 to 0.065, illustrating why one 200-replicate draw is
not stable calibration. The later fixed-instrument analytic target also failed
at three-quarters censoring: its rejection rate was 0.095 and its one-sided
exact lower bound was 0.0631, above the nominal 0.05. The descriptive two-sided
interval began at 0.0582. A separate fixed-limit family-four differential
remained elevated at 0.090.

The boundary atom is a reproducible diagnostic, not a finite-sample gate.
Self–Liang's 50:50 mixture is an asymptotic result and does not require
uncorrelated nuisance scores. Published finite-sample mixed-model results make
both the atom and the positive tail design-dependent. In the retained
family-size sweep, one cell had an atom of 0.325 and a rejection rate of exactly
0.050; requiring an atom of one half would fail a correctly sized test.

Neither the investigation-only held-fit change nor the realised-quantile
censoring harness explained the family-four differential. Comparing the code
immediately before `e4bd51f` with the current code on the exact same 40 target seeds changed no
point-mass membership or rejection decision. Fixing the instrument limit before
the outcome was drawn changed the family-four atom from 0.400 to 0.390 and the
rejection rate from 0.085 to 0.090. The harness was wrong for an instrument, but
not causal here.

The several-component analytic test now says exactly what it is: the asymptotic
Self–Liang 50:50 mixture. It returns `asymptotic_mixture_50_50`, while the
released one-component compatibility route keeps `mixture_50_50`. A
several-component request is refused if an untested component or the residual
rests on its bound, because that is not the one-boundary problem the reference
describes.

The corrected target campaign measures the labelled asymptotic test directly
at the two intended censoring shares, with instrument limits fixed before any
outcomes are drawn. Its LRT atom is diagnostic only. Completed outer attempts
are written to a build- and source-bound checkpoint outside
`release-evidence`, and a restart accepts no stale, duplicate or unknown row.
All 800 analytic attempts completed. Null rejection was 0.055 at 52% censoring
and 0.095 at 75%; the latter cell failed its declared level rule. Coverage was
0.980 and 0.960, and the operational LRT atoms were 0.465 and 0.390.

That failure applies to the analytic p-value for this exact 1,909-person,
four-component target at 75% expected censoring. It must not be reported for
that analysis without a design-specific simulated null tied to the model,
design, source and seed. It is not a universal censoring threshold: the labelled
asymptotic test remains available on other designs, and fit, interval and the
released one-component route are unaffected.

`CensoredComponentModel.bootstrap` remains available as an experimental,
design-specific alternative. It fits the constrained null, simulates the
latent outcome at complete per-row limits, refits both hypotheses and reports
an add-one Monte Carlo p-value. Its independent inner datasets are assigned
deterministic coordinate-derived streams and refitted with Rayon, so one outer
replicate no longer has a 999-fit serial floor and thread count does not alter
the result. The 800-coordinate bootstrap target completed, but eleven attempts
in the 75% null cell refused after an inner fit failed. The scored rejection
rate was 0.055 at both censoring shares and coverage was 0.980 and 0.960, but
the campaign permits no failed attempt, so the no-write merge failed and wrote
no qualifying evidence. The earlier 24.25-day projection described this nested
calibration before the inner loop was parallelised; the completed campaign did
not qualify a replacement p-value, so the bootstrap remains experimental.

Coverage of the interval at the full component set, with an interval taken for
**every** structured component rather than only the first:

```sh
uv run --no-project python checks/censored_components_coverage.py \
  --replicates 300 --workers 12 --no-write
```

The design is 2,400 people in villages of six -- three sibling pairs to a
village -- each contributing two records, with an expected censored share of
0.52 under a fixed limit. **Households have to
cross families here.** A household that is exactly a sibling pair makes twice
the kinship the identity plus the within-pair pattern and the household matrix
the identity plus that same pattern, so the residual is their exact combination
and nothing separates additive from household. Each home therefore takes one
person from each of two different families, which identifies the three at Gram
rank four while keeping the likelihood's blocks to a village of twelve rows.

Its nominal is 0.975 for the reason given above: the upper end of a component
proportion cannot be reached when a person contributes more than one record, so
what is being scored is the lower end.

**A truth sitting exactly on a bound is reported as a range rather than a
number**, because containment splits in two there. An interval whose lower end
is above nought shuts a truth of nought out whatever rule is applied, and those
are misses. An interval whose end is on nought is the Self-Liang question, and
`src/tobit.rs` deliberately leaves that verdict absent at several components --
more than one share can rest on nought at once, which is not the case the
one-component coverage rule scored, and an absent verdict says nobody has
measured it. Those are left undecided. The range is the most and the least
coverage the cell could have, and the cell still fails if even the most falls
short of nominal. The corrected fixed-instrument campaign completed 300
replicates in each of twelve cells with no refusal. Coverage in the eleven
interior cells ranged from 0.9533 to 0.9933, and every simultaneous exact
interval contained 0.975. The lower-bound cell retained the deliberately absent
boundary verdict in 289 cases and excluded nought in eleven; its predeclared
rule did not establish undercoverage.

**The band is simultaneous, and that is the difference between a rule and a
coin toss.** The campaign makes twelve statements at once, one per component per
cell. Drawn at 95 per cent for each of them separately, at least one is excluded
by chance 46 per cent of the time -- so a per-cell rule would fail about every
other run of a model with nothing wrong with it, which is a rule that teaches
its reader to ignore it. The rate is therefore split across the cells, so the
campaign as a whole raises a false alarm five times in a hundred. The first run
of this check made exactly the error the correction exists for: eleven cells
held, one over-covered at 0.9933, and its per-cell band excluded the nominal by
0.001.

**Superseded investigation result from 30 August 2026.** The recorded cells ran
under limits selected from their realised responses. Their 0.9567 to 0.9933
coverage band and boundary-cell counts remain in the historical evidence, but
they do not establish fixed-instrument coverage. The fixed-instrument results
above replace them as qualification evidence.

Against a published table rather than another package. This one needs no
outside software, only the Dryad deposit, and takes about an hour:

```sh
RED_DEER_DATA=/path/to/unpacked/deposit \
  .venv/bin/python checks/against_red_deer.py --evidence
```

`RED_DEER_DATA` is required. The checker never guesses a workspace or staging
location for an external scientific deposit.

`components_target_design.py` measures the several-component model at the
design it will be used on, which nothing else did. Its two other rules simulate
four hundred people in sibling pairs, so between them they had measured
nothing larger than a family of two -- while SAFS families reach a hundred and
sixty five. It reuses the reviewed,
participant-free aggregate the spatial target layout already uses: 1,792 people,
190 families with the largest at 165, and 1,140 households sized one to seven.

The household matrix crosses families, which is what makes the design hard.
A household that is exactly a sibling pair makes twice the kinship the
identity plus the within-pair pattern and the household matrix the identity plus
that same pattern, so the residual identity is their exact combination and the
three bases have rank two. Nothing separates additive from shared household
there. Real households hold people from different families, which is what the
1,140 locations record, so this design identifies what a sibling-pair simulation
cannot.

Simulation. These take minutes to hours, and several read the GOBS pedigree:

```sh
uv run --no-project python checks/one_trait_coverage.py \
  --replicates 8000 --families 100 --workers 12 \
  --truths 0 0.05 0.07 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 1 --no-write
uv run --no-project python checks/bivariate_calibration.py
uv run --no-project python checks/components_calibration.py
uv run --no-project python checks/components_target_design.py
uv run --no-project python checks/kinship_classes_calibration.py
uv run --no-project python checks/kinship_equality_calibration.py
uv run --no-project python checks/spatial_bootstrap.py
uv run --no-project python checks/spatial_target_layout.py \
  --fixture checks/design_fixtures/spatial_component_presence_target_layout.json \
  --fixture-sha256 7d0d7bb2afde1a5f7566e2fa04e47292d148d978d832b2c85fb6c4b76fe93d6b \
  --bootstrap-replicates 199 --bootstrap-seed 20260812 --no-write
uv run --no-project python checks/spatial_intervals.py
uv run --no-project python checks/gxe_calibration.py
uv run --no-project python checks/gxe_intervals.py
uv run --no-project python checks/discrete_gxe_calibration.py \
  --replicates 500 --workers 12 \
  --fixture checks/design_fixtures/discrete_gxe_target_design.json \
  --fixture-sha256 2cceae6f2c5ea3f0c3de5946daeaa55cf2f35edd6d365fd1f45499f463d4c1c6 \
  --levels 0.01 0.05 0.1 \
  --scenarios null different_genes different_scale noisier \
  --tests gene_by_environment any_difference correlation genetic residual \
  --no-write
uv run --no-project python checks/discrete_gxe_correlation_interval.py \
  --replicates 400 --people 600 --families 150 --sibs-per-family 4 \
  --workers 12 --truths 0.3 0.6 0.9 --no-write
uv run --no-project python checks/liability_calibration.py
uv run --no-project python checks/association_tail.py
uv run --no-project python checks/tobit_calibration.py
uv run --no-project python checks/tobit_coverage.py
uv run python checks/tobit_target_design.py \
  --replicates 200 --workers 6 \
  --censoring-shares 0.52 0.75 \
  --true-heritability 0.5 --true-variance 4.0 \
  --level 0.05 --nominal-coverage 0.95 --monte-carlo-confidence 0.95 \
  --base-seed 920000 --scenario-seed-offset 1000003 --rate-seed-offset 10007 \
  --maximum-failed-replicates 0 \
  --design-fixture checks/design_fixtures/one_trait_gaussian_heritability.json \
  --fixture-sha256 93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e \
  --no-write
uv run --no-project python checks/mixed_bivariate_calibration.py
uv run --no-project python checks/mixed_bivariate_coverage.py
uv run --no-project python checks/mixed_binary_censored_exact_combination.py
uv run python checks/mixed_binary_censored_target_design.py \
  --replicates 300 --workers 8 \
  --prevalence 0.254 --heritabilities 0.5 0.5 \
  --genetic-correlations 0.0 0.4 \
  --residual-correlation 0.15 --hearing-variance 3.0 \
  --censoring-shares 0.52 0.75 \
  --nominal-coverage 0.95 --nominal-level 0.05 \
  --monte-carlo-confidence 0.95 --point-recovery-standard-errors 3.0 \
  --base-seed 1210000 --cell-seed-offset 104729 \
  --replicate-seed-offset 7919 \
  --minimum-cases 10 --minimum-measured-hearing 10 \
  --design-fixture checks/design_fixtures/mixed_binary_censored_target_design.json \
  --fixture-sha256 aed92270dcfafb00a46a24bfaf1f3a2b52950fdea1c34deb3740e5060d6f3645 \
  --no-write
uv run --no-project python checks/mediation_calibration.py
uv run --no-project python checks/mediation_ascertainment.py
uv run --with numpy python checks/mediation_power.py
```

`mediation_power.py` answers a design question rather than checking a
calculation: whether to enrol several offspring per adjudicated case. It is here
because it is a simulation that has to be run and read like the rest.

Measurements that set a default rather than assert a result. Both exist because
a number in the code was chosen once and needed a reason attached:

```sh
uv run --no-project python checks/mediation_qmc_points.py
uv run --no-project python checks/mediation_family_size_cost.py
```

Reading a real relationship matrix, which the others never do. It asserts on
structure and agreement, never on a person:

```sh
uv run --no-project python checks/empirical_kinship_import.py
```

Timing, which asserts nothing:

```sh
uv run --no-project python checks/speed.py
```

Most simulation scripts take `ASTERISM_REPLICATES` and `ASTERISM_WORKERS`, and
the figures quoted above were not all produced at the script defaults. Where a
figure came from a different replicate count, this document says so.

Release-only scientific inventory support is deliberately fail closed.
`checks/external_reference_fixture.py` verifies a versioned independent-output
fixture on portable CI or refreshes it explicitly on a host with R or SOLAR.
`checks/external_reference_adapter.py` supplies canonical finite-array hashing,
exact live R/package identity, and strict optional argument parsing; its own
SHA-256 is included in every R fixture, alongside the comparison adapter hash
and the generated-input hashes. A portable verification regenerates those
inputs, calls the public Asterism API, and compares only with frozen independent
outputs. It does not look for R. The focused test repeats all five portable
commands with an empty `PATH`.

`checks/solar_reference_adapter.py` is the corresponding native-SOLAR helper.
It writes a temporary generated problem for a live refresh, records the exact
SOLAR executable and version, and makes portable verification consume only the
frozen independent output; portable verification never invokes SOLAR.

Five R-backed release rules were refreshed live with R 4.5.2 and then verified
portably. The ordinary no-argument scripts still run the human-readable live
comparison.

| rule | qualified independent software | portable result |
| --- | --- | --- |
| `against_r` | `regress` 1.3.22 | maximum estimate difference `7.68e-9`; likelihood-offset drift `4.44e-12` |
| `bivariate_against_r` | `regress` 1.3.22 | maximum reported-quantity difference `1.88e-8`; scaled gradient `5.46e-9` |
| `tobit_against_censreg` | `censReg` 0.5.38 | maximum relative difference `9.56e-9` |
| `tobit_against_mcmcglmm` | `MCMCglmm` 2.36, `Matrix` 1.7.4, `coda` 0.19.4.1 | all three Asterism points inside the 95% HPDs; heritability width `0.278`; effective size `1603.5` |
| `spatial_against_spamm` | `spaMM` 4.6.65, `geoR` 1.9.6, `jsonlite` 2.0.0 | five scenarios; maximum component/decay difference `5.24e-6` |

The deterministic envelopes are in `checks/external_fixtures/`; each retains
the prewritten acceptance rule and only the external numbers needed by
portable verification. Refresh is always explicit through
`checks/external_reference_fixture.py refresh`; the commands in `release.toml`
always use `verify`. Until an adapter and genuine live fixture exist,
`checks/release_gate_blocker.py` records the named missing evidence and exits
nonzero; neither command converts a historical result or unavailable external
program into a passing check.

## What the checks cost to run

The dense routes are quadratic in memory and the fits are cubic in the largest
family block. On the reviewed SAFS pedigree of 5,364 people the relationship
matrix takes about 230 MB and a one-trait fit takes about 0.011 s once the
model is prepared; a GOBS-sized roster of 1,909 takes about 0.004 s. A spatial
fit at n = 1,800 takes tens of seconds and its bootstrap takes hours, because
spatial correlation crosses families and leaves no block structure to exploit.

The bivariate model enumerates observation patterns, which is `3^t` in the
number of traits: two traits is the practical limit and three is out of reach
by this route.
