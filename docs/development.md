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

The sequential region approximation against a reference that does not come
from this crate. Every censored heritability rests on one conditional region
probability, exact to two coordinates and Mendell-Elston sequential truncation
above -- and all the evidence for the censored models was generated on pairs,
where the approximate branch never runs at all. This climbs a ladder of 5, 20,
50, 100 and 221 censored dimensions against a GHK simulator, which is unbiased
and carries its own standard error. The crate's own quasi-Monte Carlo rectangle
cannot serve as the reference here: it refuses above 25 dimensions.

```sh
uv run --no-project python checks/sequential_against_ghk.py
```

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
frequency at a time, not seventeen, and carries a shared-environment kernel
beside the genetic and person-level terms. A censored threshold is therefore
conditioned on the other ear and on whoever in the family was measurable at that
one frequency, rather than on a nearly complete audiogram. And a distance kernel
is non-zero for every pair, which is why it is preferred to a categorical
shared-environment indicator, so the likelihood's block stops being one family
and becomes the whole roster.

**The "fewer frequencies" worry above turned out to be the wrong one**, and the
run below is what says so. The paragraph predicted that conditioning on less
would leave more correlation among the censored residuals and move the
approximation back towards the regime where it fails. It does not. The leftover
mean absolute correlation is **0.008 fitting one frequency and 0.009 fitting
seventeen** -- the same. At matched dimensions the two designs behave almost
identically: at 50 coordinates 0.050 against 0.040 of a heritability step, and
at 100 coordinates **0.114 against 0.114**.

What the kernel actually changes is not the difficulty per coordinate but the
**number of coordinates**. Blocks made of families do not reach 200 censored
records; a block made of the whole roster does. The audiogram ladder stopped at
221 and found nothing wrong. This one was carried to 600 and found the wall
between 200 and 400 -- a wall the earlier design would have hit too, had anyone
climbed that far.

So the same ladder is climbed again under that design, with the rungs carried
past 221 to 600 because the block is now the roster:

```sh
uv run --no-project python checks/sequential_against_ghk.py --design components
```

The reference, the yardstick and the conditional-versus-marginal contrast are
the same ones described above; only the covariance being climbed is different.
It is one script and one GHK implementation deliberately: a second ladder would
be a second answer to the same question.

What it found, on 29 August 2026. The run behind the numbers below is:

```sh
uv run --no-project python checks/sequential_against_ghk.py \
  --design components --families 6 --replicates 8 --draws 400000
```

Six families, 480 people, 960 rows at 18 kHz, two thirds of ears censored, and a
shared-environment kernel held at 0.05 per kilometre. **The conditional regime is
qualified to 200 censored dimensions. 400 and 600 are out of reach.**

| censored dimension | 5 | 20 | 50 | 100 | 200 | 400 | 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| error as a share of a heritability step | 0.000 | 0.002 | 0.050 | 0.114 | **0.225** | 0.566 | 0.530 |

The reference resolves all of it: the error is 23 times the GHK standard error at
200 and 9 times it at 600, so none of these rungs is a statement about
simulation noise.

**200 passes, but only just** -- 0.225 against an allowance of 0.25. It should be
read as the edge of the qualified range and not as comfortable ground. The 600
rung is weaker again for a separate reason: only three of eight replicates could
supply 600 censored coordinates, so it rests on the heavily censored replicates
rather than a fair sample of them. It fails either way.

**The kernel's decay decides the answer, which is why it is a flag and not a
constant.** An earlier run held it at 0.25 per kilometre. At that decay the median
pair's kernel entry is 0.02: the kernel is non-zero everywhere and correlates
almost nothing, the ladder is stressed by dimension alone, and 400 passes at 0.19.
At 0.05 per kilometre the median entry is 0.47 and the same rung fails at 0.57.
A kernel that is dense in support but inert in effect will qualify a dimension it
has not tested, so `--decay` should be set to the strongest coupling the analysis
intends to fit, not the weakest.

**What that means for a shared-environment kernel.** A distance kernel is non-zero
for every pair, so it makes the likelihood's block the whole roster rather than
one family. A roster carrying more than 200 censored records at the frequency
being fitted is therefore outside this evidence. Staying inside it means
truncating the kernel beyond a distance where `exp(-lambda d)` is negligible,
which restores the block structure, and then showing that the truncation does not
move the answer. It does not mean switching the integral, per ADR 0010.

**The conditioning still earns most of it, as it did for the audiogram design.**
Conditioning leaves a mean absolute correlation of 0.008 between the censored
residuals, and it is flat across every rung, so the failure at 400 is not the
conditioning weakening. The marginal rungs, where nothing is conditioned away
and the correlation is 0.10, are worse at every dimension and reach eleven times
a heritability step at 400. So the conditioning is what makes the problem
tractable at all; what defeats it in the end is simply how many coordinates
there are.

The run takes a few hours. Reduce `--replicates` and `--draws` for an exploratory
pass, but quote the command with any number taken from it, because the recorded
evidence did not use the defaults.

Against a published table rather than another package. This one needs no
outside software, only the Dryad deposit, and takes about an hour:

```sh
.venv/bin/python checks/against_red_deer.py --evidence
```

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
