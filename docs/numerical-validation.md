# Numerical validation

The checks in `checks/` address three different questions:

1. equation checks compare a compiled calculation with an independently coded
   calculation on the same fixed inputs;
2. package comparisons measure whether two implementations compute the same
   estimand; and
3. simulation measures coverage, type-I error, power, and numerical failure.

Agreement between implementations is useful for detecting transcription and
optimisation errors, but correlated implementations can agree and still be
wrong. Calibration against known simulated truth is therefore reported
separately.

## Gaussian one-trait model

- Against R `regress` under REML at `n=350` with six fixed effects,
  heritability agreed to about `5e-9` relative, total variance to `4e-9`, and
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

## Two traits

- Against R `regress` under REML on an unbalanced synthetic problem,
  heritabilities agreed within `2.6e-9`, correlations within `1.9e-8`, and the
  independently recomputed projected KKT measure was `5.5e-9`.
- Against native SOLAR under ML with deliberately unbalanced observations, all
  four directly reported quantities agreed within `2e-7`.
- Across all 78 pairs of thirteen real GOBS traits, the largest discrepancy
  from SOLAR was `5.9e-6` and the median `1.5e-7`. SOLAR's stopping precision
  makes roughly `1e-6` a realistic real-data comparison tolerance.
- In 300 REML simulations at `n=180`, 95% coverage for the two
  heritabilities, genetic correlation, residual correlation, and phenotypic
  correlation was 0.957, 0.967, 0.960, 0.953, and 0.977. Interior correlation
  tests were close to nominal; the correlation-one boundary test was
  conservative (0.033 rejection at nominal 0.05).
- Independently implemented profile endpoints agreed with the compiled
  endpoints to about `1e-5` on an unbalanced `n=150` problem.
- At heritability 0.83, overall coverage remained near nominal, but the
  residual-correlation interval failed in 23 of 300 replicates and reached a
  bound in 94% of those computed. The phenotypic correlation computed in all
  replicates and did not reach a bound.

## Several components and kinship classes

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

## Spatial model

For 400 null data sets, each with a 199-replicate parametric bootstrap,
rejection rates were 0.003, 0.050, 0.102, 0.255, and 0.547 at nominal 0.01,
0.05, 0.10, 0.25, and 0.50. The likelihood-ratio statistic has an atom at zero,
so bootstrap p-values have an atom at one; validity is judged by rejection
rates rather than continuous uniformity.

The profiled spatial component proportion and half-distance intervals covered
at 0.984 and 0.976 in 250 simulations. However, 98.4% of half-distance
intervals reached a bound, demonstrating poor range identification. Integrating
over the range and then bootstrapping also held its level: rejection was 0.0575
at nominal 0.05 across 400 data sets.

## Gene by environment, continuous and discrete

Across 2,000 simulations per scenario, the exponential gene-by-environment
correlation test rejected 0.048 at nominal 0.05 under a flat null and 0.053
when the true log-linear surface changed genetic scale without reordering. The
random-regression boundary tests were conservative. A linear-scale rank-one
surface fitted with the wrong exponential family produced a 0.107 rejection
rate for a nominal 0.05 no-reordering test, showing that surface
misspecification can imitate genetic reordering.

Gene-by-environment 95% interval coverage ranged from 0.953 to 0.983 in the
studied settings, but 80% to 92% of genetic-correlation intervals and up to 64%
of some heritability intervals reached a bound. Point estimates were also
pulled toward correlation one near the boundary.

The discrete model was simulated 500 times per scenario on the real, unbalanced
GOBS pedigree, with sex as the environment and only the response simulated. At
500 replicates the binomial standard error on a nominal 0.05 rate is about
0.010, so a rate is compatible with its level within roughly plus or minus
0.019.

Under the complete null, rejection at nominal 0.05 was 0.042 for the principal
`gene_by_environment` test, 0.044 for equal genetic effects, 0.048 for equal
genetic variances, 0.036 for equal residual variances, and 0.040 for
`any_difference`. The fitted correlation sat on its upper bound in 52.8% of
null fits, which is why the even mixture is the right reference and a plain
chi-square would be conservative.

The scenario that matters is one sex measured with more error and identical
genetics. **Every genetic test held its level there** — 0.046 for the principal
test, 0.040 for equal genetic effects and 0.040 for equal genetic variances —
while `any_difference` and the residual test each rejected all 500 replicates.
That is the claim the two free residual variances exist to support, and it is
why `any_difference` must not be reported as a genetic finding.

Against genuine alternatives the tests separate as they should: where the genes
differ, the correlation test rejected 0.990 and the equal-variance test stayed
at level (0.062); where only the genetic scale differs, the equal-variance test
rejected 0.998 and the correlation test stayed at level (0.042).

## Binary liability

Against native SOLAR, liability heritability differed by at most 0.0134, or
0.086 SOLAR standard errors, across three synthetic cases. On 700 simulations
using the real GOBS pedigree structure, the zero-heritability test rejected
0.0529 at nominal 0.05 and 95% interval coverage was 0.9443 at true
heritability 0.25.

The two-person quadrature error was at most `1.3e-9` for liability correlation
below 0.5 and `2.3e-4` below 0.99. This motivates refusing nearly perfectly
related pairs in the current numerical method.

## Association

The refitted association mode agreed with native SOLAR to all printed digits on
three synthetic tests, including p-values `3.32e-7` and `6.36e-20`. The held
mode was more conservative for large effects (`4.53e-7` and `1.58e-18` on
those examples), motivating selective refitting.

In 3.6 million null tests on 30,000 real markers across 120 simulated
responses, observed/expected tail-count ratios were 1.024 at 0.01, 1.040 at
0.001, 1.125 at `1e-4`, and 1.361 at `1e-5`; between-response variation was
used because linked markers are not independent. The experiment did not reach
genome-wide `5e-8` calibration, which would require vastly more null tests.

## Variant-set matrix builders

- Independent NumPy calculations agree exactly on the worked weighted-linear
  and burden matrices.
- With SKAT 2.2.5, the public linear and burden score statistics agree within
  `1e-10` when evaluated from Asterism's matrices and SKAT's null residuals.
  This compares score statistics, not SKAT's unexposed full matrix, and an
  intercept-only null cannot distinguish raw from column-centred genotypes.

## Variant-set score test

Against famSKAT — `SKAT_NULL_emmaX` for the kinship-carrying null, then
`SKAT` with Davies' method — on identical data with identical weights, the two
agree to `2e-6` relative across twelve data sets spanning `p = 3.7e-3` to
`p = 0.99`. That is about the accuracy either implementation computes its own
tail to, so it is agreement to the precision available rather than to a
tolerance chosen for comfort.

The reason for building it is calibration. Simulated under the null on a
rare-variant kernel, 2,500 replicates, 300 people:

| kernel | test | rejection at 0.05 | at 0.01 |
| --- | --- | --- | --- |
| linear | likelihood ratio | 0.0436 | 0.0088 |
| linear | score | 0.0484 | 0.0076 |
| burden | likelihood ratio | 0.0200 | 0.0028 |
| burden | score | 0.0512 | 0.0080 |

The burden kernel is rank one, which is where assuming a 50:50 reference goes
furthest wrong: the likelihood ratio rejected 0.020 against a nominal 0.05,
seven binomial standard errors low. The score test, whose reference is computed
from the kernel's own eigenvalues, sits on nominal.

## The weighted chi-square tail

Checked against Ruben's series expansion, which writes the statistic as a
mixture of ordinary chi-squares and shares no machinery with
characteristic-function inversion.

- Where an exact answer exists -- one weight, or equal weights -- it is taken
  exactly, agreeing with the chi-square tail to `1e-19`.
- Across 83 cases where the series provably converged, the worst relative
  difference was `1.8e-3`, at a tail of `4.9e-8`.

**Every published route fails somewhere, and the failures were measured rather
than assumed.** Davies' inversion refuses to run at tight accuracy settings and
returns nought at its own default where the answer is `5.7e-7` -- which is the
case a burden kernel produces. Ruben's series stops early when the weights span
a wide range: on eighteen weights spanning eighty to one it returned `1.3e-9`
where a Monte Carlo of eight million draws gives `1.54e-4 +/- 8.6e-6` and
Asterism gives `1.60e-4`.

The series has an exact convergence test, because its coefficients sum to one,
so the check drops it as a reference wherever that mass falls short rather than
comparing against a number that has quietly stopped early.

## Latent mediation model

Low-dimensional family log likelihoods are independently reconstructed from
the structural covariance and one- or two-dimensional Gaussian rectangle
probabilities. Compiled values agree within `2e-8`; central derivatives for all
five free parameters agree within `3e-6`. A continuous underflow case remains
finite on the log scale.

Two-person rectangles far into the tail are computed on the log scale by
conditional quadrature rather than by differencing four corner probabilities,
which cancel there. Against an independent high-order Gauss-Legendre value the
log likelihood agrees within `8e-11` at log probabilities of `-25`, `-32`,
`-62`, `-231` and `-512`. The corner-difference route is kept above a
probability of `1e-9`, where it is the more accurate of the two, and the
integration method is reported per family so a reader can tell which was used.

Higher-dimensional quasi-Monte Carlo evaluations are deterministic and are
averaged over eight shifted copies of the Halton sequence. Against an
independently derived trivariate value the 8,192-point estimate agrees within
`4e-5`. Accuracy falls off with the discrete dimension: at 8,192 points the
measured error against scipy was `7e-5` at dimension four, `5e-4` at eight and
`3e-3` at twelve.

`maximum_qmc_batch_range` is the spread across those eight shifted copies. It
is a stability statistic and not an error bound, but shifting is what makes it
informative at all: contiguous blocks of one unshifted sequence share their
bias, so their spread understated the true error by three to four orders of
magnitude at dimension twelve.

No simulation presently establishes coverage or type-I error for the vertical
estimand `a b`, so the model reports numerical diagnostics and point estimates
without a calibrated interval or p-value.

## Scale and cost

The dense routes are quadratic in memory and the fits are cubic in the largest
family block. On the reviewed SAFS pedigree of 5,364 people the relationship
matrix takes about 230 MB and a one-trait fit takes about 0.011 s once the
model is prepared; a GOBS-sized roster of 1,909 takes about 0.004 s. A spatial
fit at n = 1,800 takes tens of seconds and its bootstrap takes hours, because
spatial correlation crosses families and leaves no block structure to exploit.

The bivariate model enumerates observation patterns, which is `3^t` in the
number of traits: two traits is the practical limit and three is out of reach
by this route.

## Running the checks

Every script in `checks/` is listed here. A script missing from this list is a
script nobody runs, which is how the association comparison sat broken and
unnoticed through a change to the interface it calls.

The tests, and the equation checks that need only NumPy:

```sh
cargo test --release
uv run --no-project pytest tests/ -q
uv run --locked --no-sync python checks/matrix_builders.py
uv run --no-project python checks/against_latent_mediation.py
uv run --no-project python checks/against_mixture_tail.py
uv run --no-project python checks/bivariate_reference.py
uv run --no-project python checks/bivariate_intervals_against_reference.py
```

Comparisons against another package. These need native SOLAR, or R with
`regress` or `SKAT` 2.2.5, and fail rather than skip when it is
absent:

```sh
uv run --no-project python checks/against_r.py
uv run --no-project python checks/bivariate_against_r.py
uv run --no-project python checks/against_solar.py
uv run --no-project python checks/bivariate_against_solar.py
uv run --no-project python checks/liability_against_solar.py
uv run --no-project python checks/association_against_solar.py
uv run --locked --no-sync python checks/against_skat_builders.py
uv run --locked --no-sync python checks/against_famskat.py
```

Simulation. These take minutes to hours, and several read the GOBS pedigree:

```sh
cargo run --release --bin coverage
uv run --no-project python checks/bivariate_calibration.py
uv run --no-project python checks/components_calibration.py
uv run --no-project python checks/kinship_classes_calibration.py
uv run --no-project python checks/kinship_equality_calibration.py
uv run --no-project python checks/spatial_bootstrap.py
uv run --no-project python checks/spatial_intervals.py
uv run --no-project python checks/gxe_calibration.py
uv run --no-project python checks/gxe_intervals.py
uv run --no-project python checks/discrete_gxe_calibration.py
uv run --no-project python checks/liability_calibration.py
uv run --no-project python checks/association_tail.py
```

Timing, which asserts nothing:

```sh
uv run --no-project python checks/speed.py
```

Most simulation scripts take `ASTERISM_REPLICATES` and `ASTERISM_WORKERS`, and
the figures quoted above were not all produced at the script defaults. Where a
figure came from a different replicate count, this document says so.
