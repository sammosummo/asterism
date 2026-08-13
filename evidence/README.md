# Evidence

Results of checks that take too long to be ordinary tests, kept so that a claim
about Asterism's behaviour points at a number somebody can find.

Nothing here is a status, a gate or a certificate. A file in this directory says
what was measured, on what, when, and with which seed. It says nothing about
whether anything is approved.

## `bivariate-against-r-2026-08-11.json`

The compiled Rust two-trait optimiser against R's `regress`: REML, sixty
six-person families, one trait observed for 360 people and the other for 320.
Produced by `checks/bivariate_against_r.py`.

The two routes use different parameterisations and different optimisers. The two
heritabilities agree within 2.6e-9 and the genetic and residual correlations
within 1.9e-8. Asterism's independently recomputed, response-scale-invariant
projected KKT measure is 5.5e-9. This establishes fixed-problem fidelity and
numerical convergence; it is not interval coverage or authority for a JASA
analysis.

## `coverage-2026-08-11.json`

The coverage check: simulate datasets whose heritability is known, fit every
one, and count how often the 95 per cent interval contains the truth. Produced
by `cargo run --release --bin coverage`, which prints this JSON at the end of
its table.

Two rosters, both of three-generation families of fourteen — a founding couple,
three children, three spouses married in, six grandchildren:

- `n350`, twenty-five families, which is about the size of the SAFS analyses.
- `n1400`, a hundred families.

Twelve truths from nought to one, 8,000 replicates each, REML, every replicate
scored. A cell passes when the Clopper–Pearson interval on its coverage
overlaps [0.940, 0.960].

**All twelve cells pass at n = 1400. Ten of twelve pass at n = 350**, the
exceptions being truths of 0.05 and 0.07, which over-cover. See
`docs/adr/0004` for what that means and why it is expected.

Each cell also records `naive_boundary_coverage` at the two bounds: what the
coverage would have been had a lower endpoint of nought been taken to mean the
interval contains nought. It over-covers at both bounds, which is the
measurement behind the recipe rather than an argument for it.

Each cell also records `beta_coverage`: the Wald interval coverage of every
fixed effect, one by one rather than summarised. Summarising them by their
largest departure from 0.95 was tried first and was wrong — the extreme of many
noisy estimates sits a couple of standard errors out by chance, which invented a
shortfall three times the real one.

The seeds are in the file. The whole thing reruns in about twenty seconds.

## `against-r-2026-08-11.json`

Asterism against R's `regress`, REML, n = 350, six fixed effects, three
datasets. Produced by `checks/against_r.py`.

Heritability agrees to about 5e-9 relative, total variance to 4e-9, every fixed
effect to 1e-9. The log-likelihoods differ by −316.114855422 on all three, which
is −(n − p)/2 · log(2π) to within 5e-12 — the constant `regress` omits and
Asterism keeps. A stable but unrecognisable offset would have meant something
else was going on; a recognisable one means the difference is bookkeeping.

This is fidelity rather than correctness. Two implementations agreeing shows
they compute the same function, and the predecessor's 2e-11 agreement with
itself was correlated error.

## `against-solar-2026-08-11.json`

Asterism's ML fit against native SOLAR, three datasets of twenty-five families
with six fixed effects. Produced by `checks/against_solar.py`.

Heritability agrees to within 5e-7, which is as much as SOLAR's seven printed
significant figures can demonstrate. The log-likelihoods differ by
n/2 · log(2π) — the constant SOLAR drops. REML drops (n − p)/2 · log(2π)
instead, which is why the R comparison expects a different number.

SOLAR silently overrules a sex that contradicts a parental role. The check
derives sex from role and then verifies against `pedindex.out` rather than
trusting it, because a rewritten sex means the two fits no longer share a design
and the comparison is void without saying so.

## `bivariate-against-solar-2026-08-12.json`

The two-trait ML fit against native SOLAR, twenty-five families, n = 350, with
every seventh person lacking the second trait. Produced by
`checks/bivariate_against_solar.py`.

All four reported quantities agree to within 2e-7. This is the strongest of the
two-trait fidelity checks because SOLAR reports the genetic and residual
correlations directly, so nothing is re-expressed on the way to the comparison —
unlike the R check, which carries covariances that have to be turned into
ratios.

The data are unbalanced deliberately. Eigen-simplification per family block
survives extra traits through the Kronecker structure but breaks when people are
missing different traits, so a balanced check would pass without touching the
part most likely to be wrong.

## `bivariate-against-solar-real-2026-08-12.json`

The same comparison on real GOBS material rather than simulated: thirteen
traits, all seventy-eight pairs of them, both engines reading identical inputs.

Largest disagreement on any reported quantity across all seventy-eight pairs is
5.9e-06; the median is 1.5e-07. The traits are inverse-normalised once in Python
and both engines read the result, so nothing in the agreement can come from the
transform.

**Real data agrees about an order of magnitude less closely than simulated
data**, and that is SOLAR rather than Asterism. A parallel check on the Acoustic
v116 univariate batch found SOLAR's own termination on heritability looser than
its printing — log-likelihoods equal to 1e-8 while heritabilities differed by
8e-7, worth about 1e-11 in likelihood at that standard error. So a tolerance
calibrated on synthetic data, where agreement reaches 1e-7, is too tight for
real-data comparisons. Expect around 1e-6 and treat a tighter demand as a
statement about SOLAR's stopping rule rather than about correctness.

The file also carries a check that is not a comparison with other software: each
heritability is estimated twelve times, once per partner, on a slightly
different set of people. The widest spread across partners is 0.022.

## `bivariate-calibration-2026-08-12.json`

Coverage and test calibration for the two-trait model, which is a different
question from everything above. The comparisons show Asterism computes the same
numbers as software already trusted; this asks whether the uncertainty around
those numbers means what it says. Produced by `checks/bivariate_calibration.py`,
which takes about a quarter of an hour.

Thirty families of six, n = 180, REML. Interval coverage is 0.957, 0.967, 0.960,
0.953 and 0.977 against a nominal 0.95 — the last being the phenotypic
correlation, slightly conservative.

Three tests are calibrated, one per branch of the arithmetic. Against zero, an
interior point, a plain chi-squared applies and the genetic correlation rejects
0.005, 0.052 and 0.122 at the one, five and ten per cent levels, with
Kolmogorov–Smirnov against uniform giving p = 0.62. Against one, where the null
sits on a bound and the Self–Liang mixture applies, it rejects 0.007, 0.033 and
0.068, with the statistic exactly zero in 55 per cent of samples against the
mixture's expected 50.

The third is the phenotypic correlation, which has no coordinate to pin and is
profiled by substitution instead. Its null is chosen to be awkward: a phenotypic
correlation of nought could come from both components being nought, which would
never exercise the substitution, so the genetic correlation is 0.45 and the
residual −0.4044 and only the sum is nought. It rejects 0.018, 0.055 and 0.107,
with Kolmogorov–Smirnov giving p = 0.71.

**This check earns its runtime.** It found two faults nothing else did: a units
disagreement between the free and constrained fits that made every interval
zero-width, and the same disagreement in the test, which under a true null
rejected about half the time at every level. Neither was caught by the test
suite, by agreement with SOLAR, or by reading the code.

## `phenotypic-correlation-against-solar-2026-08-12.json`

The derived phenotypic correlation against SOLAR's, on all seventy-eight real
GOBS pairs. Largest difference 5.1e-07, median 2.9e-08.

It is derived rather than estimated — the genetic and residual covariances add
and the total variances divide out — so this check cost nothing to run: it is
arithmetic on fits that already existed. Neither Asterism nor SOLAR gives it a
standard error, an interval or a test.

## Elsewhere

Asterism's own evidence is all simulated or GOBS. The univariate real-data check
against SOLAR on the Acoustic v116 batch — nineteen frozen results, seventeen
better-ear thresholds and two PTA composites, reproducing to 8.2e-7 on
heritability — lives with that analysis rather than here, in
`jasa-high-frequency-heritability/reports/`. It is kept there deliberately, so
that using Asterism does not enlarge Asterism.

## `coverage-at-the-real-design-point-2026-08-12.json`

The committed calibration simulates heritabilities of 0.6 and 0.35. The real GOBS
fits run to 0.83, which is nearer a bound, so coverage was measured there too
rather than assumed to carry over. It does carry over: 0.960, 0.953, 0.937, 0.971
and 0.953 against a nominal 0.95, every one inside the band.

**What changes at high heritability is not coverage but which quantity is
fragile.** A heritability of 0.83 leaves little residual variance in that trait,
so the residual correlation is poorly determined: its interval failed to compute
in 23 of 300 replicates and hit a bound in 94 per cent of the rest. The derived
phenotypic correlation computed in all 300 and hit a bound in none.

So where a heritability is high, the phenotypic correlation is the more reliable
of the two to report and the residual correlation the less — the opposite of what
one being derived and the other estimated would suggest. The GOBS height and
weight fit sits in exactly that regime.

## `bivariate-intervals-against-reference-2026-08-12.json`

The compiled profile intervals against independent Python ones, REML, n = 150,
unbalanced. Produced by `checks/bivariate_intervals_against_reference.py`.

**This exists because neither external comparator computes these.** SOLAR gives
standard errors and `regress` gives variance components, so the two-trait
intervals were the one thing Asterism reports that rested on calibration alone.
The two answer different questions and the package wants both: calibration says
a 95 per cent interval contains the truth 95 per cent of the time, which is that
the recipe works; it cannot say the compiled arithmetic agrees with an
independent route to the same definition, because two implementations can both
cover at 95 per cent and still disagree case by case.

All eight endpoints agree to about 1e-5, against a tolerance of 2e-3 set for what
two bisections over two optimisers can honestly demonstrate.

It covers the four estimated quantities and not the phenotypic correlation, which
the reference has no constrained fit for — pinning that one means substituting
for the residual correlation rather than holding a parameter. It rests instead on
its calibration, on its chain rule being checked against a central difference,
and on the four it is built from agreeing here.

## `components-calibration-2026-08-12.json`

One trait and three components — additive, household and residual — where every
other calibration here is one or two components. Produced by
`checks/components_calibration.py`.

Nothing carries over from the two-trait calibration. The intervals are reached by
a different substitution, and the test is a boundary test where the two-trait
correlation tests were interior ones.

Two hundred sibling pairs, n = 400, households of four holding two pairs each.
Interval coverage is 0.945 for the additive share and 0.955 for the household
share against a nominal 0.95. Under a true null — no household effect at all —
the test rejects 0.013, 0.051 and 0.104 at the one, five and ten per cent levels.
The statistic is exactly nought in 41 per cent of samples against the mixture's
expected 50.

It also records power, which is not calibration but decides whether any of it is
worth running: **a test that never rejects is perfectly calibrated.** A true
household share of 0.2 is found 90 per cent of the time at the five per cent
level.

**Correcting a claim made when this was first written**, that the households
must contain unrelated people or none of it works. That is not the condition.
Separability is a property of the whole design and not of any one household: the
three matrices A, H and I have to be linearly independent across all of it. Two
households that are each a single pair identify the model between them if the
pairs differ in relatedness, since a sibling pair gives 0.5·σ²_A + σ²_H and a
spouse pair gives σ²_H, and neither could do it alone.

Unrelated co-residents are therefore the sharpest contrast rather than a
requirement — full sibs living with a half sib do just as well, because 0.5 and
0.25 differ and that is enough. What genuinely fails is a design in which every
co-resident pair in the entire sample has the same relatedness, because A is then
exactly 0.5·H + 0.5·I.

The real GOBS design is far from that. Of 351 households with two or more
measured people, 173 are a pair at relatedness 0.5, 81 a pair at nought and 83
carry varied relatedness within them, and over the whole design 68 per cent of A
is not explained by any combination of H and I.

## `spatial-bootstrap-2026-08-12.json`

The parametric bootstrap for no spatial variance, and whether it holds its level.
Produced by `checks/spatial_bootstrap.py`, which takes about an hour: four
hundred null data sets, each put through a complete 199-replicate bootstrap, so
roughly eighty thousand spatial fits.

**Nothing else here could stand in for it.** When the spatial variance is nought
the decay rate is absent from the likelihood, so the statistic has neither a
chi-squared nor a chi-bar-squared null and there is no table to read it against.
Simulating the reduced model is the only reference there is.

It holds: rejection rates of 0.003, 0.050, 0.102, 0.255 and 0.547 against nominal
0.01, 0.05, 0.10, 0.25 and 0.50, every one inside its band.

**The criterion took three attempts and the first two were wrong**, which is
worth recording because both failures looked like faults in the bootstrap and
were faults in the check.

Comparing the p-values against a continuous uniform failed at D = 0.2725. The
statistic cannot go below nought and is exactly nought in 27.3 per cent of null
data sets — the spatial variance fits to nothing — and every one of those gets a
p-value of exactly one, because no simulated statistic can fail to reach nought.
The Kolmogorov–Smirnov statistic was measuring that atom and nothing else, to
three decimal places.

Removing the atom and asking again failed too, at D = 0.3141, for the same reason
one step removed: the bootstrap replicates carry the atom as well, so no replicate
sitting at nought can exceed an observed statistic above nought, and the largest
p-value such a data set can reach is about one minus the atom. Rescaling by that
brings the statistic to 0.089.

What a p-value must satisfy is validity — P(p ≤ α) ≤ α — not uniformity, and a
bootstrap p-value from a statistic with an atom is valid without being uniform.
The rejection rates are the criterion; the distribution is reported because it is
informative.

## `spatial-intervals-2026-08-13.json`

Coverage of the two things a spatial fit reports beside its p-value: the share of
variance and the distance at which the correlation halves. 250 replicates at
n = 300, produced by `checks/spatial_intervals.py`.

Both cover — 0.984 for the share and 0.976 for the half distance against a
nominal 0.95 — **and the second number should not be read as saying the range is
well estimated.** 98.4 per cent of the half-distance intervals have an endpoint
at a bound. They cover because they are wide enough to reach the edge of the
allowed range almost every time, which is coverage without information. The
share's intervals reach a bound in 5.6 per cent.

That matches what a single fit showed earlier: simulated with a true half
distance of 35 km, the estimate came back at 6.

**So a spatial result should carry the share with its interval and its
bootstrapped p-value, and the range as a point estimate only.** An interval on
the range would look like a result and is not one.

One thing the share's interval cannot do either: its lower endpoint reaching
nought is not a test of whether there is a spatial effect. Under that null the
decay rate is unidentified and the deviance has no chi-squared reference, which
is the whole reason the test is bootstrapped.
