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

Thirty families of six, n = 180, REML. Interval coverage is 0.957, 0.967, 0.960
and 0.953 against a nominal 0.95. The test of a correlation against zero — an
interior point, so a plain chi-squared — rejects 0.005, 0.052 and 0.122 at the
one, five and ten per cent levels, with Kolmogorov–Smirnov against uniform
giving p = 0.62. Against a correlation of one, where the null sits on a bound and
the Self–Liang mixture applies, it rejects 0.007, 0.033 and 0.068, with the
statistic exactly zero in 55 per cent of samples against the mixture's expected
50.

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
