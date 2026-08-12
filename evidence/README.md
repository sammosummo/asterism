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
