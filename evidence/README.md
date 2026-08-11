# Evidence

Results of checks that take too long to be ordinary tests, kept so that a claim
about Asterism's behaviour points at a number somebody can find.

Nothing here is a status, a gate or a certificate. A file in this directory says
what was measured, on what, when, and with which seed. It says nothing about
whether anything is approved.

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

The seeds are in the file. The whole thing reruns in about twenty seconds.
