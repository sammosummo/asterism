# The models

One page per model in the public release manifest, each carrying how to use
it, the equations behind it, a truthful summary of what has been measured, and
its interface. Inclusion here is not itself a support claim: the
several-component censored model is supported in 0.2 for fitting, intervals and
an explicitly asymptotic test. Its analytic p-value must not be
reported for the exact failed 1,909-person, four-component target at 75%
expected censoring without a design-specific simulated null; this is not a
universal censoring threshold. The bootstrap remains experimental. Published
versioned support and public-but-unsupported objects are kept separate in
[api-support.md](../api-support.md).

| Model | Page |
| --- | --- |
| One trait, one variance component | [one-trait.md](one-trait.md) |
| Several covariance components | [several-components.md](several-components.md) |
| Two traits | [two-traits.md](two-traits.md) |
| Binary traits | [binary.md](binary.md) |
| Censored traits | [censored.md](censored.md) |
| Censored traits, several components | [censored-components.md](censored-components.md) |
| Mixed pairs | [mixed.md](mixed.md) |
| Gene by environment, measured | [gxe-measured.md](gxe-measured.md) |
| Gene by environment, binary | [gxe-binary.md](gxe-binary.md) |

The [validation record](../validation.md) preserves the comparisons and
simulations behind those summaries, including historical results that have been
withdrawn as qualification evidence.

What every model shares — the notation, the Gaussian likelihood and the
numerical implementation — is in
[statistical-methods.md](../statistical-methods.md). What is common to running
the checks is in [development.md](../development.md). Models Asterism does not
support are in [outside-support.md](../outside-support.md).

Each page's **Interface** section is generated from the package's own
docstrings by `tools/build_model_pages.py`, using the entry points
`release.toml` declares. Everything above that marker is written by hand.
