# The models

One page per supported model, each carrying everything that model needs: how to
use it, the equations behind it, a summary of what has been measured about it,
and its interface.

| Model | Page |
| --- | --- |
| One trait, one variance component | [one-trait.md](one-trait.md) |
| Several covariance components | [several-components.md](several-components.md) |
| Two traits | [two-traits.md](two-traits.md) |
| Binary traits | [binary.md](binary.md) |
| Censored traits | [censored.md](censored.md) |
| Mixed pairs | [mixed.md](mixed.md) |
| Gene by environment, measured | [gxe-measured.md](gxe-measured.md) |
| Gene by environment, binary | [gxe-binary.md](gxe-binary.md) |

Every comparison and simulation behind those summaries is in the [validation
record](../validation.md), in full.

What every model shares — the notation, the Gaussian likelihood and the
numerical implementation — is in
[statistical-methods.md](../statistical-methods.md). What is common to running
the checks is in [development.md](../development.md). Models Asterism does not
support are in [outside-support.md](../outside-support.md).

Each page's **Interface** section is generated from the package's own
docstrings by `tools/build_model_pages.py`, using the entry points
`release.toml` declares. Everything above that marker is written by hand.
