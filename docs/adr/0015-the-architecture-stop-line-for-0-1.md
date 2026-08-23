# 15. The architecture stop line for 0.1

**Status: accepted.** Asterism 0.1 waits for the shared interval implementation,
including intervals for mean-diagonal component proportions; reconciled
convergence rules across supported models; consistent refusal of non-finite
inputs; migration of supported analysis scripts to the documented Python
interface; green formatting, Clippy, Ruff, Python-style, test and installed-wheel
checks; and a complete statistical methods specification and evidence tied to
the measured code.

The documentation part of the gate includes a canonical bibliography,
resolvable citations, defined equation symbols, equation-to-record mappings,
focused identity tests and a 0.1 changelog entry. Installed-wheel tests cover
CPython 3.13 and 3.14 on macOS arm64 and `manylinux_2_17_x86_64`.

The same stop line includes the subject-order commitment already promised by
ADRs 0002 and 0003, immutable release-build identity, the structured support
manifest, supported-design preflight and the standard analysis receipt. These
are completion of existing interface and reproducibility contracts, not the
larger object redesign deferred below.

The larger Python/Rust object redesign, complete typed-error overhaul and file
splitting do not block 0.1. They may follow as deliberate architectural work;
performing them now would enlarge the release without changing its supported
scientific results. The joint repeated-audiogram model likewise remains on its
later-release branch and does not block 0.1.
