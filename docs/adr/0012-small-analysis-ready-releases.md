# 12. Small analysis-ready releases and two verification tiers

**Status: accepted.** Each Asterism release has a modest set of supported
analyses, all of which must pass their scientific release checks. Capabilities
outside the set remain deferred or explicitly not ready rather than partly
activating inside one permanently incomplete release. Every change passes the
codebase-quality gate, while a release also passes the independent checks,
simulations, evidence and real-design checks required by its supported analyses.
This keeps expensive scientific checks tied to the claims they support without
making every change rerun every scientific campaign.

The supported analyses in **Asterism 0.1: trait modelling** are one-trait
Gaussian heritability; several covariance components; bivariate genetic
correlation; continuous and discrete gene-by-environment; binary liability;
censoring-corrected one-trait audiogram heritability; the genetic correlation
between a binary psychiatric diagnosis and a censored hearing threshold; and
spatial-component presence testing.
Continuous gene-by-environment support covers the exponential and
random-regression models after an independent full-fit and target-design check;
the powered-exponential models remain outside 0.1. Discrete
gene-by-environment support covers the genetic-correlation interval and the
calibrated genetic tests after an independent full-fit check, while
group-specific heritabilities remain descriptive. Spatial support covers the
presence test after a target-layout check, while range remains descriptive
because it is often weakly identified.
Several-component support reports mean-diagonal contributions and proportions,
with intervals for those same quantities, when every relationship matrix has a
positive mean diagonal. Zero-diagonal kinship-class matrices instead report
coefficients and contrasts. Raw coefficient proportions remain diagnostic and
are never presented as variance shares.
One-trait censored audiogram support includes 16 and 18 kHz only after interval
simulations at their observed censoring levels, approximately 52 and 75 per
cent, pass. Mixed bivariate support includes the intended pairing of a binary
psychiatric diagnosis with a censored hearing threshold only after that exact
combination is checked; its supported result is the genetic correlation and the
other fitted quantities remain descriptive.
Prediction, association and variant-set discovery, inferential latent
mediation, autoregressive modelling and the joint repeated-audiogram
decomposition belong to later releases.

Support applies within the design range actually measured: sample size, family
size, relationship-matrix properties, censoring and trait type. A design outside
that range needs its own design-specific check before its results are
reportable. Pre-release checks may use the real pedigree, matrices, missingness,
covariates and censoring pattern with simulated outcomes whose truth is known;
the actual outcomes are fitted only after release, so their results cannot
influence the acceptance rules.

The supported design ranges and the pass rules that establish them live in one
machine-readable release manifest. A preflight compares a proposed analysis
with those ranges before fitting and names the missing design-specific check
when the design falls outside them. Each supported analysis has a small
synthetic end-to-end example which uses only the documented Python interface,
runs against the built wheel and produces the same receipt as a real analysis.

Every supported analysis is also covered by the statistical methods
specification. It gives the estimand, notation, model and likelihood,
parameterisation, estimator, interval and test, boundary behaviour, assumptions,
measured limitations and primary references. Equations are rendered as
mathematics rather than approximated in prose or code blocks, and author-year
citations resolve to a complete bibliography with stable DOI or archival links
where available. The document must be detailed enough to audit the
implementation and support a manuscript method; an API description alone does
not satisfy this release requirement.

`docs/references.bib` is the canonical bibliography. The methods specification
uses author-year citations, preferring primary methodological sources with DOI,
PMID or stable archival links. It distinguishes a published method from an
Asterism-specific implementation choice or simulation result. Automation checks
that citation keys resolve and bibliography entries carry the required
identifiers. Every equation defines its symbols and maps each reportable
quantity to its public fit-record field; focused identity tests protect
relationships that could drift from the implementation. These checks do not
pretend to prove prose scientifically correct.

Once every configured check passes, release is automatic: the commit is tagged
and a release page in the private GitHub repository retains wheels for the
development Mac and a portable `manylinux_2_17_x86_64` ABI3 wheel, together
with their source commit, checksums and a summary of the checks that passed. It
contains no scientific data. Reportable analyses install one of those saved
wheels and never build from a mutable working checkout. No separate human
approval is required.
Every scientific check therefore has a machine-readable pass rule fixed before
the release run; a check without one blocks automatic release.
The Linux wheel is always built and tested in Linux automation. When a supported
analysis will run on Medusa, release also waits for that wheel to pass a small
non-participant import-and-fit smoke test there.

Mac and Linux runs must agree exactly on outcome status, refusal codes,
boundary states and field presence. Numerical values agree within pre-written,
model-specific tolerances rather than bit for bit. Each supported target-sized
analysis must also complete within an explicit feasible time-and-memory budget
on its intended host. **Withdrawn by ADR 0019: no resource budget gates a
release.** Broader performance measurements are retained as
benchmarks but small changes do not block release until a stable runner and its
normal variability have been measured.

The 0.1 Python contract is standard, GIL-enabled CPython 3.13 and 3.14 on macOS
arm64 and `manylinux_2_17_x86_64`, declared as
`requires-python = ">=3.13,<3.15"`. The exact built wheels are installed and
tested on both Python versions on both platforms. PyPy and free-threaded CPython
remain outside 0.1 until separately measured.

Medusa's architecture and glibc version are verified when it is reachable. If
the portable wheel cannot run there, any release needed for a Medusa analysis
waits for a compatible saved artifact and its smoke test; a mutable source
installation is never substituted.

Every real analysis's caller-side wrapper writes a standard machine-readable
receipt beside its controlled analysis outputs. The receipt identifies the
package version, source commit, saved-wheel checksum and dependency-lock
checksum; records the model settings, a non-identifying design summary, input
commitments and the fit record; and gives the final reportable, diagnostic-only
or refused outcome. It contains no participant data and is not copied to the
Asterism release page. Numerical fitting classes remain in-memory calculations
and perform no file I/O.

A development checkout has one canonical public version, `0.1.0.dev0`, rendered
as the SemVer-equivalent `0.1.0-dev.0` where Cargo requires it. Only a deliberate
change of that canonical version to exact `0.1.0` starts release and creates
`v0.1.0`. The release manifest is authoritative; automation renders and verifies
the Python package, compiled module, wheel and Cargo metadata against it and the
tag. The release commit reruns
every scientific check supporting 0.1 because the existing evidence does not
identify the exact code it measured. Later releases may reuse evidence only
when recorded source and dependency hashes prove that the relevant calculation
is unchanged. When the version-change commit passes, automation creates its tag
and private release page without another approval. Models outside the supported
analyses still pass the ordinary formatting, style, unit, package and
installation checks because they remain shipped and importable; only their
expensive scientific checks are left for their later release.

The versioned repository is the 0.1 documentation product: its README,
supported-analysis table, statistical methods specification, API reference,
synthetic examples and release notes must be complete. A separate documentation
website does not block 0.1.

Every release also has a changelog entry recording its supported analyses and
quantities, new evidence, compatibility changes, deprecations and known
limitations. Automation generates the factual support and evidence inventory
from the release manifest; its interpretation is written in plain language.
