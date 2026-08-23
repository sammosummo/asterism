# Asterism

Quantitative genetics software for the Mathias Lab.

## Language

**Asterism**:
One Rust crate named `asterism` with a Python interface around it. One product,
one version, one name. What Asterism is at any moment is what its decision
record describes; it gains a capability when that record gains it, which happens
when a planned analysis needs it and not before.
_Avoid_: engine, SOLAR replacement, platform, framework

**Supported analyses**:
The named analyses and reportable quantities a release promises are ready.
Capabilities outside the set remain deferred or explicitly not ready and do
not block the release.
_Avoid_: analysis envelope, feature-complete scope, universal readiness

**Analysis-ready release**:
A fixed, installable build whose supported analyses pass their scientific
release checks and whose codebase-quality gate passes. It makes no claim about
other analyses.
_Avoid_: feature-complete release, finished package

**Scientific release checks**:
The evidence and real-design checks required for the supported analyses in a
release. They need not cover every capability Asterism contains.
_Avoid_: analysis-envelope gate, full validation suite, qualification gate

**Codebase-quality gate**:
The repeatable package-wide checks a build must pass before it can be an
analysis-ready release. It establishes engineering integrity, not scientific
evidence for a reported quantity.
_Avoid_: beauty, polish

**Supported design range**:
The measured sample sizes, family sizes, relationship-matrix properties,
censoring levels and trait types within which a supported analysis may produce
a reportable result. The range is machine-readable and belongs to a particular
release and its evidence.
_Avoid_: generally supported, should work, analysis envelope

**Reportable result**:
A result from a supported analysis, fitted with a fixed release build to a
design inside its supported design range, whose required fits and profile
evaluations converged and whose pre-written acceptance rules passed. A
converged boundary estimate may be reportable; a failed fit may not.
_Avoid_: successful-looking fit, usable result

**Diagnostic-only result**:
An inspectable fit record that preserves a finite candidate and the reason it
did not become reportable. It is evidence for diagnosis, never permission to
quote its estimates, intervals, tests or predictions as findings.
_Avoid_: partial result, report with caution

**Refused result**:
A standard-analysis outcome carrying a stable refusal code when the inputs,
estimand or fit cannot produce even a usable diagnostic candidate. The public
estimator may raise the documented stable exception; the caller-side wrapper
records it as the refusal. A refusal is an outcome rather than a missing result.
_Avoid_: crash, failed silently

**Analysis receipt**:
A machine-readable record stored with an analysis that identifies the saved
wheel and dependencies, commits to the inputs without copying them, records the
model and non-identifying design summary, retains the fit record when a
candidate exists or the refusal code when none does, and states whether the
outcome was reportable, diagnostic-only or refused.
_Avoid_: release evidence, log file, results dump

**Statistical methods specification**:
The versioned scientific account of each supported analysis: estimand,
notation, model and likelihood, parameterisation, estimator, interval and test,
boundary behaviour, assumptions, measured limitations and primary references.
Its equations and citations must be sufficient to audit the implementation and
write a manuscript method without reconstructing the method from source code.
_Avoid_: methods overview, API guide, prose notes

**Component**:
A named relationship matrix with its estimated coefficient. Their product is
the component's covariance contribution. Components multiply freely; traits
are what cost.
_Avoid_: variance term, effect

**Mean-diagonal contribution**:
A component's coefficient multiplied by its relationship matrix's mean
diagonal. It is invariant to positive rescaling of that matrix and is the basis
of a reportable component proportion when every mean diagonal is positive.
_Avoid_: raw coefficient share, generic variance share

**Relationship matrix**:
The known matrix a component scales. Additive kinship, household, genomic
relationship, marker IBD and dominance are all this one thing.

**Builder**:
A convenience that makes a standard relationship matrix from data — kinship from
a pedigree, household from `hhid`. Asterism's interface takes the matrix, so a
builder is a courtesy rather than the boundary. Curating SAFS pedigrees belongs
to the `pedigrees` project, not here.

**The general estimator**:
The definition of correct. Where a fast route disagrees with it, the fast route
has the bug and never the reverse.

**Fast route**:
An optimisation for a case common enough to be worth special code. It is never
an alternative definition, and every result records which route produced it.
_Avoid_: fast path, lane, backend

**Independent check**:
A second implementation of the same calculation. Agreement between the two
proves fidelity and never correctness; disagreement is the informative outcome.
Any capability added to Asterism arrives with one.

A check that cannot run is not one. The matrix builders arrived with checks
against R's `SKAT` and `rres`, neither of which was installed, so their stated
agreement could not be reproduced by anyone. Both packages were installed and
both checks run and pass; the lesson kept is that a check is only evidence once
somebody has run it on the machine that makes the claim.
_Avoid_: oracle, reference lane

**The coverage check**:
A simulation that fits many datasets whose true heritability is known and counts
how often the interval contains it. It is what chooses an interval recipe, and
nothing about reading the code can tell you the same thing.
_Avoid_: EXIT gate, qualification, ratified

**The interval**:
The profile-likelihood interval for a reported quantity. One module builds it
and one record carries it, whichever family asked: what changes between families
is the objective being profiled and nothing else. Its recipe belongs to `0004`
and its shape to `0011`; neither the recipe's name nor its endpoints are a
contract term, because the coverage check has to be free to change them.

A boundary point belongs to the interval only where a coverage simulation has
scored it there. An absent verdict means nobody has measured it yet, not that
the question does not apply.
_Avoid_: error bar, bounds, CI

**Fit record**:
What an estimator returns in memory. It carries the model's scientific fields,
diagnostics and immutable build and subject-order identity; it carries a seed
only when the calculation is stochastic. Fitting classes perform no file I/O.
_Avoid_: receipt, artefact, run record

**Deferred capability**:
Something Asterism does not do yet and will do when an analysis needs it.
Survival, longitudinal, signal detection and prospective design analysis are all
deferred. The list carries no order, priority or dates, because a list with an
order is a roadmap.

Liability and threshold models, gene-by-environment, spatial, BLUP and
association have each left this list by an amendment naming the analysis that
needed them.

The variant-set builders left the list on 16 August 2026, named by the gene and
variant finding effort: gene and pathway scans on the neuroimaging traits, the
biomarkers and the auditory traits, with association follow-up on anything that
comes up. `AssociationModel` already does the follow-up.

**Linkage was considered on the same day and dropped, on measured power rather
than on taste.** This is separate from the gene and pathway scans, which stay.
 The local-IBD builders were removed with it. Simulated at 900
people with a rare variant of large effect, and comparing each method at the
threshold a real scan demands — about 5e-5 for linkage against 5e-8 genome-wide
or 2.5e-6 gene-based — linkage lost in every case tried. For a variant private
to one family, where its indifference to the kind of variant would have been
worth something, its power was nought while a gene-based test reached 0.568.
Where linkage finally worked, at a three standard deviation effect shared
across fifteen families, association was already certain to find it.

Two things follow. Whole-genome sequencing removes most of the reason to infer
descent at all, since the variants can simply be observed. And where a variant
cannot be observed — a repeat expansion, a structural variant, an unmappable
region — the answer is a caller built for that kind of variant, not a method
that ignores the question, because linkage has no power there either.

The latent mediation model left the list on 16 August 2026, named by the
mediation grant, which is the analysis it was built for.

Tobit left the list on 18 August 2026, named by the extended high-frequency
audiogram. An audiometer stops at its maximum output, so a threshold nobody
reached is a bound rather than a value, and the share of thresholds that hits
one climbs with frequency: 0.2% at 500 Hz, 16% at 12.5 kHz, 52% at 16 kHz and
75% at 18 kHz. The documented `tobit_*` Python functions fit one such trait and
return the heritability of the complete variable — the number there would have
been had the instrument reached far enough — which is not the number a Gaussian
fit to values replaced by their limit returns. The documented
`mixed_bivariate_*` functions fit a pair in any combination of continuous,
binary and censored, named by the genetic correlation between a psychiatric
diagnosis and hearing.

The variant-set kernels stay for the same effort, and Asterism will grow the
test that belongs with them. Using SKAT through R would mean treating the
method as a black box; the point of having it here is to be able to change it
for traits and designs that the published form does not cover.

**SOLAR**:
A comparator, never the definition of correct. Differences between Asterism and
SOLAR are recorded rather than treated as defects.

**The Python interface**:
The boundary between the crate and Python. It takes tables or arrays already in
memory and returns a fit record. Asterism knows nothing of file formats,
databases or the tables that go into papers.
_Avoid_: the seam, bindings layer
