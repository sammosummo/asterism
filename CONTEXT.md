# Asterism

Quantitative genetics software for the Mathias Lab.

## Language

**Asterism**:
One Rust crate named `asterism` with a Python interface around it. One product,
one version, one name. What Asterism is at any moment is what its decision
record describes; it gains a capability when that record gains it, which happens
when a planned analysis needs it and not before.
_Avoid_: engine, SOLAR replacement, platform, framework

**Component**:
An estimated variance contribution, named and bound to a known relationship
matrix. Components multiply freely; traits are what cost.
_Avoid_: variance term, effect

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

**Fit record**:
What a fit returns, held in memory and carrying its seed. The Python interface
may write it down; Asterism writes nothing by itself.
_Avoid_: receipt, artefact, run record

**Deferred capability**:
Something Asterism does not do yet and will do when an analysis needs it.
Survival, longitudinal, Tobit, signal detection and prospective design analysis
are all deferred. The list carries no order, priority or dates, because a list
with an order is a roadmap.

Liability and threshold models, gene-by-environment, spatial, BLUP and
association have each left this list by an amendment naming the analysis that
needed them.

Two capabilities arrived on 15 August 2026 without doing so: the latent
mediation model and the variant-set and local-IBD matrix builders. No analysis
was named for either. That is recorded here rather than tidied away, because
the rule exists to stop this project growing faster than it can be checked, and
reading the rule beside the two exceptions is the only way the next decision
gets made with that in view.

**SOLAR**:
A comparator, never the definition of correct. Differences between Asterism and
SOLAR are recorded rather than treated as defects.

**The Python interface**:
The boundary between the crate and Python. It takes tables or arrays already in
memory and returns a fit record. Asterism knows nothing of file formats,
databases or the tables that go into papers.
_Avoid_: the seam, bindings layer
