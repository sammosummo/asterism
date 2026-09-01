# 18. Asterism is published with the papers that cite it

**Status: accepted.** ADR 0001's decision 23 already required this, in one
sentence: Asterism is an internal tool, but "it must be citable and reproducible
when a paper uses it — a licence, a version and an archived copy at
publication". ADR 0012 then settled the release in detail, carried the **version**
forward at length, and dropped the **licence** and the **archived copy** without
saying so. It put a release page in a private repository in their place.

Nothing was decided against them; they were simply lost between two records. So
the repository has no licence, no citation file and no archive, and a paper
citing Asterism today has nothing to cite. This record restores the two that
went missing and settles what publication means.

## The licence is MIT

MIT, not the dual MIT/Apache-2.0 that ADR 0001 recorded as a working
recommendation. The recommendation was never a decision, and one permissive
licence is easier to explain to a journal, a collaborator and a reader than two.

The copyright line names Sam, and it stays that way: he holds the copyright,
not Boston Children's Hospital. This was left open when the record was first
written and has since been settled. `LICENSE` needs no change.

## The provenance question is closed

ADR 0001 left two things open and one of them was "whether the contributor
provenance record that document requires was ever written". It is not written,
and it does not need to be, because the premise behind it does not hold.

**Asterism is original work.** It depends on third-party packages — NumPy,
PyO3, nalgebra, the L-BFGS-B translation and the rest — which are ordinary
dependencies used under their own licences and named in the lock files. It
derives from nothing else.

SOLAR keeps appearing in these records for a historical reason rather than a
legal one: two of the three predecessors this package replaced carried SOLAR in
their names, and a document in one of them examined whether the published SOLAR
tree could be reused. The answer there was that it could not be, and Asterism
did not. **Asterism has no affiliation with SOLAR.**

Comparing answers with another program is not derivation from it. ADR 0006 says
why the comparisons exist — agreement proves fidelity, never correctness — and
the same reasoning is why Asterism is checked against R's `regress`, `censReg`
and SKAT, against `spaMM` and `geoR`, against MCMCglmm, and against a published
red deer analysis, as well as against SOLAR. SOLAR has exactly the standing R
has in this package: a program whose answers are worth agreeing with. It is
named more often only because it is the tool this laboratory is leaving.

No record needs to raise this again.

## Public when the papers are, and not before

The repository becomes public at the same time as the papers that cite it, so
that a reader who follows a citation finds the software. Not earlier: a build
whose checks are not configured refuses to produce a receipt at all, which is
not useful to a stranger, and there is no reason to answer questions about it
before there is a paper to anchor them.

The archive is a DOI, minted by Zenodo from a public GitHub release. Zenodo
applies no quality bar of its own — the bar is the release manifest's, and it
stays exactly where ADR 0012 put it. A DOI is a permanent name for an exact
artefact, not a claim that the artefact is good. Each paper cites the version
DOI of the release it actually ran; `CITATION.cff` carries the concept DOI.

**Publication does not relax a single scientific gate.** A public release is
still refused until its pass rules pass. (Design ranges, named here when this
was written, were removed by [ADR 0020](0020-no-design-range-gates-a-result.md);
what a receipt records is now
[ADR 0021](0021-the-receipt-states-facts-not-a-verdict.md).)

## What publication actually took

Done on 26 August 2026, in this order:

- The repository was made public.
- Zenodo archived the release. The concept DOI, 10.5281/zenodo.22113714, is in
  `CITATION.cff`; each paper cites the version DOI of the release it ran, which
  for 0.1.0 is 10.5281/zenodo.22113715. Zenodo only archives releases created
  after its switch is flipped, so the existing GitHub release had to be deleted
  and recreated from the same assets. The tag never moved.
- The `Private :: Do Not Upload` classifier came off at 0.1.1. It is baked into
  a built wheel's metadata, not only `pyproject.toml`, so removing it needs a
  rebuild: the 0.1.0 wheels carry it and PyPI would refuse them. That is why
  the distributed version is not the first archived one.

The copyright question this record left open is settled: Sam holds it, and
`LICENSE` needed no change.

Resolved before 0.2: the distribution name `asterism` was later taken on PyPI
by unrelated software. Asterism therefore publishes checksummed wheels on its
GitHub release page; choosing a distinct PyPI distribution name is a separate
future decision. The installed import remains `asterism`.

## One question this record does not settle

Whether Asterism should have a documentation site of its own, rather than the
Markdown that GitHub renders in place.

It is left open because the answer depends on something not decided here. A
site is wanted when readers arrive who will not open a repository — and for the
audience the README names, a quantitative geneticist reading a paper's methods,
GitHub's own rendering may be enough. There is also a practical constraint:
GitHub Pages needs a paid plan on a private repository, and this account does
not pay for hosted minutes, which is why its Actions are switched off
altogether. Once the repository is public, Pages becomes free and the question
is purely whether the site earns its upkeep.
