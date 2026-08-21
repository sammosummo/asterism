# PROTOTYPE — Aim 2 grant recruitment design

**Throwaway design experiment, 21 August 2026.** This directory is not
production Asterism, the canonical Aim 2 campaign, or grant text. Generated
state belongs under `artifacts/PROTOTYPE_WIPE_ME/`. Do not lift a number into
the grant unless the full matching null and alternative cells have completed,
their raw results have been retained, and the inferential qualification gate
below has been cleared.

The shareable logic view is
`recruitment_design_prototype.html`. Double-click it; it is one self-contained
file and needs no server. It lets a non-developer inspect the complete design
state and walk through the comparisons without running the simulation.

The current execution boundary is recorded in
`PREFLIGHT_RECEIPT_2026-08-21.md`. It records an intentionally terminated local
cost probe and the Medusa clearance gate; it is not a power verdict.

## The exact question

**What are the measured level and power of the grant design that has actually
been settled, using ordered CDR, with and without the proposed family-history
recruitment extension?**

The experiment answers that question across a specified recruitment-yield sensitivity:
one, one and a half, or two connected adult first-degree relatives per fixed set
of 250 index cases. It also runs binary probable AD as the prespecified outcome
sensitivity and two hearing-measurement bounds. The extension is a 250-person
**structural scenario**, not a demonstrated recruitment yield.

It does not revise the Aims, Strategy, budget, study decisions, or the existing
`campaigns/aim2` results.

## What is fixed, and what is varied

The confirmed core target is up to 600 Acoustic participants, 250 unrelated
clinically adjudicated index cases, and the enrolled adult first-degree
relatives of those cases. Every relative remains connected to the index case in
the relationship matrix. In particular, two relatives make a trio; the second
relative never becomes a singleton.

| Mean enrolled relatives per case | Index-family structure | Core target | With 250-person structural extension |
| ---: | --- | ---: | ---: |
| 1.0 | 250 dyads | 1,100 | 1,350 |
| 1.5 | 125 dyads and 125 trios | 1,225 | 1,475 |
| 2.0 | 250 trios | 1,350 | 1,600 |

These are target counts: `cells.json` and the per-design manifest are the
machine-readable authority for a run. Each result must retain participant and
role counts, family-unit counts, the family-size distribution, and the check
that every claimed relative tie is non-zero in the relationship matrix.

The fitted pedigree projection keeps every claimed index-relative tie and every
structural extension-to-reporter anchor inside the same fitted unit. To stay
within the existing size-eight numerical envelope it can project other SAFS
relationships across unit boundaries to zero, including first-degree ties. The
manifest counts those omitted source ties explicitly. This is therefore a
local-family structural projection, not a claim that the eventual full
relationship matrix has been represented without loss.

The case-relative generator uses adult offspring for both relative positions;
it therefore preserves the complete first-degree structure but does not model
the eventual sibling-versus-child mix. The index case and first relative have
an observed dementia outcome; the second relative contributes hearing,
genotype and covariates without a required dementia outcome, matching the
current Strategy boundary. Acoustic top-up ages are projected from the observed
age distribution, while extension ages are an explicit simulation assumption.

The full grid has 72 cells:

| Factor | Values |
| --- | --- |
| Mean case-relative yield | `1.0`, `1.5`, `2.0` |
| Structural extension | absent, present (+250 people) |
| AD outcome | ordered CDR primary, binary probable-AD sensitivity |
| Hearing measurement | ABR information bound, audiogram information bound |
| Truth | null loading (`a=0, b=.4`), null hearing-to-AD path (`a=.6, b=0`), vertical `a*b=.24` (`a=.6, b=.4`) |

Thus `3 × 2 × 2 × 2 × 3 = 72`. Every design/outcome/measurement combination
has both branches of the vertical union null and the `0.24` alternative at the
same alpha and replicate count. Each cell uses 200 Monte Carlo replicates and 50
parametric-bootstrap draws. Refusals remain in the 200-replicate denominator.
The summary must show rejections, refusals and Monte Carlo uncertainty, rather
than reporting only a proportion.

For ordered CDR, a person's age-indexed dementia rate `r` is divided across CDR
0, 0.5, 1, 2 and 3 as `[1-2r, r, .56r, .31r, .13r]`. CDR 1 or worse is the
case threshold used for affected-proband conditioning. This is a simulation
scenario, not an observed future-cohort distribution. The binary arm preserves
the same age-indexed dementia rate and proband conditioning.

The available Asterism likelihood conditions on the named affected index case.
The settled SR06 decision defines ascertainment more broadly—eligibility,
invitation order, invitation and enrolment as well as case qualification. These
fixed-yield cells therefore assume that relative invitation and enrolment are
ignorable after diagnosis, stage, source, route and covariates. The structural
extension is fitted population-unconditioned because the report-based roster
and its selection denominator do not yet exist. Results answer the conditional
structural question under those assumptions; they do not establish the full
recruitment-event likelihood.

The hearing arms bracket information. They do not describe two expected
protocols, and the all-audiogram arm must not be described as the realised
protocol. The settled endpoint is the instrument-harmonised
conventional-frequency hearing-loss classification, reached by behavioural
audiometry where it can be completed and passive ABR otherwise, with
route-specific error explicit. The runner retains the existing campaign's
placeholder audiometry error variance `.15`, ABR sensitivity `.80`, and ABR
specificity `.85`; these are scenario assumptions, not measured bridge
parameters. No covariates are simulated, so the direction or size of the
planned adjustment's effect on power remains unmeasured.

## Boundary around the extension

`extension=present` asks what 250 additional, structurally connected eligible
SAFS/Acoustic relatives would add **if recruited and measured**. It does not say
that 250 people are recruitable, that their relationships have been normalised,
or that the extension is feasible within the staffing and adjudication budget.
The builder selects unexamined first- or second-degree pedigree members of an
Acoustic participant with a positive authoritative family-history flag; it does
not identify the relative named in the restricted free text. Do not call these
simulated people “family-history recruits”.

The live flags establish that a route exists: 175 of 401 Acoustic respondents
reported Alzheimer's disease or another dementia in the family, across 33 of
46 represented families. The relationship descriptions remain restricted free
text and normalisation is still open. The budget record also identifies a fixed
recruitment envelope and an unresolved choice between Biggs adjudication and RII
CDR staging for apparently affected referrals. A completed power grid therefore
measures incremental information only; it does not establish recruitment,
staging, adjudication, or budget feasibility.

## Run it

All commands below start at the repository root. The generated directory is
deliberately named as disposable prototype state.

Build the cell manifest and identifier-free designs:

```sh
uv run --no-project python campaigns/aim2/prototype_grant_design_20260821/build_designs.py \
  --out-dir campaigns/aim2/prototype_grant_design_20260821/artifacts/PROTOTYPE_WIPE_ME/designs \
  --cells campaigns/aim2/prototype_grant_design_20260821/cells.json
```

Run one deliberately small smoke cell. Its result checks plumbing only and is
not power or level evidence:

```sh
uv run --no-project python campaigns/aim2/prototype_grant_design_20260821/run_cell.py \
  --cells campaigns/aim2/prototype_grant_design_20260821/cells.json \
  --cell 0 \
  --design-dir campaigns/aim2/prototype_grant_design_20260821/artifacts/PROTOTYPE_WIPE_ME/designs \
  --out-dir campaigns/aim2/prototype_grant_design_20260821/artifacts/PROTOTYPE_WIPE_ME/results \
  --replicates 2 \
  --bootstrap 5 \
  --workers 2
```

On Medusa, place `run_cell.py`, `cells.json`, `submit.sbatch` and the built
`designs/` directory in one flat work directory. Build a Linux virtual
environment there from the recorded Asterism source commit and retain the
extension-module checksum; never copy the macOS `.venv/` to Medusa. A one-cell
pilot overrides the array on submission:

```sh
WORK=/net/lus01/home/smathias/asterism-grant-prototype-20260821
sbatch --array=0 --export=ALL,WORK="$WORK" submit.sbatch
```

The full run uses the script's `0-71` array and its fixed 200 replicates, 50
bootstrap draws, 120 CPUs and eight-hour limit:

```sh
sbatch --export=ALL,WORK="$WORK" submit.sbatch
```

Any subset can be resumed without changing the script, for example
`sbatch --array=0,17,35-41 --export=ALL,WORK="$WORK" submit.sbatch`. Medusa
writes one untouched JSON file per cell under `$WORK/results/`. Copy those
files into
`campaigns/aim2/prototype_grant_design_20260821/artifacts/PROTOTYPE_WIPE_ME/results/`
before summarising; do not hand-edit them.

Summarise returned raw results:

```sh
uv run --no-project python campaigns/aim2/prototype_grant_design_20260821/summarise.py
```

The derived summary is
`artifacts/PROTOTYPE_WIPE_ME/summary.json`; the scientific answer and unresolved
gate are written to `PROTOTYPE_VERDICT.md`. Missing, duplicate, mismatched, or
short cells prevent a complete verdict.

## Evidence and status boundaries

- The prototype branch is `codex/mediation-grant-prototype`; the inspected
  starting commit is `9e988b48bd7cfdc20a4bbaacfdb45d002e028b53`. Raw cell
  results must record the code commit that actually generated them. A dirty or
  mismatched execution tree is not grant evidence.
- The earlier 0.660 ordered-CDR result remains a measured result only for its
  explicit 1,600-person campaign scenario. That campaign used 200 probands, 300
  offspring, 600 Acoustic participants and 500 otherwise unexamined pedigree
  relatives; it also separated every second offspring from the proband. It is
  not a result for this confirmed grant architecture and must not be carried
  over or multiplied by an information ratio.
- The vertical test is the only hypothesis test here, at alpha `.025`.
  Horizontal pleiotropy remains an estimand with a 97.5% confidence set and is
  outside this power grid. A flat or unbounded set means not separately
  estimable, never zero.
- This is still a design prototype. At the inspected commit, the human-facing
  README still says no calibrated `a*b` p-value or interval is available even
  though the source now exposes `test_vertical`; that contract/code mismatch is
  itself an unresolved status boundary. A completed grid does not by itself make
  the inferential model analysis-ready. Independent agreement, level/coverage
  qualification and the prespecified profile-inference implementation remain
  separate gates.
- No participant identifiers or restricted clinical values belong in
  `cells.json`, the design files, raw results, HTML, summary, or verdict.
- No file in the mediation grant directory is edited by this prototype.

## Primary-source map

- Settled inferential roles, hearing endpoint, ordered-CDR scenario and
  case-relative recruitment: `/Users/samuelmathias/MathiasLab/staging/studies/planned/mediation/DECISIONS.md:417-448`.
- Grant sample and outcome language:
  `/Users/samuelmathias/MathiasLab/staging/studies/planned/mediation/grant-applications/SPECIFIC_AIMS_REVISED_2026-08-19.md:31-43`.
- The warning against mixing designs or multiplying unmeasured information,
  and the 200-replicate/refusal standard:
  `/Users/samuelmathias/MathiasLab/staging/studies/planned/mediation/grant-applications/RESEARCH_STRATEGY_AIM2_REVISIONS_2026-08-19.md:13-31,63-79,146-195`.
- Live family-history evidence and the unresolved normalisation step:
  `/Users/samuelmathias/MathiasLab/staging/studies/planned/mediation/preliminary/sr09-family-history/README.md:10-53,94-124`.
- Extension cost, staffing limit and unresolved staging/adjudication boundary:
  `/Users/samuelmathias/MathiasLab/staging/studies/planned/mediation/grant-applications/BUDGET_IMPACT_OF_DESIGN_CHANGE_2026-08-19.md:17-29,122-176`.
- The old campaign's participant definitions and lost second-offspring tie:
  `campaigns/aim2/build_design.py:12-25,42-53,140-152,212-244`.
- Existing measurement placeholders and ordered-CDR shares:
  `campaigns/aim2/campaign_levers.py:63-74,115-118`.
- The old ordered-CDR null and alternative cells:
  `campaigns/aim2/results/aim2_levers_cell_12.json:1-14` and
  `campaigns/aim2/results/aim2_levers_cell_13.json:1-14`.
- Current latent-mediation interpretation and inferential boundary:
  `README.md:307-331`; agreement proves fidelity rather than correctness:
  `docs/adr/0006-agreement-proves-fidelity.md:8-16,34-44`.
