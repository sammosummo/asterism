# Domain docs

How the engineering skills should consume this repository's domain documentation
when exploring the code. Asterism is single-context: one `CONTEXT.md` and one
`docs/adr/`, both at the root.

## Before exploring, read these

- **`CONTEXT.md`** at the root — the project's vocabulary, including the terms it
  deliberately avoids.
- **`docs/adr/`** — read the decision records that touch the area you are about to
  work in.

If either is missing, **proceed silently**. Do not flag the absence and do not
suggest creating them upfront. The `/domain-modeling` skill (reached via
`/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily, when
terms or decisions actually get resolved.

## File structure

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-variance-components-estimator-parameterisation.md
│   └── 0002-validation-on-the-prepared-model.md
├── src/          ← the Rust crate
└── python/       ← the Python interface
```

## Use the glossary's vocabulary

When your output names a domain concept — in an issue title, a refactor proposal,
a hypothesis, a test name — use the term as defined in `CONTEXT.md`. Do not drift
to the synonyms the glossary explicitly avoids; each `_Avoid_` line is there
because that word was tried and found wanting.

If the concept you need is not in the glossary yet, that is a signal. Either you
are inventing language the project does not use, in which case reconsider, or
there is a real gap, in which case note it for `/domain-modeling`.

## Flag ADR conflicts

If your output contradicts an existing decision record, surface it rather than
silently overriding it:

> _Contradicts ADR-0006 (agreement proves fidelity) — but worth reopening because…_
