# 2. Validation happens once on the prepared model, and cannot be switched off

**Status:** accepted 7 August 2026 in Astrarium, adopted into Asterism on
11 August 2026. Adopted for its substance; the wording is brought into line with
`CONTEXT.md`, which drops "lane", "seam", "oracle" and "gate".

Arrays reach Asterism only through a **prepared model** built once from the fixed
effect design and the relationship matrices. Building it validates — exact
symmetry, positive semi-definiteness through the eigendecomposition the fit needs
anyway with a −1e-9 floor, finiteness, shapes, design rank — and decomposes.
Every fit then runs against it.

Because the expensive check is fused with a decomposition the fit already
requires, validation is effectively free. That is why **no `validate=False`
exists**: a prepared model being a validated model is an invariant rather than a
default. Validating on every call was rejected at 90 times the fit cost — a
1,266 ms eigendecomposition against a 13 ms Cholesky at n = 1904 — and a hidden
cache keyed on the matrices was rejected as invisible state.

## Consequences worth calling out

- **The objective is fixed by the interface and never inferred from the data.**
  The old engine detected binary and ordered responses unconditionally and
  silently swapped objective, which was verified to give h² of 0.885 against
  0.581 on identical data. Under decision 21 of `0001` that code does not come
  across at all, so the hazard is now absent rather than merely unreachable. A
  binary-looking response yields a warning carried on the record, never a
  different likelihood.
- **A relationship matrix is validated as mathematics, not as provenance.** No
  assertion that the diagonal is at least one, and no kinship-range check, so an
  empirical genomic relationship matrix needs no change to the interface. That
  the intended matrix is twice the kinship is documentation.
- **Conversions are exact or refused.** Integer and 32-bit floats widen to 64-bit
  and column-major converts to row-major silently, because those are value-exact.
  Non-finite values, mismatched shapes and a rank-deficient design fail closed
  with stable error codes. No silent dropping of rows: complete cases are the
  caller's business, before the prepared model is built.
- **Every check is implemented once, in Rust.** The Python interface marshals and
  translates errors, and does no validation of its own.
- **Row alignment is guarded by a subject-order commitment** — a sha256 over the
  identifiers in fit order, accepted when the model is built and echoed on the
  result. Raw arrays cannot carry identifiers, so this is what replaces
  re-projecting them.
