# 5. A boundary fit is reported by state, keeps one interval shape, and has no standard error

**Status:** accepted 7 August 2026 in Astrarium, adopted into Asterism on
11 August 2026. Decision 5 of `0001` handles a component sitting on its bound by
refitting the nested model; this governs how whatever comes back is presented.

A fit whose h² lands on a bound is reported rather than repaired. The record
carries `boundary` as one of interior, lower or upper — a state, not advice, with
no warning attached. The interval keeps **the same shape it always has**, so a
lower-boundary fit reads `[0.0, u]` with the lower endpoint flagged as limited.
Standard errors are a per-parameter mapping from which one that is undefined at
the optimum is **absent**: never NaN, never zero, never a one-sided value dressed
as symmetric, with the reason one field away in `boundary`.

## Consequences worth calling out

- **No results that change shape.** A record whose type varies at the boundary
  invites conditional handling downstream, which is exactly what scoring the
  coverage check unconditionally — every replicate counted, no boundary cells
  dropped — exists to forbid.
- **Uncertainty at a bound is the profile interval's job** (`0004`: defined at the
  bound and calibrated there), not a curvature standard error's. That is why the
  missing standard error is a correct report rather than a gap to fill.
- Decision 14 of `0001` adds a second reason. Standard errors come from a Hessian
  taken by differencing the analytic gradient, so at a boundary they would be
  both degenerate and differenced.
- Every field here is fillable by an independent check as well as by the crate,
  which is what `0006` requires.
