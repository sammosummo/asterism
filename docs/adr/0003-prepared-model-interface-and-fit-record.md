# 3. The interface is a sealed prepared model returning a lean frozen record

**Status:** accepted 7 August 2026 in Astrarium, adopted into Asterism on
11 August 2026. Decision 7 of `0001` says the same thing less specifically and
defers to this.

The interface is `model = asterism.prepare(X, K)` and then `model.fit(y)`: an
object holding the validated decomposition sealed inside Rust, with arrays
crossing as buffers rather than as text. The sealed object is the only form in
which "a prepared model is a validated model" (`0002`) is physically
enforceable, and buffers are the natural crossing — a relationship matrix at
pilot scale takes about 10 ms as a buffer against seconds as 65 MB of JSON, with
identical exactness either way.

**No one-shot `fit(y, X, K)` in the first release.** One way in. Both
predecessors grew their interface one convenience at a time.

A fit returns a **lean frozen record**: identity (the estimator echoed as data,
n, the echoed subject-order commitment), estimates (the variance components,
the fixed effects in design-column order, the full log-likelihood), diagnostics
(converged, boundary state, warnings), and reserved slots for the interval and
standard errors. The estimator is chosen when the fit runs, REML by default and
ML available.

## Consequences worth calling out

- **The both-implementations rule.** A field only one implementation can produce
  does not go on the record. This is what makes an independent check checkable
  field by field, rather than two implementations describing the same answer
  differently.
- **Riches stay off the record.** Residual kurtosis, condition numbers, the
  optimiser trace, KKT flags — reachable through diagnostic entry points if ever
  wanted, not carried on every fit. A 44-key dictionary and a 3,040-line schema
  are the two graves this walks between.
- **Additions are optional fields and never breaking**, which is the valve that
  lets the record start lean without being frozen forever.
- **Interval and boundary semantics are deliberately absent here.** The record
  reserves the slots; what goes in them belongs to `0004` and `0005`.
- Under decision 24 of `0001` the record is returned in memory and Asterism
  writes nothing itself. The Python interface may write it down.
