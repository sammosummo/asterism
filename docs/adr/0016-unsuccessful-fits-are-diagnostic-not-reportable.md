# 16. Unsuccessful fits are diagnostic, not reportable

**Status: accepted.** A supported analysis has three possible outcomes:
reportable, diagnostic-only and refused. A reportable result has valid inputs,
lies inside the release's supported design range, has finite required fields,
and has converged both its free fit and every fit used for its required
inference. An interval has no failed profile evaluations. Additional
analysis-specific requirements come from the pre-written machine-readable pass
rule.

A finite candidate that does not meet those conditions is retained as a
diagnostic-only fit record, together with its convergence verdict, stopping
reason and other diagnostics. The standard analysis path does not place its
point estimates, standard errors, tests, intervals or predictions in a report
table. There is no override which relabels `converged: false` as reportable.
Invalid inputs, an unidentified or undefined estimand, or the absence of a
usable candidate produce a stable refused result instead.

The existing public estimators may express that refusal as a stable exception;
the typed-error overhaul remains deferred. The caller-side analysis wrapper
catches the documented stable exception and records its code and message as a
refused outcome. It does not turn unexpected programming errors into scientific
refusals.

A converged boundary estimate is not an unsuccessful fit. It remains
reportable when the supported boundary rule passes; its boundary state and
limited interval endpoints are retained, and a quantity which is undefined at
that boundary is absent rather than replaced by zero or NaN.

The fit record remains inspectable because diagnosing a failed search is real
work and discarding its best candidate would make that work harder. Keeping the
record does not weaken the reporting rule: the outcome field, analysis receipt
and standard reporting path all distinguish diagnostic evidence from a
scientific finding.
