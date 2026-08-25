# 21. The receipt states facts, not a verdict

**Status: accepted.** ADR 0016 gave a supported analysis three outcomes —
reportable, diagnostic-only and refused — and told the reader that a
diagnostic-only fit's estimates do not belong in a report table. That part of
ADR 0016 is superseded, and ADR 0017's receipt vocabulary follows.

## What it did

`run_analysis` inspected the fit it had just made, decided whether the result
was fit to publish, and stamped the receipt with the answer. A converged fit
with finite values and no failed profile was `reportable`. Anything else was
`diagnostic-only`, and the documentation said its numbers must not enter a
report.

## Why it goes

**Deciding what may be reported is not the software's job.** It is the
scientist's, and then the reviewers'. A package that fits a model and hands
back an estimate has no standing to say whether that estimate belongs in a
paper. No statistical software does this, because the question is not one a
program can answer.

**The verdict was a summary of facts the receipt already had.** Convergence,
finiteness, presence of each declared quantity and profile success were all
recorded. Collapsing them into one word threw away which of them failed and
added nothing.

**A verdict invites arguing with it and hides the reason.** A user who
disagreed with `diagnostic-only` had no way to see what drove it, and no way to
weigh a nonconverged boundary fit against a failed profile point. Those are
different situations and deserve different judgements.

## What replaces it

The receipt states what happened. `outcome` is `fitted` when the estimator
returned a record and `refused` when it did not. A fitted outcome carries four
facts:

| Fact | True when |
| --- | --- |
| `converged` | the free fit and every inference fit reported convergence |
| `all_quantities_present` | every quantity the manifest declares is present |
| `all_values_finite` | every numerical value in the record is finite |
| `all_profiles_evaluated` | no profile evaluation failed |

`failures` names the field behind each fact that is false. Anyone who wants the
old verdict can compute it from these in one line; anyone who wants to weigh
them differently now can.

The manifest field follows the same reasoning. `reportable_quantities` becomes
`supported_quantities`, because what it always meant was which quantities a
release supports, not which ones a scientist may write down.

## What this does not change

The release still requires all four facts of every synthetic receipt it ships,
and the independent verifier still recomputes them from the retained fit rather
than trusting what the receipt claims. A development build still refuses before
fitting, and a refusal is still an outcome rather than a missing result. The
fit record is still retained in full when a fact is false, which is what makes
diagnosing a failed search possible.

What ADR 0016 said about refusals stands: invalid inputs, an unidentified
estimand or no usable candidate produce a refused outcome carrying a stable
code, the public estimators may express that as a documented stable exception,
and unexpected programming errors are never presented as scientific refusals. A
converged boundary estimate is still an ordinary fit, with its boundary state
and limited endpoints retained.
