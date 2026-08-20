# 11. One interval module, and one interval record

**Status:** accepted 20 August 2026. Nothing has been built.

**Decided by:** Sam, across three rounds of questioning on 20 August 2026,
following an architecture review of the whole crate. Every decision below is his
answer, not an agent's inference.

**Extends `0004` into the ground that record's own status line leaves open.**
`0004` settles the interval for the one-trait case and says in its first
paragraph that the general case still needs re-deriving. This is that
re-derivation. It amends `0005` in exactly one respect — the name of the
endpoint flag — and confirms the rest of it.

**Numbered 11 rather than 10** because `0010` is claimed by the censored
repeated-measures model on the branch in pull request #11, which is expected to
land first.

## What the review found

`0004` chose one interval recipe by measurement. That recipe is currently
implemented seven times, over seven record types, and the copies are close
enough that the differences between them are almost all accidental:

- `components.rs:1079–1104` and `bivariate.rs:2276–2303` differ by two comment
  lines and nothing else.
- `gxe.rs:1331–1344` and `liability.rs:929–942` differ by two tokens: sixty
  iterations against forty, and a tolerance of 1e-7 against 1e-5.
- One paragraph of doc comment is copy-pasted verbatim into four of the seven
  records.

Two consequences had already arrived by the time the review ran.

**A repair reached five families and missed the sixth.** Reading a profile
evaluation that failed as an infinite deviance narrows the interval without
saying so, which is why five families count failures into `profile_failures`.
`spatial.rs:1362–1365` still reads a failure as infinite, and `SpatialInterval`
has no field to report it in.

**The mixture verdict exists in one family out of seven.**
`contains_lower_bound` and `contains_upper_bound` live only on
`prepared::Interval`. `0004` records that this pair was added after an
adversarial review found the recipe otherwise unimplementable from the record
alone, and `src/bin/coverage.rs:263` is what scores it. By `0004`'s own
argument, the other six families' intervals are not checkable from the record.

The same endpoint is called `lower_limited` in five families and
`lower_at_bound` in two, and `ComponentModel` publishes both — `lower_at_bound`
from `interval()` and `lower_limited` from `contrasts()`, from the same Rust
field.

## The decision, in eight parts

**1. One module.** The recipe becomes `src/interval.rs`, exporting one record
named `Interval`. It is not folded into `deviance.rs`, which holds distributions
and tail probabilities; a bracketing search is neither.

**2. The interface is a closure.** A family hands the module the profiled
log-likelihood at a held value — `impl Fn(f64) -> Option<f64>` — together with
its fitted estimate and its bounds. Nothing else. The existing signatures
already collapse to this: `component`, `quantity`, `index`, `integrated` and
`start_from` are all resolved before bracketing begins, so by the time the
search starts every family is holding one number and asking for a likelihood.

A trait was rejected. It would drag `Reported`, `SpatialQuantity` and their
siblings into a shared vocabulary that has no reason to exist, and it would make
each family's internals part of a shared surface rather than its own business.

**3. One record, with absent rather than variant fields.** Every field lives on
the one record, and a family leaves inapplicable ones absent through `Option`.
This is `0005`'s rule — no results that change shape, because a record whose
type varies invites conditional handling downstream — applied to the general
case. A core record with per-family extensions was rejected for reintroducing
exactly that.

**4. `lower_limited` survives and `lower_at_bound` retires.** It is the majority
in Rust, it is what `prepared` uses, and it is the word `0005` itself uses:
the lower endpoint flagged as limited. This is the one respect in which this
record amends `0005`, and it amends it towards `0005`'s own wording.

**5. The mixture verdict means measured, not applicable.** A family fills
`contains_lower_bound` only where a coverage simulation has actually scored the
interval there, and leaves it absent otherwise. `0004` chose this recipe by
measurement rather than by argument, and `CONTEXT.md` is blunter still: the
coverage check is what chooses an interval recipe, and nothing about reading the
code can tell you the same thing. So the field turns on one family at a time, as
the simulation is run, rather than everywhere at once because the geometry looks
similar. The Self–Liang mixture is the right null for a parameter on its bound;
whether a genetic correlation at one, or a decay rate at its bound, is that same
situation is a question for a simulation and not for a reviewer.

**6. One tolerance and one iteration cap.** 1e-9 and eighty iterations, subject
to measuring what the tightest setting costs before it is adopted. The copies
currently run 1e-9, 1e-7, 1e-5, 1e-4 times the fitted value, and 1e-15; and 200,
80, 80, 80, 60, 60 and 40 iterations. None of the differences carries a comment
explaining itself, which is the evidence that they are accidental rather than
chosen.

**7. `prepared` becomes a client rather than the owner.** The one-trait model
gives up the name `Interval` and goes through the module like everything else.
This is the only route by which the mixture verdict becomes available to the
other families, which is most of the value here. It is also the risky half:
`prepared.rs` is 805 lines with no test module of its own, covered only from
outside by `tests/one_trait.rs`, and it is the model the coverage simulation
scores.

**8. The repair lands with the consolidation.** `spatial` inherits failure
counting in the same piece of work. Preserving the miss to keep the change
clean would be choosing the wrong thing, given that the miss is the argument for
doing this at all. The spatial checks are rerun and their evidence replaced as
part of the same job, not later.

## Consequences worth calling out

- **Endpoints move in every family, not only in `spatial`.** Unifying the
  tolerances is a change in what the search returns, so "the consolidation
  changed nothing" will not be true and must not be claimed. Fidelity is
  established the other way round: pin every family's current endpoints as a
  seeded test first, then require every difference afterwards to have a stated
  reason. The two that will move most are `liability`, at 1e-5, and `spatial`,
  at 1e-4 times the fitted value — which on a heritability of a half is 5e-5,
  large enough to show in a printed interval.

- **Nothing outside the crate breaks on the rename.** `gobs-heritability` is the
  only project that imports Asterism, and it unpacks `_core` tuples
  positionally, so it breaks on tuple order and never on a dictionary key. The
  one thing that fails is `tests/test_python_interface.py:62–69`, which asserts
  the key set exactly. That is the test doing its job.

- **The coverage check becomes the thing that turns a field on.** Under part 5,
  `contains_lower_bound` absent means nobody has measured it here yet. That is a
  weaker claim than the field currently makes for `prepared`, and a more honest
  one for everybody else.

- **Two more families are arriving.** The branch in pull request #11 adds a
  censored model and a mixed bivariate one, each carrying its own copy of this
  machinery, and neither uses `src/convergence.rs`. This work waits for that
  branch so that it consolidates nine families rather than seven, and so that
  those two are folded in rather than rewritten.

- **This record does not reopen `0004`.** The recipe is unchanged: a profile
  likelihood, the deviance crossing a chi-square threshold, and the Self–Liang
  mixture deciding whether a boundary point belongs. What changes is that it is
  written once, and that the general case now has somewhere to live.
