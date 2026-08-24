# 19. No resource budget gates a release

**Status: accepted.** ADR 0012 required that "each supported target-sized
analysis must also complete within an explicit feasible time-and-memory budget
on its intended host". That sentence is withdrawn. The machinery that enforced
it — the manifest section, the measurement runner, the release gates in both
release tools, and their tests — is removed.

## Why it goes

ADR 0012 asked for the budget and, in the next breath, said that performance
"should not block release until a stable runner and its normal variability have
been measured". Nobody measured that variability. The manifest nonetheless
turned the first half of the sentence into four enforced ceilings and ignored
the second, so a release could fail on a number that no one had established was
meaningful.

Sam's objection is the plain one, and it is right: statistical software is not
normally released against a wall-clock ceiling. A variance-components fit that
takes longer than expected is a thing to investigate, not a reason to refuse to
publish a correct result. Tying the two together means either a slow machine
blocks a sound release, or the ceiling is set loose enough to never fire, in
which case it was never a gate.

## What replaces it

Nothing, deliberately. Timings are still recorded by the checks that produce
them — the G×E target design reports 615 seconds, the censored coverage runs
report their own — and those numbers stay in release evidence as observations.
They are read by people, not by a gate.

If a future release wants a performance contract, it needs the measurement ADR
0012 asked for first: a stable runner, its normal variability, and a threshold
derived from that variability rather than guessed.

## What this does not change

The scientific gates stand. Every supported analysis still needs its checks to
pass, its design range measured, and its cross-platform agreement within
pre-written tolerances. Removing a performance ceiling removes nothing that
bears on whether a result is correct.
