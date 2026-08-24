# 20. No design range gates a result

**Status: accepted.** The supported design range is removed. ADR 0012
introduced it, ADR 0016 and ADR 0017 built on it, and ADR 0012's preflight
that "compares a proposed design" with it is withdrawn along with the public
`preflight_analysis`.

## What it did

Each supported analysis carried measured bounds — smallest and largest sample,
largest family, trait types, censoring levels. `run_analysis` compared the
caller's design with those bounds. Outside them it returned `outcome: refused`
with `fit_record: None`. It did not warn. It did not fit.

## Why it goes

**No statistical software does this.** `lme4` fits your model and warns you.
SOLAR does not ask whether your pedigree resembles the authors' test cases.
What real software does is attach a warning to a result you asked for. It does
not decline to compute.

**The upper bound is incoherent.** More data does not make an estimate less
trustworthy. Refusing at 1,910 people because the checks ran at 1,909 is not
caution.

**It would have refused this project's own work.** Ranges derived from what
each check happened to run at put the household model's ceiling at 1,792
people. The real study has 1,909. Asterism would have declined to produce a
receipt for the analysis it was written for.

**Nobody would call the public function.** `preflight_analysis` asked a user to
check their design before fitting. That is not a question anyone asks.

## What replaces it

Nothing gates on design. What each check measured is still recorded, in that
check's own `design_facts` and its evidence, where a reader can see it. That is
a record, not a claim about a boundary, and nothing consults it at run time.

A private `_release_state` keeps the two checks that remain defensible: the
pass rules must be configured, and the build must be a release rather than a
development build. A development build still cannot produce a reportable
result, which has nothing to do with how large anyone's study is.

## What this does not change

Every scientific gate stands. Supported analyses still need their checks to
pass, and their results still agree across platforms within pre-written
tolerances. Removing a bound on sample size removes nothing that bears on
whether a number is right.
