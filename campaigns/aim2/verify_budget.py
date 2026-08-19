"""Check the budget impact arithmetic against the costing it derives from.

Every figure in the impact note is recomputed from the two rates that anchor it
-- $225 participant payment and $50 transport, both taken from what was actually
paid on Acoustic -- and from the site F&A rates and totals in
`SUBAWARD_COSTING_2026-08-12.md`. A budget arithmetic error is the kind that
survives review, because nobody re-adds a table that looks finished.
"""

from pathlib import Path

BASE = Path("/Users/samuelmathias/MathiasLab/staging/studies/planned/mediation/"
            "grant-applications")
NOTE = (BASE / "BUDGET_IMPACT_OF_DESIGN_CHANGE_2026-08-19.md").read_text()

PAYMENT, TRANSPORT = 225.0, 50.0
SA_FA = 0.575
SA_SUBAWARD = 2_892_296
UTRGV_SUBAWARD = 409_211
AWARD = 4_364_327
BCH_DIRECTS, BCH_FA = 597_090, 465_730

problems = []


def want(value: str, why: str) -> None:
    if value not in NOTE:
        problems.append(f"{why}: {value} is not in the note")


# The anchor totals must reconcile, or nothing built on them is safe.
if BCH_DIRECTS + SA_SUBAWARD + UTRGV_SUBAWARD + BCH_FA != AWARD:
    problems.append("the 12 August award total does not reconcile")

# The existing participant line.
current = 500 * (PAYMENT + TRANSPORT)
if current != 137_500:
    problems.append(f"the current participant line is {current:,.0f}, not 137,500")

# One extra relative per case: 250 more participants.
added_directs = 250 * PAYMENT + 250 * TRANSPORT
added_fa = added_directs * SA_FA
increment = added_directs + added_fa
if added_directs != 68_750:
    problems.append(f"directs added is {added_directs:,.0f}, not 68,750")
want("68,750", "directs added")
want("56,250", "payment added")
want("12,500", "transport added")
want(f"{added_fa:,.0f}", "F&A added")

one = AWARD + increment
two = AWARD + 2 * increment
print(f"increment per 250 participants: {increment:,.2f}")
print(f"award with second relative:     {one:,.2f}")
print(f"award with both changes:        {two:,.2f}")

# The note rounds; check it rounds to something the arithmetic supports.
for value, label in ((one, "second relative"), (two, "both changes")):
    candidates = {f"{value:,.0f}", f"{int(value):,}"}
    if not any(c in NOTE for c in candidates):
        problems.append(f"{label}: none of {candidates} appears in the note")

if f"{SA_SUBAWARD + increment:,.0f}" not in NOTE:
    problems.append("the revised San Antonio subaward is not stated correctly")

# The consortium discrepancy that was found while reading.
stated, actual = 2_757_593, SA_SUBAWARD + UTRGV_SUBAWARD
if actual - stated != 543_914:
    problems.append(f"the consortium gap is {actual - stated:,}, not 543,914")
want("543,914", "consortium gap")
want("3,301,507", "consortium sum")
if stated - UTRGV_SUBAWARD != 2_348_382:
    problems.append("the implied earlier San Antonio figure is wrong")
want("2,348,382", "implied earlier figure")

# A second research assistant, quoted as roughly $508,000.
second_ra = 322_820 * (1 + SA_FA)
if abs(second_ra - 508_000) > 1_000:
    problems.append(f"a second assistant is {second_ra:,.0f}, not roughly 508,000")

# Recruitment rates.
if 750 / 4 != 187.5 or 1000 / 4 != 250:
    problems.append("the accrual arithmetic is wrong")

if problems:
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
else:
    print("\nevery figure in the budget impact note reconciles")
