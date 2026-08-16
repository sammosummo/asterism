"""Are the multi-component intervals and the boundary test calibrated?

`checks/bivariate_calibration.py` does this for two traits and two components.
This does it for one trait and three: additive, household and residual. Nothing
carries over — the intervals are reached by a different substitution, and the
test is a boundary test where the two-trait correlation tests were interior ones.

Three things are measured, all by simulating data whose answer is known.

**Interval coverage.** Simulate under known shares, fit, and count how often each
95 per cent interval contains the value used to generate the data.

**The boundary test under a true null.** Simulate with no household effect at all
and count how often the test says there is one. The null sits on the edge of the
parameter space, so the reference is the Self–Liang half-and-half mixture rather
than a plain chi-squared, and getting that wrong is not subtle: treating it as a
plain chi-squared would reject about twice as often as it should.

**Power, which is not calibration but decides whether any of this is worth
running.** A test that never rejects is perfectly calibrated. Simulating with a
household effect present and counting how often it is found says whether the
design can see one at all.

**What makes the two components separable is not what it first looks like.**
The obvious answer — that the households must contain unrelated people — is
wrong, and so is the framing that puts the question inside a single household.

The condition is that the three matrices A, H and I are linearly independent
across the whole design. It is a property of the design pooled together, not of
any one household. Two households, each a single pair, identify the model between
them if their pairs differ in relatedness: a sibling pair contributes a
covariance of 0.5·σ²_A + σ²_H and a spouse pair contributes σ²_H, and those two
equations plus the diagonal give three for three unknowns. Neither household
could do it alone.

So unrelated co-residents give the sharpest contrast but are not required. Full
sibs living with a half sib separate the components perfectly well, because 0.5
and 0.25 are two different numbers and one is enough variation. The one case that
genuinely fails is a design where **every** co-resident pair in the whole sample
has the **same** relatedness — siblings throughout, say — because A is then
exactly 0.5·H + 0.5·I and no amount of data recovers what has been added
together.

The real GOBS households are nowhere near that. Of 351 households holding two or
more measured people, 173 are a pair at relatedness 0.5, 81 a pair at nought, and
83 have varied relatedness within them; over the whole design 68 per cent of A is
not explained by any combination of H and I. The design used below has each
household holding two sibling pairs from different families, which is a similar
kind of contrast.

Run with:

    uv run --no-project python checks/components_calibration.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from asterism import _core

PAIRS = 200
PER_HOUSEHOLD = 4
INTERVAL_REPLICATES = 200
TEST_REPLICATES = 300
LEVELS = (0.01, 0.05, 0.10)

TRUTH = dict(additive=0.4, household=0.2, residual=0.4)
NAMES = ("additive", "household", "residual")


def structure() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sibling pairs, in households of four holding two pairs each."""
    n = 2 * PAIRS
    relationship = np.eye(n)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5
    household = np.eye(n)
    for group in range(n // PER_HOUSEHOLD):
        block = slice(group * PER_HOUSEHOLD, (group + 1) * PER_HOUSEHOLD)
        household[block, block] = 1.0
    return relationship, household, np.ones((n, 1)), n


def draw(n: int, shares: dict, seed: int) -> np.ndarray:
    """One data set, drawn from the model rather than mixed by hand.

    Each sibling's genetic value is correlated one half with its sib, which is
    what the relationship matrix says. Building the response any other way risks
    a sample the model cannot represent — a sibling correlation above a half
    needs a heritability above one, and the fit can only reach it by driving the
    residual to nought.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    if shares["household"] > 0:
        for group in range(n // PER_HOUSEHOLD):
            block = slice(group * PER_HOUSEHOLD, (group + 1) * PER_HOUSEHOLD)
            y[block] += np.sqrt(shares["household"]) * rng.standard_normal()
    half = np.sqrt(0.5)
    for pair in range(n // 2):
        common = rng.standard_normal()
        for member in range(2):
            genetic = half * common + half * rng.standard_normal()
            y[2 * pair + member] += (
                np.sqrt(shares["additive"]) * genetic
                + np.sqrt(shares["residual"]) * rng.standard_normal()
            )
    return y


def coverage(relationship, household, design, n) -> dict:
    contained = {name: 0 for name in NAMES[:2]}
    attempted = {name: 0 for name in NAMES[:2]}
    complete = 0
    started = time.perf_counter()
    for replicate in range(INTERVAL_REPLICATES):
        y = draw(n, TRUTH, 5000 + replicate)
        whole = True
        for index, name in enumerate(NAMES[:2]):
            try:
                lower, upper, _, _, _ = _core.component_interval(
                    [relationship, household], design, y, index, True
                )
            except Exception:
                whole = False
                continue
            attempted[name] += 1
            if lower <= TRUTH[name] <= upper:
                contained[name] += 1
        if whole:
            complete += 1
        if (replicate + 1) % 25 == 0:
            print(
                f"  {replicate + 1} replicates, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"contained": contained, "attempted": attempted, "complete": complete}


def boundary(relationship, household, design, n, present: bool) -> dict:
    """The test of the household component, with and without one to find."""
    shares = dict(TRUTH)
    if not present:
        # No household effect at all; its share goes to the residual so the
        # total still comes to one.
        shares = dict(
            additive=TRUTH["additive"],
            household=0.0,
            residual=TRUTH["residual"] + TRUTH["household"],
        )
    p_values, refused = [], 0
    started = time.perf_counter()
    for replicate in range(TEST_REPLICATES):
        y = draw(n, shares, (9000 if present else 4000) + replicate)
        try:
            _, p_value, _, _ = _core.component_test(
                [relationship, household], design, y, 1, True
            )
            p_values.append(p_value)
        except Exception:
            # The test refuses where another component has itself gone to
            # nought, because the mixture assumes only one is on the boundary.
            # Those replicates are counted out and reported, not counted as
            # failures to reject.
            refused += 1
        if (replicate + 1) % 100 == 0:
            print(
                f"  {replicate + 1} replicates, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"p_values": p_values, "refused": refused}


def main() -> int:
    relationship, household, design, n = structure()
    failures = []

    print(
        f"One trait, three components, REML. {PAIRS} sibling pairs, n = {n}, "
        f"households of {PER_HOUSEHOLD}\nholding two pairs each, so half of "
        f"every household's co-resident pairs share no genes.\n"
        f"Truth: additive {TRUTH['additive']}, household {TRUTH['household']}, "
        f"residual {TRUTH['residual']}.\n"
    )

    print("Profile interval coverage, nominal 95 per cent.")
    covered = coverage(relationship, household, design, n)
    print(
        f"\n  {covered['complete']} of {INTERVAL_REPLICATES} replicates gave "
        "every interval.\n"
    )
    print(f"  {'component':<12}{'coverage':>10}{'binomial 95%':>22}{'of':>8}")
    interval_results = {}
    for name in NAMES[:2]:
        counted = covered["attempted"][name]
        if counted == 0:
            failures.append(f"no interval was computed for {name}")
            continue
        hit = covered["contained"][name] / counted
        interval_results[name] = hit
        error = np.sqrt(0.95 * 0.05 / counted)
        low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
        inside = low <= hit <= high
        print(
            f"  {name:<12}{hit:>10.3f}   [{low:.3f}, {high:.3f}]  "
            f"{'ok' if inside else ('conservative' if hit > high else 'UNDER')}"
            f"{counted:>8}"
        )
        if hit < low:
            failures.append(f"{name} covers {hit:.3f}, below the band")

    print("\n\nThe household test with no household effect to find.")
    null = boundary(relationship, household, design, n, present=False)
    null_p = np.array(null["p_values"])
    tested = len(null_p)
    print(
        f"\n  {tested} of {TEST_REPLICATES} replicates tested"
        + (f", {null['refused']} refused" if null["refused"] else "")
        + ".\n"
    )
    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    rates = {}
    for level in LEVELS:
        rate = float((null_p < level).mean())
        rates[str(level)] = rate
        error = np.sqrt(level * (1 - level) / tested)
        lower, upper = level - 1.96 * error, level + 1.96 * error
        # Only over-rejection fails. A boundary test that rejects too rarely
        # under-calls a household effect, which costs power rather than
        # manufacturing a finding.
        print(
            f"  {level:>8.2f}{rate:>12.3f}   [{lower:.3f}, {upper:.3f}]  "
            f"{'ok' if lower <= rate <= upper else ('conservative' if rate < lower else 'OVER')}"
        )
        if rate > upper:
            failures.append(
                f"the household test rejects {rate:.3f} at the {level:.2f} level, "
                "above the band"
            )
    atom = float((null_p >= 1.0).mean())
    print(
        f"\n  The statistic was exactly nought in {atom:.1%} of samples; the "
        "mixture expects about 50%."
    )

    print("\n\nThe same test with a household effect present. This is power, not")
    print("calibration: a test that never rejects is perfectly calibrated.")
    powered = boundary(relationship, household, design, n, present=True)
    power_p = np.array(powered["p_values"])
    detected = float((power_p < 0.05).mean())
    print(f"\n  {len(power_p)} replicates; a true household share of "
          f"{TRUTH['household']} was found in {detected:.1%} at the five per cent level.")
    if detected < 0.5:
        failures.append(
            f"the test finds a household share of {TRUTH['household']} only "
            f"{detected:.0%} of the time; this design cannot see one"
        )

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nIntervals cover, the boundary test holds its level, and it can find")
    print("an effect that is there. Calibration is a different question from the")
    print("agreement with SOLAR and R, which do not fit this model at all.")

    print(
        json.dumps(
            {
                "estimator": "reml",
                "pairs": PAIRS,
                "people": n,
                "household_size": PER_HOUSEHOLD,
                "design_note": (
                    "each household holds two sibling pairs from different families, "
                    "so half of its co-resident pairs share no genes; the real GOBS "
                    "households are 28 per cent unrelated pairs"
                ),
                "truth": TRUTH,
                "intervals": {
                    "replicates_requested": INTERVAL_REPLICATES,
                    "replicates_complete": covered["complete"],
                    "nominal": 0.95,
                    "coverage": interval_results,
                },
                "household_test_under_the_null": {
                    "replicates_requested": TEST_REPLICATES,
                    "replicates_tested": tested,
                    "replicates_refused": null["refused"],
                    "rule": "self_liang_50_50_mixture",
                    "rejection_rates": rates,
                    "atom_at_zero": atom,
                },
                "power": {
                    "true_household_share": TRUTH["household"],
                    "detected_at_five_per_cent": detected,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
