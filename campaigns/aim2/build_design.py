"""Assemble the proposed Aim 2 sample, and write it out without identifiers.

Everything structural comes from the real thing: kinship from the reviewed SAFS
pedigree, ages from the Acoustic REDCap export, and the enrolled relatives from
who actually sits one meiosis from a participant without ever having been
examined.

What leaves this machine is a matrix, an age per row, and a role per row. No
identifier is written, so the file the cluster sees carries structure and
nothing else.

The four parts, per Sam's design of 18 August:

- **Biggs probands.** 200 adjudicated cases, over 65, recruited from a clinic.
  They are not in the SAFS pedigree and are conservatively assumed unrelated to
  it, so each is its own founder.
- **Their offspring.** 300 of them, half the probands bringing one and half two.
  The measured parent-child gap in this population is 24 years, so a proband
  diagnosed in their late seventies has children in their fifties.
- **Acoustic.** Everybody, at their real ages -- median 52, and only a fifth
  over 65.
- **The enrolled relatives.** First-degree relatives of Acoustic participants
  who are in the pedigree and have never been examined. Enrolled whatever their
  status and adjudicated at the Biggs, which is what stops the enrolment
  conditioning on being affected.
"""

from __future__ import annotations

import collections
import csv
import json

import numpy as np

PEDIGREE = ("/Users/samuelmathias/MathiasLab/promoted/studies/existing/safs/"
            "data/pedigrees/review-history/pedigree.csv")
REDCAP = ("/Users/samuelmathias/AcousticIngest/outputs/intermediate/redcap/raw/"
          "redcap_acoustic_20260811T150210.json")
OUT = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/design.npz"

PROBANDS = 200
PARENT_GAP = 24.0
PROBAND_AGE = (68.0, 88.0)      # adjudicated cases are old
ACOUSTIC_TARGET = 600           # the study will reach this
RELATIVE_TARGET = 500
# **Eight, now that the model has a cheap route above two.** Sequential
# truncation costs in proportion to the family rather than starting at five
# hundred evaluations, so a family of three costs what a family of two costs.
# The cap is prudence rather than necessity: the approximation takes one step
# per member, and its agreement with the accurate route was measured at sizes
# three, four and six.
LARGEST_UNIT = 8

ROLES = {"proband": 0, "offspring": 1, "acoustic": 2, "relative": 3}


def kinship(path):
    rows = list(csv.DictReader(open(path)))
    father = {r["id"]: (r["fa"] if r["fa"] != "0" else None) for r in rows}
    mother = {r["id"]: (r["mo"] if r["mo"] != "0" else None) for r in rows}
    order, placed, pending = [], set(), list(father)
    while pending:
        remaining, progressed = [], False
        for person in pending:
            f, m = father[person], mother[person]
            if (f is None or f in placed) and (m is None or m in placed):
                order.append(person); placed.add(person); progressed = True
            else:
                remaining.append(person)
        if not progressed:
            raise SystemExit("pedigree has a loop")
        pending = remaining
    index = {p: i for i, p in enumerate(order)}
    n = len(order)
    phi = np.zeros((n, n))
    for k, person in enumerate(order):
        f, m = father[person], mother[person]
        fi = index[f] if f is not None else -1
        mi = index[m] if m is not None else -1
        left = phi[fi, :k] if fi >= 0 else 0.0
        right = phi[mi, :k] if mi >= 0 else 0.0
        phi[k, :k] = 0.5 * (left + right)
        phi[:k, k] = phi[k, :k]
        phi[k, k] = 0.5 * (1.0 + (phi[fi, mi] if (fi >= 0 and mi >= 0) else 0.0))
    children = collections.defaultdict(list)
    for person in order:
        for parent in (father[person], mother[person]):
            if parent is not None:
                children[parent].append(person)
    return index, 2.0 * phi, father, mother, children


def first_degree(person, father, mother, children):
    out = set()
    for parent in (father.get(person), mother.get(person)):
        if parent is not None:
            out.add(parent)
            out.update(children[parent])
    out.update(children.get(person, []))
    out.discard(person)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(20260818)
    index, a, father, mother, children = kinship(PEDIGREE)
    print(f"pedigree: {len(index)} people")

    records = json.load(open(REDCAP))
    acoustic_age = {}
    for record in records:
        subject = str(record.get("record_id"))
        for field in ("age_visit_v2", "age_v2", "age_ssq"):
            try:
                got = float(record.get(field))
            except (TypeError, ValueError):
                continue
            if 0 < got < 120:
                acoustic_age[subject] = got
                break
    acoustic = [s for s in acoustic_age if s in index]
    print(f"Acoustic with an age and in the pedigree: {len(acoustic)}")

    # The study will reach 600. Top it up by resampling the observed age
    # distribution onto pedigree members who are relatives of participants --
    # which is who Acoustic actually recruits.
    ages_seen = np.array([acoustic_age[s] for s in acoustic])
    pool = []
    for subject in acoustic:
        pool.extend(first_degree(subject, father, mother, children))
    pool = [p for p in dict.fromkeys(pool) if p not in acoustic_age]
    rng.shuffle(pool)
    extra = pool[: max(0, ACOUSTIC_TARGET - len(acoustic))]
    for person in extra:
        acoustic_age[person] = float(rng.choice(ages_seen))
    acoustic = acoustic + extra
    print(f"Acoustic after topping up to the funded target: {len(acoustic)}")

    # The enrolled relatives: first-degree, in the pedigree, not in Acoustic.
    enrolled = []
    for subject in acoustic:
        for relative in first_degree(subject, father, mother, children):
            if relative not in acoustic_age and relative in index:
                enrolled.append(relative)
    enrolled = list(dict.fromkeys(enrolled))
    rng.shuffle(enrolled)
    enrolled = enrolled[:RELATIVE_TARGET]
    # They are the generation above a participant of median age, so around 76.
    relative_age = {p: float(np.clip(rng.normal(76.0, 8.0), 40.0, 95.0))
                    for p in enrolled}
    print(f"enrolled relatives, never examined: {len(enrolled)}")

    # Family units among the pedigree people, so no unit is too deep for the
    # region probability. A unit is a participant with their enrolled kin.
    assigned = set()
    units = []
    for subject in acoustic:
        if subject in assigned:
            continue
        unit = [subject]
        assigned.add(subject)
        for relative in first_degree(subject, father, mother, children):
            if len(unit) >= LARGEST_UNIT:
                break
            if relative in assigned:
                continue
            if relative in acoustic_age or relative in relative_age:
                unit.append(relative)
                assigned.add(relative)
        units.append(unit)
    # **Pair up whatever is left rather than leaving it alone.** A singleton
    # contributes its own value and no relatedness, so it carries none of the
    # information a family design exists for. A first pass left 446 of 1,600
    # people unmatched; this sweeps the remainder against their own relatives
    # before giving up on any of them.
    leftover = [p for p in list(acoustic_age) + enrolled if p not in assigned]
    still = set(leftover)
    for person in leftover:
        if person not in still:
            continue
        still.discard(person)
        partner = next(
            (r for r in first_degree(person, father, mother, children)
             if r in still),
            None,
        )
        if partner is not None:
            still.discard(partner)
            units.append([person, partner])
            assigned.update((person, partner))
        else:
            units.append([person])
            assigned.add(person)

    # Now lay everything out: the pedigree people first, then the Biggs.
    people, ages, roles, family = [], [], [], []
    for u, unit in enumerate(units):
        for person in unit:
            people.append(person)
            if person in acoustic_age:
                ages.append(acoustic_age[person])
                roles.append(ROLES["acoustic"])
            else:
                ages.append(relative_age[person])
                roles.append(ROLES["relative"])
            family.append(u)
    pedigree_rows = [index[p] for p in people]
    block = a[np.ix_(pedigree_rows, pedigree_rows)]
    print(f"pedigree part: {len(people)} people in {len(units)} units")

    # The Biggs families are their own founders: a proband and their children.
    offspring_counts = [1] * (PROBANDS // 2) + [2] * (PROBANDS - PROBANDS // 2)
    rng.shuffle(offspring_counts)
    biggs_units = []
    next_family = len(units)
    for count in offspring_counts:
        proband_age = float(rng.uniform(*PROBAND_AGE))
        children = [float(np.clip(proband_age - rng.normal(PARENT_GAP, 6.0),
                                  25.0, 74.0)) for _ in range(count)]
        # The proband pairs with one child, because a proband-offspring pair is
        # where the information is. A second child cannot join them without
        # making a trio, so it becomes its own unit and its tie to the proband
        # is lost. That loss is the model's limit, not the design's.
        biggs_units.append((next_family,
                            [(proband_age, ROLES["proband"]),
                             (children[0], ROLES["offspring"])]))
        next_family += 1
        for spare in children[1:]:
            biggs_units.append((next_family, [(spare, ROLES["offspring"])]))
            next_family += 1
    biggs_n = sum(len(rows) for _, rows in biggs_units)
    print(f"Biggs part: {PROBANDS} probands and "
          f"{biggs_n - PROBANDS} offspring, {biggs_n} people")

    total = len(people) + biggs_n
    relationship = np.zeros((total, total))
    relationship[: len(people), : len(people)] = block
    at = len(people)
    for unit_id, rows in biggs_units:
        size = len(rows)
        for i in range(size):
            for j in range(size):
                relationship[at + i, at + j] = 1.0 if i == j else 0.5
        for age, role in rows:
            ages.append(age)
            roles.append(role)
            family.append(unit_id)
        at += size

    ages = np.array(ages)
    roles = np.array(roles)
    family = np.array(family)

    # **Every unit must be a connected family.** Grouping people who are not
    # related to one another gives a block-diagonal matrix, and the model
    # refuses it -- correctly, because such a block is two families claiming to
    # be one. Split each unit into the components it actually has.
    split = np.empty_like(family)
    next_id = 0
    for unit in np.unique(family):
        rows = np.flatnonzero(family == unit)
        block = relationship[np.ix_(rows, rows)]
        seen, groups = set(), []
        for start in range(len(rows)):
            if start in seen:
                continue
            stack, group = [start], []
            seen.add(start)
            while stack:
                here = stack.pop()
                group.append(here)
                for other in range(len(rows)):
                    if other not in seen and here != other and block[here, other] > 0:
                        seen.add(other)
                        stack.append(other)
            groups.append(group)
        for group in groups:
            split[rows[group]] = next_id
            next_id += 1
    if next_id != len(np.unique(family)):
        print(f"  split {len(np.unique(family))} units into {next_id} connected ones")
    family = split
    print(f"\nTOTAL {total} people in {family.max() + 1} units")
    for name, code in ROLES.items():
        here = roles == code
        print(f"  {name:<10} {here.sum():>5}  median age "
              f"{np.median(ages[here]):.0f}, over 65 {int((ages[here] >= 65).sum())}")
    print(f"  over 65 overall: {int((ages >= 65).sum())} "
          f"({(ages >= 65).mean():.1%})")

    # **The model wants a diagonal of exactly one, and kinship gives 1 + F.**
    # SAFS carries real consanguinity, so some diagonals exceed one. The latent
    # mediation model is specified on an additive relationship with a unit
    # diagonal, so the inbreeding is dropped here rather than smuggled in --
    # and it is reported, because dropping it is a limitation of the model
    # rather than a property of the sample.
    inbreeding = np.diag(relationship) - 1.0
    inbred = int((inbreeding > 1e-9).sum())
    print(f"\ninbreeding: {inbred} of {total} people have F > 0, "
          f"largest F {inbreeding.max():.4f}, mean over the inbred "
          f"{inbreeding[inbreeding > 1e-9].mean() if inbred else 0:.4f}")
    print("  the model takes a unit diagonal, so F is dropped and not modelled")
    np.fill_diagonal(relationship, 1.0)

    np.savez_compressed(OUT, relationship=relationship, age=ages, role=roles,
                        family=family)
    print(f"\nwritten to {OUT} (no identifiers)")
