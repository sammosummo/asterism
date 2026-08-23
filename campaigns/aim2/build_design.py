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

type ParentMap = dict[str, str | None]
type ChildrenMap = collections.defaultdict[str, list[str]]
type KinshipResult = tuple[
    dict[str, int], np.ndarray, ParentMap, ParentMap, ChildrenMap
]

PEDIGREE: str = (
    "/Users/samuelmathias/MathiasLab/promoted/studies/existing/safs/"
    "data/pedigrees/review-history/pedigree.csv"
)
"""Located the reviewed pedigree used only to derive relationship structure."""
REDCAP: str = (
    "/Users/samuelmathias/AcousticIngest/outputs/intermediate/redcap/raw/"
    "redcap_acoustic_20260811T150210.json"
)
"""Located the Acoustic export used only to derive participant ages."""
OUT: str = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/design.npz"
"""Selected the identifier-free archive written for the simulation campaign."""

PROBANDS: int = 200
"""Set the number of independently recruited adjudicated probands."""
PARENT_GAP: float = 24.0
"""Set the measured mean age difference between parents and offspring."""
PROBAND_AGE: tuple[float, float] = (68.0, 88.0)  # adjudicated cases are old
"""Bounded simulated proband ages to the intended older clinic population."""
ACOUSTIC_TARGET: int = 600  # the study will reach this
"""Set the funded target size for the Acoustic contribution."""
RELATIVE_TARGET: int = 500
"""Set the intended number of enrolled first-degree relatives."""
# **Eight, now that the model has a cheap route above two.** Sequential
# truncation costs in proportion to the family rather than starting at five
# hundred evaluations, so a family of three costs what a family of two costs.
# The cap is prudence rather than necessity: the approximation takes one step
# per member, and its agreement with the accurate route was measured at sizes
# three, four and six.
LARGEST_UNIT: int = 8
"""Capped connected family units at the largest measured integration size."""

ROLES: dict[str, int] = {
    "proband": 0,
    "offspring": 1,
    "acoustic": 2,
    "relative": 3,
}
"""Mapped non-identifying recruitment roles to stable integer codes."""


def kinship(path: str) -> KinshipResult:
    """Derive twice-kinship and pedigree relations from a reviewed pedigree.

    Args:
        path: CSV path containing id, father and mother columns.

    Returns:
        Row index, additive relationship matrix, parent maps and child map.

    Raises:
        SystemExit: If the pedigree contains a directed ancestry loop.
    """
    with open(path, encoding="utf-8", newline="") as stream:
        rows: list[dict[str, str]] = list(csv.DictReader(stream))
        """Loaded the reviewed pedigree rows from the source CSV."""
    father: ParentMap = {
        row["id"]: (row["fa"] if row["fa"] != "0" else None) for row in rows
    }
    """Mapped each person to an optional recorded father."""
    mother: ParentMap = {
        row["id"]: (row["mo"] if row["mo"] != "0" else None) for row in rows
    }
    """Mapped each person to an optional recorded mother."""
    order: list[str] = []
    """Initialised the parent-before-child pedigree order."""
    placed: set[str] = set()
    """Initialised the people already placed in topological order."""
    pending: list[str] = list(father)
    """Initialised the people still awaiting placement."""
    while pending:
        remaining: list[str] = []
        """Initialised people whose parents had not yet been placed."""
        progressed: bool = False
        """Tracked whether this topological pass placed anybody."""
        for person in pending:
            f, m = father[person], mother[person]
            """Recovered the person's optional recorded parents."""
            if (f is None or f in placed) and (m is None or m in placed):
                order.append(person)
                placed.add(person)
                progressed = True
                """Placed the person after every recorded parent."""
            else:
                remaining.append(person)
                """Deferred the person until a later topological pass."""
        if not progressed:
            raise SystemExit("pedigree has a loop")
        pending = remaining
        """Advanced to the people still waiting for their parents."""
    """Constructed a complete parent-before-child pedigree ordering."""
    index: dict[str, int] = {person: position for position, person in enumerate(order)}
    """Mapped each pedigree identifier to its relationship-matrix row."""
    n: int = len(order)
    """Counted people in the reviewed pedigree."""
    phi: np.ndarray = np.zeros((n, n))
    """Initialised the tabular kinship matrix."""
    for k, person in enumerate(order):
        f, m = father[person], mother[person]
        """Recovered the person's optional recorded parents."""
        fi: int = index[f] if f is not None else -1
        """Located the father's row or marked him as absent."""
        mi: int = index[m] if m is not None else -1
        """Located the mother's row or marked her as absent."""
        left: np.ndarray | float = phi[fi, :k] if fi >= 0 else 0.0
        """Selected the father's kinship with people already processed."""
        right: np.ndarray | float = phi[mi, :k] if mi >= 0 else 0.0
        """Selected the mother's kinship with people already processed."""
        phi[k, :k] = 0.5 * (left + right)
        """Derived the person's kinship with each earlier pedigree member."""
        phi[:k, k] = phi[k, :k]
        """Mirrored the new kinship row into the symmetric column."""
        phi[k, k] = 0.5 * (1.0 + (phi[fi, mi] if (fi >= 0 and mi >= 0) else 0.0))
        """Derived self-kinship including parental consanguinity."""
    """Filled the complete pedigree kinship matrix in topological order."""
    children: ChildrenMap = collections.defaultdict(list)
    """Initialised children grouped by each recorded parent."""
    for person in order:
        for parent in (father[person], mother[person]):
            if parent is not None:
                children[parent].append(person)
                """Recorded the person as a child of this known parent."""
    """Built the child lookup used for first-degree relative discovery."""
    return index, 2.0 * phi, father, mother, children


def first_degree(
    person: str,
    father: ParentMap,
    mother: ParentMap,
    children: ChildrenMap,
) -> set[str]:
    """Find a person's parents, siblings and children.

    Args:
        person: Pedigree identifier whose relatives are requested.
        father: Optional father by pedigree identifier.
        mother: Optional mother by pedigree identifier.
        children: Children grouped by parent identifier.

    Returns:
        First-degree relative identifiers, excluding the person.
    """
    out: set[str] = set()
    """Initialised the person's first-degree relative set."""
    for parent in (father.get(person), mother.get(person)):
        if parent is not None:
            out.add(parent)
            out.update(children[parent])
    """Added known parents and the siblings reached through each parent."""
    out.update(children.get(person, []))
    """Added the person's known children."""
    out.discard(person)
    """Removed the focal person if a pedigree anomaly introduced them."""
    return out


if __name__ == "__main__":
    rng: np.random.Generator = np.random.default_rng(20260818)
    """Created the deterministic generator for every sampled design choice."""
    index, relationship_source, father, mother, children_by_parent = kinship(PEDIGREE)
    """Derived additive relationships and immediate pedigree relations."""
    print(f"pedigree: {len(index)} people")

    with open(REDCAP, encoding="utf-8") as stream:
        records: list[dict[str, object]] = json.load(stream)
        """Loaded the raw Acoustic records used to derive observed ages."""
    acoustic_age: dict[str, float] = {}
    """Initialised valid Acoustic ages by source subject identifier."""
    for record in records:
        subject: str = str(record.get("record_id"))
        """Normalised the source record identifier for pedigree matching."""
        for field in ("age_visit_v2", "age_v2", "age_ssq"):
            try:
                got: float = float(record.get(field))
                """Parsed the first candidate age field as a numeric value."""
            except (TypeError, ValueError):
                continue
            if 0 < got < 120:
                acoustic_age[subject] = got
                """Retained the first plausible age for this source subject."""
                break
    """Collected one plausible age per source record wherever available."""
    acoustic: list[str] = [subject for subject in acoustic_age if subject in index]
    """Selected aged Acoustic participants present in the reviewed pedigree."""
    print(f"Acoustic with an age and in the pedigree: {len(acoustic)}")

    # The study will reach 600. Top it up by resampling the observed age
    # distribution onto pedigree members who are relatives of participants --
    # which is who Acoustic actually recruits.
    ages_seen: np.ndarray = np.array([acoustic_age[subject] for subject in acoustic])
    """Captured the observed Acoustic age distribution for target-size imputation."""
    pool: list[str] = []
    """Initialised candidate first-degree relatives for Acoustic target top-up."""
    for subject in acoustic:
        pool.extend(first_degree(subject, father, mother, children_by_parent))
    """Collected first-degree relatives of every observed Acoustic participant."""
    pool = [p for p in dict.fromkeys(pool) if p not in acoustic_age]
    """Deduplicated candidates and removed everybody with an observed Acoustic age."""
    rng.shuffle(pool)
    """Randomised otherwise exchangeable target top-up candidates deterministically."""
    extra: list[str] = pool[: max(0, ACOUSTIC_TARGET - len(acoustic))]
    """Selected only enough relatives to reach the funded Acoustic target."""
    for person in extra:
        acoustic_age[person] = float(rng.choice(ages_seen))
        """Imputed an observed-distribution age for this unexamined top-up person."""
    """Assigned ages to every selected target top-up person."""
    acoustic = acoustic + extra
    """Extended the Acoustic design contribution to its funded target size."""
    print(f"Acoustic after topping up to the funded target: {len(acoustic)}")

    # The enrolled relatives: first-degree, in the pedigree, not in Acoustic.
    enrolled: list[str] = []
    """Initialised first-degree relatives eligible for new enrolment."""
    for subject in acoustic:
        for relative in first_degree(subject, father, mother, children_by_parent):
            if relative not in acoustic_age and relative in index:
                enrolled.append(relative)
                """Retained an unexamined pedigree relative for possible enrolment."""
    """Collected eligible relatives of the complete Acoustic target sample."""
    enrolled = list(dict.fromkeys(enrolled))
    """Deduplicated relatives reached from more than one participant."""
    rng.shuffle(enrolled)
    """Randomised otherwise exchangeable relatives deterministically."""
    enrolled = enrolled[:RELATIVE_TARGET]
    """Restricted enrolled relatives to the prespecified recruitment target."""
    # They are the generation above a participant of median age, so around 76.
    relative_age: dict[str, float] = {
        person: float(np.clip(rng.normal(76.0, 8.0), 40.0, 95.0)) for person in enrolled
    }
    """Simulated bounded ages for newly enrolled older-generation relatives."""
    print(f"enrolled relatives, never examined: {len(enrolled)}")

    # Family units among the pedigree people, so no unit is too deep for the
    # region probability. A unit is a participant with their enrolled kin.
    assigned: set[str] = set()
    """Initialised pedigree people already assigned to a family unit."""
    units: list[list[str]] = []
    """Initialised connected pedigree family units for the campaign design."""
    for subject in acoustic:
        if subject in assigned:
            continue
        unit: list[str] = [subject]
        """Started a family unit from this unassigned Acoustic participant."""
        assigned.add(subject)
        for relative in first_degree(subject, father, mother, children_by_parent):
            if len(unit) >= LARGEST_UNIT:
                break
            if relative in assigned:
                continue
            if relative in acoustic_age or relative in relative_age:
                unit.append(relative)
                assigned.add(relative)
                """Added an eligible first-degree relative without exceeding the cap."""
        units.append(unit)
        """Stored the participant-centred family unit."""
    """Built capped participant-centred units from the pedigree contribution."""
    # **Pair up whatever is left rather than leaving it alone.** A singleton
    # contributes its own value and no relatedness, so it carries none of the
    # information a family design exists for. A first pass left 446 of 1,600
    # people unmatched; this sweeps the remainder against their own relatives
    # before giving up on any of them.
    leftover: list[str] = [
        person for person in list(acoustic_age) + enrolled if person not in assigned
    ]
    """Collected target participants not reached by the first unit-building pass."""
    still: set[str] = set(leftover)
    """Tracked leftover people still awaiting a connected unit."""
    for person in leftover:
        if person not in still:
            continue
        still.discard(person)
        partner: str | None = next(
            (
                relative
                for relative in first_degree(person, father, mother, children_by_parent)
                if relative in still
            ),
            None,
        )
        """Selected an unassigned first-degree partner when one remained."""
        if partner is not None:
            still.discard(partner)
            units.append([person, partner])
            assigned.update((person, partner))
            """Stored and marked a connected two-person leftover unit."""
        else:
            units.append([person])
            assigned.add(person)
            """Stored an irreducible singleton after exhausting related partners."""
    """Assigned every leftover target participant to a final family unit."""

    # Now lay everything out: the pedigree people first, then the Biggs.
    people: list[str] = []
    """Initialised pedigree identifiers in final design-row order."""
    ages: list[float] = []
    """Initialised ages aligned to the final design rows."""
    roles: list[int] = []
    """Initialised non-identifying recruitment roles aligned to design rows."""
    family: list[int] = []
    """Initialised family-unit codes aligned to the final design rows."""
    for unit_index, unit in enumerate(units):
        for person in unit:
            people.append(person)
            if person in acoustic_age:
                ages.append(acoustic_age[person])
                roles.append(ROLES["acoustic"])
            else:
                ages.append(relative_age[person])
                roles.append(ROLES["relative"])
            family.append(unit_index)
    """Laid out the complete pedigree contribution in connected-unit order."""
    pedigree_rows: list[int] = [index[person] for person in people]
    """Mapped final pedigree people back to relationship-matrix rows."""
    pedigree_block: np.ndarray = relationship_source[
        np.ix_(pedigree_rows, pedigree_rows)
    ]
    """Extracted the additive relationship block for included pedigree people."""
    print(f"pedigree part: {len(people)} people in {len(units)} units")

    # The Biggs families are their own founders: a proband and their children.
    offspring_counts: list[int] = [1] * (PROBANDS // 2) + [2] * (
        PROBANDS - PROBANDS // 2
    )
    """Assigned half the probands one offspring and half two offspring."""
    rng.shuffle(offspring_counts)
    """Randomised the proband family shapes deterministically."""
    biggs_units: list[tuple[int, list[tuple[float, int]]]] = []
    """Initialised non-pedigree clinic units with age and role only."""
    next_family: int = len(units)
    """Selected the first unused family-unit code after the pedigree contribution."""
    for count in offspring_counts:
        proband_age: float = float(rng.uniform(*PROBAND_AGE))
        """Simulated an older adjudicated-proband age within the specified range."""
        child_ages: list[float] = [
            float(np.clip(proband_age - rng.normal(PARENT_GAP, 6.0), 25.0, 74.0))
            for _ in range(count)
        ]
        """Simulated bounded offspring ages around the measured parent gap."""
        # The proband pairs with one child, because a proband-offspring pair is
        # where the information is. A second child cannot join them without
        # making a trio, so it becomes its own unit and its tie to the proband
        # is lost. That loss is the model's limit, not the design's.
        biggs_units.append(
            (
                next_family,
                [
                    (proband_age, ROLES["proband"]),
                    (child_ages[0], ROLES["offspring"]),
                ],
            )
        )
        """Stored the informative proband-offspring pair as one connected unit."""
        next_family += 1
        """Advanced to the next unused family-unit code."""
        for spare in child_ages[1:]:
            biggs_units.append((next_family, [(spare, ROLES["offspring"])]))
            next_family += 1
            """Advanced past the singleton offspring's family-unit code."""
        """Stored any second offspring as a separate founder unit."""
    """Built every independent Biggs clinic-family contribution."""
    biggs_n: int = sum(len(rows) for _, rows in biggs_units)
    """Counted people in the complete Biggs contribution."""
    print(
        f"Biggs part: {PROBANDS} probands and "
        f"{biggs_n - PROBANDS} offspring, {biggs_n} people"
    )

    total: int = len(people) + biggs_n
    """Counted the complete proposed Aim 2 sample."""
    relationship: np.ndarray = np.zeros((total, total))
    """Initialised the final additive relationship matrix."""
    relationship[: len(people), : len(people)] = pedigree_block
    """Placed the reviewed-pedigree relationships in the leading block."""
    at: int = len(people)
    """Selected the first matrix row reserved for the Biggs contribution."""
    for unit_id, rows in biggs_units:
        size: int = len(rows)
        """Counted people in this independent Biggs unit."""
        for i in range(size):
            for j in range(size):
                relationship[at + i, at + j] = 1.0 if i == j else 0.5
                """Set one relationship entry within this independent Biggs unit."""
        """Encoded founder self-relationships and parent-offspring relatedness."""
        for age, role in rows:
            ages.append(age)
            roles.append(role)
            family.append(unit_id)
        """Appended non-identifying Biggs ages, roles and unit codes."""
        at += size
        """Advanced the next Biggs matrix position by this unit's size."""
    """Laid every Biggs unit into the final design arrays and relationship matrix."""

    age_array: np.ndarray = np.array(ages)
    """Converted aligned ages to the numeric array written to the design archive."""
    role_array: np.ndarray = np.array(roles)
    """Converted aligned role codes to the numeric array written to the archive."""
    family_array: np.ndarray = np.array(family)
    """Converted aligned family-unit codes to the working numeric array."""

    # **Every unit must be a connected family.** Grouping people who are not
    # related to one another gives a block-diagonal matrix, and the model
    # refuses it -- correctly, because such a block is two families claiming to
    # be one. Split each unit into the components it actually has.
    split: np.ndarray = np.empty_like(family_array)
    """Initialised corrected unit codes aligned to every design row."""
    next_id: int = 0
    """Initialised the next connected-component unit code."""
    for unit in np.unique(family_array):
        unit_rows: np.ndarray = np.flatnonzero(family_array == unit)
        """Selected rows carrying this provisional family-unit code."""
        unit_block: np.ndarray = relationship[np.ix_(unit_rows, unit_rows)]
        """Extracted relationships among people in the provisional unit."""
        seen: set[int] = set()
        """Initialised provisional-unit positions assigned to a component."""
        groups: list[list[int]] = []
        """Initialised connected components within the provisional unit."""
        for start in range(len(unit_rows)):
            if start in seen:
                continue
            stack: list[int] = [start]
            """Initialised depth-first traversal from this unvisited person."""
            group: list[int] = []
            """Initialised positions belonging to the current component."""
            seen.add(start)
            while stack:
                here: int = stack.pop()
                """Selected the next connected position to expand."""
                group.append(here)
                """Added this position to the current connected component."""
                for other in range(len(unit_rows)):
                    if (
                        other not in seen
                        and here != other
                        and unit_block[here, other] > 0
                    ):
                        seen.add(other)
                        stack.append(other)
                """Queued every newly reached relative in the provisional unit."""
            """Completed the connected component reached from this start."""
            groups.append(group)
            """Stored the completed connected component."""
        """Partitioned the provisional unit into relationship-connected components."""
        for group in groups:
            split[unit_rows[group]] = next_id
            """Assigned one stable unit code to this connected component."""
            next_id += 1
            """Advanced to the next connected-component unit code."""
    """Reassigned every design row to a genuinely connected family unit."""
    if next_id != len(np.unique(family_array)):
        print(
            f"  split {len(np.unique(family_array))} units into "
            f"{next_id} connected ones"
        )
    family_array = split
    """Replaced provisional unit codes with the connected-component codes."""
    print(f"\nTOTAL {total} people in {family_array.max() + 1} units")
    for name, code in ROLES.items():
        role_mask: np.ndarray = role_array == code
        """Selected design rows carrying this recruitment role."""
        print(
            f"  {name:<10} {role_mask.sum():>5}  median age "
            f"{np.median(age_array[role_mask]):.0f}, over 65 "
            f"{int((age_array[role_mask] >= 65).sum())}"
        )
    """Reported the count and age distribution of each recruitment role."""
    print(
        f"  over 65 overall: {int((age_array >= 65).sum())} "
        f"({(age_array >= 65).mean():.1%})"
    )

    # **The model wants a diagonal of exactly one, and kinship gives 1 + F.**
    # SAFS carries real consanguinity, so some diagonals exceed one. The latent
    # mediation model is specified on an additive relationship with a unit
    # diagonal, so the inbreeding is dropped here rather than smuggled in --
    # and it is reported, because dropping it is a limitation of the model
    # rather than a property of the sample.
    inbreeding: np.ndarray = np.diag(relationship) - 1.0
    """Calculated each included pedigree person's inbreeding coefficient."""
    inbred: int = int((inbreeding > 1e-9).sum())
    """Counted design rows with non-negligible pedigree inbreeding."""
    print(
        f"\ninbreeding: {inbred} of {total} people have F > 0, "
        f"largest F {inbreeding.max():.4f}, mean over the inbred "
        f"{inbreeding[inbreeding > 1e-9].mean() if inbred else 0:.4f}"
    )
    print("  the model takes a unit diagonal, so F is dropped and not modelled")
    np.fill_diagonal(relationship, 1.0)
    """Removed inbreeding from the relationship diagonal as the model requires."""

    np.savez_compressed(
        OUT,
        relationship=relationship,
        age=age_array,
        role=role_array,
        family=family_array,
    )
    """Wrote only relationship, age, role and unit arrays, with no identifiers."""
    print(f"\nwritten to {OUT} (no identifiers)")
