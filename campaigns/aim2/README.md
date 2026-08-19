# Aim 2 power campaign

Level and power for the latent mediation test, on the design settled 18 August
2026 — the improved one, with the family-history recruitment route rather than
the design as the application currently reads.

`report.html` is the write-up. Everything else here is what produced it, kept so
the numbers can be re-made rather than trusted.

## What is here

| file | what it does |
| --- | --- |
| `build_design.py` | Builds `design.npz` from the real pedigrees: a relationship matrix, an age, a role and a family unit per person, and **no identifiers**. |
| `campaign.py` | The power grid — two measurement arms by six truths, one array task per cell. |
| `campaign_level.py` | The level grid — two arms by three truths, all with the direct path at nought. |
| `aim2_array.sbatch`, `aim2_level.sbatch` | Submit each grid to Medusa, one node and 120 workers per cell. |
| `power_curve.py` | Turns the grid into the smallest detectable effect and the sample multiple needed for 80 per cent. |
| `all_levels.py` | Every null cell with its Monte Carlo error, working out per cell which test it is the level *of*. |
| `route_one_size.py` | Both integrators at one family size, for checking the cheap route against the accurate one. |
| `results/` | One JSON per cell, as written by the array. |

## Two things worth knowing before changing anything

**`campaign_level.py` is a separate file from `campaign.py` on purpose.** The
power array was still running when the level grid was written, and array tasks
load their script when they start — editing `campaign.py` in place would have
had the last cells compute something different from the first, with nothing in
the output to say so.

**The direct path lives in the cells, not in `FIXED`.** `FIXED` is merged last,
so anything named in both is decided by `FIXED`. The original grid pinned
`c_prime` at 0.2 there, in every cell including both of its nulls — which is why
the direct-path test's level was never measured anywhere, and why naming
`c_prime` in a cell would have been silently ignored. It has been moved out for
the level grid. Move it back and the level cells quietly stop testing anything.

## What the campaign found

The design does not reach 80 per cent power anywhere in the range considered.
The best case is 55 per cent, with audiograms at a mediated effect of 0.24;
reaching 80 per cent there needs about 1.7 times the sample. Every null cell
holds its level, so those figures are readable rather than an artefact.

The more serious finding is the direct-path test. It rejects between 0.06 and
0.12 where the true direct path is 0.2, so a second grid asked what size of one
*would* be detectable:

| direct path | ABR only | audiogram |
| --- | --- | --- |
| 0 (level) | 0.025 | 0.030 |
| 0.3 | 0.115 | 0.155 |
| 0.5 | 0.300 | 0.330 |
| 0.7 | 0.500 | 0.545 |

Eighty per cent is reached nowhere. The two grids then meet on a comparison that
states the asymmetry exactly: with audiograms a direct path of 0.7 gives 0.545,
the same power to three decimals as a mediated effect of 0.24. The two estimands
are equally detectable only when the direct one is about 2.9 times larger —
and they are commensurable, because the model sums them into one outcome
loading.

Separating vertical from horizontal pleiotropy is what Aim 2 is for, so this is
a design question. It is the weak identification the family-count ladder of 18
August found, now with a number on it.
