# Prototype preflight receipt — 21 August 2026

This receipt records what was actually verified before the Aim 2 grant-design
power grid could run. It is not a power result, a scientific-release receipt,
or evidence that the family-history extension is feasible.

## Captured prototype

- Isolated branch: `codex/mediation-grant-prototype`.
- Inspected starting commit:
  `9e988b48bd7cfdc20a4bbaacfdb45d002e028b53`.
- Executable prototype commit:
  `cca65b3d4ec82a1b3f4db848db33958111350214`.
- Frozen cell/manifests commit:
  `058515660f34f46e5c7e8aeddbb3d8db309ded5b`.
- `cells.json` SHA-256:
  `75b59fb10a5ec6618171e7851eb8c35c8267650fbad6d2fb8bd6870eec61fb41`.
- `run_cell.py` SHA-256:
  `a9bc23a23c745fb64397882d677bcab170e89f5aa1414ebda173183d96386af6`.

The manifest contains 72 unique cells: three case-relative yields, two
extension states, two outcomes, two hearing-information bounds, and both faces
of the vertical union null plus the `a*b=.24` alternative. Every cell specifies
200 outer replicates, 50 requested bootstrap draws and alpha `.025`. The six
designs have totals 1,100, 1,225, 1,350, 1,350, 1,475 and 1,600. Their manifests
report zero missing claimed ties and zero disconnected fitted units. They also
report source relationships projected away by the size-eight local-family
projection; the projection is not represented as the complete pedigree.

The self-contained HTML logic prototype passed its reducer/derive acceptance
checks. It retains connected case trios, refuses a detached second offspring,
and refuses to promote the structural extension to demonstrated feasibility.
Its SHA-256 is
`7fe220f2af4695ef9c140ae296d07aaa2f76a92ded1dc6493d66ae56c05757da`.

## Local execution probe

Four earlier one-replicate, zero-bootstrap cells checked plumbing only. The
fail-closed summariser correctly rejects them because their execution settings
do not match `cells.json`; `PROTOTYPE_VERDICT.md` therefore remains a no-verdict
artefact.

One evidence-settings pilot was started locally for cell 0: confirmed core,
mean yield 1.0, ordered CDR, all-audiogram information bound, and the
`a=0, b=.4` null face. It requested 200 outer replicates, 50 bootstrap draws and
10 workers. The ordered runner reported:

- 25/200 returned rows after about 21 minutes (`51.3 s` elapsed per returned
  row);
- 50/200 returned rows after about 26 minutes (`31.4 s` elapsed per returned
  row);
- no 75/200 checkpoint after at least 41 minutes 19 seconds, while all 10
  workers remained approximately 96–100 per cent CPU-active.

The process was then deliberately terminated because this bounded probe had
already shown that a local 14-core Mac was not a responsible substitute for the
planned 120-core-per-cell array. `run_cell.py` writes the result atomically only
after all 200 rows finish, so the aborted probe produced no result and no partial
row may be treated as level evidence. The existing `cell_000.json` remains the
earlier one-replicate smoke payload and is still rejected by the summariser.

The cached local Linux containers cannot build the current source without new
downloads: none contains Linux Rust/Cargo 1.97 or maturin, and no source-matched
Linux wheel is cached. The macOS extension was source-matched and sufficient for
the cost probe, but it does not clear the accepted Linux-wheel smoke boundary.

## Medusa gate

No prototype file was transferred to Medusa and no job was submitted.

Live access could not be verified because GlobalProtect reported no tunnel and
the documented BatchMode route timed out. More importantly, the 20 August Texas
storage-cleanup receipt records removal of the old Asterism checkout and Rust
toolchain and says not to resume transfers or compute until the administrators
give the all-clear. The all-clear request remains a draft in the inspected local
record.

The smallest valid next sequence is therefore:

1. obtain the Texas administrator all-clear and an approved work location;
2. connect the VPN and repeat the read-only Medusa host, queue, quota and
   toolchain probe;
3. stage the checksum-verified committed source and six identifier-free NPZ
   designs;
4. build and smoke-test a fresh source-matched Linux wheel;
5. run one complete 200/50 pilot cell and retain its raw JSON;
6. submit the remaining cells only after that pilot clears the eight-hour
   envelope; and
7. require all 72 hash-matched cells before `summarise.py` can issue a verdict.

Until those gates clear, measured power for the settled confirmed-core design
and the extension increment is **not available**. The earlier `.660` result
remains evidence only for its different 1,600-person campaign composition.
