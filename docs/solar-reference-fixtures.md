# Native SOLAR reference fixtures

Five scientific checks have a strict two-mode adapter for native SOLAR:

- `against_solar`
- `bivariate_against_solar`
- `liability_against_solar`
- `mixed_bivariate_against_solar`
- `spatial_against_solar`

They use generated, participant-free inputs only. A live refresh is qualified
against the fixed executable `/usr/local/bin/solar` and the exact banner
identity `SOLAR Eclipse 9.0.0 (2022-01-01)`. Another path, version, incomplete
report, or failed scientific comparison stops the refresh before a fixture is
written.

Each JSON fixture under `checks/external_fixtures/` records:

- the exact comparison-adapter SHA-256;
- stable hashes for generated arrays and native input files;
- the SHA-256 of `checks/solar_reference_adapter.py`;
- the native executable path and version;
- raw native outputs, without substituting Asterism-derived quantities; and
- acceptance rules written in the adapter before the external result is used.

The liability check requires positive native standard errors for its two
interior cases. Its fixed zero-heritability negative control instead requires
both implementations to return exactly zero, because a bound estimate has no
native standard error. The 0.25-standard-error acceptance bar for the interior
cases is unchanged.

The spatial check freezes two distinct facts. Its block-diagonal kernel is a
same-model comparison. Its dense kernel documents that native SOLAR silently
drops cross-pedigree covariance and therefore reproduces its own block result;
that dense run is a tested limitation, not evidence that SOLAR fits Asterism's
cross-family spatial model.

## Generated inputs must be quantised, and text has no tolerance

A fixture is verified by regenerating its inputs and checking their identities,
so a fixture is only portable if generation is. `array_identity` already
quantises floating-point arrays to twelve decimals before hashing, which is why
relationship matrices and covariates regenerate identically on Mac and Linux.

Native input **files** escaped that, because they are text and text has no
tolerance. The liability check rendered a raw generated age at twelve decimal
places. A draw differs between the two platforms in its last bit, about seven
parts in a thousand million million on a number of order fifty, and that is
enough to change the twelfth decimal whenever a value sits near the rounding
boundary. Over six hundred draws one of them did, so the phenotype table hashed
on the Mac could not be regenerated on Linux and the check could only ever
verify where it was made. Nothing scientific differed: the binary phenotype
itself agreed exactly, and refreshing after quantisation returned every native
heritability, log likelihood and p-value unchanged to six significant figures,
with only two standard errors moving in their fifth and fourth.

A generated value that reaches a native input file is therefore quantised at
generation, far coarser than the arithmetic underneath it. The liability check
quantises age at six decimals, which is nine orders of magnitude above the
difference between platforms.

The other four checks pass on both platforms today. Any of them that writes a
raw generated float into a native input file has the same latent fault, which
would appear the first time a value landed near a boundary. None has been
checked for it.

## Refresh with native SOLAR

Run one explicit live refresh from the repository root, substituting the check
and adapter names from the list above:

```console
uv run python checks/external_reference_fixture.py refresh \
  --check-id against_solar \
  --adapter checks/against_solar.py \
  --fixture checks/external_fixtures/against_solar.json
```

Review the tool version, generated-input identities, raw outputs and acceptance
before retaining the written fixture. Do not change a tolerance after seeing a
live result. Formatting or editing an adapter after refresh makes its fixture
stale by design.

## Verify portably

Portable release verification recomputes Asterism and consumes only the frozen
native outputs:

```console
uv run python checks/external_reference_fixture.py verify \
  --check-id against_solar \
  --adapter checks/against_solar.py \
  --fixture checks/external_fixtures/against_solar.json
```

The adapter response must state `external_tool_invoked: false` and
`asterism_recomputed: true`. The fixture driver also rejects a stale adapter,
changed generated input, missing provenance, malformed external output, or a
failed scientific acceptance rule. Running an adapter with no arguments remains
the useful qualified live comparison for local investigation; release rules use
only explicit `verify` mode.
