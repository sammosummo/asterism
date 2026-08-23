# 17. Machine-readable support and analysis receipts

**Status: accepted.** Each release carries one structured manifest mapping its
supported analyses to their reportable quantities, documented Python entry
points, measured design ranges, required checks, pre-written pass rules and
evidence. Release automation reads this record rather than reconstructing the
claim from prose or from which models happen to be importable.

Before fitting, the standard analysis path compares the proposed design with
the manifest. Falling outside a measured range names the additional
design-specific check required and prevents reportable status; it does not
claim that the method is generally invalid. Every supported analysis also has
a small synthetic example which uses only the public Python interface, installs
the built wheel and exercises this preflight and receipt path in automation.

Every real analysis's caller-side wrapper automatically stores a JSON receipt
with its controlled outputs. It identifies the package version and source
commit, saved-wheel SHA-256, dependency-lock checksum, model settings,
non-identifying design summary, input commitments, the fit record when present
and the final reportable, diagnostic-only or refused outcome. It contains no
participant data and never goes on the Asterism release page. Numerical fitting
classes perform no file I/O; the wrapper proves which fixed calculation
produced a result without turning Asterism into a data or file-format manager.

The fit record is required for reportable and diagnostic-only outcomes and is
absent for a refusal that produced no usable candidate. The refusal code and
message take its place. Asterism supplies the serialisable receipt data and the
standard wrapper template; the analysis project owns the output path and writes
the JSON beside its controlled outputs.

Both this receipt and the support manifest carry `schema_version: 1`. Optional
fields may be added within version 1, but existing fields retain their names and
meanings throughout 0.1.x. Readers refuse an unsupported schema version rather
than inferring its shape.

Each fit record remains independently identifiable without becoming a full run
manifest. It carries the immutable Asterism version, source commit,
release-build status and the subject-order commitment promised by ADRs 0002 and
0003. The analysis receipt adds the artifact, dependency, input, configuration
and consumer-script commitments that belong to the caller rather than the
in-memory estimator.
