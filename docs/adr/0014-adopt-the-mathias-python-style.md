# 14. Adopt the Mathias Python style

**Status: accepted.** Asterism adopts [the repository's Python style
guide](../python-style.md), including comprehensive local annotations,
standalone triple-quoted documentation after non-trivial expressions and blocks,
and avoidance of private or single-use helper functions. These distinctive
rules are deliberate rather than legacy material. The predecessor guide's
PyCharm requirement is removed because the resulting code, not the editor, is
the contract; obsolete workspace instructions are replaced by Asterism's
current `uv` and Ruff configuration.

This consciously reverses the 8 August 2026 workspace-migration decision to
retire the predecessor guide without replacement. On 21 August 2026 Sam
explicitly restored these project-local rules, said that he liked the unusual
rules, and directed that the guide be brought into Asterism and edited as
necessary. It is an Asterism convention now, not a claim that the retired guide
remained lab-wide policy.

The codebase-quality gate applies the guide to every maintained Python file in
the package, tests, scientific checks and campaigns. Ruff permits the
intentional standalone strings by disabling `B018`, and an Asterism style check
enforces the rules Ruff cannot express. A narrowly placed exception is allowed
only with a specific reason; generated results, logs, HTML and third-party
material are outside the gate.

The adopted repository contained thousands of pre-existing violations, and a
literal mechanical retrofit would have replaced scientific code with
unreviewable annotations and generic prose. The migration therefore began with
an exact, machine-readable debt ratchet that identified each legacy construct
by semantic fingerprint and refused both new debt and stale records after an
improvement. That migration is now complete and its acceptance inventory has
been removed. The ordinary gate and the retained `--no-baseline` spelling both
report every current violation and require zero debt before release readiness
can pass.
