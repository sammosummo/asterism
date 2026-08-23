# 13. Scientific support does not remove public models

**Status: accepted.** The supported analyses for a release define its scientific
claims, not which models can be imported. A model outside the current release
may remain public for simulations, power calculations and diagnostic work until
it is deliberately deprecated and its consumers are migrated. Conversely,
every script for a supported analysis must use the documented Python interface,
never the private positional `_core` functions, and must exercise a deterministic
smoke path against the saved release wheel. The Python interface is the only
supported programming interface in 0.1; direct Rust use is not a product
contract.

The API reference documents every public Python object. Each entry is labelled
`supported in 0.1`, `public but outside 0.1 scientific support`, or
`deprecated`; being importable never implies scientific support. No public
object remains undocumented.

Before 0.1.0, a breaking repair to that interface is allowed only when every
known consumer is migrated in the same change. From 0.1.0 through the 0.1
series, documented Python names, method signatures and existing fit-record
fields remain compatible. Additive optional fields are allowed. A breaking
change is announced by a deprecation warning for at least one minor release and
lands no earlier than 0.2.

The support manifest and analysis receipt each carry an integer schema version.
Within schema version 1, optional fields may be added but existing fields keep
their names and meanings throughout 0.1.x. An incompatible schema change uses a
new schema version; readers refuse an unsupported version rather than guessing.
