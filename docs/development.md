# Developing Asterism

For working on Asterism itself. If you only want to *use* it, the
[README](../README.md) is the whole story and none of this applies.

## Building from source

CPython 3.13 or 3.14, the repository-pinned Rust 1.97.1 toolchain, and `uv`.
Build isolation, the development environment and wheel builds all use the
locked Maturin 1.14.1 release.

```sh
uv sync --locked --all-groups
uv run maturin develop --release --locked
```

That gives an editable build. An editable build is deliberately not a
releasable one: it refuses to call any result reportable, because that status
belongs to a checksummed wheel saved by a fixed release. See
[ADR 0012](adr/0012-small-analysis-ready-releases.md).

## The gate

```sh
sh tools/check_locally.sh
```

That runs everything: Rust formatting, Clippy with warnings refused, the Rust
suite, the locked environment, the Python formatter and linter, Asterism's own
Python style rules, the generated API reference's staleness check, the release
metadata check, and the Python suite.

It runs here rather than on GitHub because Asterism's hosted automation is
switched off — the account does not pay for hosted minutes, so every automatic
run failed before it started. The workflows are kept and can still be started
by hand, but nothing starts them.

## The scientific checks

The gate above is about the codebase. The checks that measure whether the
*numbers* are right live in `checks/`, and each supported analysis names the
ones it depends on in `release.toml`.

```sh
uv run --locked python checks/against_famskat.py
```

The complete list, with what each one establishes, is in
[numerical-validation.md](numerical-validation.md).

## The generated reference

`docs/api-reference.md` is built from the package's own docstrings and must not
be edited by hand:

```sh
uv run python tools/build_api_reference.py
```

A test fails when the file on disk stops matching what the generator produces,
so a changed signature cannot ship beside a stale reference.

## Examples

Every example in `examples/` is a complete program that generates its own data
from a fixed seed. Tests require each one to run, to keep recovering whatever
truth it claims to recover, and — where the README quotes its output — to match
that quotation exactly.
