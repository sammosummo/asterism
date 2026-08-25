#!/bin/sh
# The codebase-quality gate, run here instead of on GitHub.
#
# Asterism's GitHub Actions are switched off: the account does not pay for
# hosted minutes, so every automatic run failed before it started. These are
# the same steps `.github/workflows/quality.yml` would have run, in the same
# order, so a clean run here means what a green CI run used to mean.
#
# Usage:  sh tools/check_locally.sh
set -e

say() { printf '\n=== %s ===\n' "$1"; }

say "Rust formatting"
cargo fmt --check

say "Clippy, warnings refused"
cargo clippy --locked --all-features --all-targets -- -D warnings

say "Rust suite"
cargo test --release --locked --all-features

say "Locked environment"
uv lock --check
uv sync --locked --all-groups

say "Extension used by source-level tests"
uv run maturin develop --release --locked

say "Python formatting"
uv run ruff format --check python tests checks tools examples

say "Python lint"
uv run ruff check python tests checks tools examples

say "Asterism Python style"
uv run python tools/check_python_style.py

say "API reference is current"
uv run python tools/build_api_reference.py --check

say "Release metadata"
uv run python tools/check_release.py --metadata

say "Python suite"
uv run pytest -q tests

printf '\nAll local checks passed.\n'
