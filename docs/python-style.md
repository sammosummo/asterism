# Python style

This guide defines the Python conventions for Asterism. It applies to maintained
package code, tests, scientific checks and campaign scripts. Generated results,
logs, HTML reports and third-party material are outside its scope.

## Language and runtime

Use British English in names, docstrings and documentation: `colour`, not
`color`; `standardise`, not `standardize`.

Run Python through Asterism's `uv` environment. Use `uv run ...` or a documented
`uv`-backed command; do not use system Python, mutate `.venv` by hand or install
dependencies with `pip`. Add development and scientific dependencies to the
owning `pyproject.toml` and lock them in `uv.lock`.

The editor is not part of the style contract. Ruff and Asterism's own style
check are the definitions of the mechanically enforced rules.

## Formatting and imports

Use four spaces for indentation and an 88-character line length. Let Ruff format
the code and organise imports.

Separate standard-library, third-party and local imports with blank lines. Put
`import x` before `from x import y` within each group and sort alphabetically.
Import most objects directly where they are used. Keep conventional scientific
package aliases:

```python
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
```

## Names

| Kind | Convention | Example |
| --- | --- | --- |
| Functions | `snake_case` | `simulate_phenotype` |
| Classes | `PascalCase` | `VarianceComponents` |
| Variables | `snake_case` | `n_individuals` |
| Constants | `UPPER_CASE` | `MAX_ITERATIONS` |
| Modules | `lowercase_with_underscores` | `ridge_regression.py` |
| Packages | `lowercase` | `asterism` |

Prefer whole words to abbreviations unless the value is disposable or the
notation is established in the mathematics. Names such as `n`, `h2`, `rho_g`
and `sigma2_e` are deliberate. Functions use descriptive verbs.

## Type annotations

Use modern Python syntax and comprehensive type annotations. Annotate every
function parameter and return value. Annotate the first assignment of every
named value wherever its type can be expressed, including local and temporary
values:

```python
DEFAULT_ITERATIONS: int = 1_000
CONFIG_PATH: Path | None = None


def process_data(input_data: pl.DataFrame) -> pl.DataFrame:
    results: list[dict[str, float]] = []
    """Rows accumulated while processing the input."""

    total_variance: float = compute_variance(input_data)
    """Total phenotypic variance in the input."""

    processed: pl.DataFrame = transform(input_data, total_variance)
    """Input transformed using the estimated total variance."""
    return processed
```

Do not repeat an annotation when rebinding an already annotated value. Use
built-in generic types (`list`, `dict`, `tuple`, `set`), the `|` union operator
and `Type | None` instead of legacy `typing` aliases.

Disposable loop targets, unpacking placeholders conventionally named `_`, and
values whose syntax cannot carry an annotation are exempt.

## Docstrings

Use Google-style docstrings for every public module, class and function. Include
only the sections that add information:

- `Args:` for parameters;
- `Returns:` for return values;
- `Raises:` for exceptions;
- `Yields:` for generators;
- `Attributes:` for public attributes;
- `Notes:` for mathematical or operational detail;
- `Examples:` for usage.

Private functions normally need less formal docstrings, but their purpose must
still be clear where an allowed private function exists.

## Documentation after expressions

Each non-trivial expression, transformation or small code block is immediately
followed by a standalone triple-quoted string describing what the code just
achieved. Read the code first; use the following prose to confirm its meaning.
Past tense is often clearest.

```python
session: Session = Session()
"""Created the requests session used for API calls."""

url: str = "https://example.com/api"
"""Selected the API endpoint."""

data: list[dict[str, str | int | None]] = []
"""Initialised the rows that will hold processed values."""
```

Document successive transformations as successive achievements:

```python
frame: pl.DataFrame = pl.DataFrame(source)
"""Constructed the initial data frame from the source rows."""

frame = frame.with_columns(pl.col("value") * 2)
"""Doubled the measured values."""

frame = frame.filter(pl.col("status") == "active")
"""Retained active records."""
```

Place one string after a small control-flow block when it describes the block as
a whole:

```python
for skipped_name in skipped_names:
    if skipped_name in filename:
        continue
"""Skipped files carrying a name on the exclusion list."""
```

Do not duplicate the code in prose. Trivial syntax whose meaning is already
complete—imports, `return`, `raise`, `pass`, `break`, `continue`, `else`, and a
disposable loop target—does not need its own string. Traditional `#` comments
remain appropriate for tool directives, short mathematical labels and points
where a following string cannot attach to the intended syntax.

These standalone strings are intentional documentation, not accidental useless
expressions. Ruff's `B018` rule is therefore disabled for Asterism.

## Functions and locality

Never extract a function used only once merely to shorten its caller. Keep a
calculation together when extraction would add navigation and debugging cost
without creating reuse, an independently testable concept or a real boundary.

Avoid private functions. An allowed private function must serve a concrete
purpose such as reuse, recursion, a callback or framework protocol, an
independently testable numerical primitive, or a boundary whose name materially
clarifies the caller. Public functions and classes receive the full Google-style
documentation contract.

## Zero-debt gate

The migration is complete: every maintained Python file must satisfy this guide.
The ordinary command checks the whole maintained tree and reports every current
violation:

```console
uv run python tools/check_python_style.py
```

The retained explicit spelling runs the same zero-debt check:

```console
uv run python tools/check_python_style.py --no-baseline
```

Any violation fails either command. There is no accepted debt inventory and no
command that can create one. Do not mass-insert `Any`, generic prose or
unexplained exceptions merely to satisfy the checker; each repair must express
the source's actual types and intent.

## Enforced exceptions

The style check may be bypassed only beside the smallest affected construct,
using this form:

```python
# asterism-style: allow <rule> -- <specific reason>
```

Available rule names are:

- `unannotated-local`;
- `missing-following-doc`;
- `single-use-helper`;
- `private-helper`.

The reason is mandatory. File-wide exceptions and unexplained `noqa` markers do
not satisfy this guide. Exceptions are visible debts to review, not alternative
style preferences.
