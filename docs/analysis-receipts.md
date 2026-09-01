# Analysis receipts

`asterism.run_analysis` wraps a fit in a record of what produced it. It checks
the installed wheel and the release manifest first, runs the fitting callback
only if those pass, and returns a JSON-serialisable mapping.

The record's `outcome` is one of two words:

- `fitted`: the callback returned a fit record;
- `refused`: nothing was fitted, and `refusal` names the reason.

A fitted outcome carries a `facts` mapping stating four things about it.

| Fact | True when |
| --- | --- |
| `converged` | the free fit and every inference fit reported convergence |
| `all_quantities_present` | every quantity the manifest declares for this analysis is present |
| `all_values_finite` | every numerical value in the record is finite |
| `all_profiles_evaluated` | no profile evaluation failed |

`failures` names the field behind each fact that is false. What those facts
mean for a particular analysis is the caller's to decide.

Development builds refuse before fitting, so an analysis wrapper and its tests
can be written without a mutable checkout producing a record that looks like a
release.

## Standard release receipts

Release automation runs one participant-free example for every supported
analysis through the same `run_analysis` seam:

```console
python tools/synthetic_analysis_receipts.py \
  --output release-evidence/synthetic-analysis-receipts \
  --wheel dist/asterism-0.2.0-*.whl
```

The command uses only the public Python interface. It refuses a development or
dirty build, binds every receipt to the saved wheel, embedded dependency lock,
source commit, complete synthetic-input commitment and fitted row-order
commitment, and requires all nine fits to meet every fact. It keeps all
receipts in memory until the entire set passes, then writes one strict JSON file
per analysis and a checksummed `index.json`. A partial set is never release
evidence. The independent release verifier reads every indexed file and
recomputes convergence, required-field, profile-failure and finite-value
conditions instead of trusting an aggregate `passed` flag.

This command is an installed-wheel release check, not a shortcut for real
analysis provenance. Real analysis projects still own their input commitments,
consumer commit and controlled receipt location.

## Fixed inputs

A caller installs one of the saved release wheels. Before calling
`run_analysis`, it records:

- the wheel's SHA-256;
- the dependency lock's SHA-256;
- the clean Git commit containing the consumer wrapper;
- values-free commitments to each input snapshot or file;
- the exact fitted subject order, supplied as identifiers only to
  `run_analysis`, which retains only its SHA-256 commitment.

The model settings and design summary must not contain participant values or
identifiers. Input commitments belong in the controlled analysis receipt, not
on the data-free Asterism release page.

## Caller pattern

The fitting callback composes the point estimate and every inferential result
the manifest names for that analysis. For the one-trait Gaussian path the
prepared fit already contains its interval and test:

```python
import json
from pathlib import Path

import asterism

order_sha256 = asterism.subject_order_commitment(aligned["order"])


def fit_model() -> dict[str, object]:
    prepared = asterism.prepare(
        design_matrix,
        aligned["relationship"],
        subject_order_sha256=order_sha256,
    )
    return prepared.fit(aligned["response"])


receipt = asterism.run_analysis(
    "one_trait_gaussian_heritability",
    {
        "sample_size": len(aligned["order"]),
        "largest_family": largest_family,
        "trait_type": "continuous",
        "components": "additive_relationship",
    },
    fit_model,
    model={"estimator": "reml", "covariates": covariate_names},
    provenance={
        "wheel_sha256": wheel_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "consumer_commit": consumer_commit,
        "input_commitments": input_commitments,
    },
    subject_order=aligned["order"],
)

# The analysis project, not a fitting class, owns this controlled output path.
receipt_path = Path("controlled-results/heritability.receipt.json")
receipt_path.write_text(
    json.dumps(receipt, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
```

The example write is intentionally ordinary caller code. Asterism returns a
JSON-serialisable mapping and performs no file I/O. The caller must refuse to
copy a receipt outside its controlled result area if its own model settings or
input labels contain sensitive material. Constructing the estimator inside the
callback also lets documented validation errors become refused receipts; a
model constructed before `run_analysis` remains ordinary caller code and its
exceptions are not intercepted.

## Identity checks

If a fit record already carries build or subject-order identity,
`run_analysis` verifies it and refuses a mismatch rather than overwriting it.
Documented Asterism estimator `ValueError` codes become refused receipts;
unexpected exceptions propagate so implementation faults cannot be presented
as scientific refusals. A fit whose facts are not all true is returned in full,
with its estimates, so it can be inspected.
