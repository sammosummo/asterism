# Analysis preflight and receipts

A numerical fit is not automatically a reportable analysis. The public
`asterism.run_analysis` path first compares a non-identifying design summary
with the measured limits embedded in the installed wheel, runs the fitting
callback only when that preflight passes, and returns one of three outcomes:

- `reportable`: the release and design checks passed, the free fit and every
  required inference fit converged, required values are finite, and no profile
  evaluation failed;
- `diagnostic-only`: a candidate fit exists, but it failed at least one
  reporting condition;
- `refused`: fitting was not allowed or no usable candidate was produced.

Development builds deliberately refuse before fitting. This lets an analysis
wrapper and its tests be written without letting a mutable checkout create a
result that looks released.

## Standard release receipts

Release automation runs one participant-free example for every supported
analysis through the same `run_analysis` seam:

```console
python tools/synthetic_analysis_receipts.py \
  --output release-evidence/synthetic-analysis-receipts \
  --wheel dist/asterism-0.1.0-*.whl
```

The command uses only the public Python interface. It refuses a development or
dirty build, binds every receipt to the saved wheel, embedded dependency lock,
source commit, complete synthetic-input commitment and fitted row-order
commitment, and requires all nine outcomes to be `reportable`. It keeps all
receipts in memory until the entire set passes, then writes one strict JSON file
per analysis and a checksummed `index.json`. A partial set is never release
evidence. The independent release verifier reads every indexed file and
recomputes convergence, required-field, profile-failure and finite-value
conditions instead of trusting an aggregate `passed` flag.

This command is an installed-wheel release check, not a shortcut for real
analysis provenance. Real analysis projects still own their input commitments,
consumer commit and controlled receipt location.

## Fixed inputs

A reportable caller installs one of the saved release wheels. Before calling
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
that the manifest names as reportable. For the one-trait Gaussian path the
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
as scientific refusals. Diagnostic-only records remain available for debugging,
but their estimates and inference must not enter report tables.
