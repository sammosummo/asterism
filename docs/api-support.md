# Python API support status

This inventory separates a public import from a scientifically supported
analysis. It is authoritative for the development version described by
`release.toml`; the statistical claims remain disabled while that manifest's
its supported analysis set.

The installed-wheel workflow is configured to check each supported interpreter
and platform independently. A configured workflow is not release evidence, and
those runs are not the Mac/Linux agreement check required by ADR 0012. That
distinct comparison, including exact status/field agreement and model-specific
numerical tolerances, is recorded in the release evidence.

The object docstrings and `README.md` describe signatures and ordinary use. The
statistical methods specification defines estimands, equations, inference and
limitations. “Supported in 0.1” below means that the object belongs to the
supported 0.1.1 analysis set. It does not make other importable objects
scientifically supported.

| Public object | Status | 0.1 role |
| --- | --- | --- |
| `AssociationModel` | Public but outside 0.1 scientific support | Marker association is deferred. |
| `AutoregressiveModel` | Public but outside 0.1 scientific support | Autoregressive modelling is deferred. |
| `BivariateModel` | Supported in 0.1 | Bivariate genetic correlation. |
| `ComponentModel` | Supported in 0.1 | Mean-diagonal contributions and proportions, their matching intervals, and coefficients or contrasts for zero-diagonal bases. |
| `DiscreteGxeModel` | Supported in 0.1 | Discrete-environment genetic correlation and calibrated genetic tests. |
| `GxeModel` | Supported in 0.1 | Restricted continuous G×E surfaces after the manifest's required checks pass. |
| `LatentMediationModel` | Public but outside 0.1 scientific support | Inferential latent mediation is deferred. |
| `LiabilityModel` | Supported in 0.1 | Binary-trait liability heritability and inference. |
| `PreparedModel` | Supported in 0.1 | Prepared one-trait Gaussian heritability. |
| `SpatialModel` | Public but outside 0.1 scientific support | Spatial-component presence testing is deferred. |
| `VariantSetModel` | Public but outside 0.1 scientific support | Variant-set discovery is deferred. |
| `__version__` | Supported infrastructure | Immutable public package version. |
| `align` | Supported utility | Identifier-based alignment before positional numerical fitting. |
| `build_identity` | Supported infrastructure | Immutable build identity for fit records and receipts. |
| `grouping_matrix` | Supported utility | Grouping-matrix construction: one where two rows share a group. Household from a home identifier, person-level from a person identifier. |
| `kinship_classes` | Supported utility | Zero-diagonal component bases reported through coefficients and contrasts. |
| `mixed_bivariate_fit` | Supported in 0.1 | Genetic correlation for a mixed pair with a continuous trait; the binary-with-censored pairing is deferred. |
| `mixed_bivariate_interval` | Supported in 0.1 | Interval for the mixed-pair genetic correlation. |
| `mixed_bivariate_test` | Supported in 0.1 | Test for the mixed-pair genetic correlation. |
| `prepare` | Supported in 0.1 | One-trait Gaussian model preparation. |
| `region_log_probability` | Public but outside 0.1 scientific support | Diagnostic numerical primitive, not a supported analysis. |
| `relationship_matrix` | Supported utility | Additive relationship-matrix construction. |
| `release_manifest` | Supported infrastructure | Read-only access to the exact manifest embedded in the wheel. |
| `run_analysis` | Supported infrastructure | Standard outcome and receipt-data path. |
| `subject_order_commitment` | Supported infrastructure | Non-identifying commitment to positional subject order. |
| `tobit_fit` | Supported in 0.1 | One-trait censored heritability. |
| `tobit_interval` | Supported in 0.1 | Interval for censored-trait heritability. |
| `tobit_test` | Supported in 0.1 | Test for censored-trait heritability. |
| `weighted_chi2_upper_tail` | Public but outside 0.1 scientific support | Diagnostic numerical primitive, not a supported analysis. |

No public object is deprecated in the 0.1.1 release. Existing documented
names, signatures and fit-record fields are compatibility commitments for this
release.

The models marked *public but outside 0.1 scientific support* are described in [outside-support.md](outside-support.md), with what each is for and what it does not do.
