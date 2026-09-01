# Python API support status

This inventory separates a public import from a scientifically supported
analysis, and keeps two dates apart. “Supported in 0.1” records the immutable
0.1.1 release claim. `release.toml` is authoritative for the current development
checkout. The fixed-instrument one-component censored and mixed-pair reruns now
pass. The current 0.2 scope for several censored components includes fit,
interval and the explicitly asymptotic test. Its analytic p-value is a known
failure on the exact 1,909-person, four-component target at 75% expected
censoring and must not be reported for that analysis without a design-specific
simulated null; this is not a general censoring threshold. The bootstrap remains
experimental. A published status therefore does not by itself mean that the
changed checkout is an analysis-ready release.

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

| Public object | 0.1 scientific status | Published role and current development note |
| --- | --- | --- |
| `AssociationModel` | Public but outside 0.1 scientific support | Marker association is deferred. |
| `AutoregressiveModel` | Public but outside 0.1 scientific support | Autoregressive modelling is deferred. |
| `BivariateModel` | Supported in 0.1 | Bivariate genetic correlation. |
| `CensoredComponentModel` | Public but outside 0.1 scientific support | One censored trait with any number of variance components. The current 0.2 scope includes fixed-instrument fitting, intervals and the explicitly labelled asymptotic test. The exact 1,909-person, four-component target at 75% expected censoring requires a design-specific simulated null before its p-value is reported. The bootstrap remains experimental. |
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
| `installed_extension_sha256` | Supported infrastructure | SHA-256 of the native module loaded by the current process, for exact worker and checkpoint provenance. |
| `kinship_classes` | Supported utility | Zero-diagonal component bases reported through coefficients and contrasts. |
| `mixed_bivariate_fit` | Supported in 0.1 | Genetic correlation for a mixed pair with a continuous trait; the binary-with-censored pairing is deferred. Corrected fixed-threshold development recovery checks now pass for continuous, binary, censored and censored-pair cells. |
| `mixed_bivariate_interval` | Supported in 0.1 | Interval for the mixed-pair genetic correlation. Corrected fixed-threshold coverage checks now pass for all four development pairings. |
| `mixed_bivariate_test` | Supported in 0.1 | Test for the mixed-pair genetic correlation. Corrected null-level checks now pass; the separate correlation-one boundary measurement remains historical. |
| `prepare` | Supported in 0.1 | One-trait Gaussian model preparation. |
| `region_log_probability` | Public but outside 0.1 scientific support | Diagnostic numerical primitive, not a supported analysis. |
| `relationship_matrix` | Supported utility | Additive relationship-matrix construction. |
| `release_manifest` | Supported infrastructure | Read-only access to the exact manifest embedded in the wheel. |
| `run_analysis` | Supported infrastructure | Standard outcome and receipt-data path. |
| `subject_order_commitment` | Supported infrastructure | Non-identifying commitment to positional subject order. |
| `tobit_fit` | Supported in 0.1 | One-trait censored heritability. The numerical path is unchanged, and corrected fixed-instrument recovery checks pass. |
| `tobit_interval` | Supported in 0.1 | Interval for censored-trait heritability. Corrected fixed-instrument coverage checks pass at 52% and 75% expected censoring. |
| `tobit_test` | Supported in 0.1 | Analytic one-component test retained for compatibility. Its corrected target check passes, with null rejection of 0.040 and 0.045 at 52% and 75% expected censoring. |
| `weighted_chi2_upper_tail` | Public but outside 0.1 scientific support | Diagnostic numerical primitive, not a supported analysis. |

No public object is deprecated in the 0.1.1 release. Existing documented
names, signatures and fit-record fields are compatibility commitments for this
release.

The models marked *public but outside 0.1 scientific support* are described in [outside-support.md](outside-support.md), with what each is for and what it does not do.
