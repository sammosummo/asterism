# Changelog

This file records user-visible Asterism changes. A planned entry is not a
release claim; released entries acquire a date and correspond to an immutable
tag and saved wheels.

## Unreleased

- Asterism is MIT licensed, carries a `CITATION.cff`, and is archived with a
  DOI at publication rather than before it. ADR 0018 restores the citation
  requirement ADR 0001 made and ADR 0012 dropped, and records that Asterism is
  original work with no affiliation to SOLAR or any other package.
- The censored model now takes the shared convergence rule. It had tested an
  unprojected, unscaled gradient against a threshold a thousand times looser
  than every other family, which reported about half of all null fits as
  failures because a heritability of nought rests on its own lower bound.
- Pinned interval endpoints are compared within a written tolerance rather than
  to the last digit, which is what ADR 0012 already required of two platforms.

- The release contract, support limits and pass rules are being made
  machine-readable.
- The codebase-quality, installed-wheel and fail-closed release workflows are
  being established.
- Statistical-method documentation now covers every planned 0.1 analysis with
  equations and primary references. All scientific rules have executable
  commands; their exact fixed-wheel campaigns remain
  in progress.
- Several-component fitting now refuses linearly dependent covariance bases,
  including an identity matrix confounded with the implicit residual; its
  one-component boundary test uses the identified analytic residual-only null.
- The installed-wheel matrix runs independently on Mac and Linux; the separate
  cross-platform status/field comparison and model-specific numerical
  tolerances remain deliberately unconfigured release blockers.

## 0.1.0 (2026-08-26)

### Supported analyses

- One-trait Gaussian heritability.
- Several covariance components, using scale-invariant mean-diagonal
  contributions where defined.
- Bivariate genetic correlation.
- Restricted continuous and discrete gene-by-environment analyses.
- Binary liability heritability.
- One-trait Tobit audiogram heritability.
- Genetic correlation for a mixed pair including a continuous trait.

### Compatibility

- CPython 3.13 and 3.14 on macOS arm64 and manylinux 2.17 x86_64.
- The documented Python interface is the sole supported programming interface.

### Known limitations

- The release is not yet built or qualified. Every
  scientific pass rule remains disabled in `release.toml` until measured on the
  release commit.
- Prediction, association, variant sets, inferential latent mediation,
  autoregressive modelling and joint repeated-audiogram decomposition remain
  outside 0.1 scientific support.
