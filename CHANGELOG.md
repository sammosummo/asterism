# Changelog

This file records user-visible Asterism changes. A planned entry is not a
release claim; released entries acquire a date and correspond to an immutable
tag and saved wheels.

## Unreleased

- **The two-person normal probability is one accurate integral, and a
  relationship of one is accepted.** The package carried two implementations of
  the same integral: a sixteen-point quadrature beside the liability model and
  an adaptive Owen form beside latent mediation. The first was accurate at
  thresholds of nought and nought and about seventy times worse away from them,
  reaching 4.5e-02 in the log probability at a correlation of 0.99. Because of
  it, `build` refused any off-diagonal relationship above 0.9 in the liability,
  censored and mixed-bivariate models -- so **monozygotic twins could not be
  fitted**, nor any model carrying a component a person shares in full with
  themselves, which is what a person-level component is. The integral is now
  shared, measured against two independent integrations at 53 points to
  1.7e-11, and the refusal is gone. Replacing it moved the liability model
  against its SOLAR successor by at most 1.2e-6, or 2.7e-9 of the value, and
  most in the tail where the retired arithmetic was weakest.
- **The deep tail keeps its probability.** Below 1e-3 the two-person region is
  evaluated on the log scale, because the ordinary-scale integral holds an
  absolute tolerance and so loses the logarithm as the probability shrinks. The
  region is now accurate to 6.9e-12 in the log probability out to a log
  probability of -88, where it previously returned the floor -- a likelihood
  wrong by six hundred nats, finite enough to be believed, in the tail a rare
  binary trait and a heavily censored one both occupy.
- **`CensoredComponentModel` is the Python interface to a censored trait with
  several components**, shaped like `ComponentModel`: the components and the
  design at construction, the censored data at each call. `fit` reports each
  coefficient and the mean-diagonal proportions; `interval` gives the proportion
  by default and the coefficient on request; `test` carries `nuisance_at_bound`.
  With one component it is bit-identical to `tobit_fit`. There is deliberately
  no `heritability` key: it would be the first component's coefficient whatever
  that component is, and it moves under a rescaling that the heritability does
  not.
- **`mean_diagonal_interval(index)` gives the interval to report** when
  components are compared, beside the coefficient interval that is not
  comparable across matrices whose diagonals differ. Rescaling a component's
  matrix by four describes the same model and moves its coefficient interval;
  the proportion's does not move. Where every component carries a unit diagonal
  the two are the same interval.
- **The released censored analysis was re-measured after the integral changed**,
  and all six of its pass rules hold. `censReg` agrees to 9.6e-09 relative, all
  three points sit inside `MCMCglmm`'s posterior intervals, coverage holds at
  both censoring shares, and the target-design campaign ran 800 replicates at
  1,909 people with zero failed fits of a permitted zero.
- **`grouping_matrix` builds a component from what rows share.** One where two
  rows share a group, nought where they do not, one on the diagonal. Pass
  household identifiers and it is a household matrix; pass the person each row
  belongs to and it is the person-level matrix, which is the listener kernel
  where a listener contributes two ears. The relation is sharing, and a home is
  only one thing to share. `None` is a group nobody knows: that row shares with
  nobody and keeps its diagonal, so its effect cannot be told apart from its
  residual.
- **Every component of a censored fit can be reported, not only adjusted for.**
  `coefficient_interval(index)` and `coefficient_test(index)` profile and test
  any component, and the fit record carries `mean_diagonal_contributions`,
  `mean_diagonal_total` and `mean_diagonal_proportions` on the same footing
  `ComponentModel` reports them -- which is the quantity to compare components
  by, a raw coefficient not being comparable across matrices whose diagonals
  differ. The three are absent, rather than NaN, where a component's mean
  diagonal is not positive and finite.
- **The censored test is offered at several components rather than refused**,
  and carries `nuisance_at_bound`: true where another coefficient or the
  residual also rested on nought in the null fit, which is when the 50:50
  mixture is not the reference its p-value should be read against. It was
  briefly refused there, which would have stopped the coverage check that
  scores it from ever running. The interval's boundary verdict is still filled
  at one component only, that being where it was scored.
- **Breaking, an analysis is renamed.** `one_trait_tobit_audiogram` is now
  `one_trait_censored`. Having "audiogram" in the identifier implied the model
  applies only to audiograms; nothing in a censored variance-components model is
  specific to hearing, and the same analysis fits any right- or left-censored
  trait. Extended high-frequency audiometry stays in the documentation as the
  case it was built for and as the source of its design facts. Retained 0.1.x
  evidence keeps the name it was measured under, and `evidence/README.md` says
  which analysis that is now.
- **Breaking, error codes.** Seven codes lost their `LATENT_MEDIATION_` prefix
  now that three models raise them: `NORMAL_VARIATE_NOT_FINITE`,
  `BIVARIATE_CORRELATION_INVALID`, `BIVARIATE_THRESHOLD_INVALID`,
  `BIVARIATE_PROBABILITY_UNRESOLVED`, `BIVARIATE_PROBABILITY_OUTSIDE_BOUNDS`,
  `BIVARIATE_QUADRATURE_NOT_FINITE` and `BIVARIATE_QUADRATURE_DID_NOT_CONVERGE`.
  Three codes are retired with the guard: `LIABILITY_RELATIONSHIP_TOO_CLOSE`,
  `TOBIT_RELATIONSHIP_TOO_CLOSE` and `MIXED_BIVARIATE_RELATIONSHIP_TOO_CLOSE`.

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

## 0.1.1 (2026-08-26)

- Removed the `Private :: Do Not Upload` classifier so the package can be
  distributed on PyPI. The guard existed to stop anything reaching PyPI while
  the repository was private. No other change: nothing outside the packaging
  metadata differs from 0.1.0.

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
