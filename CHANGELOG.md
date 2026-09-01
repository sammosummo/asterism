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
  by default and the coefficient on request. At several components `test`
  explicitly reports the asymptotic Self–Liang reference as
  `asymptotic_mixture_50_50` and refuses if another variance rests on its bound.
  `bootstrap` can instead simulate the complete constrained-null reference and
  report an add-one Monte Carlo p-value, but remains experimental rather than a
  0.2 release claim. Its independent inner datasets now use deterministic
  coordinate-derived streams and refit in parallel, removing the serial
  per-outer-replicate ceiling without making the answer thread-dependent.
  Several-component builds now refuse covariance bases that cannot identify
  separate coefficients, including an explicit identity that duplicates the
  residual. Bootstrap nuisance boundaries are judged from scale-invariant
  mean-diagonal proportions, so changing a matrix's units does not change the
  refusal. The fixed-instrument analytic campaign failed on the exact
  1,909-person, four-component target at 75% expected censoring, so its analytic
  p-value must not be reported for that analysis without a design-specific
  simulated null. This is not a universal censoring threshold and does not
  withdraw the generally available, explicitly asymptotic test. The optional
  999-draw bootstrap target also completed but failed because eleven outer
  attempts could not complete every inner fit, so the bootstrap remains
  experimental. With one component the fit remains bit-identical to
  `tobit_fit`, and its corrected fixed-instrument
  recovery, coverage and target checks pass. There is deliberately no
  `heritability` key: it would be the first component's coefficient whatever
  that component is, and it moves under a rescaling that the heritability does
  not.
- **`interval(..., quantity="mean_diagonal_proportion")` gives the interval to
  report** when
  components are compared, beside the coefficient interval that is not
  comparable across matrices whose diagonals differ. Rescaling a component's
  matrix by four describes the same model and moves its coefficient interval;
  the proportion's does not move. Where every component carries a unit diagonal
  the two are the same interval.
- **The published 0.1.1 censored interface and numerical path remain
  unchanged, and the corrected development checks now pass.** Earlier recovery,
  coverage and target-design generators chose their censoring limits from each
  realised response. Their records remain historical. The replacement
  fixed-instrument MCMCglmm comparison, recovery, coverage and target checks all
  pass, including null rejection of 0.040 and 0.045 at 52% and 75% expected
  censoring. The changed checkout still needs its final clean-wheel release run;
  these measurements do not alter the immutable 0.1.1 artefacts.
- **`grouping_matrix` builds a component from what rows share.** One where two
  rows share a group, nought where they do not, one on the diagonal. Pass
  household identifiers and it is a household matrix; pass the person each row
  belongs to and it is the person-level matrix, which is the listener kernel
  where a listener contributes two ears. The relation is sharing, and a home is
  only one thing to share. `None` is a group nobody knows: that row shares with
  nobody and keeps its diagonal, so its effect cannot be told apart from its
  residual.
- **Every component of a censored fit can be reported, not only adjusted for.**
  `interval(..., component=index)` profiles any component, and the fit record
  carries `mean_diagonal_proportions` beside the raw coefficients -- the former
  is the quantity to compare across matrices whose diagonals differ. It is
  absent, rather than NaN, where a component's mean diagonal is not positive
  and finite. A several-component analytic p-value is labelled asymptotic,
  rather than described as finite-sample exact, and refused when a nuisance
  variance is on a bound. Target measurements record known limitations without
  creating a censoring threshold. The optional experimental bootstrap remains
  available for a design-specific Monte Carlo reference. The interval's boundary verdict is still filled at
  one component only, that being the only case for which the compatibility
  field exists.
- **A censored trait with several variance components is in the current 0.2
  scope.** `one_trait_censored_components` supports fitting, intervals and the
  explicitly labelled asymptotic test; this unreleased development entry is not
  itself a release claim. The
  independent censored-likelihood comparison remains a conditional numerical
  comparison: on its retained fixtures, the fitted proportions agree to
  1.4e-03 at an expected quarter censored and 2.9e-03 at an expected half,
  with limits fixed before outcomes were drawn, while the wider
  log-likelihood difference measures the sequential-truncation approximation.
  The historical component-recovery, interval-coverage and target-design
  campaigns used outcome-dependent censoring limits and are retired as
  qualification evidence. Their fixed-instrument replacements have now run:
  component recovery and interval coverage pass. The analytic p-value is a
  known failure on the exact 1,909-person, four-component target at 75% expected
  censoring and requires a design-specific simulated null there. The optional
  exhaustive-bootstrap target failed its declared rule, so the bootstrap remains
  experimental. This is not a universal censoring threshold.

  Three things about it are worth knowing before use. The upper end of a
  component proportion cannot be reached when a person contributes more than one
  record, because a proportion of one leaves that person's records perfectly
  correlated and the covariance singular; the interval covers that end, so what
  it states is a lower bound and its nominal coverage target is 0.975. The
  historical analytic rejection rates of 0.055 and 0.100 came from the retired
  generator and do not qualify a reference distribution; the empirical share
  of likelihood-ratio statistics on the bound is diagnostic, not a pass rule.
  The largest censored region compared with a GHK reference remains 600 rows in
  one block, which is the largest rung climbed rather than a limit that was
  found. [ADR
  0022](docs/adr/0022-the-censored-model-generalises-in-place.md) records why
  the censored model took several components in place rather than through a
  second model beside it, and that ADR 0001 decision 16 stands.

- **Commit `e4bd51f` was an investigation-era search change, not part of the
  original 0.1.1 code and neither the cause nor the fix.** It gave a held
  several-component fit three starting points instead of one. Against its
  parent on the same 40 affected target seeds, bound membership was 12/40 in
  both versions, the operational p-value-one atom was 12/40 in both, and
  rejections were 7/40 in both. Coefficients and alternative log likelihoods
  were bit-identical; the largest null-log-likelihood change was 1.847e-09 and
  the largest p-value change 5.7e-10. Exact zero statistics changed from 3 to 6
  without changing the operational atom. One-component fits are untouched by
  construction. The change is retained as search hygiene only.

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
