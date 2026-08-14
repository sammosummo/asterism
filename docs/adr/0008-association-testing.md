# 8. Association testing in a polygenic model

**Extends ADR 0001 under decision 20. Amends nothing.**

## The analysis this names

Decision 20 requires the amendment to name the analysis before the code exists.
The shape is settled and the trait is not: **a phenotype regressed on each of
many markers in turn, with ancestry components as covariates and a polygenic
term absorbing relatedness and structure.**

```text
y = X0 b + m_j c + g + e,    V = s2 (h K + (1 - h) I)
```

`X0` carries the intercept, the ancestry components and any other covariate;
`m_j` is one marker; `c` is what is tested.

**The specific trait and marker set are to be filled in here before this is run
on anything real.** What is available to fill them with:

| markers | count | subjects |
| --- | --- | --- |
| array genotypes, `chp-genotype-subject` | 1,146,843 | 2,620 |
| whole-genome sequence, `wgs_vendor-genotype-subject` | 51,258,090 | 2,621 |
| methylation probes | 485,578 | 859 |
| expression transcripts | 22,414 | 1,241 |

The genotypes and sequence live on the bulk drive at
`Studies/Existing/SAFS/Data/Genotypes`, with array dosages beside them and two
recent WGS freezes under `WGSReacquisition`. **They are not reachable from the
workspace**: `bulk` links to the Emma Knowles drop and the pedigree evidence and
to nothing else, so a link is needed before any of this can be run. Genotypes
are restricted material and belong behind a `bulk` symlink rather than copied
in, like everything else of that kind.

Correcting an earlier version of this document, which said there was no genotype
data anywhere in the workspace: there is 9 GB of it one symlink away, and the
search that concluded otherwise looked inside `bulk` without following what
`bulk` points at.

At the measured cost the array scan is about seven minutes and the full sequence
five to seven hours with the variance components held. Refitted, the sequence is
not a run anybody would start. Whatever filtering by frequency precedes a scan
cuts both figures substantially and belongs to whatever prepares the markers.

## Why this is not the one-trait model in a loop

It could be, and for a handful of markers it should be. A prepared model already
returns every fixed effect with a standard error, a z and a two-sided p-value,
so putting a marker in the design and reading the last column has always been an
association test.

What fails at scale is the direction of reuse. A prepared model decomposes once
and refits cheaply **for a new response**; here the response never changes while
the design changes half a million times. Re-preparing per marker is 0.018 s at
n = 1,400, which is five hours for a million markers and days at larger samples.

So the variance components are fitted once under the null design and held, and
each marker becomes a weighted least squares on the rotated data. Measured on
this machine with ten ancestry components: 0.26 ms per marker at n = 1,400 and
0.36 ms at n = 1,910, so 485,578 probes take about two to three minutes and a
million markers under six.

The rotation is per family block rather than over the whole matrix, which is
what keeps it there: a pedigree relationship matrix is block diagonal, so
rotating a marker costs the sum of squared block sizes rather than the square of
the sample size.

## Holding the variance components is an assumption

It is what EMMAX and its descendants do, and it is a good approximation when no
single marker explains much of the variance — the situation a scan is in, and
exactly not the situation for a marker of large effect. Refitting under every
marker is available and is around a hundred times slower.

## Wald and the likelihood ratio, and what choosing between them means

Both were asked for and both are returned. **With the variance components held
they are algebraically identical**: the profile log likelihood in the fixed
effects is exactly quadratic when the covariance is known, so the likelihood
ratio is the square of the Wald statistic. Offering them as alternatives there
would be printing one answer twice, and a test asserts the identity rather than
a comment claiming it.

They separate only when the variance components are refitted under each marker.
So the real choice is not Wald against likelihood ratio; it is held against
refitted, and the second is the expensive one.

## What this deliberately does not do

**The marker under test is inside `K`.** Leaving out the chromosome it sits on —
one decomposition per chromosome — is the usual answer, and it is not done here:
Sam judged it too expensive for the analysis this was built for. The consequence
is a test biased towards the null, more so for a marker contributing materially
to `K`, and it is worst exactly where a scan cares most. `AssociationModel`
reports `leave_one_chromosome_out` as false so the answer sits in the record
rather than in somebody's memory.

Also out: multiple testing, which is the caller's; any marker file format, since
Asterism reads no files; and imputation quality, missingness and allele coding,
all of which belong to whatever prepares the marker matrix.

## Consequences

- `src/association.rs`, self-contained. `prepared.rs` is untouched, because it
  is the most used and most validated path in the package and a faster scan is
  not worth destabilising it.
- **The genomic inflation is one under a realistic null**, which is the check
  that matters for a scan. On the real pedigree, the real ancestry components
  and 20,000 real markers, with the response drawn from the fitted polygenic
  model and no marker effect anywhere, lambda came back at 0.991. The same scan
  on real height gives 1.168, and the null draw is what says that difference is
  the polygenicity of height rather than uncontrolled structure — the two are
  otherwise indistinguishable from the scan alone.
- Six tests also establish that an effect is recovered, that an absent one is
  not, that the two statistics coincide when held, and that markers correlated
  with family membership and nothing else are not called associations.
- Still not a measured type I error rate across the tail, which is what a
  genome-wide claim would need. A median-based lambda says nothing about
  behaviour at 5e-08.
- No comparison against an independent implementation. Under ADR 0006 that means
  it is not qualified, the same standing the liability model had this morning
  before SOLAR was run against it.
