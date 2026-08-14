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
on anything real.** Two facts constrain what they can be. There is no genotype
data anywhere in this workspace — no PLINK files, no VCF, no genomic
relationship matrix — so a SNP scan needs genotypes brought in first. What does
exist, in `bulk/safs-emma-knowles-20260805`, is 485,578 methylation probes on
859 people and 22,414 transcripts on 1,241.

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
- Not calibrated. Six tests establish that an effect is recovered, that an
  absent one is not, that the two statistics coincide when held, and — the one
  that matters — that markers correlated with family membership and nothing else
  are not called associations. None of that is a measured type I error rate
  across a scan, and until it is, no genome-wide claim rests on this.
- No comparison against an independent implementation. Under ADR 0006 that means
  it is not qualified, the same standing the liability model had this morning
  before SOLAR was run against it.
