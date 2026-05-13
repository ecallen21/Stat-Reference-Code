# Binomial Test (Reference §3.22)

`H₀ : p = p₀` for `x` successes in `n` independent Bernoulli trials. Three variants here, in increasing order of approximation:

| Variant | How | When |
|---------|-----|------|
| **Exact binomial** | Sum the binomial PMF in the tail(s); two-sided uses the "method of small p-values" (sum over `k` with `P(K=k) ≤ P(K=x)`). Matches `scipy.stats.binomtest` and R's `binom.test`. | Small `n`; gold standard; always safe. |
| **Mid-p** | Exact tail minus `½·P(observed)`. Less conservative than exact (Lancaster 1961). | When the exact test feels over-conservative. |
| **Normal-approx z** | `z = (p̂ − p₀) / √(p₀(1−p₀)/n)` — same as the one-proportion z-test in `techniques/z-tests`. Optional Yates continuity correction. | Large `n` and `np₀(1−p₀) ≳ 10`; closes the gap to the exact test. |

For the **confidence interval** on `p`, use Wilson or Clopper–Pearson — see `techniques/rates-proportions`. The Clopper–Pearson CI *is* the inversion of the exact test.

## Why three?

- **Exact** is always correct in size, but discreteness makes it strictly conservative — the actual type I error rate is *less* than the nominal α.
- **Mid-p** trades guaranteed conservatism for actual type I error closer to α (on average); recommended by Berry & Armitage as a sensible default.
- **Normal-approx** is fast and analytical (you get a z and a CI on the difference), but coverage is poor near the boundaries (`p₀` close to 0 or 1) and for small `n`.

You'll see all three give slightly different p-values on small data — that's intrinsic to discrete tests, not a bug.

## Files
- `python/binomial_test.py` — from-scratch exact / mid-p / normal-approx (with optional Yates); compares against `scipy.stats.binomtest`.
- `r/binomial_test.R` — from-scratch versions + `stats::binom.test` (exact) and `stats::prop.test` (normal approx).
- PySpark: N/A — for very large `n`, the normal approximation is essentially perfect and the one-proportion z-test in `techniques/z-tests/pyspark/` covers the distributed case.

## Run
```
python techniques/binomial-test/python/binomial_test.py
Rscript techniques/binomial-test/r/binomial_test.R
```

**Refs:** Clopper & Pearson, "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial," *Biometrika* 26(4), 1934; Lancaster, "Significance Tests in Discrete Distributions," *JASA* 56(294), 223–234, 1961.
