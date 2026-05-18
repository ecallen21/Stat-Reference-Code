# Kolmogorov–Smirnov Test (Reference §6.7)

Distribution-equality tests built on the **sup norm** of empirical CDF differences.

## One-sample

`H₀: F = F₀` for a **fully specified** continuous CDF `F₀`.

`Dₙ = supₓ |F_n(x) − F₀(x)|`

Under H₀, `√n · Dₙ` converges to the Kolmogorov distribution (`scipy.stats.kstwobign`).

## Two-sample

`H₀: F_X = F_Y`.

`D_{m,n} = supₓ |F̂_X(x) − F̂_Y(x)|`

Under H₀, `√(mn/(m+n)) · D` ~ Kolmogorov.

## Properties

- **Distribution-free** under H₀ (continuous data; ties hurt power).
- **Consistent** against any difference in distribution.
- **Lower power than tests targeting a specific alternative** — Mann–Whitney beats K–S for pure location shift; the chi-square goodness-of-fit can be more powerful for binned categorical-like data.
- Sensitive to differences in the **middle** of the distribution (where the ECDF is steep). Anderson–Darling weights the tails more.

## Critical caveat: fitted parameters

If you compute `F₀` using parameters estimated from the **same** data (e.g. "is x normal with sample mean and SD?"), the standard K–S p-value is **invalid** — it's far too conservative. Use **Lilliefors** for that case (`techniques/normality-tests`), which has its own critical values for fitted-parameter K–S.

## Files
- `python/kolmogorov_smirnov.py` — from-scratch one-sample (against any callable CDF) and two-sample K–S with the asymptotic Kolmogorov p-value; matches `scipy.stats.kstest` / `ks_2samp` to a few significant figures (scipy uses an exact small-sample distribution).
- `r/kolmogorov_smirnov.R` — from-scratch + base `ks.test`.
- `pyspark/kolmogorov_smirnov.py` — one-sample via `pyspark.mllib.stat.Statistics.kolmogorovSmirnovTest` (only supports normal as of current MLlib); two-sample via grid-evaluated ECDFs.

## Run
```
python techniques/kolmogorov-smirnov/python/kolmogorov_smirnov.py
Rscript techniques/kolmogorov-smirnov/r/kolmogorov_smirnov.R
python techniques/kolmogorov-smirnov/pyspark/kolmogorov_smirnov.py
```

**Refs:** Kolmogorov, "Sulla determinazione empirica di una legge di distribuzione," *Giorn. Inst. Ital. Attuari* 4, 83–91, 1933; Smirnov, "Table for Estimating the Goodness of Fit of Empirical Distributions," *Ann. Math. Stat.* 19(2), 279–281, 1948.
