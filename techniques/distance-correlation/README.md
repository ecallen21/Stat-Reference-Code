# Distance Correlation (Reference §4.8)

Székely, Rizzo & Bakirov (2007). Detects **any** dependence between two variables — linear, monotone, or wildly nonlinear. Unlike Pearson / Spearman / Kendall:

**`dCor(x, y) = 0` if and only if `x` and `y` are independent.**

(For the others, zero correlation only rules out a specific *form* of dependence.)

## Algorithm — `O(n²)`

1. Pairwise distance matrices `a_{ij} = |xᵢ − xⱼ|`, `b_{ij} = |yᵢ − yⱼ|`.
2. **Double-center** each: subtract row mean, column mean, add grand mean.
   `A = a − a_{i·} − a_{·j} + a_{··}`, similarly `B`.
3. Distance covariance squared: `dCov²(x, y) = mean(A · B)` (element-wise product).
4. Distance variances: `dVar²(x) = mean(A · A)`, `dVar²(y) = mean(B · B)`.
5. **Distance correlation**: `dCor = √(dCov² / √(dVar²(x) · dVar²(y)))`.

`dCor ∈ [0, 1]`. For *scalar* `x, y`, `dCor = 1` only when `y` is a linear function of `x`. The definition extends naturally to vectors of arbitrary dimension (just use Euclidean distance in step 1).

## Significance

The asymptotic distribution of `n · dCov²` under independence has a known form (a weighted sum of independent χ²₁), but is awkward to invert. The standard test is a **permutation test**: shuffle `y`, recompute `dCor`, compare. p = fraction of permuted statistics `≥` observed.

## Files
- `python/distance_correlation.py` — from-scratch double-centered distance matrices + permutation p-value; demos two cases: independent `N(0,1)` (p large) and `y = x² + noise` (Pearson `r ≈ 0`, `dCor ≈ 0.5`, `p ≈ 0`).
- `r/distance_correlation.R` — from-scratch + `energy::dcor` / `energy::dcor.test`.
- PySpark: N/A — `O(n²)` pairwise distance matrices are infeasible at Spark scale without batching. Sub-sample the data to ~10⁴ rows and run the Python version.

## Run
```
python techniques/distance-correlation/python/distance_correlation.py
Rscript techniques/distance-correlation/r/distance_correlation.R
```

**Refs:** Székely, Rizzo & Bakirov, "Measuring and Testing Dependence by Correlation of Distances," *Annals of Statistics* 35(6), 2769–2794, 2007; Székely & Rizzo, "Brownian Distance Covariance," *Annals of Applied Statistics* 3(4), 1236–1265, 2009.
