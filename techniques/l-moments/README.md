# L-Moments & Probability-Weighted Moments (Reference §1.24)

**L-moments** are linear combinations of order statistics that play the role of conventional moments but are far more robust and **always exist** — even for heavy-tailed distributions with infinite variance. Standard tools in hydrology and extreme-value analysis (flood return periods), and well-suited to small samples and distribution selection (L-moment ratio diagrams).

## Probability-weighted moments (PWMs)
Unbiased sample estimator (Landwehr et al.), with `x₍₁₎ ≤ … ≤ x₍ₙ₎`:
`b_r = (1/n) · Σᵢ [ C(i−1, r) / C(n−1, r) ] · x₍ᵢ₎`, so `b₀ = mean`.

## L-moments from PWMs
`L1 = b₀` · `L2 = 2b₁ − b₀` · `L3 = 6b₂ − 6b₁ + b₀` · `L4 = 20b₃ − 30b₂ + 12b₁ − b₀`

## L-moment ratios
| Ratio | Definition | Analogue of | Range |
|-------|-----------|-------------|-------|
| L-CV (`τ`) | `L2 / L1` | coefficient of variation | (for positive data) |
| L-skewness (`τ₃`) | `L3 / L2` | skewness | (−1, 1) |
| L-kurtosis (`τ₄`) | `L4 / L2` | kurtosis | roughly (−¼, 1) |

These are much less sensitive to a single outlier than the conventional moment-based `g1`/`g2` (compare with `techniques/shape-skewness-kurtosis`). They also **uniquely characterize** a distribution and are used to fit GEV / GPD / Wakeby etc. via `pel*` functions in R's `lmom`.

## Files
- `python/l_moments.py` — from-scratch unbiased PWMs and L1–L4 plus the ratios; verifies against `lmoments3.lmom_ratios` when installed (`pip install lmoments3`)
- `r/l_moments.R` — from-scratch + `lmom::samlmu`; notes on `lmomco::lmoms`, `Lmoments`
- PySpark: N/A (an order-statistic computation on a single sample; not a distributed-compute use case)

## Run
```
python techniques/l-moments/python/l_moments.py
Rscript techniques/l-moments/r/l_moments.R
```

**Refs:** Hosking, "L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics," *JRSS-B* 52(1), 105–124, 1990; Hosking & Wallis, *Regional Frequency Analysis*, Cambridge University Press, 1997.
