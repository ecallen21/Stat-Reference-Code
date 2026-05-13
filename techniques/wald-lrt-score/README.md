# Wald, Likelihood-Ratio, and Score (Rao) Tests (Reference §3.18, §3.30, §3.31, §3.33)

The three classical likelihood-based tests of `H₀: θ = θ₀` in a parametric model with log-likelihood `ℓ(θ)`. All three are **asymptotically equivalent** under H₀ (each converges to χ²_p), but they differ in what you have to compute and in finite-sample behavior.

## The three statistics

Let `θ̂` = MLE, `U(θ) = ∂ℓ/∂θ` = score function, `I(θ) = −E[∂²ℓ/∂θ²]` = Fisher information.

| Test | Statistic (1-D) | Evaluated at |
|------|-----------------|--------------|
| **Wald** | `W = (θ̂ − θ₀)² · I(θ̂)` | the MLE |
| **Likelihood ratio** | `−2 log Λ = 2 (ℓ(θ̂) − ℓ(θ₀))` | both |
| **Score (Rao)** | `S = U(θ₀)² / I(θ₀)` | the null only |

All are `~ χ²_p` under H₀ (p = number of restrictions). The multivariate forms swap the squares for quadratic forms in `I` (or its inverse).

## When each is convenient

- **Wald** — you already have the MLE and its SE; one quick formula. The default in regression output (`coef/se` z-scores are Wald). Watch out: **Wald is not invariant to reparameterization** and behaves poorly near boundaries or with skewed likelihoods.
- **LRT** — needs *both* the constrained and unconstrained fits. Invariant to reparameterization and usually the most accurate for moderate `n`. Standard tool for nested-model comparison (`anova(full, null, test = "LRT")` in R).
- **Score** — needs only the **null** fit. Useful when fitting under H₁ is hard (e.g. testing whether a variance component is zero — fitting that null is easy, fitting the full mixed model is hard). The classical Lagrange-multiplier test in econometrics.

## Boundary case

If `θ̂` lands at a boundary (e.g. `p̂ = 0` in a binomial), Wald's `I(θ̂)` blows up and `W → ∞`; the LRT and score remain well-defined. This is one of the reasons people prefer the LRT for "exact" small-sample inference. See the demo for a `0/20` binomial.

## Worked example: binomial proportion

For `Binomial(n, p)` with `x` successes, testing `H₀: p = p₀`:
- `ℓ(p) = x log p + (n − x) log(1 − p)` (with `0 log 0 = 0`)
- `U(p) = x/p − (n − x)/(1 − p)`
- `I(p) = n / (p(1 − p))`

The demo prints the three p-values for `x = 60 / n = 100 / p₀ = 0.5` (all three agree closely) and for the boundary case `x = 0 / n = 20 / p₀ = 0.1` (Wald breaks, LRT and score are reasonable).

## Files
- `python/wald_lrt_score.py` — generic 1-D implementations of all three tests + binomial worked example; cross-checks the score-form against `statsmodels.stats.proportion.proportions_ztest(prop_var = p0)`.
- `r/wald_lrt_score.R` — same generics + binomial example; notes on `lmtest::lrtest` and the standard GLM idiom `anova(fit_full, fit_null, test = "LRT")`.
- PySpark: N/A — purely model-theoretic.

## Run
```
python techniques/wald-lrt-score/python/wald_lrt_score.py
Rscript techniques/wald-lrt-score/r/wald_lrt_score.R
```

**Refs:** Wald, "Tests of Statistical Hypotheses Concerning Several Parameters When the Number of Observations is Large," *Trans. AMS* 54(3), 426–482, 1943; Neyman & Pearson, "On the Use and Interpretation of Certain Test Criteria for Purposes of Statistical Inference," *Biometrika* 20A, 175–240 & 263–294, 1928; Rao, "Large-Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation," *Math. Proc. Cambridge Philos. Soc.* 44(1), 50–57, 1948.
