# Overdispersion Tests for Count GLMs (Reference §7.35, §7.42, §7.54)

Three complementary diagnostics for "is Var(Y) really equal to E[Y] (Poisson assumption)?"

## 1. Pearson dispersion estimate

`φ̂ = χ²_P / (n − p)`,   `χ²_P = Σ (yᵢ − μ̂ᵢ)² / V(μ̂ᵢ)`

| `φ̂` | Interpretation |
|------|----------------|
| `≈ 1` | Poisson assumption looks OK |
| > ~1.5 | Likely overdispersion — investigate |
| ≫ 2 | Strong overdispersion — Poisson SEs definitely wrong |

A quick first look. Some authors use a chi-square approximate test on `χ²_P` against `χ²_{n−p}`, but on large `n` this trivially rejects.

## 2. Cameron–Trivedi score test (the formal version)

Test H₀: `Var(Y) = μ` vs. H₁: `Var(Y) = μ + α · g(μ)`.

| Form | `g(μ)` | Alternative |
|------|--------|-------------|
| **NB1** | `1` | Linear variance: `Var = μ + α` |
| **NB2** | `μ` | Quadratic variance: `Var = μ + α·μ²` (the standard NB) |

The score statistic is asymptotically standard normal under H₀. **NB2 is the default** — it's the alternative against the standard negative-binomial parameterization. A positive significant `T` means overdispersion; a negative one (rare) means underdispersion.

## 3. Likelihood-ratio Poisson vs. NB

Fit both models, compare `LRT = 2 (ll_NB − ll_Poisson)`.

The wrinkle: NB nests Poisson at `θ → ∞` (or equivalently `α = 0`), which is a **boundary** point. The standard `χ²_1` p-value is conservative by a factor of 2; the correct reference distribution is a 50:50 mixture of `χ²_0` and `χ²_1`, which means **halving the standard p-value**. The Python implementation uses the boundary-corrected form.

## Workflow

1. Fit Poisson; check `φ̂` from the Pearson residuals.
2. If `φ̂ > 1.5` or you suspect overdispersion a priori, run the **score test** (`NB2`) for a formal check.
3. Fit NB and compare via boundary-adjusted LRT.
4. If NB still looks bad (excess zeros), consider zero-inflated / hurdle models (§7.14–7.19, deferred).

## Files
- `python/overdispersion_tests.py` — all three diagnostics from scratch; demo on clean Poisson (passes all checks) and NB-generated data (all three flag overdispersion strongly).
- `r/overdispersion_tests.R` — from-scratch + notes on `AER::dispersiontest` and `performance::check_overdispersion`.
- PySpark: N/A.

## Run
```
python techniques/overdispersion-tests/python/overdispersion_tests.py
Rscript techniques/overdispersion-tests/r/overdispersion_tests.R
```

**Refs:** Cameron & Trivedi, "Regression-Based Tests for Overdispersion in the Poisson Model," *J. Econometrics* 46(3), 347–364, 1990; Cameron & Trivedi, *Regression Analysis of Count Data*, 2nd ed., 2013.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
