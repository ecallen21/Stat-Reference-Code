# Bayesian Model Comparison (Reference §14.20, §14.21, §14.22)

Compare Bayesian models by their **out-of-sample predictive performance** — specifically the expected log pointwise predictive density (`elpd`). All three criteria below estimate `elpd` from posterior draws.

## DIC (Spiegelhalter et al. 2002)

```
DIC = −2 log p(y | θ̄) + 2 p_D
p_D = 2 (log p(y | θ̄) − E_post[log p(y | θ)])
```

Simple and popular in the mid-2000s but sensitive to parameterization; largely deprecated in favor of WAIC / LOO.

## WAIC (Watanabe 2010; Gelman, Hwang, Vehtari 2014)

```
lpd       = Σ_i log E_post[p(y_i | θ)]
p_WAIC    = Σ_i Var_post[log p(y_i | θ)]
elpd_WAIC = lpd − p_WAIC
WAIC      = −2 elpd_WAIC
```

Fully Bayesian, invariant to reparameterization, but requires per-observation log-likelihood evaluations.

## PSIS-LOO (Vehtari, Gelman, Gabry 2017)

Approximates leave-one-out cross-validation using importance weights `r_i = 1 / p(y_i | θ_s)`, then smooths the top tail by fitting a generalized Pareto. Reports **Pareto-k** diagnostics per observation — `k > 0.7` warns that the LOO estimate is unreliable and that observation should be exactly refit.

```
elpd_LOO = Σ_i log E_LOO[p(y_i | θ)]
LOO      = −2 elpd_LOO
```

**Standard recommendation**: PSIS-LOO first; fall back to WAIC when LOO Pareto-k's blow up on many observations.

## Bayes factors

```
B_12 = p(y | M_1) / p(y | M_2)          ratio of marginal likelihoods
```

Extremely sensitive to prior width — a `Normal(0, 10²)` vs `Normal(0, 100²)` prior on the same slope can shift the log-Bayes-factor by many units. Undefined for improper priors. Prefer WAIC / LOO for **predictive** comparison; use Bayes factors only when priors are seriously informative and comparison of **hypotheses** (not predictions) is the goal.

## Files

- `python/bayesian_model_comparison.py` — from-scratch WAIC, PSIS-LOO (with simplified Pareto tail fit), and DIC on an `S × N` log-likelihood matrix. Demo: correct model (intercept + slope) beats the intercept-only alternative by ΔWAIC ≈ −75, ΔLOO ≈ −77 on n = 200.
- `r/bayesian_model_comparison.R` — `loo::waic`, `loo::loo`, `loo::loo_compare` for standard production usage.

## When to use

- Comparing two or more Bayesian models fit to the same observations.
- Reporting predictive gap between a full and a nested model.
- Diagnosing influential observations (Pareto-k in LOO).

## Assumptions & caveats

- All models must be fit on **exactly the same** observations.
- Log-likelihood matrix must be **pointwise** — one column per observation, one row per posterior draw.
- Differences (Δelpd) matter more than absolute values; a rule of thumb is that Δelpd ≥ 4 SE is a meaningful preference.

## Run

```
python techniques/bayesian-model-comparison/python/bayesian_model_comparison.py
Rscript techniques/bayesian-model-comparison/r/bayesian_model_comparison.R
```

**Refs:** Spiegelhalter, D.J. et al. "Bayesian measures of model complexity and fit." *J. R. Stat. Soc. B* 64(4), 583–639, 2002; Watanabe, S. "Asymptotic equivalence of Bayes cross validation and widely applicable information criterion." *JMLR* 11, 3571–3594, 2010; Gelman, A., Hwang, J. & Vehtari, A. "Understanding predictive information criteria for Bayesian models." *Stat. Comput.* 24(6), 997–1016, 2014; Vehtari, A., Gelman, A. & Gabry, J. "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." *Stat. Comput.* 27(5), 1413–1432, 2017.

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
