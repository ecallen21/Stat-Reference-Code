# Bayesian Model Averaging (Reference §14.26)

Instead of picking one "best" model and ignoring the uncertainty about which model is right, **BMA** weights predictions and parameter estimates by each model's posterior probability.

```
E[Δ | y]  = Σ_k Pr(M_k | y) · E[Δ | y, M_k]
Var[Δ|y]  = Σ_k Pr(M_k | y) · Var[Δ | y, M_k]                    (within-model uncertainty)
          + Σ_k Pr(M_k | y) (E[Δ | y, M_k] − E[Δ | y])²          (between-model uncertainty)
```

## Model posterior

```
Pr(M_k | y) = p(y | M_k) · Pr(M_k) / Σ_j p(y | M_j) · Pr(M_j)
```

The marginal likelihood `p(y | M_k)` integrates out parameters — often expensive. **BIC approximation** (Schwarz 1978):

```
log p(y | M_k) ≈ log p(y | θ̂_k, M_k) − (d_k / 2) log n

Pr(M_k | y) ∝ exp(−BIC_k / 2) · Pr(M_k)
```

## Posterior inclusion probability (PIP)

```
PIP_j = Σ_{k : j ∈ M_k}  Pr(M_k | y)
```

Interpretation: `PIP_j > 0.5` — variable `j` is included in models with majority posterior mass; `PIP_j > 0.9` — very strong evidence for inclusion.

## Files

- `python/bayesian_model_averaging.py` — enumerate all `2^p` subsets, fit OLS, BIC-approx BMA weights, and PIP per variable. Demo (n = 100, p = 5, true non-zeros x1, x2, x4): PIPs = (1.00, 1.00, 0.10, 1.00, 0.10); top model (weight 0.81) is exactly `x1 + x2 + x4`.
- `r/bayesian_model_averaging.R` — `BMA::bicreg` (BIC) or `BAS::bas.lm` (full marginal likelihood; supports Zellner g-prior and Zellner-Siow priors).

## When to use

- **Small-to-moderate p** where enumeration or MCMC over models is tractable (`p ≤ 20` roughly for full enumeration).
- **Prediction with honest uncertainty** — BMA intervals include between-model variance, which single-model intervals miss.
- **Variable-importance quantification** via PIP — an alternative to LASSO-style hard selection.
- **Sensitivity analysis** — how much does the coefficient of interest change across the credible model set?

## When NOT to use

- **Large p** — number of subsets is `2^p`; use MCMC-over-models (`BAS::bas.lm(method = "MCMC")`) or a spike-and-slab prior.
- **When one model dominates** — the top model's weight > 0.95, BMA and single-model inference agree; use the top model.
- **Priors matter a lot** here: a uniform prior over `2^p` models puts most mass on medium-size models, which may not be substantively justified. Use a Beta-Binomial prior on model size for a more neutral default.

## Assumptions & caveats

- **BIC assumes** regular MLE conditions; poor for boundary problems, mixture models, or with informative priors on individual coefficients.
- **Same y across models** — BMA is undefined when different candidate models have different outcome scales.
- **Interpretability**: BMA coefficients don't come from a single model, so they're not directly interpretable as effects; use them for prediction and PIP for selection.

## Run

```
python techniques/bayesian-model-averaging/python/bayesian_model_averaging.py
Rscript techniques/bayesian-model-averaging/r/bayesian_model_averaging.R
```

**Refs:** Raftery, A.E. "Bayesian model selection in social research." *Sociol. Methodol.* 25, 111–163, 1995; Hoeting, J.A. et al. "Bayesian model averaging: a tutorial." *Stat. Sci.* 14(4), 382–401, 1999; Clyde, M.A. & George, E.I. "Model uncertainty." *Stat. Sci.* 19(1), 81–94, 2004.

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
