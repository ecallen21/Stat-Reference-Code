# GLM Diagnostics (Reference §7.40, §7.41, §7.55)

Three tools that together cover most of what you need to check a GLM fit.

## 1. Pearson and deviance residuals

| Residual | Formula | Use |
|----------|---------|-----|
| Pearson | `r_P = (y − μ) / √V(μ)` | Direct standardization of the raw residual; large values flag points where the variance assumption is violated |
| Deviance | `r_D = sign(y − μ) · √dᵢ`, `dᵢ` = unit deviance contribution | Squared deviance residuals sum to the **deviance**; the natural "log-likelihood-distance" residual |

Both should be roughly N(0, 1) if the model is correct (for *continuous-data* GLMs — see RQR below for discrete).

## 2. Hosmer–Lemeshow goodness-of-fit (logistic specifically)

1. Sort observations by fitted probability `p̂`.
2. Bin into `g = 10` deciles.
3. For each decile compute observed counts of `y = 0/1` and expected counts (sum of `p̂` / `1 − p̂` in the bin).
4. `χ² = Σ_{bins} (obs − exp)² / exp` ~ `χ²_{g−2}` under H₀ (model fits well).

**Use with caution.** For large `n`, the test rejects with trivial misspecification. For small `n`, the binning depends on tied probabilities (try different `g`). Pair with calibration plots — they're more informative than the single p-value.

## 3. Randomized Quantile Residuals (Dunn–Smyth 1996)

Standard Pearson / deviance residuals for **discrete** outcomes (binary, count) have a step structure that makes residual plots a mess of horizontal bands. RQRs fix this:

`uᵢ = uniform(F(yᵢ − 1), F(yᵢ))`,  `rᵢ = Φ⁻¹(uᵢ)`

The random "jitter" inside the discrete-CDF step is what gives RQRs a continuous distribution. Under a correct model, `rᵢ ~ N(0, 1)` exactly — **any** pattern in their Q-Q plot, residuals-vs-fitted plot, or normality test signals model misspecification (wrong link, overdispersion, zero-inflation, omitted nonlinearity, …).

This is what the popular **DHARMa** R package automates via simulation; the closed-form RQR formula here is the original Dunn–Smyth construction.

## Files
- `python/glm_diagnostics.py` — Pearson + deviance residuals (logistic & Poisson examples), Hosmer–Lemeshow with the table per decile, RQR for Poisson / binomial / negative-binomial.
- `r/glm_diagnostics.R` — same from-scratch versions; notes on `ResourceSelection::hoslem.test` and `DHARMa::simulateResiduals` (the canonical R workflow).
- PySpark: N/A — residual analysis on the full sample.

## Run
```
python techniques/glm-diagnostics/python/glm_diagnostics.py
Rscript techniques/glm-diagnostics/r/glm_diagnostics.R
```

**Refs:** Hosmer & Lemeshow, "A Goodness-of-Fit Test for the Multiple Logistic Regression Model," *Communications in Statistics* A9, 1043–1069, 1980; Dunn & Smyth, "Randomized Quantile Residuals," *J. Computational and Graphical Statistics* 5(3), 236–244, 1996; Hartig, `DHARMa`: Residual Diagnostics for Hierarchical (Multi-Level / Mixed) Regression Models, R package documentation.

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
