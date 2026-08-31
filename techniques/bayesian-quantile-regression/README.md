# Bayesian Quantile Regression (Reference §33.2)

Yu & Moyeed (2001) showed that the **asymmetric Laplace** likelihood

```
p(y | μ, σ, τ) = (τ (1 − τ) / σ) · exp( − ρ_τ((y − μ) / σ) )
```

turns MLE into ordinary quantile regression at level `τ` (`ρ_τ` = check
loss). Combined with a Gaussian prior on the coefficient vector, this
yields a proper posterior sampled by MCMC.

## Advantages over frequentist QR

- **Full posterior** — credible intervals, quantile-of-quantile
  reporting.
- **Priors + regularisation + hierarchical structure** trivial to add.
- **Small n** handled without asymptotic sandwich SEs.

## When to use

- **Any QR problem where n is modest** and you want credible intervals.
- **Hierarchical / grouped quantile regression** across subgroups.
- **Priors from a previous study** — pooled Bayesian analysis.

## When NOT to use

- **Enormous n** — frequentist QR is faster; asymptotic CIs match.
- **Non-symmetric residual concerns** — the ALD is a working
  likelihood; extreme residual shapes may bias the posterior.

## Files

- `python/bayesian_quantile_regression.py` — from-scratch asymmetric-
  Laplace log-likelihood + Metropolis sampler for `(β, log σ)`.
  Demo on heteroscedastic data at `τ ∈ {0.10, 0.50, 0.90}` — slopes
  ~0.33-0.41 (true 0.5), intercepts spread −0.29 → 0.94 → 2.34
  reflecting the growing conditional scale.
- `r/bayesian_quantile_regression.R` — `bayesQR` / `brms(family="asym_laplace")`;
  `pymc` / `numpyro` (Python).

## Assumptions & caveats

- **ALD is a working likelihood** — the resulting posterior is a
  *quantile-based pseudo-Bayesian* object, not literally the posterior
  over the population quantile if residuals are truly non-ALD.
- **Scale parameter `σ`** — must be given a prior (log-normal here).
- **Metropolis step size** — tune for acceptance around 40-60 %;
  the demo hits 60-70 % which is on the accept-too-often side but
  demonstrates the shape.
- **Multiple quantiles** — fit each `τ` independently; posterior
  monotonicity is not enforced (see `additive-quantile-regression`).
- **Cross-quantile inference** requires a joint model (Chen 2009).

## Related in this repo

- `quantile-regression` — the frequentist parent.
- `additive-quantile-regression` — flexible non-linear extension.
- `expectile-regression` — asymmetric squared-loss sibling.
- `censored-quantile-regression` — extension for censored outcomes.
- `bayesian-linear-regression`, `bayesian-glms`, `hmc-nuts` — the
  Bayesian machinery.

## Run

```
python techniques/bayesian-quantile-regression/python/bayesian_quantile_regression.py
Rscript techniques/bayesian-quantile-regression/r/bayesian_quantile_regression.R
```

**Refs:** Yu, K. & Moyeed, R. "Bayesian quantile regression." *Statistics and Probability Letters*, 2001; Koenker, R. *Quantile Regression*, Cambridge University Press, 2005; Chen, C. "Quantile regression modelling for longitudinal data." *Biometrics*, 2009.

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
