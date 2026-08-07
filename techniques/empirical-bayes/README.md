# Empirical Bayes (Reference §14.17, §14.18)

**EB** plugs a **point estimate** of the hyperprior into an otherwise Bayesian analysis instead of putting a full hyperprior on the hyperparameters (which is fully-Bayes / hierarchical). Cheap, fast, and often gives essentially the same shrinkage as a full hierarchical fit.

## Beta – Binomial EB

Each group `j` has an observed rate `y_j / n_j`.

```
θ_j     ~ Beta(α, β)                     (hyperprior; α, β estimated)
y_j     ~ Binomial(n_j, θ_j)
```

Method-of-moments on the sample rates gives `α̂, β̂`. Plug in:

```
θ_j | y ~ Beta(α̂ + y_j, β̂ + n_j − y_j)
```

Groups with small `n_j` shrink strongly toward the overall rate; large-`n_j` groups barely move.

## James–Stein estimator

For `J ≥ 3` independent Normal estimates `y_j ~ N(θ_j, σ²)`, the shrinkage estimator

```
θ̂_j^JS = ȳ + (1 − (J − 3) σ² / Σ(y_j − ȳ)²) (y_j − ȳ)
```

**dominates** the raw `y_j` in total squared error — the classical Stein paradox. Equivalent to plug-in EB with a `Normal(μ, τ²)` hyperprior on the `θ_j`.

## Files

- `python/empirical_bayes.py` — method-of-moments EB Beta-Binomial and both scalar and heteroscedastic James-Stein variants. Demos: EB reduces RMSE from 0.061 to 0.039 on 20 batting-average-style rates (35% reduction); James-Stein reduces MSE from 1.37 to 1.07 (22% reduction) on J = 10 Normal estimates.
- `r/empirical_bayes.R` — same in base R.

## When to use

- Many small groups, each with weak data (batting averages, ad-click rates, county-level rare-disease incidence).
- When a full hierarchical MCMC is overkill and shrinkage alone is enough.
- Small-area estimation.
- Genomics: EB on p-values (`limma`, `EBSeq`), on effect sizes.

## Trade-offs vs full Bayes

- **EB is faster** — no MCMC.
- **EB underestimates hyperparameter uncertainty** — treats `(α, β)` as known once plugged in.
- **Fully Bayesian** puts a hyperprior on `(α, β)` and integrates over it → wider posterior intervals for `θ_j` in small-J regimes.
- For large J (say J ≥ 50), EB and fully Bayes agree closely.

## Run

```
python techniques/empirical-bayes/python/empirical_bayes.py
Rscript techniques/empirical-bayes/r/empirical_bayes.R
```

**Refs:** Robbins, H. "An empirical Bayes approach to statistics." *Proc. 3rd Berkeley Symp. Math. Stat. Prob.* 1, 157–163, 1956; Stein, C. "Inadmissibility of the usual estimator for the mean of a multivariate normal distribution." *Proc. 3rd Berkeley Symp.* 1, 197–206, 1956; Efron, B. & Morris, C. "Data analysis using Stein's estimator and its generalizations." *JASA* 70(350), 311–319, 1975; Efron, B. *Large-Scale Inference*, Cambridge, 2010.

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
