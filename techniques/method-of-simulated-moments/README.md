# Method of Simulated Moments (MSM) (Reference §45.7)

McFadden (1989); Pakes & Pollard (1989). When theoretical moments
`m(θ) = E[g(X; θ)]` are intractable but we can **simulate** draws
`X_s(θ)`, replace the integral by a Monte-Carlo average and minimise
a GMM criterion:

    m̂(θ) = (1/S) Σ_s g(X_s(θ))
    θ̂    = argmin_{θ} (m̂(θ) − ḡ)' W (m̂(θ) − ḡ)

with `ḡ` the empirical data moments and `W` a weighting matrix
(identity or two-step efficient GMM).

## Files

- `python/method_of_simulated_moments.py` — matches mean + variance
  of a log-normal via simulation + Nelder-Mead from scratch. Demo
  (n=500, true (μ, σ) = (0.30, 0.40)): MSM recovers (0.295, 0.396)
  vs direct MLE (0.289, 0.406) — machinery works, and this is what
  you need when the likelihood is intractable.
- `r/method_of_simulated_moments.R` — `gmm::gmm`, `momentfit`,
  `BLPestimatoR` (R); `pyblp`, `statsmodels.sandbox.regression.gmm`
  (Python).

## When to use

- **Likelihood involves intractable integrals** — BLP / mixed-logit
  demand, DSGE macro, auction models, dynamic games.
- **Latent variables** — random utility, unobserved heterogeneity;
  simulate them out.
- **You have a good simulator** — cheap, unbiased, and covers the
  moment space.

## When NOT to use

- **Likelihood is tractable** — MLE is more efficient than MSM.
- **Simulator is expensive** — every criterion evaluation runs `S`
  draws; use quasi-Monte-Carlo or fewer moments.
- **Weak identification** — moments carry little information about
  θ; add more moments or refine choice.

## Assumptions & caveats

- **Common random numbers** — fix the seed across `θ` evaluations so
  the criterion is smooth (Rosenbrock, McFadden 1989).
- **Consistency at large S**  — MSM is consistent as `n → ∞` even
  for fixed `S`, but efficiency requires `S → ∞`.
- **Asymptotic variance** — sandwich with an extra `(1 + 1/S)`
  factor accounts for simulation noise.
- **Choice of moments** — use theory + auxiliary regressions;
  weakly-identifying moments blow up the criterion.

## Related in this repo

- `gmm-general` — the non-simulated cousin.
- `arellano-bond-gmm` — dynamic-panel GMM.
- `indirect-inference` — MSM's cousin using auxiliary-model
  estimates as pseudo-moments.
- `abc-approximate-bayesian` — likelihood-free Bayesian analogue.

## Run

```
python techniques/method-of-simulated-moments/python/method_of_simulated_moments.py
Rscript techniques/method-of-simulated-moments/r/method_of_simulated_moments.R
```

**Refs:** McFadden, D. "A method of simulated moments for estimation of discrete response models without numerical integration." *Econometrica*, 57(5): 995-1026, 1989; Pakes, A. & Pollard, D. "Simulation and the asymptotics of optimization estimators." *Econometrica*, 57(5): 1027-1057, 1989.

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
