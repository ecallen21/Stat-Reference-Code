# Approximate Bayesian Computation (Reference §14.27)

**"Likelihood-free" Bayesian inference.** Applicable when the likelihood `p(y | θ)` is intractable — no closed form, too expensive to evaluate — but simulating `y | θ` from the model is easy. Common in population genetics, epidemiology, agent-based simulators, and physical models.

## Rejection ABC (Pritchard et al. 1999)

```
for i = 1, ..., N:
    θ_i ~ prior
    y_sim = simulator(θ_i)
    if dist(summary(y_sim), summary(y_obs)) ≤ ε:
        keep θ_i    as an approximate posterior draw
```

- **Small ε** → tighter approximation, more rejections.
- **Standard practice**: generate `N` candidates, keep the closest `q · N` fraction (`q = 0.001` to `0.01`).

## Regression adjustment (Beaumont-Zhang-Balding 2002)

Locally regress `θ` on the summary distance and back-project to the observed summary. Removes the finite-ε bias:

```
θ_i - β̂ᵀ (s_i - s_obs)      with local Epanechnikov weights
```

## Sequential ABC methods

- **ABC-MCMC** — Markov chain over `θ` with likelihood replaced by ABC acceptance ratio.
- **ABC-SMC** — population of particles moved through a shrinking ε schedule (Sisson-Fan-Tanaka 2007).
- **Neural ABC / SNPE** — train a neural density estimator to approximate the posterior directly.

## Files

- `python/abc_approximate_bayesian.py` — rejection ABC + Beaumont local-linear regression adjustment on a Normal-mean toy problem. On n = 50, true μ = 3: ABC posterior mean 3.13, 95% CI (2.84, 3.42) — matches the analytic Normal-Normal posterior 3.13, (2.85, 3.41).
- `r/abc_approximate_bayesian.R` — `abc::abc` (Csillery-Blum-Gaggiotti-François package; supports rejection, local-linear, and neural-network regression adjustment).

## When to use

- **Simulators without a likelihood**: coalescent models, gravitational-wave simulators, cellular automata.
- **Doubly intractable likelihoods** (Potts model, Markov random fields).
- **Rapid prototyping of Bayesian inference** when you can simulate but can't yet evaluate the density.

## Choosing summary statistics

- **Sufficient summaries** (if available) preserve full posterior information.
- **Insufficient but low-dimensional** summaries lose information but keep the acceptance rate manageable.
- **Automatic summary selection**: semi-automatic ABC (Fearnhead & Prangle 2012), or a pilot neural network.

## Assumptions & caveats

- **Curse of dimensionality**: acceptance rate collapses with the dimension of the summary vector; keep the summary short.
- **ε and quantile choice** matter — check sensitivity by comparing across `q ∈ {0.001, 0.01, 0.05}`.
- **ABC is an APPROXIMATION**: with insufficient summaries the ABC posterior is not the true posterior, only a partial-information posterior.

## Run

```
python techniques/abc-approximate-bayesian/python/abc_approximate_bayesian.py
Rscript techniques/abc-approximate-bayesian/r/abc_approximate_bayesian.R
```

**Refs:** Pritchard, J.K. et al. "Population growth of human Y chromosomes: a study of Y chromosome microsatellites." *Mol. Biol. Evol.* 16(12), 1791–1798, 1999; Beaumont, M.A., Zhang, W. & Balding, D.J. "Approximate Bayesian computation in population genetics." *Genetics* 162(4), 2025–2035, 2002; Sisson, S.A., Fan, Y. & Beaumont, M.A. *Handbook of Approximate Bayesian Computation*, CRC, 2018.

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
