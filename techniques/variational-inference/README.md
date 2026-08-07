# Variational Inference (Reference §14.24, §14.25)

Approximate the posterior `p(θ | y)` with a simpler family `q(θ; φ)` by **maximizing the evidence lower bound**:

```
ELBO(φ) = E_q[log p(y, θ)] − E_q[log q(θ; φ)]
```

Maximizing the ELBO minimizes `KL(q ‖ p)`. Vastly faster than MCMC on big data; trades off exactness (`q` rarely equals `p`), and can systematically **underestimate posterior variance** because KL(q ‖ p) penalizes q putting mass where p is small.

## Mean-field VI

Restrict `q` to a product of independent factors:

```
q(θ) = Π_j q_j(θ_j)
```

Coordinate-ascent updates cycle each `q_j = exp(E_{−j}[log p(y, θ)]) / Z_j`. For conjugate models this gives closed-form recurrences — the Bayesian analog of Gibbs sampling.

## ADVI (Automatic Differentiation VI, Kucukelbir et al. 2017)

Fix `q` to a diagonal Gaussian in a transformed unconstrained space, use **pathwise / reparameterization gradients** of the ELBO, optimize with Adam. The production choice in Stan / PyMC / NumPyro.

```
θ = m + s · ε,    ε ~ N(0, I)
∇_m ELBO = E_ε[∇_θ log p(y, θ)]
∇_s ELBO = E_ε[∇_θ log p(y, θ) · ε] + 1/s     (entropy adds 1/s per dimension)
```

## Files

- `python/variational_inference.py` — reparameterization-gradient mean-field Gaussian VI on a Normal-Normal problem (recovers posterior mean 2.632 vs exact 2.628 and posterior sd 0.141 vs exact 0.141 to three decimals) and closed-form CAVI on Beta-Binomial.
- `r/variational_inference.R` — analytic CAVI on Beta-Binomial and Normal-Normal in base R. Production: `rstan(algorithm="meanfield")`, `brms(algorithm="fullrank")`.

## When to prefer VI

- Very large datasets where MCMC's per-iteration cost is prohibitive.
- Rough posterior summaries during model prototyping (fast ELBO ascent).
- Downstream tasks where a fast point estimate + rough uncertainty is enough.

## When NOT to use VI

- Publication-grade uncertainty quantification — mean-field VI systematically **underestimates variances** (and misses correlations by construction).
- Multimodal posteriors — mean-field Gaussian collapses onto a single mode.
- Small-data regimes where MCMC is fast enough anyway.

## Diagnostics

- Track the ELBO over iterations — it should increase monotonically (up to Monte Carlo noise).
- Compare to a short MCMC run — VI-vs-MCMC posterior means should agree; sds are a rougher check.
- Pareto-smoothed importance-sampling correction (PSIS-VI, Yao et al. 2018) can salvage biased VI.

## Run

```
python techniques/variational-inference/python/variational_inference.py
Rscript techniques/variational-inference/r/variational_inference.R
```

**Refs:** Jordan, M.I. et al. "An introduction to variational methods for graphical models." *Mach. Learn.* 37, 183–233, 1999; Blei, D.M., Kucukelbir, A. & McAuliffe, J.D. "Variational inference: a review for statisticians." *JASA* 112(518), 859–877, 2017; Kucukelbir, A. et al. "Automatic differentiation variational inference." *JMLR* 18(1), 430–474, 2017.

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
