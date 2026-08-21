# Spatial GLM — Poisson-CAR (Reference §23.x extra)

Generalised linear model with a **spatially structured random effect** on the linear predictor. Poisson version (disease mapping / rate modelling):

```
y_i ~ Poisson( E_i · exp( x_iᵀ β + u_i ) )
u   ~ N(0, σ²  (D − α W)⁻¹)                (CAR)
```

- `E_i` = expected count (offset, e.g. population × baseline rate).
- `x_i` = fixed-effect covariates.
- `u_i` = latent spatial random effect drawn from a CAR (Leroux, ICAR, BYM) prior.
- `α ∈ [0, 1)` interpolates iid (`α = 0`) and improper ICAR (`α = 1`).

Binomial-CAR (logit link) fits the same skeleton with a Bernoulli / binomial likelihood; Gaussian-CAR reduces to the classical spatial linear model.

## Fitting

- **MAP / PIRLS** (this module) — coordinate ascent: Newton for β, penalised IRLS for u with the CAR quadratic penalty, closed-form σ² conditional. Fast, no draws.
- **MCMC (CARBayes)** — Gibbs / adaptive-random-walk MH; produces posterior draws for β, u, α, σ².
- **INLA** — Integrated Nested Laplace Approximation; fast Bayes for latent Gaussian models; the workhorse for large disease-mapping models.
- **Variational Bayes** — mean-field q(u) q(β); scales further at the cost of variance underestimation.

## When to use

- **Disease mapping** — small-area cancer / infection / mortality rates with unstable local counts (BYM is the standard).
- **Ecology** — species counts per grid cell; environment covariates + spatial residual.
- **Crime rate** modelling per beat / block with covariates.
- **Any rate / count / binary outcome** with spatially correlated residual structure that a purely fixed-effect GLM would leave in the residuals.

## Files

- `python/spatial_glm.py` — from-scratch Poisson-CAR via PIRLS + CAR penalty. Demo (10×10 grid, α=0.95, true β = [−0.5, 0.7], σ² = 1.5): β_x_hat = 0.78 (true 0.70), σ²_hat = 1.06, cor(û, u) = 0.95. Intercept 0.23 vs true −0.5 — a known confounding between the intercept and the mean of u under a CAR without sum-to-zero constraint.
- `r/spatial_glm.R` — `CARBayes::S.CARleroux / S.CARbym`, `INLA::inla(family='poisson', f(id, model='bym'))`, `spaMM::HLCor`.

## Assumptions & caveats

- **Intercept ↔ E[u] confounding** — CAR without a sum-to-zero constraint on u leaves the overall level of u unidentified from the intercept. ICAR fixes it with a sum-to-zero constraint (`bym2` in INLA); Leroux CAR with `α < 1` is proper but still weakly identified. When the intercept matters substantively, constrain u.
- **α is often weakly identified** — set to a fixed high value (0.95, 0.99) or place a strong prior (e.g. `Beta(a, b)` favouring near-1).
- **Rate confounding with the offset** — always include `log(E)` as an offset, not as a covariate.
- **Spatial smoothing hides genuine local excess** — check residuals; consider `spatial-scan-cluster` (Kulldorff) alongside for hot-spot detection.
- **Zero counts** — handled correctly by Poisson; if there are many structural zeros, consider zero-inflated variants (`CARBayes::ZIP.CARleroux`).
- **Neighbourhood definition matters** — the estimated σ² and u depend on W (rook / queen / distance); a sensitivity analysis is standard practice.

## Run

```
python techniques/spatial-glm/python/spatial_glm.py
Rscript techniques/spatial-glm/r/spatial_glm.R
```

**Refs:** Besag, J., York, J. & Mollié, A. "Bayesian image restoration, with two applications in spatial statistics." *Ann. Inst. Stat. Math.* 43(1), 1–20, 1991; Lee, D. "CARBayes: An R package for Bayesian spatial modelling with conditional autoregressive priors." *J. Stat. Softw.* 55(13), 2013; Riebler, A. et al. "An intuitive Bayesian spatial model for disease mapping that accounts for scaling." *Stat. Meth. Med. Res.* 25(4), 1145–1165, 2016.

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
