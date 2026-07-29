# Generalized Linear Mixed Models (Reference §12.3; also covers §12.23 ordinal GLMM)

Extends LMM to non-Gaussian outcomes via a link function:

```
g(μ_{ij})  =  X_{ij}' β  +  u_i             u_i ~ N(0, σ²_u)
```

- Binary → logit link (this file).
- Count → log link (Poisson GLMM).
- Ordinal (§12.23) → cumulative-logit link with subject-specific thresholds.

## Marginal likelihood — no closed form

Unlike LMM, integrating out `u_i` is not analytic. Two standard approximations:

- **Laplace approximation** — second-order Taylor at the mode of the integrand. Fast; used by `glmer` by default.
- **Adaptive Gauss-Hermite quadrature** — sum the integrand at K nodes (Liu-Pierce 1994). `K = 5–15` for random-intercept models. More accurate than Laplace, more expensive.

This file implements **Gauss-Hermite** on a random-intercept binomial GLMM; extends easily to Poisson (swap the per-cluster log-likelihood).

## Subject-specific vs. population-averaged

The GLMM `β` is **subject-specific / conditional** — "the effect of X on cluster i's own outcome, given its random effect." Different from GEE, which reports the **population-averaged / marginal** effect. For a nonlinear link the two differ (see [`gee`](../gee) for the marginal complement).

## §12.23 Cumulative Link Mixed Model (ordinal GLMM)

Same structure as binary GLMM, but the link is the cumulative-logit with K-1 fixed thresholds:

```
logit P(Y ≤ k | X, u)  =  α_k − X'β − u
```

Fit is analogous — R's `ordinal::clmm` handles it cleanly; the Python file's Gauss-Hermite driver extends by summing the ordinal per-cluster log-likelihood.

## Files

- `python/generalized_linear_mixed_models.py` — Gauss-Hermite marginal-likelihood MLE for binary random-intercept GLMM. Recovers σ_u nearly exactly on synthetic data; the intercept picks up some finite-sample bias typical of non-adaptive Gauss-Hermite.
- `r/generalized_linear_mixed_models.R` — thin wrappers around `lme4::glmer` (binary / Poisson) and `ordinal::clmm` (ordinal).

## Assumptions

- Random effects are normal (adaptive quadrature = better if not).
- Correct link.
- Enough clusters (`n_cluster ≥ 20`) and events per cluster to identify σ_u.

## Run

```
python techniques/generalized-linear-mixed-models/python/generalized_linear_mixed_models.py
Rscript techniques/generalized-linear-mixed-models/r/generalized_linear_mixed_models.R
```

**Refs:** Breslow, N.E. & Clayton, D.G. "Approximate inference in generalized linear mixed models." *JASA* 88(421), 9–25, 1993; Liu, Q. & Pierce, D.A. "A note on Gauss-Hermite quadrature." *Biometrika* 81(3), 624–629, 1994; Bolker, B.M., Brooks, M.E., Clark, C.J. *et al.* "Generalized linear mixed models: a practical guide for ecology and evolution." *Trends Ecol. Evol.* 24(3), 127–135, 2009.

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
