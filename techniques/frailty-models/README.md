# Shared Gamma Frailty Cox Model (Reference §11.26)

For **clustered survival data** (patients within hospitals, teeth within a mouth, event recurrences within a subject), a shared frailty adds a multiplicative cluster-level random effect `u_c` to the Cox hazard:

```
h(t | X, u_c)  =  h₀(t) · u_c · exp(X·β)

u_c ~ Gamma(1/θ, 1/θ)         (mean 1, variance θ)
```

- `θ = 0` → ordinary Cox (no clustering).
- Larger `θ` → more between-cluster variability.

## Why bother

If you ignore clustering:
- **SEs are too small** — treats correlated within-cluster events as independent.
- **β estimates can be biased** if cluster size correlates with `X`.

## Estimation approach here

**Moment estimator** (Klein-Moeschberger §13.4) — transparent, single-step, robust to demonstrate frailty presence:

```
1. Fit ordinary Cox → β̂, baseline H₀
2. Per cluster c:
     D_c = observed events
     E_c = Σᵢ H₀(tᵢ)·exp(Xᵢ β̂)     (expected events)
     û_c = D_c / E_c
3. θ̂ = Var(û_c) / Mean(û_c)²        (CV² of observed-to-expected ratios)
```

**Demo:** with 40 clusters, per-cluster frailty `Var(u)/Mean(u)² = 0.66`, the moment estimator recovers `θ̂ = 0.71`.

For a fully joint EM estimator of `(β, θ)` with proper SEs and a boundary-adjusted LR test for `θ = 0`, use R's **`survival::coxph(..., + frailty(cluster, distribution = "gamma"))`** — mirrored in the R file.

## Files

- `python/frailty_models.py` — moment-based frailty variance + per-cluster frailty estimates. Reuses [`cox-ph`](../cox-ph)'s `fit_cox`.
- `r/frailty_models.R` — thin wrapper around `survival::coxph(... + frailty())`.

## Assumptions

- Frailties are shared within cluster and independent across clusters.
- Gamma frailty (mean 1) is a convenient conjugate choice; log-normal frailty is an alternative (not shipped).
- Enough clusters and events per cluster to estimate `θ` — a few clusters or `< 5` events per cluster gives an unreliable estimate.

## Run

```
python techniques/frailty-models/python/frailty_models.py
Rscript techniques/frailty-models/r/frailty_models.R
```

**Refs:** Vaupel, J.W., Manton, K.G. & Stallard, E. "The impact of heterogeneity in individual frailty on the dynamics of mortality." *Demography* 16(3), 439–454, 1979; McGilchrist, C.A. & Aisbett, C.W. "Regression with frailty in survival analysis." *Biometrics* 47(2), 461–466, 1991; Klein, J.P. & Moeschberger, M.L. *Survival Analysis*, 2nd ed., Springer, 2003 (Ch. 13); Duchateau, L. & Janssen, P. *The Frailty Model*, Springer, 2008.

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
