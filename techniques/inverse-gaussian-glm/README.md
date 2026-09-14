# Inverse-Gaussian GLM (Reference §47.325)

Tweedie (1957); McCullagh & Nelder (1989). GLM with an
INVERSE GAUSSIAN response distribution:

```
Y ~ IG(μ, λ)      E[Y] = μ,  Var(Y) = μ³ / λ
```

The canonical link is `1/μ²`, but log-link is more common in
practice. Used for skewed positive-only outcomes with variance
that grows CUBICALLY with the mean (rare events, extreme skew,
waiting-time reliability).

## Files

- `python/inverse_gaussian_glm.py` — IRLS fit with a
  log-link on n=500 IG-simulated data (via Michael-
  Schucany-Haas transform). Recovers β̂ ≈ (0.79, 2.20) vs
  true (1.0, 2.0) and dispersion φ̂ ≈ 0.51 vs true 0.5.
- `r/inverse_gaussian_glm.R` —
  `stats::glm(family=inverse.gaussian(link='log'))`,
  `MASS`, `gamlss` (R);
  `statsmodels.genmod.families.InverseGaussian`,
  from-scratch (Python).

## When to use

- **Positive continuous outcomes with heavy right tail** —
  reliability / lifetime data.
- **Waiting times** — first-passage-time interpretation of IG.
- **Insurance / actuarial** — beside Tweedie for pure
  positive-continuous data.

## When NOT to use

- **Data with zeros** — IG support is (0, ∞); use Tweedie or
  hurdle model.
- **Symmetric or light-tailed data** — Normal / Gamma fit
  better.
- **Very small samples** — MLE for IG is sensitive to
  outliers.

## Assumptions & caveats

- **Link choice** — log-link is practical; canonical
  `1/μ²` link is theoretically motivated but numerically
  tricky.
- **Variance ~ μ³** — larger observations have much greater
  variance; check residual plots vs fitted values.
- **Parametrisation** — some sources use (μ, σ²) with
  Var = σ² μ³; others (μ, λ) with Var = μ³ / λ; the two are
  reciprocal.
- **MLE convergence** — IRLS is stable with log-link; identity
  link risks negative μ.

## Related in this repo

- `tweedie-glm-regression` — Tweedie(p=3) recovers IG.
- `gamma-regression` — closest relative for skewed positive
  data.
- `parametric-survival`, `deep-survival-network` — survival
  cousins.
- `gamlss` — generalised additive model for scale/shape.

## Run

```
python techniques/inverse-gaussian-glm/python/inverse_gaussian_glm.py
Rscript techniques/inverse-gaussian-glm/r/inverse_gaussian_glm.R
```

**Refs:** Tweedie, M.C.K. "Statistical properties of inverse Gaussian distributions I / II." *Ann. Math. Statist.*, 28(2): 362-377, 1957; McCullagh, P. and Nelder, J.A. *Generalized Linear Models*, 2nd ed., Chapman & Hall, 1989.

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
