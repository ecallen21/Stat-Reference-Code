# Copulas (Reference §38.9)

Nelsen (2006), Joe (2015). Sklar's theorem: any joint CDF factors as

```
F(x_1, x_2) = C( F_1(x_1), F_2(x_2) )
```

with **C** a copula — a joint CDF on `[0, 1]^d` with uniform
margins. This cleanly separates **dependence structure** from
**marginal distributions**, so you can pair any marginals with any
copula.

## Three parametric families

- **Gaussian(ρ)** — symmetric, no tail dependence. Everyday
  workhorse; misfires on joint extremes.
- **Clayton(θ > 0)** — lower-tail dependence; useful for joint
  crashes / defaults.
- **Gumbel(θ ≥ 1)** — upper-tail dependence; useful for joint
  extremes (storms, prices).

Kendall's τ links to the parameter cleanly:
`Clayton: τ = θ/(θ + 2)`, `Gumbel: τ = 1 − 1/θ`, `Gaussian: τ = (2/π) arcsin(ρ)`.

## Fitting

Two-stage IFM (Inference Function for Margins):

1. Estimate margins (empirically via ranks → pseudo-uniforms
   `u_i = R_i / (n + 1)`).
2. MLE the copula parameter on the pseudo-uniforms.

## When to use

- **Risk aggregation** — model losses jointly with different
  marginals + one copula.
- **Financial dependence** — modelling joint extremes when Gaussian
  correlation understates tail risk.
- **Insurance / actuarial** — dependence between claim types.
- **Hydrology / climate** — joint distributions of flood + wind.

## When NOT to use

- **Independence really holds** — a copula adds no signal, just
  parameters.
- **High-dimensional dependence** — Archimedean copulas scale
  poorly; use vine (pair-copula) constructions.
- **Very heavy-tailed univariate marginals with limited data** —
  marginal estimation dominates copula fit uncertainty.

## Files

- `python/copulas.py` — Sklar refresher + MLE for Gaussian, Clayton,
  Gumbel families on pseudo-observations. Demo (n=800, true
  Clayton θ=2, N(0,1) × Exp(1) margins): AIC ranks **Clayton −630**
  < Gaussian −441 < Gumbel −305; MLE θ̂_Clayton = 1.85 vs
  Kendall-τ implied θ̂ = 1.84.
- `r/copulas.R` — `copula`, `VineCopula`, `rvinecopulib` (R);
  `copulas`, `pyvinecopulib` (Python).

## Assumptions & caveats

- **Model selection** — AIC / BIC across families; goodness-of-fit
  tests (`gofCopula` in R) for the chosen one.
- **Tail behaviour** — a Gaussian copula assigns zero tail
  dependence even at ρ close to 1 — a documented failure in the
  2008 crisis.
- **Pseudo-observations vs parametric margins** — pseudo-uniforms
  eat degrees of freedom; use parametric margins where they fit.
- **Discrete margins** — Sklar's theorem is unique only for
  continuous margins; discrete data require careful handling
  (jittering or Genest-Nešlehová 2007 corrections).

## Related in this repo

- `gaussian-graphical-model` — sparse dependence for many variables.
- `mixture-models` — alternative dependence-through-latent-classes.
- `extreme-value-theory` — pairs with copulas for joint extremes.

## Run

```
python techniques/copulas/python/copulas.py
Rscript techniques/copulas/r/copulas.R
```

**Refs:** Nelsen, R.B. *An Introduction to Copulas*, 2nd ed., Springer, 2006; Joe, H. *Dependence Modeling with Copulas*, Chapman & Hall/CRC, 2015; Sklar, A. "Fonctions de répartition à n dimensions et leurs marges." *Publ Inst Statist Univ Paris*, 1959.

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
