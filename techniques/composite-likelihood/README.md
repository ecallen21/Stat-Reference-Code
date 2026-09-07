# Composite Likelihood (Reference §46.13)

Lindsay (1988); Varin, Reid & Firth (2011). When the full likelihood
`p(y₁, …, y_n; θ)` is intractable, work with a **product of
low-dimensional tractable factors**:

    L_C(θ; y) = Π_{S ∈ 𝒮} L_S(θ; y_S)^{w_S}

## Common forms

| Name | Factor |
|---|---|
| Pairwise marginal | `Π_{i<j} p(y_i, y_j)` |
| Pairwise conditional | `Π_{i<j} p(y_i | y_j)` |
| Full conditional (pseudolikelihood) | `Π_i p(y_i | y_{-i})` |

The composite score is **unbiased**; the MLE-analogue is
**consistent** and **asymptotically normal** with **Godambe sandwich**
variance:

    Var(θ̂) = H(θ)⁻¹ · J(θ) · H(θ)⁻¹
    H = −E[∇² log L_C]    (sensitivity)
    J = Var(∇ log L_C)    (variability)

## Files

- `python/composite_likelihood.py` — bivariate-marginal pairwise CL
  for the mean of an exchangeable correlated sample; comparison
  against the full multivariate-normal MLE. Demo (n=30, ρ=0.5):
  full MLE and pairwise CL both recover μ̂ = 1.41 (truth 1.5);
  sample mean coincides by symmetry.
- `r/composite_likelihood.R` — `CompRandFld`, `spBayes`, `geoR`,
  ordinal + composite, `coxme` (R); from-scratch, `statsmodels`
  fallback (Python).

## When to use

- **Spatial / spatiotemporal models** — full likelihood requires
  huge covariance factorisation; pairwise CL is O(n²) rather than
  O(n³).
- **Longitudinal binary / ordinal** — full likelihood requires
  numerical integration of latent variables; pairwise is closed form.
- **Network / relational data** — Ising models, exponential-family
  random graphs, ergm pseudolikelihood.
- **Frailty / mixed-effect models** — pairwise conditional CL is
  standard in survival with correlated event times.

## When NOT to use

- **Efficiency is critical** — CL is asymptotically less efficient
  than the full MLE (loss usually 5–20 %).
- **Full likelihood is tractable** — no reason to abandon it.
- **Poor first-stage estimator** — CL can inherit / amplify bias if
  the underlying model is misspecified.

## Assumptions & caveats

- **Correct marginal / conditional factors** — misspecification
  matters just as much as for full MLE.
- **Weights `w_S`** — uniform is standard; optimal weighting depends
  on `H`, `J`.
- **Godambe variance** — the usual Fisher inverse is WRONG; you must
  use the sandwich because the score is not a proper likelihood
  score.
- **Information matrix equality fails** — likelihood-ratio and AIC
  need modification (Varin & Vidoni 2005 composite BIC).

## Related in this repo

- `gmm-general` — moment-based cousin.
- `empirical-likelihood` — nonparametric alternative.
- `ergm-exponential-random-graph` — a canonical CL user.
- `frailty-models`, `gee`, `generalized-linear-mixed-models` — CL
  appears as a fallback / diagnostic here.

## Run

```
python techniques/composite-likelihood/python/composite_likelihood.py
Rscript techniques/composite-likelihood/r/composite_likelihood.R
```

**Refs:** Lindsay, B.G. "Composite likelihood methods." *Contemporary Mathematics*, 80: 221-239, 1988; Varin, C., Reid, N. & Firth, D. "An overview of composite likelihood methods." *Statistica Sinica*, 21: 5-42, 2011.

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
