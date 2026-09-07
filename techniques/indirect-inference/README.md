# Indirect Inference (Reference §45.8)

Gouriéroux, Monfort & Renault (1993); Smith (1993). When the
structural model's likelihood is intractable, use an **auxiliary
tractable model** as an "instrument":

1. Fit the auxiliary model to the **data** → `β̂`.
2. For candidate `θ`, simulate `S` datasets from the structural
   model, fit the auxiliary model to each → `{β̂_s(θ)}`, average to
   `β̄(θ)`.
3. Choose `θ` so `β̄(θ)` matches `β̂`:

       θ̂ = argmin (β̂ − β̄(θ))' W (β̂ − β̄(θ))

## Files

- `python/indirect_inference.py` — MA(1) `y_t = ε_t + θ · ε_{t−1}`
  estimated by matching AR(2) auxiliary coefficients from scratch.
  Demo (T=800, θ=0.6, S=40 simulations): II recovers θ̂ = 0.573
  vs truth 0.60. Prints the auxiliary AR(2) coefficients and the
  theoretical `θ/(1+θ²)` for orientation.
- `r/indirect_inference.R` — `indirectInference`, `gmm::gmm` with
  identity weight, `yuima` (R); from-scratch Python.

## When to use

- **Structural macro models** — DSGE, financial-market models with
  latent variables.
- **Simulation-only models** — agent-based / general-equilibrium.
- **Auxiliary model available** — pick something rich enough to
  identify θ (e.g. VAR / AR(p) for MA / ARMA structural).

## When NOT to use

- **Likelihood tractable** — use MLE; II is less efficient.
- **Auxiliary model too small** — II is inconsistent if the
  auxiliary doesn't "encompass" the structural parameters.
- **Simulator too slow** — every criterion call runs `S` sims.

## Assumptions & caveats

- **Common random numbers** — hold seeds fixed across θ evaluations
  so the criterion is a smooth function of θ.
- **Efficiency** — II is asymptotically as efficient as MLE when the
  auxiliary model encompasses the structural; less otherwise.
- **Variance formula** — Gouriéroux et al. sandwich accounts for
  simulation noise via `(1 + 1/S)` factor.
- **Identification** — the auxiliary parameters must be sensitive to
  θ; verify by inspecting `∂β̄/∂θ`.

## Related in this repo

- `method-of-simulated-moments` — moments-based cousin.
- `gmm-general` — the exact-moments framework II fits into.
- `abc-approximate-bayesian` — likelihood-free Bayesian analogue.
- `arima`, `arfima`, `garch` — auxiliary models often used with II.

## Run

```
python techniques/indirect-inference/python/indirect_inference.py
Rscript techniques/indirect-inference/r/indirect_inference.R
```

**Refs:** Gouriéroux, C., Monfort, A. & Renault, E. "Indirect inference." *Journal of Applied Econometrics*, 8(S1): S85-S118, 1993; Smith, A.A. "Estimating nonlinear time-series models using simulated vector autoregressions." *Journal of Applied Econometrics*, 8(S1): S63-S84, 1993.

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
