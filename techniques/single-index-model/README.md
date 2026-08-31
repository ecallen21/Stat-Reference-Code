# Single-Index Model (Reference §33.8)

**Semiparametric regression** with a linear index and an unknown link
function. Ichimura (1993) proved consistent estimation of the direction
`β` under mild conditions.

## Model

```
Y = g( Xᵀ β ) + ε           g : ℝ → ℝ unknown, β ∈ ℝᵖ.
```

`β` is identified only up to scale (fix by `‖β‖ = 1`). `g` is a
**nonparametric** function estimated inside the least-squares
objective:

```
min_β  Σ_i ( y_i − ĝ₋ᵢ( x_iᵀ β ) )²
```

with `ĝ₋ᵢ` a leave-one-out kernel smoother (Nadaraya-Watson typically).

## Convergence rates

- **`β̂`** converges at the parametric rate `√n`, despite the
  nonparametric `g`.
- **`ĝ`** converges at the univariate nonparametric rate `n^(2/5)`.

## When to use

- **A single linear combination of predictors** is expected to drive
  the response.
- **Sparsity in nonlinearity** — the response is a *smooth* function
  of a 1-D projection, avoiding the curse of dimensionality.
- **Interpretability** — the estimated direction `β̂` can be reported
  like a linear-model coefficient.

## When NOT to use

- **Multiple relevant indices** — use multi-index / projection-pursuit
  regression.
- **Interactions matter** — the model can't represent `f(x₁ x₂)`
  without inflating β dimensionality.
- **Very high dim** — bandwidth choice becomes very sensitive.

## Files

- `python/single_index_model.py` — from-scratch grid-search over the
  unit 2-sphere + Nadaraya-Watson leave-one-out smoother. Synthetic
  demo with true direction `(1, 2)/√5` and `g(u) = sin(3u)`. Result:
  **direction alignment `|cos| = 1.0000`** (perfect); nonparametric
  `ĝ` tracks the true link at most `z` values within noise.
- `r/single_index_model.R` — `np::npindex`, `SemiPar` (R); `sisreg`
  (Python).

## Assumptions & caveats

- **Bandwidth** matters — too small = wiggly `ĝ`, too large = flat.
  Rule-of-thumb start: `1.06 σ n^(-1/5)`.
- **Non-convex objective** — grid or global search over the unit
  sphere for small `p`; for larger `p` use quasi-Newton with random
  restarts (Ichimura's iterative procedure).
- **Identifiability** — β is direction-only; the demo reports
  `|cos(angle)|` because signs can flip.
- **Standard errors** — sandwich SEs from Ichimura's asymptotics; the
  demo doesn't compute them.

## Related in this repo

- `varying-coefficient-model` — coefficients that vary in a covariate.
- `additive-quantile-regression` — additive-decomposition alternative.
- `deep-mlp-backprop` — nonparametric multi-index cousin.
- `nadaraya-watson-kernel-regression` (if present) — the smoother
  used inside.
- `local-regression-loess` — local polynomial smoothing alternative.

## Run

```
python techniques/single-index-model/python/single_index_model.py
Rscript techniques/single-index-model/r/single_index_model.R
```

**Refs:** Ichimura, H. "Semiparametric least squares (SLS) and weighted SLS estimation of single-index models." *Journal of Econometrics*, 1993; Härdle, W., Hall, P. & Ichimura, H. "Optimal smoothing in single-index models." *Annals of Statistics*, 1993.

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
