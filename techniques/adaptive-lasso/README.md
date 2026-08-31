# Adaptive LASSO (Reference §32.12)

Zou (2006). Two-step fix for LASSO's bias:

1. Compute an initial estimate `β̂_init` — OLS (`n > p`) or ridge /
   LASSO (`p ≫ n`).
2. Fit weighted LASSO with weights `w_j = 1 / |β̂_init_j|^γ` (γ > 0,
   often 1):

```
argmin_β  ‖ y − X β ‖²  +  λ · Σ_j w_j |β_j|
```

Weights **de-bias** large signals (small weight ⇒ little shrinkage)
and **hard-shrink** noise (huge weight ⇒ forced to zero). Under mild
conditions Adaptive LASSO enjoys the **oracle property**: it selects
the true support AND asymptotically behaves like OLS on that support.

## When to use

- **Sparse regression with signal recovery** — you care about the
  magnitudes.
- **Alternative to SCAD / MCP** that stays convex given the initial
  weights.
- **Grouped / structured extensions** exist (Zou-Zhang 2009).

## When NOT to use

- **Very few observations** — the initial estimate is noisy; weights
  blow up.
- **Highly correlated features** — the initial estimate may point at
  the wrong feature; weights lock in the mistake.
- **Just prediction** — plain LASSO or ridge is fine.

## Files

- `python/adaptive_lasso.py` — from-scratch weighted-LASSO
  coordinate descent + ridge initialisation. Demo `n=200, d=30`, 3
  signals `[3.0, -2.5, 2.0]`:
  **LASSO mean-bias 0.196 → aLASSO 0.061 (3× less) → oracle-OLS
  0.036**; all three methods recover the correct support.
- `r/adaptive_lasso.R` — `glmnet(penalty.factor = 1 / |init|^γ)` (R);
  `celer` / weighted `sklearn.Lasso` (Python).

## Assumptions & caveats

- **Initial estimate must be consistent** — OLS / ridge / LASSO all
  work; ridge is a robust default in `p ≫ n`.
- **`γ` tuning** — γ = 1 is Zou's default; larger γ = harsher weight
  contrast.
- **Ε-guard on the weights** — `1 / |β̂_init|^γ` blows up at 0; add
  `+ ε` to the denominator.
- **Multi-step re-adaptation** (M-adaptive LASSO) can further reduce
  bias.
- **CV on `λ`** — the initial estimate depends on its own tuning; a
  two-level CV is safest.

## Related in this repo

- `ridge-lasso-elasticnet` — the convex parents.
- `scad-mcp-penalties` — nonconvex-penalty alternative with the same
  oracle goal.
- `debiased-lasso`, `model-x-knockoffs`, `stability-selection` —
  inference / selection alternatives.
- `sure-independence-screening` (if present) — a screening step
  before LASSO in very high-dim settings.

## Run

```
python techniques/adaptive-lasso/python/adaptive_lasso.py
Rscript techniques/adaptive-lasso/r/adaptive_lasso.R
```

**Refs:** Zou, H. "The adaptive LASSO and its oracle properties." *JASA*, 2006; Zou, H. & Zhang, H.H. "On the adaptive elastic-net with a diverging number of parameters." *Annals of Statistics*, 2009.

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
