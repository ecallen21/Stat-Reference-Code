# The Delta Method (Reference §3.29)

A first-order Taylor approximation that gives you the SE / CI for a **function of estimated parameters** when you already know the joint covariance of the parameters themselves.

## Setup

`θ̂ →ᵈ N(θ, V)` (the MLE is asymptotically normal with covariance `V`). For a smooth scalar function `g`:

`g(θ̂) →ᵈ N(g(θ), ∇g(θ)ᵀ · V · ∇g(θ))`

So **SE(g(θ̂)) ≈ √(∇g · V · ∇g)** — the **delta-method SE**.

## Why this is everywhere

It's the engine behind:
- The classical **log-OR Wald CI** for a 2×2 table: `Var(log(ad/bc)) ≈ 1/a + 1/b + 1/c + 1/d` falls right out.
- SEs for **transformed regression coefficients** — e.g. the ratio of two β's, `exp(β)` in a Poisson GLM, marginal effects.
- The CV variance approximation used in `techniques/coefficient-of-variation`.
- Most "Wald CI" output you've ever seen from statistical software for nonlinear transformations.

## When to reach for it (and when not to)

✅ Use it when:
- You have `θ̂` and `V` (or SE/cov from a model summary), and need a CI on `g(θ̂)`.
- `g` is smooth near `θ̂` and the normal approximation works on the natural scale of `g`.

⚠️ Don't use it when:
- `θ̂` is near a boundary (e.g. variances near 0, proportions near 0/1).
- `g` is highly nonlinear (e.g. ratios with denominators near 0).
- You can compute a likelihood-based CI or bootstrap CI instead — those are usually more accurate.

For one-sided sensitivity (e.g. `log(OR)`), build the CI on the **transformed** scale (where normality is best) and back-transform. The CI on `OR` is asymmetric; on `log OR` it's symmetric.

## What's in this file

- `delta_se(g, theta, V)` — numerical-gradient SE; works for any scalar `g` and any `V`.
- `delta_ci(g, theta, V, conf)` — point estimate + Wald CI.
- Worked example 1: **log-OR CI** from a 2×2 table. Side-by-side with the closed-form `√(Σ 1/cell)` — same answer, by construction, since the textbook formula *is* the delta method on `log(ad/bc)` with Poisson cell variances.
- Worked example 2: **SE of CV** = `sd / mean` from `Var(mean)` and `Var(sd)`.

## Files
- `python/delta_method.py` — generic numerical-gradient implementation; log-OR closed-form vs. delta cross-check; CV SE example.
- `r/delta_method.R` — same generics + `car::deltaMethod` (idiomatic, accepts a formula expression in the fit's coefficients) demo on a linear model.
- PySpark: N/A — purely a downstream transform of a covariance matrix.

## Run
```
python techniques/delta-method/python/delta_method.py
Rscript techniques/delta-method/r/delta_method.R
```

**Refs:** Casella & Berger, *Statistical Inference*, 2nd ed., 2002 (§5.5); Oehlert, "A Note on the Delta Method," *The American Statistician* 46(1), 27–29, 1992.
