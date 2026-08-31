# SCAD + MCP Nonconvex Penalties (Reference §32.2)

LASSO shrinks large coefficients as much as small ones — **biased for
signals**. Fan & Li (2001) and Zhang (2010) introduced folded-concave
penalties that **vanish for large `|β|`** — unbiased for signals,
sparse for zeros, and enjoy the **oracle property**.

## SCAD (Smoothly Clipped Absolute Deviation) derivative

```
p'_λ(t) = λ                              0 ≤ t ≤ λ
         (a λ − t)₊ / (a − 1)            λ ≤ t ≤ a λ
         0                                t > a λ           (a = 3.7)
```

## MCP (Minimax Concave Penalty) derivative

```
p'_λ(t) = λ − t / γ            t ≤ γ λ
         0                      t > γ λ                   (γ = 3)
```

## Fit via Local Linear Approximation (LLA, Zou-Li 2008)

Linearise the nonconvex penalty at `β_k`; each LLA step is a
**weighted LASSO** with weights `w_j = p'_λ(|β_k_j|)`. Iterating a
few times converges to a local optimum with the oracle property under
mild conditions.

## When to use

- **Sparse regression with signal recovery** — you care about the
  magnitude and identity of true predictors.
- **Post-selection inference** — SCAD/MCP behave like plain OLS on the
  selected support asymptotically (oracle property).
- **Cox / GLM extension** — `ncvreg` handles both.

## When NOT to use

- **Predictive accuracy alone** — LASSO / elastic-net is often just as
  good and easier.
- **You need a global optimum** — SCAD / MCP objectives are non-convex.
- **Very small `n`** — LLA convergence is unreliable.

## Files

- `python/scad_mcp_penalties.py` — from-scratch coordinate-descent
  weighted LASSO + LLA loop for SCAD & MCP. Demo `n=200, d=30`, 3
  non-zero signals `[3.0, -2.5, 2.0]`. All three methods recover the
  correct support (TP = 3/3, FP = 0). Mean signal bias:
  **LASSO 0.138 → SCAD 0.036 → MCP 0.036** (~4× less bias).
- `r/scad_mcp_penalties.R` — `ncvreg` (R reference); `celer`,
  `pyglmnet` (Python).

## Assumptions & caveats

- **Non-convex objective** — solutions are local; warm-start from
  LASSO path.
- **`a`, `γ` tuning** — SCAD `a = 3.7`, MCP `γ = 3` are Fan-Li /
  Zhang defaults; not scale-invariant, so standardise `X` first.
- **Standard errors** — post-selection inference from the oracle
  distribution; or sandwich on the refitted OLS.
- **Grouped / structured penalties** — group-SCAD, sparse-group-MCP
  extensions exist.

## Related in this repo

- `ridge-lasso-elasticnet` — the convex parent.
- `adaptive-lasso` — a two-step alternative with a similar oracle
  property.
- `debiased-lasso` — post-LASSO inference.
- `model-x-knockoffs`, `stability-selection` — FDR-controlled
  selection alternatives.

## Run

```
python techniques/scad-mcp-penalties/python/scad_mcp_penalties.py
Rscript techniques/scad-mcp-penalties/r/scad_mcp_penalties.R
```

**Refs:** Fan, J. & Li, R. "Variable selection via nonconcave penalized likelihood and its oracle properties." *JASA*, 2001; Zhang, C.-H. "Nearly unbiased variable selection under minimax concave penalty." *Annals of Statistics*, 2010; Zou, H. & Li, R. "One-step sparse estimates in nonconcave penalized likelihood models (LLA)." *Annals of Statistics*, 2008.

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
