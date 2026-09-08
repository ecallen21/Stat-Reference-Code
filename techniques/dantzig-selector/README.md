# Dantzig Selector (Reference §6.18)

Candès & Tao (2007). Sparse linear regression alternative to lasso:

    min ‖β‖₁    s.t.  ‖X' (y − X β)‖_∞  ≤  λ

The dual constraint bounds the maximum absolute correlation of any
covariate with the residuals. Solved as a **linear program** in
polynomial time.

## vs Lasso

- **Lasso**:  `min ½ ‖y − Xβ‖² + λ ‖β‖₁`.
- **Dantzig**: LP formulation; similar theoretical guarantees under
  restricted-isometry; solutions close but not identical.

## Files

- `python/dantzig_selector.py` — LP via `scipy.optimize.linprog`
  with `β = u − v, u, v ≥ 0` from scratch. Demo (n=100, p=40,
  true support {0, 5, 12}, coefs (1.5, −1.0, 0.8)): at λ=0.10,
  Dantzig recovers exactly the correct 3-sparse support with
  coefficients (1.449, −0.896, 0.836).
- `r/dantzig_selector.R` — `flare::slim(method='dantzig')`,
  `hdi::lasso.proj` (R); cvxpy, `scipy.optimize.linprog`,
  from-scratch (Python).

## When to use

- **High-dim linear regression** (p > n).
- **Feature selection** with theoretical guarantees.
- **Alternative to lasso** when convex-LP solvers are preferred
  over coordinate-descent.
- **Post-selection inference** — Dantzig-selection + debiased
  variants give asymptotic CIs.

## When NOT to use

- **Very large p** — LP `O(p³)` per solve; use lasso / coordinate
  descent for scale.
- **Non-sparse truth** — like lasso, biased toward sparse
  solutions.
- **Correlated features** — RIP fails; use group / elastic-net
  variants.

## Assumptions & caveats

- **RIP / RE condition** on X — required for exact recovery
  guarantees.
- **λ selection** — cross-validate or use theoretical rate
  `λ = c σ √(log p / n)`.
- **Model-selection consistency** requires irrepresentable-like
  conditions.
- **Bias correction** — pair with debiased-lasso for post-
  selection inference.

## Related in this repo

- `ridge-lasso-elasticnet`, `scad-mcp-penalties`,
  `adaptive-lasso`, `debiased-lasso`, `group-lasso`,
  `fused-lasso`, `stability-selection`,
  `sure-independence-screening` — penalised / sparse regression
  cousins.

## Run

```
python techniques/dantzig-selector/python/dantzig_selector.py
Rscript techniques/dantzig-selector/r/dantzig_selector.R
```

**Refs:** Candès, E. & Tao, T. "The Dantzig selector: statistical estimation when p is much larger than n." *Annals of Statistics*, 35(6): 2313-2351, 2007; Bickel, P.J., Ritov, Y. & Tsybakov, A.B. "Simultaneous analysis of lasso and Dantzig selector." *Annals of Statistics*, 37(4): 1705-1732, 2009.

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
