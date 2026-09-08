# Orthogonal Matching Pursuit (Reference §47.123)

Pati, Rezaiifar & Krishnaprasad (1993). Greedy sparse regression:

    1. r = y, S = ∅.
    2. j* = argmax|X_j'r|; S ← S ∪ {j*}.
    3. β_S = argmin ‖y − X_S β_S‖²    (least-squares refit).
    4. r = y − X_S β_S.
    5. Repeat until |S| = K or ‖r‖ < tol.

Contrast with LARS (equiangular direction, single-column update):
OMP fully refits every step and lands closer to OLS-on-support at
each K.

## Files

- `python/orthogonal_matching_pursuit.py` — from-scratch OMP.
  Demo (n=200, p=20, true support {0, 3, 7, 15}, coefs (1.5, −1.0,
  0.8, −0.6)):
  - K=1: {0} added; residual ‖.‖ = 1.37
  - K=4: **exact true support recovered**; β̂ = (1.485, −0.993,
    0.774, −0.591) — matches truth to 3 decimals.
- `r/orthogonal_matching_pursuit.R` — custom in R + `glmnet` as
  baseline; `sklearn.linear_model.OrthogonalMatchingPursuit`
  (Python).

## When to use

- **Fixed sparsity budget** (compressed sensing, sensor selection).
- **Fast greedy alternative to L1**.
- **Very tall design matrices** — one dot product per iter.
- **Dictionary learning** — OMP is the standard coding step
  in K-SVD.

## When NOT to use

- **Highly correlated features** — OMP is greedy; may pick a
  substitute early and never correct.
- **When RIP fails** — LASSO / Dantzig with recovery guarantees
  more reliable.
- **When you need the full regularisation path** — LARS gives
  it cheaply.

## Assumptions & caveats

- **Standardise X** columns to unit norm.
- **Stopping**: fixed K, tolerance on ‖r‖, or an information
  criterion.
- **Uniqueness**: solution unique given RIP-like design.
- **Extensions**: Stagewise OMP (StOMP), Batch OMP, GraphOMP.

## Related in this repo

- `lars-least-angle-regression`, `coordinate-descent-lasso`,
  `ridge-lasso-elasticnet`, `adaptive-lasso`,
  `scad-mcp-penalties`, `debiased-lasso`,
  `dantzig-selector`, `stability-selection`, `group-lasso`,
  `fused-lasso` — sparse regression family.
- `sure-independence-screening`, `model-x-knockoffs`,
  `post-selection-inference` — feature-selection cousins.
- `sparse-pca`, `nmf`, `matrix-completion-svt` — sparse/low-rank
  neighbours.

## Run

```
python techniques/orthogonal-matching-pursuit/python/orthogonal_matching_pursuit.py
Rscript techniques/orthogonal-matching-pursuit/r/orthogonal_matching_pursuit.R
```

**Refs:** Pati, Y.C., Rezaiifar, R. & Krishnaprasad, P.S. "Orthogonal matching pursuit: recursive function approximation with applications to wavelet decomposition." *Asilomar Conf on Signals, Systems and Computers*, 1993; Tropp, J.A. "Greed is good: Algorithmic results for sparse approximation." *IEEE Trans Info Theory* 50(10): 2231-2242, 2004.

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
