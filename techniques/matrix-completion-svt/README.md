# Matrix Completion via SVT (Reference §47.43)

Cai, Candès & Shen (2010). Recover a low-rank matrix from a subset
Ω of its entries by nuclear-norm minimisation

    min ‖X‖_*   s.t.  P_Ω(X) = P_Ω(M).

SVT iterates

    Y_{k+1} = Y_k + δ · P_Ω(M − X_k)
    X_{k+1} = D_τ(Y_{k+1})              (soft-threshold singular values)

Convergent to the nuclear-norm minimum when τ, δ chosen per CCS
heuristic. Exact recovery under RIP-type conditions on Ω.

## Files

- `python/matrix_completion_svt.py` — SVT loop with SVD +
  soft-threshold, using CCS defaults `τ = 5√(nm)`, `δ = 1.2/frac`.
  Demo (50×40 rank-3 truth, 62% observed):
  exact recovery — rank 3, relative error ≈ 0.0000.
- `r/matrix_completion_svt.R` — `softImpute` (Mazumder-Hastie-
  Tibshirani), `nlSVMc` (R); `fancyimpute`, from-scratch (Python).

## When to use

- **Recommender systems** — user × item matrix with observed ratings.
- **Sensor / survey non-response** — impute low-rank blocks.
- **Image / video inpainting** — small missing regions on low-rank
  frame representations.
- **Genomics** — low-rank expression matrices with dropouts.

## When NOT to use

- **Very large matrices** — SVD per iter is O(nmk); use ALS
  (`softImpute`), coordinate methods, or ADMM at scale.
- **Non-low-rank truth** — nuclear norm biased toward low rank;
  add sparse component (RPCA) if data is low-rank + sparse.
- **MNAR** — SVT assumes MCAR sampling of Ω; use IPW or joint
  models for informative missingness.

## Assumptions & caveats

- **RIP / RUB** on sampling operator P_Ω — required for exact
  recovery guarantees.
- **τ selection** — CCS `5√(nm)`; too small → high rank; too large
  → oversmoothing.
- **δ step size** — the paper's `1.2/frac` is a conservative default;
  Nesterov acceleration or ADMM converge faster.
- **Bias** — like lasso, soft-thresholding shrinks singular values;
  consider iterative reweighted / SVP for less-biased alternatives.

## Related in this repo

- `robust-pca` — low-rank + sparse (Candès-Li-Ma-Wright 2011).
- `nmf`, `sparse-pca`, `probabilistic-pca` — factorisation cousins.
- `random-projections`, `dimensionality-reduction` — low-rank
  approximation via sketching.
- `ridge-lasso-elasticnet`, `group-lasso` — nuclear-norm's
  sparsity-inducing sibling.

## Run

```
python techniques/matrix-completion-svt/python/matrix_completion_svt.py
Rscript techniques/matrix-completion-svt/r/matrix_completion_svt.R
```

**Refs:** Cai, J.-F., Candès, E.J. & Shen, Z. "A singular value thresholding algorithm for matrix completion." *SIAM J Optim* 20(4): 1956-1982, 2010; Candès, E.J. & Recht, B. "Exact matrix completion via convex optimization." *FoCM* 9: 717-772, 2009.

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
