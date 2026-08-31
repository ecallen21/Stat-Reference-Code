# Robust PCA (Reference §25.13)

Candès, Li, Ma & Wright (2011). Decompose `M = L + S` into a **low-rank
part** `L` and a **sparse** corruption `S`:

```
min_{L, S}  ‖L‖_*  +  λ ‖S‖_1     s.t.  M = L + S.
```

Solved by **Principal Component Pursuit** (ADMM / IALM). Recovers `L`
exactly under mild incoherence + sparsity of `S`.

## When to use

- **Video background modelling** — background = low-rank, moving
  objects = sparse.
- **Face illumination artefacts** — surface + shadows.
- **Sensor spikes** in industrial monitoring.
- **Any low-rank matrix contaminated with gross but sparse errors.**

## When NOT to use

- **Dense noise** — RPCA assumes sparse contamination; use PPCA / PCA.
- **Very low signal-to-noise** — the low-rank part must be
  distinguishable.

## Files

- `python/robust_pca.py` — from-scratch ADMM with soft-thresholding
  + singular-value thresholding. Demo `n=40, d=30, rank=2`, 10 %
  entries corrupted at ±3: **rank recovered = 2**, `L` relative
  Frobenius error 0.0000, all 119 corruptions identified with 0 FP.
- `r/robust_pca.R` — `rpca`, `pcaPP`, `robustbase` (R); `splitpca`,
  `ristretto` (Python).

## Assumptions & caveats

- **`λ = 1 / √(max(n, d))`** — Candès' default suggested value.
- **`μ` (ADMM penalty)** — chosen via `0.25 · nd / ‖M‖_1`.
- **Convergence** — checks `‖M − L − S‖_F / ‖M‖_F < tol`.
- **Non-convex extensions** — accelerated schemes (Lin 2010 IALM,
  Yuan-Yang 2013 approximate).
- **Streaming variants** — Feng-Xu-Yan 2013 online RPCA.

## Related in this repo

- `probabilistic-pca`, `sparse-pca`, `dimensionality-reduction-pca` —
  PCA family.
- `gaussian-graphical-model`, `fused-lasso` — sibling low-rank +
  sparse decompositions.
- `robust-regression`, `mm-estimators-robust` — robust statistics
  cousin.

## Run

```
python techniques/robust-pca/python/robust_pca.py
Rscript techniques/robust-pca/r/robust_pca.R
```

**Refs:** Candès, E.J., Li, X., Ma, Y. & Wright, J. "Robust principal component analysis?" *JACM*, 2011; Lin, Z., Chen, M. & Ma, Y. "The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices." *arXiv:1009.5055*, 2010.

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
