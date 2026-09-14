# Randomized SVD (Reference §47.341)

Halko, Martinsson & Tropp (2011). Approximate the top-k
singular triplets of a large matrix by projecting onto a
`k + p` dimensional random subspace, then doing an exact SVD
of the compressed `(k + p) × n` matrix:

```
Omega ~ N(0, 1)         (n × (k+p))
Y = A @ Omega             (range sketch)
Q = orth(Y)               (QR)
B = Qᵀ @ A                (small)
U_b, S, Vᵀ = svd(B)       (exact but small)
U = Q @ U_b
```

Power iteration `Y ← (A Aᵀ)^q Y` sharpens singular-value
decay. Cost is `O(mn(k+p))` vs `O(mn min(m,n))` for full SVD.

## Files

- `python/randomized_svd.py` — 500×300 matrix (effective
  rank 20). Random SVD ~ 6× faster than full SVD and matches
  the top-25 singular values to two decimals; low-rank
  reconstruction error near-identical to exact truncated SVD.
- `r/randomized_svd.R` — `rsvd::rsvd` (R);
  `sklearn.utils.extmath.randomized_svd`, from-scratch (Python).

## When to use

- **Large low-rank matrices** — recommender systems, term-
  document, gene-expression, image compression.
- **Streaming / one-pass approximations** — random projections
  compose with sketching.
- **Approximate PCA / spectral clustering** on n > 10⁴ points.

## When NOT to use

- **Slowly-decaying spectrum** — random sketch needs decay for
  good error; use randomised subspace iteration q ≥ 4.
- **Very tall / thin matrices** — CX / CUR or column-selection
  may be preferable.
- **When exact singular vectors matter** — the error scales
  with the (k+p+1)st singular value.

## Assumptions & caveats

- **Oversampling p** — 5-10 is standard; error decays with p.
- **Power iterations q** — 1-2 suffices when spectrum decays
  polynomially; 4+ for near-flat spectra (stabilise with
  intermediate QR).
- **Structured randomness** — SRFT / Gaussian projections give
  provable O(√(k/(k+p))) tail error bounds.
- **Reproducibility** — fix the random seed; results depend
  on Omega.

## Related in this repo

- `pca`, `sparse-pca`, `robust-pca` — statistical dim-reduction
  cousins.
- `nystrom-approximation` — kernel-matrix analogue.
- `random-projections` — Johnson-Lindenstrauss lemma.
- `matrix-completion-svt` — thresholded SVD for missing data.

## Run

```
python techniques/randomized-svd/python/randomized_svd.py
Rscript techniques/randomized-svd/r/randomized_svd.R
```

**Refs:** Halko, N., Martinsson, P.G. and Tropp, J.A. "Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions." *SIAM Rev.*, 53(2): 217-288, 2011.

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
