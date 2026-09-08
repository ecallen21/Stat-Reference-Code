# Matrix Factorisation with ALS for RecSys (Reference §47.118)

Hu, Koren & Volinsky (2008); explicit-feedback SVD roots in Funk
(2006). Represent user-item interactions R as low-rank U V^T with
IMPLICIT feedback confidences:

    min_{U, V}  Σ c_{u,i} (p_{u,i} − u_uᵀ v_i)² + λ(‖U‖² + ‖V‖²)
    p_{u,i} = 𝟙{R > 0},   c_{u,i} = 1 + α R_{u,i}.

Alternate closed-form user / item updates. Explicit-feedback SVD is
sum-of-squares on RATED entries only.

## Files

- `python/matrix_factorization_als_recsys.py` — from-scratch
  implicit-ALS with weighted per-row updates. Demo (50×40 implicit
  matrix, 15 % density):
  - **ALS held-out hit@10 = 100 %**
  - Popularity baseline hit@10 = 59 %.
- `r/matrix_factorization_als_recsys.R` — `recosystem`,
  `sparseSVD` (R); `implicit`, `LightFM`, `Surprise` (Python).

## When to use

- **Implicit feedback**: clicks, plays, views without explicit
  ratings — recommender-system foundation.
- **Cold-start ranking** with sparse interactions.
- **Item / user similarities** via learned embeddings.
- **Very large but sparse matrices** — ALS parallelises well.

## When NOT to use

- **Explicit ratings without missing-at-random assumption** — use
  weighted / IPW MF or hierarchical Bayesian MF.
- **Rich side information** — factorisation machines / neural CF /
  two-tower.
- **Sequential recommendation** — session RNNs / SASRec dominate.

## Assumptions & caveats

- **α, λ tuning** — validate on held-out interactions; α controls
  confidence-in-positive.
- **Cold-start** for new users/items — need content-based fallback.
- **Popularity bias** — MF may amplify already-popular items;
  reranking / diversification often needed.
- **Non-negative variant** (NMF) forces interpretability.

## Related in this repo

- `neural-collaborative-filtering` — nonlinear extension.
- `nmf`, `probabilistic-pca`, `matrix-completion-svt`,
  `sparse-pca`, `robust-pca`, `pca` — factorisation cousins.
- `patient-similarity-network`, `node2vec-deepwalk`,
  `latent-space-network` — graph-embedding neighbours.
- `bayesian-hierarchical-models`, `variational-inference` —
  Bayesian MF alternatives (Salakhutdinov-Mnih 2008).

## Run

```
python techniques/matrix-factorization-als-recsys/python/matrix_factorization_als_recsys.py
Rscript techniques/matrix-factorization-als-recsys/r/matrix_factorization_als_recsys.R
```

**Refs:** Hu, Y., Koren, Y. & Volinsky, C. "Collaborative filtering for implicit feedback datasets." *ICDM*, 2008; Koren, Y., Bell, R. & Volinsky, C. "Matrix factorization techniques for recommender systems." *Computer* 42(8): 30-37, 2009.

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
