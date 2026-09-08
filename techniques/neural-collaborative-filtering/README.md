# Neural Collaborative Filtering (Reference §47.119)

He, Liao, Zhang, Nie, Hu & Chua (2017). Replaces the inner product
of matrix factorisation with a NEURAL NETWORK on user + item
embeddings:

    GMF:    ŷ = σ( Σ_k (u_k * v_k) )                      (generalised MF)
    MLP:    ŷ = σ( f_θ([u; v]) )                          (feed-forward)
    NeuMF:  concat(GMF output, MLP output) → logistic     (best of both)

Extended by two-tower models (Youtube 2019), Transformer-based
sequential recsys (SASRec, BERT4Rec), and factorization machines.

## Files

- `python/neural_collaborative_filtering.py` — compares GMF /
  concat-linear / concat-MLP heads on a NONLINEAR truth
  `y = 1(‖u − v‖ < 2.5)`. Demo (100 users × 60 items, dim 4):
  - GMF     linear  ≈ chance / MLP small lift
  - concat  linear  ≈ chance / **MLP captures the nonlinear boundary**
  - neumf   MLP acc = 0.60.
- `r/neural_collaborative_filtering.R` — no first-class R port;
  `recommenders` (Microsoft), `neural_collaborative_filtering`
  reference (Python).

## When to use

- **Nonlinear user-item interactions** where MF's dot-product is
  inadequate.
- **Rich embeddings + side features** (age, tags, category).
- **When you want the best of MF + neural** — NeuMF ensemble.
- **Session / sequence** — extends to SASRec / BERT4Rec.

## When NOT to use

- **Tiny data / cold-start** — MF or content-based simpler.
- **Very strict serving-latency** — MF's O(d) dot product is hard
  to beat; keep neural for scoring re-rank.
- **Interpretability** — deep model harder to explain than
  factorised similarity.

## Assumptions & caveats

- **Negative sampling** critical — random negatives can be too
  easy; use popularity- or in-batch-hard negatives.
- **Regularisation** for shared embeddings across GMF / MLP heads.
- **Cold-start** — pair with content encoders (two-tower).
- **Evaluation** — hit@K / NDCG@K with leave-one-out; avoid
  temporal leakage.

## Related in this repo

- `matrix-factorization-als-recsys` — the linear baseline NCF
  generalises.
- `neural-network-mlp`, `residual-connections`,
  `label-smoothing`, `mixup` — DL toolkit.
- `two-tower` (not present) analogue: `word2vec-skipgram`
  co-training; `bpr` alternative via `matrix-factorization-als-recsys`.
- `contrastive-learning`, `contrastive-predictive-coding`,
  `byol-simsiam` — SSL representation cousins.

## Run

```
python techniques/neural-collaborative-filtering/python/neural_collaborative_filtering.py
Rscript techniques/neural-collaborative-filtering/r/neural_collaborative_filtering.R
```

**Refs:** He, X., Liao, L., Zhang, H., Nie, L., Hu, X. & Chua, T.-S. "Neural collaborative filtering." *WWW*, 2017.

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
