# Siamese Networks (Reference §47.152)

Bromley et al. (1994); Koch, Zemel & Salakhutdinov (2015). Two
identical sub-networks share weights and process a **pair** of
inputs; a similarity head compares the embeddings:

    L_contrastive(x1, x2, y) = y · D²  +  (1 − y) · max(0, m − D)²

with D = ‖f(x1) − f(x2)‖₂. Enables one-shot / few-shot learning by
scoring similarity of a query against a handful of reference
exemplars — no per-class training needed.

## Files

- `python/siamese_networks.py` — linear-embedding Siamese with
  Hadsell-Chopra-LeCun contrastive loss and balanced pos / neg
  sampling.
  - **Iris**: 3-way one-shot acc raw = 0.816, Siamese 2-D
    = **0.958**.
  - **Noisy synthetic** (K = 4, d = 20, signal in first 2 dims):
    4-way one-shot raw = 0.34, Siamese 2-D = **1.000**.
- `r/siamese_networks.R` — recommends
  `pytorch-metric-learning.ContrastiveLoss`, keras Siamese
  examples, `tensorflow-similarity`.

## When to use

- **One-shot / few-shot** classification and verification (face,
  signature, product IDs).
- **Duplicate detection** (near-duplicate document / image
  retrieval).
- **Pairwise ranking / similarity scoring** at test time.

## When NOT to use

- **Fixed, closed-label** classification with abundant data —
  cross-entropy is simpler.
- **Many-way comparisons** where triplet loss is more informative.
- **Very small pair budgets** — pair sampling dominates variance.

## Assumptions & caveats

- **Weight sharing** (twin networks) is essential — asymmetric
  copies rarely help.
- **Margin m** ~ 1 typical; tune per problem.
- **Pair sampling balance** — random pairs are ~99 % negative in
  many-class problems; balance or use hard-negative mining.
- **Distance metric** — Euclidean is default; cosine works well
  after L2 normalisation.

## Related in this repo

- `deep-metric-learning-triplet` — triplet-based sibling.
- `simclr-contrastive`, `contrastive-learning`,
  `byol-simsiam` — self-supervised cousins using the same
  positive / negative construction.
- `barlow-twins` — negative-pair-free redundancy-reduction
  alternative.

## Run

```
python techniques/siamese-networks/python/siamese_networks.py
Rscript techniques/siamese-networks/r/siamese_networks.R
```

**Refs:** Bromley, J. et al. "Signature verification using a Siamese time delay neural network." *NIPS*, 1994; Koch, G., Zemel, R. & Salakhutdinov, R. "Siamese neural networks for one-shot image recognition." *ICML DL Workshop*, 2015; Hadsell, R., Chopra, S. & LeCun, Y. "Dimensionality reduction by learning an invariant mapping." *CVPR*, 2006.

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
