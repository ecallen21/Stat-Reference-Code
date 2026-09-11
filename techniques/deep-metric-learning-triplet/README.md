# Deep Metric Learning with Triplet Loss (Reference §47.151)

Schroff, Kalenichenko & Philbin (2015, FaceNet). Learns an
embedding f(x) such that same-class points cluster together and
different-class points separate:

    L(a, p, n) = max(0, ‖f(a) − f(p)‖² − ‖f(a) − f(n)‖² + margin)

**Semi-hard mining** picks (a, p, n) where n is farther than p but
still within the margin — the strongest, most informative gradient
signal.

## Files

- `python/deep_metric_learning_triplet.py` — linear-embedding
  triplet demo with random-sampled triplets:
  - **Iris**, PCA 2-D kNN acc = 0.967   Triplet 2-D = 0.973.
  - **Noisy synthetic** (K = 3, d = 20, signal in first 2 dims):
    PCA 2-D collapses on noise (**acc 0.44**), while Triplet 2-D
    recovers the signal (**acc 1.000**).
- `r/deep_metric_learning_triplet.R` — recommends
  `pytorch-metric-learning`, `tensorflow-similarity`,
  `keras.losses.TripletSemiHardLoss`.

## When to use

- **Face / person / product re-identification** where the label
  space is huge and constantly changing.
- **One-shot / few-shot classification** via nearest-embedding
  matching.
- **Retrieval** where inter-item distance is the deliverable.

## When NOT to use

- **Small, closed label sets** — plain cross-entropy is simpler.
- **Highly imbalanced classes** without careful triplet sampling
  — collapse or ignoring rare classes is a common failure.
- **Objects with no natural notion of similarity**.

## Assumptions & caveats

- **Semi-hard mining** is critical — random triplets waste
  gradient; hardest-mining collapses the embedding.
- **Margin** ~ 0.2-1.0 empirically; tune per problem.
- **L2 embedding normalisation** (unit sphere) common in
  FaceNet-style pipelines.
- **Batch construction** — need enough per-class samples for
  in-batch mining to have candidates.

## Related in this repo

- `siamese-networks` — pairwise (contrastive) cousin.
- `simclr-contrastive`, `contrastive-learning` — self-supervised
  cousins.
- `barlow-twins`, `byol-simsiam` — non-contrastive SSL
  alternatives.
- `bertscore-chrf-metrics` — embedding-based text similarity.

## Run

```
python techniques/deep-metric-learning-triplet/python/deep_metric_learning_triplet.py
Rscript techniques/deep-metric-learning-triplet/r/deep_metric_learning_triplet.R
```

**Refs:** Schroff, F., Kalenichenko, D. & Philbin, J. "FaceNet: A unified embedding for face recognition and clustering." *CVPR*, 2015; Hermans, A., Beyer, L. & Leibe, B. "In defense of the triplet loss for person re-identification." *arXiv:1703.07737*, 2017.

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
