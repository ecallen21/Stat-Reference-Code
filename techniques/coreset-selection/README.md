# Coreset Selection (Reference §47.44)

Feldman & Langberg (2011); Mirzasoleiman et al (2020). A **coreset**
S ⊆ D with weights w is a compressed proxy such that for every
hypothesis f,

    |L(f; S, w) − L(f; D)| ≤ ε · L(f; D).

For k-means, sensitivity sampling picks x_i with probability ∝ its
worst-case share of clustering cost. Yields O(k log n / ε²)
size coreset with (1 + ε)-approximation guarantees.

## Files

- `python/coreset_selection.py` — Feldman-Langberg-style
  sensitivity coreset for k-means: k-means++ seeds → per-point
  sensitivity → weighted sampling. Demo (n=5000, k=5 clusters in
  2D):
  - m=  50: cost error 24.1 %
  - m= 200: cost error  4.0 %
  - m= 500: cost error  0.4 %
- `r/coreset_selection.R` — `coreset` (limited), `submodlib` in
  Python; from-scratch here.

## When to use

- **Scaling k-means, GMM, linear regression** to billions of rows.
- **Streaming / online learning** — merge-and-reduce coresets.
- **Active learning + budget** — pre-select an informative subset
  to label.
- **Model retraining pipelines** — retrain on coresets for speed.

## When NOT to use

- **Small n** — no scaling benefit; use full data.
- **Complex hypothesis class** (deep nets) — sensitivity bounds
  vacuous; use gradient-matching (CRAIG) instead.
- **Non-decomposable losses** — AUC, ranking — Feldman-Langberg
  framework doesn't directly apply.

## Assumptions & caveats

- **Decomposable loss** L(f; D) = Σ ℓ(f; x_i) — key assumption.
- **Sensitivity bound** — the ratio ℓ(f; x_i) / L(f; D) — must be
  bounded uniformly in f; needs a good pilot solution.
- **Random seed sensitivity** — coreset varies per draw; report
  multiple runs.
- **Weight variance** — 1/(m p_i) has heavy right tail when p_i
  small; consider capped-weight variants.

## Related in this repo

- `k-means`, `k-medoids`, `gaussian-mixture-models` — target models
  for k-means coresets.
- `active-learning-query-strategies` — related budgeted-labelling
  toolset.
- `subsampling`, `bootstrap-optimism-correction` — general
  resampling.
- `random-projections`, `sparse-pca` — companion data-compression
  techniques.

## Run

```
python techniques/coreset-selection/python/coreset_selection.py
Rscript techniques/coreset-selection/r/coreset_selection.R
```

**Refs:** Feldman, D. & Langberg, M. "A unified framework for approximating and clustering data." *STOC*, pp. 569-578, 2011; Mirzasoleiman, B., Bilmes, J. & Leskovec, J. "Coresets for data-efficient training of machine learning models." *ICML*, 2020.

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
