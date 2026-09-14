# Shapelet Transform (Reference §47.292)

Ye & Keogh (2009); Hills et al (2014). For each candidate
subsequence `s` of length `L`, compute its DISTANCE to every
training series (best-matching subseries), then rank
candidates by how well the distance SPLITS classes:

```
quality(s) = information_gain(class | dist_to_s)
```

Retain the top-K shapelets; transform each series to a K-dim
feature vector of shapelet distances; train any tabular
classifier.

## Files

- `python/shapelet_transform.py` — Info-gain-based shapelet
  discovery + logistic-regression classifier. Demo: 2-class
  60-obs sine problem (freq 1 vs freq 3). Top-3 shapelets
  identified with info-gain > 0.5; downstream logistic gets
  100% train/test accuracy. Shapelets are HUMAN-INSPECTABLE
  subseries of the training set.
- `r/shapelet_transform.R` — reticulate + `sktime` (R);
  `sktime.transformations.panel.shapelets`, `tslearn.shapelets`
  (learning-shapelets LTS), from-scratch (Python).

## When to use

- **Interpretable time-series classification** — shapelets
  are literal subseries you can plot.
- **Small n / medium series length** — brute force fits.
- **Domain-informed features** — provide seed shapelets from
  domain knowledge.

## When NOT to use

- **Very long series without pruning** — O(n · m · L)
  quality evaluation is expensive; use random-projection
  variants.
- **Where ROCKET beats it** — modern TSC benchmarks
  (Bagnall et al 2020) often place ROCKET / MultiRocket
  above shapelet transform on accuracy.
- **Non-euclidean semantics** — DTW-based shapelets
  (LFDS) may be more appropriate.

## Assumptions & caveats

- **Info-gain threshold-search** is O(n log n) per candidate;
  cache sorted arrays.
- **z-normalisation** — canonical; without it, amplitude
  becomes a feature.
- **Shapelet length pool** — 3-5 candidate lengths tuned by
  domain scale.
- **Learning shapelets** (Grabocka 2014) makes them
  differentiable; may generalise better than discrete search.

## Related in this repo

- `rocket-random-conv-features` — random-feature cousin.
- `matrix-profile-anomaly` — subsequence-similarity toolkit.
- `dynamic-time-warping` — elastic-distance baseline.
- `random-forest`, `gradient-boosting` — downstream
  classifiers after shapelet transform.

## Run

```
python techniques/shapelet-transform/python/shapelet_transform.py
Rscript techniques/shapelet-transform/r/shapelet_transform.R
```

**Refs:** Ye, L. and Keogh, E. "Time series shapelets: A new primitive for data mining." In *KDD*, pp. 947-956, 2009; Hills, J., Lines, J., Baranauskas, E., Mapp, J. and Bagnall, A. "Classification of time series by shapelet transformation." *Data Min. Knowl. Discov.*, 28(4): 851-881, 2014.

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
