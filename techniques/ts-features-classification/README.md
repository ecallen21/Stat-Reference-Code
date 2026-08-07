# Time Series Features and Classification (Reference §13.39, §13.41)

Two closely related tasks:

## Feature extraction

Transform each time series into a **fixed-length feature vector** so downstream classifiers, regressors, and clustering can treat them as ordinary tabular data. Common features:

- **Distribution**: mean, sd, min, max, median, IQR, skewness, kurtosis.
- **Trend / structure**: OLS slope on time, ACF at lags 1, 2, 5.
- **Complexity**: number of peaks, approximate entropy, sample entropy.
- **Spectrum**: energy per frequency band, dominant period.
- **Wavelet**: energy per scale, entropy of the wavelet coefficients.

Production toolkits: **`tsfeatures`** in R, **`tsfresh`** in Python (which extracts ~800 candidate features).

## Time series classification

Two dominant families, both benchmarked in Bagnall et al. (2017):

- **Feature-based** — extract features, train a standard classifier (random forest / gradient-boosted trees / SVM).
- **Distance-based** — **1-NN with DTW distance** is the historical baseline that still competes with (and often beats) modern deep-learning approaches on many UCR datasets.

Newer strong performers: HIVE-COTE, ROCKET, InceptionTime.

## Files

- `python/ts_features_classification.py` — 14 hand-rolled features (distribution, trend, ACF, peaks, spectral energy) plus 1-NN classifiers using either those features or DTW distance. Demo (3-class sine / cosine / random-walk, N = 90): both classifiers hit 94% test accuracy.
- `r/ts_features_classification.R` — `tsfeatures::tsfeatures` for the standard R feature set.

## Choosing between feature-based and DTW

- **Feature-based**: fast at test time; works with unequal-length series; features are inspectable and explainable.
- **DTW-based**: needs `O(N M)` per pair at test time; captures fine-grained shape; typically 5–10% better on hard shape-based benchmarks.
- **Hybrid**: use features to prune candidates, then DTW to break ties.

## Assumptions & caveats

- Feature methods depend on the choice of features — no single set is universally best; `tsfresh` overkills and needs feature selection.
- DTW works on **shape**, not scale — z-normalize each series first if magnitude differences are noise.
- Classification benchmarks on the UCR archive; report accuracy, F1, and a proper train/test split.

## Run

```
python techniques/ts-features-classification/python/ts_features_classification.py
Rscript techniques/ts-features-classification/r/ts_features_classification.R
```

**Refs:** Fulcher, B.D. & Jones, N.S. "hctsa: a computational framework for automated time-series phenotyping using massive feature extraction." *Cell Systems* 5(5), 527–531, 2017; Bagnall, A. et al. "The great time series classification bake off." *Data Min. Knowl. Disc.* 31(3), 606–660, 2017.

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
