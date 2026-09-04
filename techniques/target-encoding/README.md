# Target / Mean Encoding (Reference §41.11)

Micci-Barreca (2001). Encode a high-cardinality categorical feature
by replacing each level with a **shrunken estimate of the target
mean** at that level:

```
enc(level) = (n_l · ȳ_l + k · ȳ) / (n_l + k)
```

The smoothing `k` pulls rare-level estimates toward the grand mean.
Variants:

- **Leave-one-out** — remove the current row before computing its
  encoding; reduces target leakage.
- **Weight of Evidence (WOE)** — binary-target log-odds version
  `log(P(x|y=1) / P(x|y=0))`.

## When to use

- **High-cardinality categoricals** (zip code, product SKU) where
  one-hot encoding blows up the feature count.
- **Gradient boosting** — target encoding is often the single most
  effective feature for a tree model.

## When NOT to use

- **Small samples** — encoding is noisy; smoothing helps but
  regularisation is a shore-up, not a fix.
- **Time series without care** — encode using only past data to
  avoid leakage.

## Files

- `python/target_encoding.py` — smoothed target encoding (k=10) +
  leave-one-out + Weight of Evidence for binary y. Demo (5 levels
  A-E, n = 40/30/20/8/2): rare levels D (n=8, ȳ=−0.88) and E (n=2,
  ȳ=−2.87) are shrunk toward the grand mean; WOE for A = +2.5,
  E = −11.4.
- `r/target_encoding.R` — `vtreat::mkCrossFrameCExperiment`,
  `recipes::step_lencode_mixed`/`step_lencode_bayes`,
  `embed::step_woe` (R); `category_encoders.TargetEncoder`/
  `LeaveOneOutEncoder`/`WOEEncoder` (Python).

## Assumptions & caveats

- **Leakage** — target encoding on the training set + inference-
  time re-encoding must use the same fitted encoding, not the test
  set.
- **Cross-validation folds** — fit encoding on each fold's training
  portion separately when doing CV; libraries handle this.
- **Rare categories** — even smoothed, rare-level estimates are
  noisy; consider grouping into an "other" level.
- **Overfitting** — tree-based models can memorise target-encoded
  categoricals; regularise `k` or add noise (`category_encoders`
  `sigma`).

## Related in this repo

- `feature-hashing`, `dummy-contrast-coding` — alternative
  encodings.
- `bayesian-glms`, `james-stein-shrinkage` — the shrinkage rationale
  behind smoothing.

## Run

```
python techniques/target-encoding/python/target_encoding.py
Rscript techniques/target-encoding/r/target_encoding.R
```

**Refs:** Micci-Barreca, D. "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems." *ACM SIGKDD Explorations*, 2001.

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
