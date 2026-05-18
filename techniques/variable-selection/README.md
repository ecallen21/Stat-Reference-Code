# Variable Selection (Reference §5.8, §5.18, §5.19, §5.36)

> ⚠️ **Read §5.36 first.** Classical stepwise selection is **problematic** for inference: inflated R², biased coefficients, p-values that aren't valid for the post-selection model. The methods here are implemented for **completeness and understanding** — the modern preference is regularization (`techniques/regularization` — lasso / elastic net) or pre-specified, theory-driven variable choice.

## Methods

| Method | Search | Cost | When useful |
|--------|--------|------|-------------|
| **Forward stepwise** | start empty; add the predictor that best improves the criterion; stop when none improve | `O(p²)` fits | When `p > n` or `p ≫ n` and you need a quick screen |
| **Backward stepwise** | start full; drop the predictor whose removal best improves the criterion | `O(p²)` fits | When `p < n` and you can fit the full model |
| **Bidirectional** | each step considers both add and drop | `O(p²)` fits | Compromise |
| **Best subsets** | evaluate **all** `2^p` models | `O(2^p)` fits | `p ≲ 20`; the most defensible classical choice |

## Selection criteria

| Criterion | Formula | Interpretation |
|-----------|---------|----------------|
| **AIC** | `n·log(RSS/n) + 2p` | Prediction-focused; tends to keep more terms |
| **BIC** | `n·log(RSS/n) + log(n)·p` | "True model" focused; heavier penalty, more parsimony |
| **Adjusted R²** | `1 − (1−R²)(n−1)/(n−p)` | The R² that already penalizes adding predictors |

Lower AIC/BIC is better; higher adjusted R² is better. AIC and BIC differ in the penalty constant: `2` vs. `log n`. For `n ≥ 8`, BIC penalizes more strongly.

## The post-selection inference problem

After choosing variables by their data-driven significance, the **printed p-values are invalid** — they're computed *as if* the model had been specified in advance. Some signs of trouble:

- **Inflated effect sizes** for kept variables (winner's curse).
- **Wildly unstable selection across bootstrap resamples**.
- **Inferential statements that don't hold** if you ran the same procedure on a held-out sample.

What to do instead, in roughly increasing order of effort:

1. Specify the model in advance based on theory; don't search.
2. Use **lasso / elastic net** (`techniques/regularization`) with **CV-chosen λ**.
3. For prediction-only goals, evaluate on a **held-out test set** — selection bias matters less.
4. For valid inference after data-driven selection, see **post-selection inference** literature (Berk et al. 2013, Lee et al. 2016) and the `selectiveInference` R package.

## Files
- `python/variable_selection.py` — from-scratch forward / backward stepwise, exhaustive best-subsets, AIC / BIC / adjusted-R² criteria. Demos on data where the true non-zero indices are `[0, 1, 4, 7]`; forward AIC recovers them, BIC drops the small `−0.3` effect.
- `r/variable_selection.R` — from-scratch versions + `stats::step` (forward / backward via AIC) and `leaps::regsubsets` (the canonical best-subsets implementation).
- PySpark: N/A — for large `n` and `p`, prefer regularization (`techniques/regularization/pyspark/`) which IS the modern variable-selection answer at scale.

## Run
```
python techniques/variable-selection/python/variable_selection.py
Rscript techniques/variable-selection/r/variable_selection.R
```

**Refs:** Akaike, "A New Look at the Statistical Model Identification," *IEEE TAC* 19(6), 716–723, 1974; Schwarz, "Estimating the Dimension of a Model," *Ann. Stat.* 6(2), 461–464, 1978; Harrell, *Regression Modeling Strategies*, 2nd ed., 2015 (Ch. 4–5 — the most prominent critique of stepwise); Berk, Brown, Buja, Zhang & Zhao, "Valid Post-Selection Inference," *Ann. Stat.* 41(2), 802–837, 2013.

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
