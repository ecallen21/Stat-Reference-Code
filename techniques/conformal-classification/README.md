# Conformal Classification — APS / RAPS (Reference Ch 29 UQ)

Turn any pretrained classifier into a **set-valued predictor with
finite-sample coverage guarantee** `P(y ∈ set(x)) ≥ 1 − α`, distribution
free, requiring only exchangeability of the calibration and test data.

## Adaptive Prediction Sets (APS, Romano-Sesia-Candès 2020)

For each `(x, y)`, sort the softmax probabilities in decreasing order and
compute the cumulative probability up to (and including) the true class:

```
s(x, y) = Σ_{k : p̂(k) ≥ p̂(y)} p̂(k)   −   U · p̂(y)      (U ~ U[0,1])
```

The `U` term is a **randomised tie-break** that gives *exact* (not just
conservative) marginal coverage.

**Calibrate**: on a held-out `(x_i, y_i)_{i=1..n_cal}` compute all `s_i`
and set

```
q̂ =  ⌈(n_cal + 1)(1 − α)⌉ / n_cal   quantile of {s_i}
```

**Predict**: for a new `x*`, include class `y` in the set iff `s(x*, y) ≤ q̂`.

## RAPS (Angelopoulos 2021)

APS produces very large sets on easy problems (thousands of classes).
RAPS adds a **regularisation** to the score:

```
s(x, y) = Σ_{k ≤ rank(y)} p̂_(k)(x)  +  λ · max(rank(y) − k_reg + 1, 0)
```

which discourages including deep-tail classes. Coverage guarantee is
preserved; average set size shrinks.

## When to use

- **Any softmax classifier** that you want a coverage guarantee on
  (medical AI, image classification, LLM answer selection).
- **Post-hoc** — you don't retrain the model.
- **Selective prediction** — take action only when `|set(x)| = 1`;
  otherwise abstain.

## Files

- `python/conformal_classification.py` — from-scratch **APS** with the
  randomised tie-break, calibration on a held-out set, empirical
  coverage vs target on a synthetic 4-class problem:
  **α = 0.05 → cov 0.952 (target 0.95); α = 0.10 → cov 0.894 (target 0.90);
  α = 0.20 → cov 0.790 (target 0.80)**. Mean set sizes 3.16 / 2.54 / 1.81.
- `r/conformal_classification.R` — `reticulate` + `mapie` / `puncc` /
  `torchcp`; `conformalClassification` R package for basic split-conformal.

## Assumptions & caveats

- **Exchangeability required** — the coverage guarantee breaks under
  covariate shift. See `covariate-shift-adaptation` for weighted
  conformal (Tibshirani 2019).
- **Coverage is marginal**, not conditional on `x`. Conditional
  coverage in general is provably impossible without extra assumptions
  (Vovk 2012, Lei & Wasserman 2014).
- **Set size vs miscoverage trade-off** — RAPS with a large `λ` shrinks
  sets but the same coverage guarantee still holds.
- **Randomised tie-break** — remove `U` if you need deterministic sets;
  coverage becomes conservative (≥ 1 − α exactly).
- **Split** vs **cross-conformal**: split needs a big calibration set;
  cross-conformal (CV+) uses all data but is more compute.

## Related in this repo

- `conformal-prediction` — the regression version + CQR + jackknife+.
- `jackknife-plus` — cross-conformal prediction intervals.
- `covariate-shift-adaptation` — weighted conformal under shift.
- `selective-prediction` — a natural user of APS sets (abstain when |set|>1).
- `calibration-scaling` — post-hoc calibration that combines well with
  APS to shrink set size.

## Run

```
python techniques/conformal-classification/python/conformal_classification.py
Rscript techniques/conformal-classification/r/conformal_classification.R
```

**Refs:** Romano, Y., Sesia, M. & Candès, E. "Classification with valid and adaptive coverage." *NeurIPS*, 2020; Angelopoulos, A. et al. "Uncertainty sets for image classifiers using conformal prediction (RAPS)." *ICLR*, 2021; Vovk, V., Gammerman, A. & Shafer, G. *Algorithmic Learning in a Random World*, Springer 2005.

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
