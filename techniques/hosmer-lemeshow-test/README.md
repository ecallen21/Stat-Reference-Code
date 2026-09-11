# Hosmer-Lemeshow Test (Reference §47.256)

Hosmer & Lemeshow (1980). Assess calibration of a binary
probability model by binning predictions into `g` deciles of
risk and comparing observed vs expected events per bin:

```
H = Σ_{k=1}^g (O_k - E_k)² / (E_k (1 - E_k / n_k))
```

Under H₀ (model calibrated), `H ~ χ²(g − 2)`. Small p-value
⇒ poor fit. Standard for logistic-regression diagnostics; use
with caution — sensitive to `g` and weak power in large `n`.

## Files

- `python/hosmer_lemeshow_test.py` — Chi-square implementation
  with decile pooling. Demo: n=3000, 8-feature LR on 70/30
  outcome. Well-fit model: H = 5.03, df = 8, p = 0.755 (do NOT
  reject H₀). Deliberately mis-calibrated `p² ` predictions:
  H = 316, df = 8, p ≈ 0 (REJECT). Decile table shows observed
  ≈ expected per bin for the well-fit model.
- `r/hosmer_lemeshow_test.R` — `ResourceSelection::hoslem.test`,
  `generalhoslem::logitgof`, `rms::val.prob`,
  `performance::performance_hosmer` (R); statsmodels manual,
  from-scratch (Python).

## When to use

- **Logistic-regression diagnostics** — the historical default.
- **Small-to-moderate n** (500–5000) where the χ² approximation
  is trustworthy.
- **Quick calibration screen** paired with a calibration plot.

## When NOT to use

- **Very large `n`** — the test has excessive power, flagging
  clinically-irrelevant miscalibration. Use calibration slope /
  intercept instead.
- **Machine-learning models** with non-decile risk
  distributions — bin choice is sensitive; use Spiegelhalter's
  z or calibration curves.
- **External-validation** primary evidence — pair with
  discrimination + calibration slope / intercept (Steyerberg
  approach).

## Assumptions & caveats

- **Bin count `g`** — 10 is convention; changing it changes the
  result. Report `g` alongside p.
- **Equal bin sizes** — ties in predicted probability can leave
  bins uneven; use `np.array_split` for row-wise split.
- **Approximate null** — the true null distribution has fewer
  df than assumed with ML-estimated coefficients; correction
  exists (Hosmer-Lemeshow adjusted).
- **Not a fitness metric** — high p is necessary but not
  sufficient for good calibration; visual plots complement.

## Related in this repo

- `calibration-plots` — visual companion (decile / loess
  smoothers).
- `discrimination-calibration` — the AUROC + calibration
  bundle.
- `model-recalibration` — how to fix a poor H-L fit.
- `proper-scoring-rules-crps` — proper-score alternative.

## Run

```
python techniques/hosmer-lemeshow-test/python/hosmer_lemeshow_test.py
Rscript techniques/hosmer-lemeshow-test/r/hosmer_lemeshow_test.R
```

**Refs:** Hosmer, D.W. and Lemeshow, S. "Goodness-of-fit tests for the multiple logistic regression model." *Comm. Stat. — Theory Methods*, 9(10): 1043-1069, 1980.

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
