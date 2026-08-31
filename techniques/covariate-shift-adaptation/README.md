# Covariate-Shift Adaptation (Reference Ch 29 UQ)

**Covariate shift**: `p_test(x) ≠ p_train(x)` but `p_test(y | x) =
p_train(y | x)`. Shimodaira (2000) showed that importance-weighting the
training log-likelihood by the **density ratio** `w(x) = p_test(x) /
p_train(x)` recovers a consistent estimator of the target-distribution
risk when the model is misspecified.

## Density-ratio estimator via pooled classifier (Bickel 2007)

Label all training samples 0 and all test samples 1, fit any probabilistic
classifier `p̂(y = 1 | x)`, and use

```
ŵ(x) = ( p̂(y=1 | x) / p̂(y=0 | x) ) · ( n_train / n_test )
```

Then reweight the downstream loss:

```
E_test[ ℓ(f(x), y) ]  =  E_train[ w(x) · ℓ(f(x), y) ]
```

## Estimator families

- **Pooled classifier trick** (Bickel-Bruckner-Scheffer 2007) — the demo
  here; only needs a good classifier for train-vs-test.
- **KLIEP** (Sugiyama 2008) — direct density-ratio estimation minimising
  KL to a mixture-of-basis-functions ratio; robust for higher-dim `x`.
- **uLSIF / RuLSIF** (Kanamori 2009, Yamada 2011) — squared-loss ratio;
  closed-form + robust variants.
- **KMM — Kernel Mean Matching** (Huang 2007) — match feature means in
  RKHS; convex QP.

## When to use

- **Deployment population differs from training** (older cohorts, new
  hospitals, new years, new geographies).
- **Model is misspecified** — the demo shows a 46 % test-MSE reduction
  when a linear model fits a quadratic truth.
- **Well-specified model + no shift** — IW does very little; keep OLS.

## When NOT to use

- **Support mismatch** — if `p_train` gives near-zero mass in regions of
  `p_test`, the weights explode and variance dominates bias reduction.
  Clip weights or use robust variants (RuLSIF, doubly-robust).
- **Concept shift** (`p(y | x)` changes) — IW does not fix this; need
  online adaptation.
- **Label shift** (`p(y)` changes) — use BBSE / RLLS instead.

## Files

- `python/covariate_shift_adaptation.py` — from-scratch pooled-classifier
  logistic regression → density-ratio weights → weighted OLS on a
  **misspecified linear model** fitting a quadratic truth with x biased
  low in training, high in test. Result: **plain OLS MSE 13.61 →
  IW-OLS MSE 7.32 (46.2 % reduction)**.
- `r/covariate_shift_adaptation.R` — `densratio` R package (KLIEP,
  uLSIF, RuLSIF); Python `adapt` toolkit alternatives.

## Assumptions & caveats

- **Support overlap** — a hard prerequisite; check `w_max` before trusting.
- **Weight clipping** — capping at `w ≤ w_max` (say 10) improves
  finite-sample stability at the cost of a small bias.
- **Effective sample size** = `(Σw)² / Σw²`; if this is a small fraction
  of `n_train`, the IW estimator has very high variance.
- **Density-ratio fit dominates error** — a mis-specified ratio is
  worse than no weighting; validate on a held-out test-similar subset.
- **Conformal prediction under shift** — weighted split conformal
  (Tibshirani et al. 2019) uses the same `w(x)` to restore coverage.

## Related in this repo

- `conformal-prediction`, `conformal-classification` — the weighted
  variants use exactly this density ratio.
- `class-imbalance` — a special-case reweighting.
- `linear-regression` / `logistic-regression` — the base models that
  gain from re-weighting when misspecified.
- `epistemic-aleatoric` — the shift contributes to *epistemic* uncertainty.

## Run

```
python techniques/covariate-shift-adaptation/python/covariate_shift_adaptation.py
Rscript techniques/covariate-shift-adaptation/r/covariate_shift_adaptation.R
```

**Refs:** Shimodaira, H. "Improving predictive inference under covariate shift by weighting the log-likelihood function." *JSPI*, 2000; Sugiyama, M., Suzuki, T. & Kanamori, T. *Density Ratio Estimation in Machine Learning*, Cambridge U.P., 2012; Bickel, S., Brückner, M. & Scheffer, T. "Discriminative learning for differing training and test distributions." *ICML*, 2007; Tibshirani, R.J. et al. "Conformal prediction under covariate shift." *NeurIPS*, 2019.

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
